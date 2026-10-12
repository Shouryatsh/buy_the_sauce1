"""NarrativeAgent: plain-English summary of positions, signals and risk. Template-based (no LLM)."""

from __future__ import annotations

import asyncio
import json
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from ..db import Database
from ..events import AgentStatus
from ..strategies.wheel import WheelPosition
from .base import BaseAgent


def build_narrative(db: Database, today: date, kill_switch: bool = False) -> str:
    lines: list[str] = [f"Daily summary for {today.isoformat()}"]
    if kill_switch:
        lines.append("KILL SWITCH IS ENGAGED: no new orders will be sent.")

    states = [WheelPosition.model_validate(r["payload"]) for r in db.wheel_states()]
    active = [p for p in states if p.state.value != "SEARCHING"]
    if active:
        lines.append("Wheel positions:")
        for p in active:
            if p.leg:
                dte = (p.leg.expiry - today).days
                what = f"short {p.leg.strike:g}{p.leg.right} expiring {p.leg.expiry} ({dte} DTE)"
            else:
                what = "holding shares, no call open"
            basis = f", cost basis {p.cost_basis:.2f}" if p.cost_basis else ""
            lines.append(f"  - {p.symbol}: {p.state.value}, {what}{basis}; premium collected this cycle ${p.cycle_premium:,.0f}.")
    else:
        lines.append("No open wheel positions.")
    done = sum(p.cycles for p in states)
    pnl = sum(p.realized_pnl for p in states)
    if done:
        lines.append(f"Completed wheel cycles: {done}, realized P&L ${pnl:,.0f}.")

    straddles = [p for p in db.open_positions() if p["strategy"] == "straddle"]
    for p in straddles:
        lines.append(f"Straddle open on {p['symbol']} ({p['meta'].get('mode', '?')}), net premium ${p['open_premium']:,.0f}.")

    since = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    journal = [j for j in db.journal(500) if j["ts"] >= since]
    if journal:
        by_verdict: dict[str, int] = {}
        for j in journal:
            by_verdict[j["verdict"] or "pending"] = by_verdict.get(j["verdict"] or "pending", 0) + 1
        mix = ", ".join(f"{n} {v}" for v, n in sorted(by_verdict.items()))
        lines.append(f"Signals in the last 24h: {len(journal)} ({mix}).")
        for j in journal[:3]:
            lines.append(f"  - {j['strategy']} {j['symbol']} {j['action']}: {j['reason']} [{j['verdict'] or 'awaiting risk'}]")
        blocked = [j for j in journal if j["verdict"] == "block"]
        for j in blocked[:2]:
            lines.append(f"  - BLOCKED {j['symbol']}: {', '.join(json.loads(j['risk_reasons'] or '[]'))}")
    else:
        lines.append("No signals in the last 24h.")

    util = db.latest_utilization()
    if util:
        hot = sorted(((k, v) for k, v in util.items() if v >= 0.8), key=lambda kv: -kv[1])
        lines.append("Risk: " + (", ".join(f"{k} at {v:.0%} of limit" for k, v in hot) if hot else "all limits below 80% utilization."))

    bad = [h for h in db.agent_health() if h["status"] in ("error", "degraded")]
    if bad:
        lines.append("Agent issues: " + "; ".join(f"{h['agent']} {h['status']} ({h['last_error']})" for h in bad))
    return "\n".join(lines)


class NarrativeAgent(BaseAgent):
    name = "narrative"

    def latest(self) -> Optional[str]:
        return self.db.kv_get("narrative:latest")

    def generate(self) -> str:
        text = build_narrative(self.db, self.ctx.today(), self.db.kill_switch_engaged())
        self.db.kv_set("narrative:latest", text)
        self.db.kv_set("narrative:generated_at", datetime.now(timezone.utc).isoformat())
        return text

    async def run(self) -> None:
        self.beat(AgentStatus.RUNNING)
        while True:
            try:
                self.generate()
                self.beat(AgentStatus.RUNNING, ran=True)
            except Exception as exc:  # noqa: BLE001
                self.degrade(exc, "narrative")
            waited = 0.0
            while waited < self.settings.narrative_interval_seconds:
                step = min(self.settings.heartbeat_interval_seconds, self.settings.narrative_interval_seconds - waited)
                await asyncio.sleep(step)
                waited += step
                self.beat(None)
