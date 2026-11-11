"""
stockscreen/alerts.py — alert rules and the de-duplicating state machine.

Alert kinds
  low_price               passes the fundamental screen and at least one lookback is in the low-price zone
  healthy_pullback        opportunity class C
  reversal_confirmation   opportunity class D
  fundamental_deterioration  a stock that previously qualified (passed) no longer does

State machine, per (symbol, kind), advanced once per scan with the freshly
evaluated record:

  condition TRUE,  state inactive, cooldown over  -> FIRE (create alert, state active)
  condition TRUE,  state inactive, cooling down   -> suppressed (no alert, state stays inactive)
  condition TRUE,  state active                   -> nothing (same unchanged condition: deduplicated)
  condition FALSE, state active                   -> state inactive (the stock left the zone; re-entry may fire again)
  condition FALSE, state inactive                 -> nothing
  condition UNKNOWN (stale/insufficient/error)    -> nothing, state unchanged, never fires

"UNKNOWN" is deliberate: an alert is never sent when the underlying data is stale or
the condition cannot be verified, and missing data never counts as "left the zone".
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from . import store

KINDS = {
    "low_price": "Low-price alert",
    "healthy_pullback": "Healthy-pullback alert",
    "reversal_confirmation": "Reversal-confirmation alert",
    "fundamental_deterioration": "Fundamental deterioration alert",
}

UNKNOWN = None


def _condition(kind: str, ev: dict, cfg) -> Optional[bool]:
    """True / False, or UNKNOWN when the data can't support a decision."""
    row = ev["row"]
    status = row["status"]
    max_age = cfg.alert_max_data_age_days
    age = row.get("data_age_days")
    fresh = age is not None and age <= max_age and not row.get("stale") and not row.get("price_stale")

    if kind == "fundamental_deterioration":
        # "previously qualified" is tracked by the 'qualified' state, handled in process()
        if status == "PASSED":
            return False
        if status == "FAILED":
            return True if fresh else UNKNOWN
        return UNKNOWN                      # stale / insufficient / error: unverifiable

    # the three price-based kinds need a verifiable PASSED screen + fresh price data
    if status != "PASSED" or not fresh:
        return False if status == "FAILED" else UNKNOWN
    zone = row["price_zone"]
    if zone == "Insufficient history":
        return UNKNOWN
    low = zone in ("Deep value-zone candidate", "Low-price candidate")
    if kind == "low_price":
        return low
    if kind == "healthy_pullback":
        return row["opp_code"] == "C"
    if kind == "reversal_confirmation":
        return row["opp_code"] == "D"
    return UNKNOWN


def build_alert(kind: str, ev: dict) -> dict:
    row = ev["row"]
    pcts = [p for p in (row.get("pct_52w"), row.get("pct_2y"), row.get("pct_3y")) if p is not None]
    lows = row.get("zone_lookbacks") or ""
    if kind == "fundamental_deterioration":
        trigger = "no longer meets the fundamental screen"
        reason = row.get("reasons") or "fundamental criteria failed"
    else:
        trigger = f"in the low-price zone on: {lows}" if lows else "in the low-price zone"
        if row.get("pos_52w") is not None:
            trigger += f" (52-week range position {row['pos_52w']:.0f}%)"
        reason = row.get("opp_why") or trigger
    return {
        "symbol": row["symbol"], "name": row.get("name"), "kind": kind, "price": row.get("price"),
        "trigger_text": trigger, "percentile": min(pcts) if pcts else None, "fcf_yield": row.get("fcf_yield"),
        "fund_status": f"{row['status']} ({row.get('fund_trend')})", "tech_status": row.get("tech_trend"),
        "reason": reason, "data_asof": row.get("latest_quarter_end"), "price_asof": row.get("price_asof"),
    }


def process(evals: dict[str, dict], cfg, now: Optional[datetime] = None, path: Optional[str] = None) -> dict:
    """Advance the state machine for every evaluated record. Returns counts."""
    now = now or datetime.now()
    stats = {"fired": 0, "deduplicated": 0, "suppressed_cooldown": 0, "cleared": 0, "unverifiable": 0}
    cooldown = timedelta(days=cfg.alert_cooldown_days)
    from .universe import excluded
    ex = excluded()
    for sym, ev in evals.items():
        if ev.get("error") or sym.upper() in ex:
            continue
        row = ev["row"]
        # remember which stocks currently qualify so deterioration can be detected later
        q_state = store.get_alert_state(sym, "qualified", path)
        if row["status"] == "PASSED":
            if not q_state["active"]:
                store.set_alert_state(sym, "qualified", True, triggered=now.isoformat(timespec="seconds"), path=path)
        for kind in cfg.alert_kinds:
            cond = _condition(kind, ev, cfg)
            if kind == "fundamental_deterioration" and cond is True and not q_state["active"]:
                cond = False                # it never qualified, so it didn't deteriorate
            st = store.get_alert_state(sym, kind, path)
            if cond is UNKNOWN:
                stats["unverifiable"] += 1
                continue
            if cond:
                if st["active"]:
                    stats["deduplicated"] += 1
                    continue
                last = st["last_triggered"]
                if last and now - datetime.fromisoformat(last) < cooldown:
                    stats["suppressed_cooldown"] += 1
                    continue
                store.add_alert({**build_alert(kind, ev), "created_at": now.isoformat(timespec="seconds")}, path)
                store.set_alert_state(sym, kind, True, triggered=now.isoformat(timespec="seconds"), path=path)
                stats["fired"] += 1
                if kind == "fundamental_deterioration":      # it no longer qualifies
                    store.set_alert_state(sym, "qualified", False, cleared=now.isoformat(timespec="seconds"), path=path)
            else:
                if st["active"]:
                    store.set_alert_state(sym, kind, False, cleared=now.isoformat(timespec="seconds"), path=path)
                    stats["cleared"] += 1
    return stats
