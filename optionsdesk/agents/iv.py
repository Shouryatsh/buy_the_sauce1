"""IVAgent: ATM IV, IV rank/percentile from stored history, straddle-implied expected move."""

from __future__ import annotations

import math
from datetime import date
from typing import Optional, Sequence

from ..events import AgentStatus, ChainSnapshot, IVMetrics, OptionQuote
from ..strategies.pricing import implied_vol, iv_percentile, iv_rank
from .base import BaseAgent, Context

TARGET_DTE = 30
MIN_DTE = 5


def _quote_iv(q: OptionQuote, spot: float, asof: date, r: float) -> Optional[float]:
    if q.iv and q.iv > 0:
        return q.iv
    T = (q.expiry - asof).days / 365.0
    return implied_vol(q.mid, spot, q.strike, T, r, q.right) if q.mid else None


def atm_snapshot(snap: ChainSnapshot, asof: date, r: float = 0.045) -> Optional[dict]:
    """ATM IV and straddle mid for the expiry nearest 30 DTE (at least 5 DTE out)."""
    spot = snap.underlying_price
    by_exp: dict[date, list[OptionQuote]] = {}
    for q in snap.quotes:
        if (q.expiry - asof).days >= MIN_DTE:
            by_exp.setdefault(q.expiry, []).append(q)
    for exp in sorted(by_exp, key=lambda e: abs((e - asof).days - TARGET_DTE)):
        qs = by_exp[exp]
        strikes = sorted({q.strike for q in qs if q.right == "C"} & {q.strike for q in qs if q.right == "P"},
                         key=lambda k: abs(k - spot))
        if not strikes:
            continue
        k = strikes[0]
        call = next(q for q in qs if q.right == "C" and q.strike == k)
        put = next(q for q in qs if q.right == "P" and q.strike == k)
        ivs = [v for v in (_quote_iv(call, spot, asof, r), _quote_iv(put, spot, asof, r)) if v]
        if not ivs:
            continue
        straddle = (call.mid + put.mid) if call.mid and put.mid else None
        return {"iv": sum(ivs) / len(ivs), "expiry": exp, "strike": k, "straddle": straddle}
    return None


def realized_vol(closes: Sequence[float], window: int = 60) -> Optional[float]:
    px = list(closes)[-(window + 1):]
    if len(px) < 21:
        return None
    rets = [math.log(b / a) for a, b in zip(px, px[1:]) if a > 0 and b > 0]
    if len(rets) < 20:
        return None
    mean = sum(rets) / len(rets)
    var = sum((x - mean) ** 2 for x in rets) / (len(rets) - 1)
    return math.sqrt(var * 252)


def compute_iv_metrics(snap: ChainSnapshot, asof: date, iv_hist: Sequence[float],
                       closes: Sequence[float]) -> Optional[dict]:
    atm = atm_snapshot(snap, asof)
    if atm is None:
        return None
    hist = list(iv_hist)
    rank, pct = iv_rank(atm["iv"], hist), iv_percentile(atm["iv"], hist)
    rv = realized_vol(closes)
    move = atm["straddle"]
    return {
        "symbol": snap.symbol, "iv_current": atm["iv"], "iv_rank": rank, "iv_percentile": pct,
        "history_days": len(hist), "expected_move_abs": move,
        "expected_move_pct": move / snap.underlying_price if move else None,
        "expected_move_expiry": atm["expiry"], "realized_vol": rv, "rv_days": min(len(closes), 61),
    }


class IVAgent(BaseAgent):
    name = "iv"

    async def run(self) -> None:
        sub = self.bus.subscribe(ChainSnapshot)
        self.beat(AgentStatus.RUNNING)
        try:
            while True:
                ev = await self.next_event(sub)
                if ev is not None:
                    await self.handle_safely(self.on_chain, ev)
        finally:
            sub.close()

    def on_chain(self, snap: ChainSnapshot) -> None:
        today = self.ctx.today()
        atm = atm_snapshot(snap, today)
        if atm is None:
            self.log.info("no usable ATM quotes", extra={"symbol": snap.symbol})
            return
        self.db.upsert_iv(snap.symbol, today, atm["iv"])
        hist = [v for _, v in self.db.iv_history(snap.symbol, until=today)]
        closes = [c for _, c in self.db.closes(snap.symbol, until=today)]
        fields = compute_iv_metrics(snap, today, hist, closes)
        if fields:
            self.bus.publish(snap.derive(IVMetrics, **fields))
