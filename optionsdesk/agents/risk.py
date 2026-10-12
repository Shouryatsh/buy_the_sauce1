"""RiskAgent: portfolio-level checks. Approves, flags (amber) or blocks every signal."""

from __future__ import annotations

from typing import Optional

from ..events import AgentStatus, ChainSnapshot, Quote, RiskDecision, Signal, Verdict
from ..risk import RiskPosition, check_signal
from ..strategies.straddle import PayoffLeg
from ..strategies.wheel import WheelPosition
from .base import BaseAgent, Context


def build_book(db) -> list[RiskPosition]:
    """Current portfolio from persisted wheel state and open straddle positions (netted per symbol)."""
    book: dict[str, RiskPosition] = {}
    for row in db.wheel_states():
        wp = WheelPosition.model_validate(row["payload"])
        legs = []
        if wp.leg is not None:
            legs.append(PayoffLeg(right=wp.leg.right, strike=wp.leg.strike, qty=-wp.leg.contracts,
                                  price=wp.leg.premium, expiry=wp.leg.expiry))
        if wp.shares or legs:
            book[wp.symbol] = RiskPosition(symbol=wp.symbol, shares=wp.shares, legs=legs)
    for p in db.open_positions():
        if p["strategy"] != "straddle":
            continue
        pos = book.setdefault(p["symbol"], RiskPosition(symbol=p["symbol"]))
        pos.legs.extend(PayoffLeg.model_validate(l) for l in p["legs"])
    return list(book.values())


class RiskAgent(BaseAgent):
    name = "risk"

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx)
        self.spots: dict[str, float] = {}

    async def run(self) -> None:
        sub = self.bus.subscribe(Signal, Quote, ChainSnapshot)
        self.beat(AgentStatus.RUNNING)
        try:
            while True:
                ev = await self.next_event(sub)
                if ev is None:
                    continue
                if isinstance(ev, Quote):
                    self.spots[ev.symbol] = ev.price
                elif isinstance(ev, ChainSnapshot):
                    self.spots[ev.symbol] = ev.underlying_price
                elif isinstance(ev, Signal):
                    await self.handle_safely(self.on_signal, ev)
        finally:
            sub.close()

    def _spots_for(self, book: list[RiskPosition]) -> dict[str, float]:
        """Live spots, falling back to the last stored close for held tickers not polled yet."""
        spots = dict(self.spots)
        for p in book:
            if p.symbol not in spots:
                rows = self.db.closes(p.symbol, limit=1)
                if rows:
                    spots[p.symbol] = rows[-1][1]
        return spots

    def on_signal(self, sig: Signal) -> RiskDecision:
        self.db.save_signal(sig)
        book = build_book(self.db)
        res = check_signal(sig, book, self._spots_for(book), self.ctx.today(), self.settings.risk)
        decision = sig.derive(RiskDecision, signal_id=sig.id, verdict=res.verdict, reasons=res.reasons,  # type: ignore[assignment]
                              utilization={k: float(v) for k, v in res.utilization.items()})
        self.db.save_risk_decision(decision)  # type: ignore[arg-type]
        self.log.info("risk decision", extra={"signal_id": sig.id, "verdict": res.verdict.value, "reasons": res.reasons})
        self.bus.publish(decision)  # type: ignore[arg-type]
        return decision  # type: ignore[return-value]
