"""WheelAgent: per-ticker wheel state machine driven by chain snapshots and order fills.

SEARCHING -> CSP_OPEN -> ASSIGNED -> CC_OPEN -> CALLED_AWAY -> SEARCHING
The agent only emits Signals; state changes when the ExecutionAgent reports a fill.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Optional, Sequence

from ..events import (
    AgentStatus, ChainSnapshot, IVMetrics, Leg, OptionQuote, OrderStatus, OrderUpdate, RiskDecision, Signal,
    SignalAction, Verdict,
)
from ..strategies.wheel import (
    Candidate, InvalidTransition, ManagementAction, WheelPosition, WheelState, entry_ok, management_actions,
    select_candidate,
)
from .base import BaseAgent, Context, IVGate

MULT = 100


def find_quote(quotes: Sequence[OptionQuote], right: str, strike: float, expiry: date) -> Optional[OptionQuote]:
    return next((q for q in quotes if q.right == right and q.strike == strike and q.expiry == expiry), None)


class WheelAgent(BaseAgent):
    name = "wheel"

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx)
        self.cfg = self.settings.wheel
        self.iv: dict[str, IVMetrics] = {}
        self.pending: dict[str, dict[str, Any]] = {}  # signal_id -> {symbol, kind, ...}
        self.inflight: dict[str, str] = {}  # symbol -> signal_id
        self.gate = IVGate(self, self.on_chain, self.settings.iv_wait_seconds)
        saved = {r["symbol"]: r for r in self.db.wheel_states()}
        self.positions: dict[str, WheelPosition] = {}
        for sym in self.settings.tickers:
            if sym in saved:
                self.positions[sym] = WheelPosition.model_validate(saved[sym]["payload"])
            else:
                self.positions[sym] = WheelPosition(symbol=sym, allow_cc_below_basis=self.cfg.allow_cc_below_cost_basis)

    # ---- plumbing ---------------------------------------------------------
    async def run(self) -> None:
        sub = self.bus.subscribe(ChainSnapshot, IVMetrics, OrderUpdate, RiskDecision)
        self.beat(AgentStatus.RUNNING)
        try:
            while True:
                ev = await self.next_event(sub)
                if ev is None:
                    continue
                if isinstance(ev, IVMetrics):
                    self.iv[ev.symbol] = ev
                    self.gate.on_iv(ev)
                elif isinstance(ev, ChainSnapshot):
                    self.gate.on_chain(ev)
                elif isinstance(ev, OrderUpdate):
                    await self.handle_safely(self.on_order, ev)
                elif isinstance(ev, RiskDecision):
                    await self.handle_safely(self.on_risk, ev)
        finally:
            sub.close()
            self.gate.cancel()

    def _persist(self, pos: WheelPosition) -> None:
        self.db.save_wheel_state(pos.symbol, pos.state.value, pos.shares, pos.cost_basis, pos.lifetime_premium,
                                 pos.cycles, pos.model_dump(mode="json"))

    # ---- decisions --------------------------------------------------------
    def on_chain(self, snap: ChainSnapshot) -> Optional[Signal]:
        pos = self.positions.get(snap.symbol)
        if pos is None:
            return None
        today = self.ctx.today()
        self._settle_expiry(pos, snap.underlying_price, today)
        if snap.symbol in self.inflight:
            return None
        sig = self.decide(pos, snap, today)
        if sig is not None:
            self.inflight[snap.symbol] = sig.id
            self._remember(sig)
            self.bus.publish(sig)
        return sig

    def _settle_expiry(self, pos: WheelPosition, spot: float, today: date) -> None:
        """Apply expiry outcome using the last stored close on/before expiry (assumption: reconcile with broker)."""
        leg = pos.leg
        if leg is None or leg.expiry >= today or pos.symbol in self.inflight:
            return
        rows = self.db.closes(pos.symbol, until=leg.expiry, limit=1)
        px = rows[-1][1] if rows else spot
        if leg.right == "P":
            if px < leg.strike:
                pos.assign_put(today)
            else:
                pos.expire_leg(today)
        elif px > leg.strike:
            pos.call_away(today)
            pos.restart()
        else:
            pos.expire_leg(today)
        self.log.info("expiry settled", extra={"symbol": pos.symbol, "state": pos.state.value, "settle_px": px})
        self._persist(pos)

    def _candidate(self, snap: ChainSnapshot, right: str, today: date, min_strike: Optional[float] = None) -> Optional[Candidate]:
        c = self.cfg
        drange = (c.put_delta_min, c.put_delta_max) if right == "P" else (c.call_delta_min, c.call_delta_max)
        return select_candidate(snap.quotes, right, snap.underlying_price, today, drange,  # type: ignore[arg-type]
                                (c.dte_min, c.dte_max), c.min_open_interest, c.max_spread_pct, min_strike)

    def _leg(self, q: OptionQuote, qty: int, price: float, signed_delta: Optional[float] = None) -> Leg:
        return Leg(right=q.right, strike=q.strike, expiry=q.expiry, quantity=qty, price=price,
                   delta=signed_delta if signed_delta is not None else q.delta, iv=q.iv)

    def decide(self, pos: WheelPosition, snap: ChainSnapshot, today: date) -> Optional[Signal]:
        n = self.cfg.contracts_per_trade
        iv = self.iv.get(snap.symbol)
        quality = {"data_quality": snap.quality.value, "iv_rank": iv.iv_rank if iv else None}

        if pos.state is WheelState.SEARCHING:
            ok, why = entry_ok(iv.iv_rank if iv else None, self.cfg)
            if not ok:
                return None
            c = self._candidate(snap, "P", today)
            if c is None:
                return None
            q = c.quote
            reason = (f"Sell {n}x {q.symbol} {q.strike:g}P {q.expiry} @ {c.mid:.2f}: |delta| {c.abs_delta:.2f}"
                      f"{' (est.)' if c.delta_estimated else ''}, {c.dte} DTE, ann. ROC {c.annualized_roc:.0%}; {why}")
            return snap.derive(Signal, strategy="wheel", symbol=snap.symbol, action=SignalAction.OPEN,  # type: ignore[return-value]
                               legs=[self._leg(q, -n, c.mid, -c.abs_delta)], reason=reason, score=c.annualized_roc,
                               capital_required=q.strike * MULT * n, max_loss=(q.strike - c.mid) * MULT * n,
                               meta={"kind": "open_csp", "delta_estimated": c.delta_estimated, **quality})

        if pos.state is WheelState.ASSIGNED:
            floor = None if self.cfg.allow_cc_below_cost_basis else pos.cost_basis
            c = self._candidate(snap, "C", today, floor)
            if c is None:
                return None
            q = c.quote
            reason = (f"Sell {pos.shares // MULT}x {q.symbol} {q.strike:g}C {q.expiry} @ {c.mid:.2f}: delta {c.abs_delta:.2f}"
                      f", {c.dte} DTE; strike >= cost basis {pos.cost_basis:.2f}")
            return snap.derive(Signal, strategy="wheel", symbol=snap.symbol, action=SignalAction.OPEN,  # type: ignore[return-value]
                               legs=[self._leg(q, -(pos.shares // MULT), c.mid, c.abs_delta)], reason=reason,
                               score=c.annualized_roc, meta={"kind": "open_cc", "delta_estimated": c.delta_estimated,
                                                             "cost_basis": pos.cost_basis, **quality})

        if pos.leg is not None:
            leg = pos.leg
            cur = find_quote(snap.quotes, leg.right, leg.strike, leg.expiry)
            mid = cur.mid if cur else None
            acts = management_actions(pos, snap.underlying_price, today, mid, self.cfg)
            if not acts:
                return None
            return self._management_signal(pos, snap, today, acts[0], cur, quality)
        return None

    def _management_signal(self, pos: WheelPosition, snap: ChainSnapshot, today: date, act: ManagementAction,
                           cur: Optional[OptionQuote], quality: dict) -> Optional[Signal]:
        leg = pos.leg
        assert leg is not None
        if cur is None or cur.mid is None:
            self.log.info("management needed but no quote for open leg", extra={"symbol": pos.symbol, "action": act.kind})
            return None
        buy_back = self._leg(cur, leg.contracts, cur.mid)
        if act.kind == "close_profit":
            return snap.derive(Signal, strategy="wheel", symbol=pos.symbol, action=SignalAction.CLOSE,  # type: ignore[return-value]
                               legs=[buy_back], reason=f"Close {pos.symbol} {leg.strike:g}{leg.right}: {act.reason}",
                               meta={"kind": "close", "trigger": act.kind, **quality})
        floor = None if (leg.right == "P" or self.cfg.allow_cc_below_cost_basis) else pos.cost_basis
        c = self._candidate(snap, leg.right, today, floor)
        if c is None:
            self.log.info("roll wanted but no candidate", extra={"symbol": pos.symbol, "trigger": act.kind})
            return None
        sign = -1 if leg.right == "P" else 1
        return snap.derive(Signal, strategy="wheel", symbol=pos.symbol, action=SignalAction.ROLL,  # type: ignore[return-value]
                           legs=[buy_back, self._leg(c.quote, -leg.contracts, c.mid, sign * c.abs_delta)],
                           reason=(f"Roll {pos.symbol} {leg.strike:g}{leg.right} {leg.expiry} -> {c.quote.strike:g}"
                                   f"{leg.right} {c.quote.expiry}: {act.reason}"),
                           score=c.annualized_roc, meta={"kind": "roll", "trigger": act.kind, **quality})

    # ---- fills ------------------------------------------------------------
    def on_risk(self, d: RiskDecision) -> None:
        if d.verdict is Verdict.BLOCK:
            self._clear(d.signal_id)

    def _clear(self, signal_id: str) -> None:
        for sym, sid in list(self.inflight.items()):
            if sid == signal_id:
                del self.inflight[sym]

    def _remember(self, sig: Signal) -> None:
        self.pending[sig.id] = {"symbol": sig.symbol, "kind": sig.meta.get("kind"), "legs": sig.legs}

    def on_order(self, u: OrderUpdate) -> None:
        if u.signal_id not in self.inflight.values():
            return
        if u.status in (OrderStatus.REJECTED, OrderStatus.CANCELLED):
            self._clear(u.signal_id)
            self.pending.pop(u.signal_id, None)
            return
        if u.status is not OrderStatus.FILLED:
            return
        info = self.pending.pop(u.signal_id, None)
        self._clear(u.signal_id)
        if info is None:
            return
        pos, today = self.positions[info["symbol"]], self.ctx.today()
        legs: list[Leg] = info["legs"]
        fills = list(u.leg_fills) or [u.fill_price or (l.price or 0.0) for l in legs]
        try:
            kind = info["kind"]
            if kind == "open_csp":
                pos.open_csp(legs[0].strike, legs[0].expiry, fills[0], abs(legs[0].quantity), today)  # type: ignore[arg-type]
            elif kind == "open_cc":
                pos.open_cc(legs[0].strike, legs[0].expiry, fills[0], today, abs(legs[0].quantity))  # type: ignore[arg-type]
            elif kind == "close":
                pos.close_leg(fills[0], today)
            elif kind == "roll":
                pos.roll(legs[1].strike, legs[1].expiry, fills[1], fills[0], today)  # type: ignore[arg-type]
        except InvalidTransition as exc:
            self.degrade(exc, f"applying fill for {u.signal_id}")
            return
        self._persist(pos)
