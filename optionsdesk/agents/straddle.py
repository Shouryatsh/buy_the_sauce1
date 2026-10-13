"""StraddleAgent: evaluates ATM straddles (long or short, optional wings) and manages exits."""

from __future__ import annotations

from datetime import date
from typing import Any, Optional

from ..earnings import load_earnings, next_event
from ..events import (
    AgentStatus, ChainSnapshot, IVMetrics, Leg, OrderStatus, OrderUpdate, RiskDecision, Signal, SignalAction,
    Verdict,
)
from ..strategies.pricing import expected_abs_move
from ..strategies.straddle import (
    MULT, PayoffLeg, breakevens, build_legs, evaluate_entry, exit_decision, max_profit_loss, net_greeks,
    net_value_per_share,
)
from .base import BaseAgent, Context, IVGate, SignalCooldown
from .wheel import find_quote


class StraddleAgent(BaseAgent):
    name = "straddle"

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx)
        self.cfg = self.settings.straddle
        self.iv: dict[str, IVMetrics] = {}
        self.events = load_earnings(self.settings.earnings_file)
        self.pending: dict[str, dict[str, Any]] = {}
        self.inflight: dict[str, str] = {}
        self.gate = IVGate(self, self.on_chain, self.settings.iv_wait_seconds)
        self.cooldown = SignalCooldown(self.settings.signal_cooldown_seconds)

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

    # ---- state ------------------------------------------------------------
    def open_position(self, symbol: str) -> Optional[dict]:
        return next((p for p in self.db.open_positions() if p["strategy"] == "straddle" and p["symbol"] == symbol), None)

    @staticmethod
    def _legs(pos: dict) -> list[PayoffLeg]:
        return [PayoffLeg.model_validate(l) for l in pos["legs"]]

    # ---- decisions --------------------------------------------------------
    def on_chain(self, snap: ChainSnapshot) -> Optional[Signal]:
        if snap.symbol not in self.settings.tickers or snap.symbol in self.inflight:
            return None
        today = self.ctx.today()
        pos = self.open_position(snap.symbol)
        sig = self._manage(pos, snap, today) if pos else self._enter(snap, today)
        if sig is not None and self.cooldown.active(SignalCooldown.key(sig)):
            return None
        if sig is not None:
            self.inflight[snap.symbol] = sig.id
            self.pending[sig.id] = {"symbol": snap.symbol, "kind": sig.meta["kind"], "legs": sig.legs,
                                    "position_id": pos["id"] if pos else sig.id, "mode": sig.meta.get("mode"),
                                    "key": SignalCooldown.key(sig)}
            self.bus.publish(sig)
        return sig

    def _enter(self, snap: ChainSnapshot, today: date) -> Optional[Signal]:
        s, mode = self.cfg, self.cfg.mode
        iv = self.iv.get(snap.symbol)
        if iv is None:
            return None
        event = next_event(self.events, snap.symbol, today)
        legs = build_legs(snap.quotes, snap.underlying_price, today, mode, s, event_date=event)
        if not legs:
            return None
        quotes = [find_quote(snap.quotes, l.right, l.strike, l.expiry) for l in legs]
        for q in quotes:
            if q is None or q.spread_pct is None or q.spread_pct > s.max_spread_pct or \
                    q.open_interest is None or q.open_interest < s.min_open_interest:
                return None  # illiquid or liquidity unknown
        spot = snap.underlying_price
        dte = (legs[0].expiry - today).days
        implied = legs[0].price + legs[1].price
        rv = iv.realized_vol
        realized = expected_abs_move(spot, rv, dte / 365.0) if rv else None
        d = evaluate_entry(mode, s, iv.iv_rank, iv.iv_current / rv if rv else None,
                           (event - today).days if event else None, dte, implied, realized)
        if not d.ok:
            return None
        mx_profit, mx_loss = max_profit_loss(legs)
        g = net_greeks(legs, spot, today)
        edge = ((realized - implied) / realized if mode == "long" else (implied - realized) / realized) if realized else 0.0
        strike = legs[0].strike
        reason = f"{mode.capitalize()} straddle {snap.symbol} {strike:g} {legs[0].expiry} ({dte} DTE)"
        reason += " with wings" if s.use_wings else ""
        if realized:
            reason += f": implied move {implied:.2f} vs RV move {realized:.2f}"
        flags = "; ".join(d.flags)
        return snap.derive(Signal, strategy="straddle", symbol=snap.symbol, action=SignalAction.OPEN,  # type: ignore[return-value]
                           legs=[Leg(right=l.right, strike=l.strike, expiry=l.expiry, quantity=l.qty, price=l.price, iv=l.iv)
                                 for l in legs],
                           reason=reason + (f" [{flags}]" if flags else ""), score=edge,
                           capital_required=(mx_loss or 0.0), max_loss=mx_loss,
                           meta={"kind": "open", "mode": mode, "breakevens": breakevens(legs), "max_profit": mx_profit,
                                 "max_loss": mx_loss, "net_delta": g.delta, "net_vega": g.vega, "net_theta": g.theta,
                                 "net_gamma": g.gamma, "implied_move": implied, "realized_move": realized,
                                 "iv_rank": iv.iv_rank, "days_to_event": (event - today).days if event else None,
                                 "entry_value": net_value_per_share(legs), "data_quality": snap.quality.value})

    def _manage(self, pos: dict, snap: ChainSnapshot, today: date) -> Optional[Signal]:
        legs = self._legs(pos)
        qs = [find_quote(snap.quotes, l.right, l.strike, l.expiry) for l in legs]
        if any(q is None or q.mid is None for q in qs):
            return None
        entry_value = net_value_per_share(legs)
        current = sum(l.qty * q.mid for l, q in zip(legs, qs))  # type: ignore[union-attr]
        dte = (legs[0].expiry - today).days
        why = exit_decision(entry_value, current, dte, self.cfg)
        if not why:
            return None
        close_legs = [Leg(right=l.right, strike=l.strike, expiry=l.expiry, quantity=-l.qty, price=q.mid, iv=q.iv)  # type: ignore[union-attr]
                      for l, q in zip(legs, qs)]
        return snap.derive(Signal, strategy="straddle", symbol=snap.symbol, action=SignalAction.CLOSE,  # type: ignore[return-value]
                           legs=close_legs, reason=f"Close {snap.symbol} straddle: {why}",
                           meta={"kind": "close", "trigger": why.split(":")[0], "data_quality": snap.quality.value})

    # ---- fills ------------------------------------------------------------
    def on_risk(self, d: RiskDecision) -> None:
        if d.verdict is Verdict.BLOCK:
            info = self.pending.pop(d.signal_id, None)
            if info is not None:
                self.cooldown.suppress(info["key"])
            self._clear(d.signal_id)

    def _clear(self, signal_id: str) -> None:
        for sym, sid in list(self.inflight.items()):
            if sid == signal_id:
                del self.inflight[sym]

    def on_order(self, u: OrderUpdate) -> None:
        if u.signal_id not in self.inflight.values():
            return
        if u.status in (OrderStatus.REJECTED, OrderStatus.CANCELLED):
            info = self.pending.pop(u.signal_id, None)
            if info is not None:
                self.cooldown.suppress(info["key"])
            self._clear(u.signal_id)
            return
        if u.status is not OrderStatus.FILLED:
            return
        info = self.pending.pop(u.signal_id, None)
        self._clear(u.signal_id)
        if info is None:
            return
        legs: list[Leg] = info["legs"]
        fills = list(u.leg_fills) or [l.price or 0.0 for l in legs]
        now = self.ctx.today().isoformat()
        if info["kind"] == "open":
            payoff = [PayoffLeg(right=l.right, strike=l.strike, qty=l.quantity, price=f, expiry=l.expiry, iv=l.iv)  # type: ignore[arg-type]
                      for l, f in zip(legs, fills)]
            self.db.save_position({
                "id": info["position_id"], "correlation_id": u.correlation_id, "symbol": info["symbol"],
                "strategy": "straddle", "status": "open", "legs": [l.model_dump(mode="json") for l in payoff],
                "opened_at": now, "open_premium": net_value_per_share(payoff) * MULT,
                "meta": {"mode": info["mode"]}})
        else:
            pos = next((p for p in self.db.open_positions() if p["id"] == info["position_id"]), None)
            if pos is None:
                return
            opened = self._legs(pos)
            # close legs mirror the open legs, so P&L per open leg = qty_open * (close_price - open_price)
            pnl = sum(l.qty * (f - l.price) for l, f in zip(opened, fills)) * MULT
            pos.update(status="closed", closed_at=now, realized_pnl=pnl, legs=[l.model_dump(mode="json") for l in opened])
            self.db.save_position(pos)
