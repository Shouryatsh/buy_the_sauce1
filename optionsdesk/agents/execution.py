"""ExecutionAgent: turns approved signals into orders. Paper by default, kill switch, full audit trail.

Safety rules enforced here (not in the strategies):
  * BLOCKed signals are never executed.
  * Kill switch (persisted in the DB) rejects every order until released.
  * mode=live additionally needs a manual confirmation (with explicit live acknowledgement).
  * IBKR backend refuses to trade if the connected account type does not match `mode`.
  * Every order row stores the reason and correlation id of the signal that caused it.
"""

from __future__ import annotations

import asyncio
from typing import Optional, Protocol, Sequence

from ..events import (
    AgentStatus, KillSwitchChanged, OrderIntent, OrderStatus, OrderUpdate, RiskDecision, Signal, Verdict, new_id,
)
from ..logs import correlation
from ..settings import Mode, Settings
from .base import BaseAgent, Context


class OrderRejected(Exception):
    pass


class OrderUnfilled(Exception):
    pass


class Broker(Protocol):
    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...
    def is_connected(self) -> bool: ...
    async def place(self, intent: OrderIntent, order_id: str) -> list[float]:
        """Return the per-share fill price for each leg, in order. Raise OrderRejected/OrderUnfilled."""
        ...
    async def cancel_all(self) -> None: ...


class SimulatedBroker:
    """Fills every leg immediately at its limit price, half a half-spread worse. Never touches IBKR."""

    def __init__(self) -> None:
        self._connected = False
        self.placed: list[OrderIntent] = []

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    async def place(self, intent: OrderIntent, order_id: str) -> list[float]:
        self.placed.append(intent)
        fills = []
        for leg in intent.legs:
            px = leg.price or 0.0
            slip = 0.5 * max(0.025, 0.02 * px)
            fills.append(round(px + slip if leg.quantity > 0 else max(px - slip, 0.01), 4))
        return fills

    async def cancel_all(self) -> None:
        return None


def verify_account_matches_mode(accounts: Sequence[str], mode: Mode) -> None:
    """IBKR paper accounts start with 'DU'. Refuse to trade if that disagrees with the configured mode."""
    if not accounts:
        raise OrderRejected("no managed accounts reported by IBKR")
    paper_like = [a.upper().startswith("DU") for a in accounts]
    if mode is Mode.PAPER and not all(paper_like):
        raise OrderRejected(f"mode=paper but connected account(s) {list(accounts)} are not paper accounts")
    if mode is Mode.LIVE and any(paper_like):
        raise OrderRejected(f"mode=live but connected account(s) {list(accounts)} include a paper account")


class IBKRBroker:
    """ib_insync limit-order execution. Not exercised against live TWS in tests (see README "Unverified")."""

    def __init__(self, settings: Settings) -> None:
        from ib_insync import IB
        self._s = settings
        self._ib = IB()

    async def connect(self) -> None:
        c = self._s.ibkr
        await self._ib.connectAsync(c.host, self._s.ibkr_port, clientId=c.client_id + 1, timeout=c.connect_timeout)
        try:
            verify_account_matches_mode(self._ib.managedAccounts(), self._s.mode)
        except OrderRejected:
            self._ib.disconnect()
            raise

    async def disconnect(self) -> None:
        if self._ib.isConnected():
            self._ib.disconnect()

    def is_connected(self) -> bool:
        return bool(self._ib.isConnected())

    async def place(self, intent: OrderIntent, order_id: str) -> list[float]:
        from ib_insync import LimitOrder, Option
        trades = []
        for leg in intent.legs:
            if leg.right == "STK" or leg.expiry is None or leg.price is None:
                raise OrderRejected("only priced option legs are supported")
            contract = Option(intent.symbol, leg.expiry.strftime("%Y%m%d"), leg.strike, leg.right, "SMART")
            if not await self._ib.qualifyContractsAsync(contract):
                raise OrderRejected(f"cannot qualify {intent.symbol} {leg.strike}{leg.right} {leg.expiry}")
            order = LimitOrder("BUY" if leg.quantity > 0 else "SELL", abs(leg.quantity), round(leg.price, 2))
            order.orderRef = order_id
            trades.append(self._ib.placeOrder(contract, order))

        async def done(t) -> None:
            while not t.isDone():
                await asyncio.sleep(0.5)

        try:
            await asyncio.wait_for(asyncio.gather(*(done(t) for t in trades)),
                                   timeout=self._s.order_fill_timeout_seconds)
        except asyncio.TimeoutError:
            for t in trades:
                if not t.isDone():
                    self._ib.cancelOrder(t.order)
            raise OrderUnfilled(f"not filled within {self._s.order_fill_timeout_seconds:.0f}s; cancelled")
        if any(t.orderStatus.status != "Filled" for t in trades):
            raise OrderRejected("; ".join(f"{t.orderStatus.status}" for t in trades))
        return [float(t.orderStatus.avgFillPrice) for t in trades]

    async def cancel_all(self) -> None:
        if self._ib.isConnected():
            self._ib.reqGlobalCancel()


def make_broker(settings: Settings) -> Broker:
    return SimulatedBroker() if settings.execution_backend == "simulated" else IBKRBroker(settings)


class ExecutionAgent(BaseAgent):
    name = "execution"

    def __init__(self, ctx: Context, broker: Broker) -> None:
        super().__init__(ctx)
        self.broker = broker
        self._signals: dict[str, Signal] = {}
        self._handled: set[str] = set()
        self._awaiting: dict[str, tuple[OrderIntent, RiskDecision]] = {}  # order_id -> (intent, decision)
        self._tasks: set[asyncio.Task] = set()

    @property
    def paper(self) -> bool:
        return self.settings.mode is Mode.PAPER

    async def close(self) -> None:
        for t in list(self._tasks):
            t.cancel()
        await self.broker.disconnect()

    async def run(self) -> None:
        # Anything left awaiting confirmation by a previous process can no longer be confirmed.
        self.db.cancel_stale_confirmations()
        sub = self.bus.subscribe(Signal, RiskDecision)
        self.beat(AgentStatus.RUNNING)
        try:
            while True:
                ev = await self.next_event(sub)
                if ev is None:
                    continue
                if isinstance(ev, Signal):
                    self._signals[ev.id] = ev
                    if len(self._signals) > 500:
                        self._signals.pop(next(iter(self._signals)))
                elif isinstance(ev, RiskDecision):
                    await self.handle_safely(self.on_decision, ev)
        finally:
            sub.close()

    # ---- kill switch ------------------------------------------------------
    async def set_kill_switch(self, engaged: bool, reason: str = "") -> None:
        self.db.set_kill_switch(engaged, reason)
        self.bus.publish(KillSwitchChanged(engaged=engaged, reason=reason))
        self.log.warning("kill switch %s", "ENGAGED" if engaged else "released", extra={"reason": reason})
        if engaged:
            for oid in list(self._awaiting):
                self._finish(oid, OrderStatus.CANCELLED, "kill switch engaged")
            await self.broker.cancel_all()

    # ---- orders -------------------------------------------------------------
    def register_signal(self, sig: Signal) -> None:
        self._signals[sig.id] = sig

    async def on_decision(self, d: RiskDecision) -> Optional[str]:
        sig = self._signals.get(d.signal_id)
        if sig is None or d.signal_id in self._handled or d.verdict is Verdict.BLOCK:
            return None
        self._handled.add(d.signal_id)
        order_id = new_id()
        intent = sig.derive(OrderIntent, signal_id=sig.id, symbol=sig.symbol, strategy=sig.strategy, legs=sig.legs,  # type: ignore[assignment]
                            reason=f"{sig.reason} | risk={d.verdict.value}" + (f" ({'; '.join(d.reasons)})" if d.reasons else ""),
                            limit_price=None)
        self.db.save_order_intent(order_id, intent, OrderStatus.PENDING_CONFIRMATION.value, self.paper)  # type: ignore[arg-type]
        self._log_order("order created", order_id, intent)  # type: ignore[arg-type]

        if self.db.kill_switch_engaged():
            self._publish(order_id, intent, OrderStatus.REJECTED, "kill switch engaged")  # type: ignore[arg-type]
            return order_id
        if self.paper and self.settings.auto_execute_paper:
            self._spawn(order_id, intent, d)  # type: ignore[arg-type]
        else:
            self._awaiting[order_id] = (intent, d)  # type: ignore[assignment]
            self._publish(order_id, intent, OrderStatus.PENDING_CONFIRMATION, "awaiting manual confirmation")  # type: ignore[arg-type]
        return order_id

    async def confirm(self, order_id: str, acknowledge_live: bool = False) -> None:
        if order_id not in self._awaiting:
            raise KeyError(f"no order awaiting confirmation: {order_id}")
        if not self.paper and not acknowledge_live:
            raise PermissionError("live orders require acknowledge_live=True")
        if self.db.kill_switch_engaged():
            self._finish(order_id, OrderStatus.REJECTED, "kill switch engaged")
            return
        intent, d = self._awaiting.pop(order_id)
        self._spawn(order_id, intent, d)

    def reject(self, order_id: str, why: str = "rejected by user") -> None:
        if order_id not in self._awaiting:
            raise KeyError(f"no order awaiting confirmation: {order_id}")
        self._finish(order_id, OrderStatus.CANCELLED, why)

    def pending_confirmations(self) -> list[str]:
        return list(self._awaiting)

    # ---- internals ------------------------------------------------------------
    def _spawn(self, order_id: str, intent: OrderIntent, d: RiskDecision) -> None:
        t = asyncio.create_task(self._submit(order_id, intent))
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

    async def _submit(self, order_id: str, intent: OrderIntent) -> None:
        with correlation(intent.correlation_id):
            try:
                if self.db.kill_switch_engaged():  # re-check right before sending
                    raise OrderRejected("kill switch engaged")
                if not self.broker.is_connected():
                    await self.broker.connect()
                self._publish(order_id, intent, OrderStatus.SUBMITTED, "submitted")
                fills = await self.broker.place(intent, order_id)
                self._publish(order_id, intent, OrderStatus.FILLED, "filled", fills)
            except OrderUnfilled as exc:
                self._publish(order_id, intent, OrderStatus.CANCELLED, str(exc))
            except OrderRejected as exc:
                self._publish(order_id, intent, OrderStatus.REJECTED, str(exc))
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self.degrade(exc, f"order {order_id}")
                self._publish(order_id, intent, OrderStatus.REJECTED, f"{type(exc).__name__}: {exc}")

    def _finish(self, order_id: str, status: OrderStatus, detail: str) -> None:
        intent, _ = self._awaiting.pop(order_id)
        self._publish(order_id, intent, status, detail)

    def _publish(self, order_id: str, intent: OrderIntent, status: OrderStatus, detail: str,
                 fills: Optional[list[float]] = None) -> None:
        fill_price = fills[0] if fills and len(fills) == 1 else None
        self.db.update_order(order_id, status.value, detail, fill_price)
        self.bus.publish(OrderUpdate(correlation_id=intent.correlation_id, order_id=order_id, signal_id=intent.signal_id,
                                     status=status, detail=detail, paper=self.paper, fill_price=fill_price,
                                     leg_fills=fills or []))
        self._log_order(f"order {status.value}", order_id, intent, detail)

    def _log_order(self, msg: str, order_id: str, intent: OrderIntent, detail: str = "") -> None:
        with correlation(intent.correlation_id):
            self.log.info(msg, extra={"order_id": order_id, "signal_id": intent.signal_id, "symbol": intent.symbol,
                                      "paper": self.paper, "reason": intent.reason, "detail": detail,
                                      "legs": [l.model_dump(mode="json") for l in intent.legs]})
