import asyncio
import json
import logging
from datetime import date, timedelta

import pytest

from optionsdesk.agents.base import Context
from optionsdesk.agents.execution import (
    ExecutionAgent, IBKRBroker, OrderRejected, OrderUnfilled, SimulatedBroker, make_broker,
    verify_account_matches_mode,
)
from optionsdesk.bus import EventBus
from optionsdesk.db import Database
from optionsdesk.events import Leg, OrderStatus, OrderUpdate, RiskDecision, Signal, SignalAction, Verdict
from optionsdesk.logs import JsonFormatter
from optionsdesk.settings import Mode, Settings

EXP = date(2025, 7, 11)


def make(mode="paper", **kw):
    extra = {"allow_live": True, "data_source": "ibkr", "execution_backend": "ibkr"} if mode == "live" else {}
    s = Settings(mode=mode, heartbeat_interval_seconds=0.05, **extra, **kw)
    return Context(settings=s, bus=EventBus(), db=Database(":memory:"))


def signal(**kw) -> Signal:
    base = dict(strategy="wheel", symbol="AAA", action=SignalAction.OPEN, reason="Sell 1x AAA 95P",
                legs=[Leg(right="P", strike=95.0, expiry=EXP, quantity=-1, price=2.0)])
    base.update(kw)
    return Signal(**base)


def decision(sig, verdict=Verdict.APPROVE, reasons=()):
    return sig.derive(RiskDecision, signal_id=sig.id, verdict=verdict, reasons=list(reasons))


class RecordingBroker(SimulatedBroker):
    def __init__(self, exc=None):
        super().__init__()
        self.exc, self.cancelled = exc, 0

    async def place(self, intent, order_id):
        if self.exc:
            raise self.exc
        return await super().place(intent, order_id)

    async def cancel_all(self):
        self.cancelled += 1


async def settle(agent):
    await asyncio.gather(*list(agent._tasks))


async def test_paper_order_fills_and_is_fully_logged(caplog):
    ctx = make()
    agent = ExecutionAgent(ctx, RecordingBroker())
    sub = ctx.bus.subscribe(OrderUpdate)
    sig = signal()
    agent.register_signal(sig)
    with caplog.at_level(logging.INFO, logger="optionsdesk.execution"):
        oid = await agent.on_decision(decision(sig, Verdict.FLAG, ["capital at 85% of limit"]))
        await settle(agent)
    statuses = [sub.queue.get_nowait().status for _ in range(sub.queue.qsize())]
    assert statuses == [OrderStatus.SUBMITTED, OrderStatus.FILLED]
    row = ctx.db.get_order(oid)
    assert row["status"] == "filled" and row["paper"] == 1 and row["correlation_id"] == sig.correlation_id
    assert "Sell 1x AAA 95P" in row["reason"] and "capital at 85%" in row["reason"] and row["fill_price"] < 2.0
    logged = [r for r in caplog.records if getattr(r, "order_id", None) == oid]
    assert logged and all(getattr(r, "reason", None) for r in logged)
    # structured log line carries the correlation id of the originating signal
    import io
    from optionsdesk.logs import correlation
    h = logging.StreamHandler(buf := io.StringIO())
    h.setFormatter(JsonFormatter())
    agent.log.addHandler(h)
    agent.log.setLevel(logging.INFO)
    agent._log_order("x", oid, agent_intent(ctx, oid))
    assert json.loads(buf.getvalue())["correlation_id"] == sig.correlation_id


def agent_intent(ctx, oid):
    from optionsdesk.events import OrderIntent
    row = ctx.db.get_order(oid)
    return OrderIntent.model_validate_json(row["payload"])


async def test_blocked_signal_is_never_executed():
    ctx = make()
    broker = RecordingBroker()
    agent = ExecutionAgent(ctx, broker)
    sig = signal()
    agent.register_signal(sig)
    assert await agent.on_decision(decision(sig, Verdict.BLOCK, ["too big"])) is None
    assert broker.placed == [] and ctx.db.list_orders() == []


async def test_decision_for_unknown_signal_or_duplicate_is_ignored():
    ctx = make()
    agent = ExecutionAgent(ctx, RecordingBroker())
    sig = signal()
    assert await agent.on_decision(decision(sig)) is None  # signal never registered
    agent.register_signal(sig)
    assert await agent.on_decision(decision(sig)) is not None
    assert await agent.on_decision(decision(sig)) is None
    await settle(agent)
    assert len(ctx.db.list_orders()) == 1


async def test_kill_switch_rejects_orders_and_never_reaches_broker():
    ctx = make()
    broker = RecordingBroker()
    agent = ExecutionAgent(ctx, broker)
    await agent.set_kill_switch(True, "test")
    assert broker.cancelled == 1
    sig = signal()
    agent.register_signal(sig)
    oid = await agent.on_decision(decision(sig))
    assert ctx.db.get_order(oid)["status"] == "rejected" and "kill switch" in ctx.db.get_order(oid)["detail"]
    assert broker.placed == []
    await agent.set_kill_switch(False)
    sig2 = signal()
    agent.register_signal(sig2)
    oid2 = await agent.on_decision(decision(sig2))
    await settle(agent)
    assert ctx.db.get_order(oid2)["status"] == "filled"


async def test_kill_switch_cancels_orders_awaiting_confirmation():
    ctx = make(auto_execute_paper=False)
    agent = ExecutionAgent(ctx, RecordingBroker())
    sig = signal()
    agent.register_signal(sig)
    oid = await agent.on_decision(decision(sig))
    assert agent.pending_confirmations() == [oid]
    await agent.set_kill_switch(True, "panic")
    assert agent.pending_confirmations() == [] and ctx.db.get_order(oid)["status"] == "cancelled"


async def test_live_mode_needs_manual_confirmation_and_acknowledgement():
    ctx = make("live")
    broker = RecordingBroker()
    agent = ExecutionAgent(ctx, broker)
    sig = signal()
    agent.register_signal(sig)
    oid = await agent.on_decision(decision(sig))
    assert broker.placed == [] and ctx.db.get_order(oid)["status"] == "pending_confirmation"
    assert ctx.db.get_order(oid)["paper"] == 0
    with pytest.raises(PermissionError):
        await agent.confirm(oid)
    assert broker.placed == []
    await agent.confirm(oid, acknowledge_live=True)
    await settle(agent)
    assert len(broker.placed) == 1 and ctx.db.get_order(oid)["status"] == "filled"


async def test_user_can_reject_pending_order_and_confirmation_is_single_use():
    ctx = make("live")
    agent = ExecutionAgent(ctx, RecordingBroker())
    sig = signal()
    agent.register_signal(sig)
    oid = await agent.on_decision(decision(sig))
    agent.reject(oid)
    assert ctx.db.get_order(oid)["status"] == "cancelled"
    with pytest.raises(KeyError):
        await agent.confirm(oid, acknowledge_live=True)


async def test_confirm_blocked_if_kill_switch_engaged_after_pending():
    ctx = make("live")
    broker = RecordingBroker()
    agent = ExecutionAgent(ctx, broker)
    sig = signal()
    agent.register_signal(sig)
    oid = await agent.on_decision(decision(sig))
    ctx.db.set_kill_switch(True, "late")  # flag flipped from elsewhere (e.g. CLI) without the agent noticing
    await agent.confirm(oid, acknowledge_live=True)
    assert broker.placed == [] and ctx.db.get_order(oid)["status"] == "rejected"


async def test_paper_can_also_require_confirmation():
    ctx = make(auto_execute_paper=False)
    broker = RecordingBroker()
    agent = ExecutionAgent(ctx, broker)
    sig = signal()
    agent.register_signal(sig)
    oid = await agent.on_decision(decision(sig))
    assert broker.placed == []
    await agent.confirm(oid)  # paper: no live acknowledgement needed
    await settle(agent)
    assert ctx.db.get_order(oid)["status"] == "filled"


@pytest.mark.parametrize("exc,status", [(OrderRejected("margin"), "rejected"), (OrderUnfilled("timeout"), "cancelled"),
                                        (RuntimeError("socket"), "rejected")])
async def test_broker_failures_become_terminal_order_states(exc, status):
    ctx = make()
    agent = ExecutionAgent(ctx, RecordingBroker(exc))
    sub = ctx.bus.subscribe(OrderUpdate)
    sig = signal()
    agent.register_signal(sig)
    oid = await agent.on_decision(decision(sig))
    await settle(agent)
    assert ctx.db.get_order(oid)["status"] == status
    last = [sub.queue.get_nowait() for _ in range(sub.queue.qsize())][-1]
    assert last.status.value == status and last.signal_id == sig.id


async def test_two_leg_roll_reports_each_leg_fill_in_order():
    ctx = make()
    agent = ExecutionAgent(ctx, RecordingBroker())
    sub = ctx.bus.subscribe(OrderUpdate)
    sig = signal(action=SignalAction.ROLL, legs=[
        Leg(right="P", strike=95.0, expiry=EXP, quantity=1, price=1.0),
        Leg(right="P", strike=93.0, expiry=EXP + timedelta(14), quantity=-1, price=2.0)])
    agent.register_signal(sig)
    await agent.on_decision(decision(sig))
    await settle(agent)
    filled = [sub.queue.get_nowait() for _ in range(sub.queue.qsize())][-1]
    assert filled.status is OrderStatus.FILLED and filled.fill_price is None
    assert filled.leg_fills[0] > 1.0 and filled.leg_fills[1] < 2.0  # buy pays up, sell gives up a little


async def test_run_marks_stale_pending_confirmations_cancelled():
    ctx = make("live")
    first = ExecutionAgent(ctx, RecordingBroker())
    sig = signal()
    first.register_signal(sig)
    oid = await first.on_decision(decision(sig))
    second = ExecutionAgent(ctx, RecordingBroker())
    task = asyncio.create_task(second.run())
    await asyncio.sleep(0.05)
    task.cancel()
    assert ctx.db.get_order(oid)["status"] == "cancelled" and "restart" in ctx.db.get_order(oid)["detail"]


async def test_run_loop_end_to_end_from_bus_events():
    ctx = make()
    agent = ExecutionAgent(ctx, RecordingBroker())
    task = asyncio.create_task(agent.run())
    await asyncio.sleep(0.02)
    sub = ctx.bus.subscribe(OrderUpdate)
    sig = signal()
    ctx.bus.publish(sig)
    ctx.bus.publish(decision(sig))
    final = None
    for _ in range(2):
        final = await sub.get(timeout=2)
    task.cancel()
    assert final.status is OrderStatus.FILLED and final.correlation_id == sig.correlation_id


def test_account_must_match_mode():
    verify_account_matches_mode(["DU123456"], Mode.PAPER)
    verify_account_matches_mode(["U1234567"], Mode.LIVE)
    with pytest.raises(OrderRejected):
        verify_account_matches_mode(["U1234567"], Mode.PAPER)  # paper mode pointed at a real account
    with pytest.raises(OrderRejected):
        verify_account_matches_mode(["DU123456"], Mode.LIVE)
    with pytest.raises(OrderRejected):
        verify_account_matches_mode([], Mode.PAPER)


async def test_default_backend_is_simulated_and_ibkr_needs_explicit_opt_in():
    assert isinstance(make_broker(Settings()), SimulatedBroker)
    assert Settings().execution_backend == "simulated" and Settings().mode is Mode.PAPER
    assert isinstance(make_broker(Settings(execution_backend="ibkr")), IBKRBroker)  # constructing never connects
