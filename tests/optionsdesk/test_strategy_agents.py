from datetime import date, timedelta

import pytest

from optionsdesk.agents.base import Context
from optionsdesk.agents.risk import RiskAgent, build_book
from optionsdesk.agents.straddle import StraddleAgent
from optionsdesk.agents.wheel import WheelAgent
from optionsdesk.backtest import synthetic_chain
from optionsdesk.bus import EventBus
from optionsdesk.db import Database
from optionsdesk.events import (
    ChainSnapshot, DataQuality, IVMetrics, OrderStatus, OrderUpdate, Quote, RiskDecision, Signal, SignalAction, Verdict,
)
from optionsdesk.settings import RiskSettings, Settings, StraddleSettings
from optionsdesk.strategies.wheel import WheelState

TODAY = date(2025, 6, 2)


def ctx_for(**kw) -> Context:
    s = Settings(tickers=["AAA"], heartbeat_interval_seconds=0.05, **kw)
    return Context(settings=s, bus=EventBus(), db=Database(":memory:"), today=lambda: TODAY)


def snap(spot=100.0, sigma=0.30, exps=(35,), symbol="AAA"):
    qs = synthetic_chain(symbol, spot, TODAY, sigma, [TODAY + timedelta(d) for d in exps], "CP", (0.7, 1.3), 0.045, [], 1.0)
    return ChainSnapshot(symbol=symbol, underlying_price=spot, quotes=qs, quality=DataQuality.DELAYED)


def ivm(**kw):
    base = dict(symbol="AAA", iv_current=0.30, iv_rank=50.0, iv_percentile=50.0, history_days=100, realized_vol=0.30)
    base.update(kw)
    return IVMetrics(**base)


def fill(sig, status=OrderStatus.FILLED, fills=None):
    return OrderUpdate(order_id="o1", signal_id=sig.id, status=status, leg_fills=fills or [l.price for l in sig.legs])


# ---- wheel -----------------------------------------------------------------

def test_wheel_opens_csp_at_target_delta_and_records_reasoning():
    ctx = ctx_for()
    agent = WheelAgent(ctx)
    agent.iv["AAA"] = ivm()
    s = snap()
    sig = agent.on_chain(s)
    assert isinstance(sig, Signal) and sig.action is SignalAction.OPEN and sig.strategy == "wheel"
    leg = sig.legs[0]
    assert leg.right == "P" and leg.quantity == -1 and leg.strike < 100
    assert 0.25 <= abs(leg.delta) <= 0.30 and 30 <= (leg.expiry - TODAY).days <= 45
    assert sig.correlation_id == s.correlation_id  # traceable back to the market-data snapshot
    assert sig.capital_required == pytest.approx(leg.strike * 100) and "ann. ROC" in sig.reason
    assert agent.on_chain(s) is None  # one signal in flight per symbol


def test_wheel_respects_iv_rank_filter():
    ctx = ctx_for()
    agent = WheelAgent(ctx)
    agent.iv["AAA"] = ivm(iv_rank=5.0)
    assert agent.on_chain(snap()) is None


def test_wheel_fill_moves_state_and_persists():
    ctx = ctx_for()
    agent = WheelAgent(ctx)
    agent.iv["AAA"] = ivm()
    sig = agent.on_chain(snap())
    agent.on_order(fill(sig, OrderStatus.SUBMITTED))
    assert agent.positions["AAA"].state is WheelState.SEARCHING  # not filled yet
    agent.on_order(fill(sig, fills=[1.85]))
    pos = agent.positions["AAA"]
    assert pos.state is WheelState.CSP_OPEN and pos.leg.premium == 1.85 and "AAA" not in agent.inflight
    saved = ctx.db.wheel_states()[0]
    assert saved["state"] == "CSP_OPEN" and saved["premium_collected"] == pytest.approx(185)
    # restart: state is reloaded
    assert WheelAgent(ctx).positions["AAA"].state is WheelState.CSP_OPEN


def test_wheel_rejected_or_blocked_clears_inflight():
    ctx = ctx_for()
    agent = WheelAgent(ctx)
    agent.iv["AAA"] = ivm()
    sig = agent.on_chain(snap())
    agent.on_order(fill(sig, OrderStatus.REJECTED))
    assert "AAA" not in agent.inflight and agent.positions["AAA"].state is WheelState.SEARCHING
    sig2 = agent.on_chain(snap())
    assert sig2 is not None
    agent.on_risk(sig2.derive(RiskDecision, signal_id=sig2.id, verdict=Verdict.BLOCK))
    assert "AAA" not in agent.inflight


def _open_csp(agent, strike=95.0, dte=35, premium=2.0):
    agent.positions["AAA"].open_csp(strike, TODAY + timedelta(dte), premium, 1, TODAY)


def test_wheel_takes_profit_at_50_percent():
    ctx = ctx_for()
    agent = WheelAgent(ctx)
    _open_csp(agent, 90.0, 35, premium=10.0)  # huge original premium => current mid is <50% of it
    sig = agent.on_chain(snap(exps=(35,)))
    assert sig.action is SignalAction.CLOSE and sig.legs[0].quantity == 1 and "captured" in sig.reason
    agent.on_order(fill(sig, fills=[0.5]))
    assert agent.positions["AAA"].state is WheelState.SEARCHING and agent.positions["AAA"].cycles == 1


def test_wheel_rolls_near_21_dte():
    ctx = ctx_for()
    agent = WheelAgent(ctx)
    s = snap(exps=(20, 35))
    cur_mid = next(q for q in s.quotes if q.right == "P" and q.strike == 95 and q.expiry == TODAY + timedelta(20)).mid
    _open_csp(agent, 95.0, 20, premium=cur_mid * 1.1)  # only ~9% captured: no profit-take
    sig = agent.on_chain(s)
    assert sig.action is SignalAction.ROLL and "21" in sig.reason
    assert [l.quantity for l in sig.legs] == [1, -1] and sig.legs[1].expiry == TODAY + timedelta(35)
    agent.on_order(fill(sig, fills=[cur_mid, 2.1]))
    pos = agent.positions["AAA"]
    assert pos.state is WheelState.CSP_OPEN and pos.leg.expiry == TODAY + timedelta(35) and pos.leg.premium == 2.1


def test_wheel_assignment_then_covered_call_above_cost_basis():
    ctx = ctx_for()
    agent = WheelAgent(ctx)
    agent.positions["AAA"].open_csp(100.0, TODAY - timedelta(1), 2.0, 1, TODAY - timedelta(36))
    ctx.db.upsert_close("AAA", TODAY - timedelta(1), 92.0)  # settled in the money
    s = snap(spot=93.0, exps=(35,))
    sig = agent.on_chain(s)
    pos = agent.positions["AAA"]
    assert pos.state is WheelState.ASSIGNED and pos.shares == 100 and pos.cost_basis == pytest.approx(98.0)
    assert sig is not None and sig.legs[0].right == "C" and sig.legs[0].strike >= 98.0  # never below cost basis
    agent.on_order(fill(sig, fills=[1.2]))
    assert pos.state is WheelState.CC_OPEN


def test_wheel_cc_returns_none_when_every_strike_below_cost_basis():
    ctx = ctx_for()
    agent = WheelAgent(ctx)
    p = agent.positions["AAA"]
    p.open_csp(130.0, TODAY - timedelta(1), 1.0, 1, TODAY - timedelta(36))
    ctx.db.upsert_close("AAA", TODAY - timedelta(1), 100.0)
    assert agent.on_chain(snap(spot=100.0)) is None  # all 25-30 delta calls are near 106-107 < basis 129
    assert p.state is WheelState.ASSIGNED


# ---- straddle --------------------------------------------------------------

def straddle_ctx(**st):
    kw = dict(mode="long", long_max_iv_rank=30, max_iv_over_rv=1.0, event_window_days_min=1, event_window_days_max=14)
    kw.update(st)
    return ctx_for(straddle=StraddleSettings(**kw))


def test_long_straddle_signal_has_breakevens_greeks_and_trace():
    ctx = straddle_ctx()
    agent = StraddleAgent(ctx)
    agent.events = {"AAA": [TODAY + timedelta(7)]}
    agent.iv["AAA"] = ivm(iv_rank=15.0, realized_vol=0.36)
    s = snap(exps=(10, 30))
    sig = agent.on_chain(s)
    assert sig is not None and sig.strategy == "straddle" and sig.correlation_id == s.correlation_id
    assert {l.right for l in sig.legs} == {"C", "P"} and all(l.quantity == 1 for l in sig.legs)
    assert all(l.expiry == TODAY + timedelta(10) for l in sig.legs)  # first expiry after the event
    be = sig.meta["breakevens"]
    assert len(be) == 2 and be[0] < 100 < be[1]
    assert sig.max_loss == pytest.approx(sig.meta["max_loss"]) and sig.max_loss > 0
    assert sig.meta["net_vega"] > 0 and sig.meta["net_theta"] < 0 and sig.meta["days_to_event"] == 7


@pytest.mark.parametrize("iv_kw,events", [
    ({"iv_rank": 80.0}, {"AAA": [TODAY + timedelta(7)]}),  # rich vol: long rejected
    ({"iv_rank": 15.0}, {}),  # no event ahead
    ({"iv_rank": 15.0, "realized_vol": 0.20}, {"AAA": [TODAY + timedelta(7)]}),  # IV/RV too high
])
def test_long_straddle_filters(iv_kw, events):
    agent = StraddleAgent(straddle_ctx())
    agent.events = events
    agent.iv["AAA"] = ivm(**{"realized_vol": 0.36, **iv_kw})
    assert agent.on_chain(snap(exps=(10, 30))) is None


def test_straddle_skips_illiquid_chain():
    agent = StraddleAgent(straddle_ctx())
    agent.events = {"AAA": [TODAY + timedelta(7)]}
    agent.iv["AAA"] = ivm(iv_rank=15.0, realized_vol=0.36)
    s = snap(exps=(10, 30))
    thin = s.model_copy(update={"quotes": [q.model_copy(update={"open_interest": 5}) for q in s.quotes]})
    assert agent.on_chain(thin) is None


def test_straddle_open_then_profit_exit_realizes_pnl():
    ctx = straddle_ctx()
    agent = StraddleAgent(ctx)
    agent.events = {"AAA": [TODAY + timedelta(7)]}
    agent.iv["AAA"] = ivm(iv_rank=15.0, realized_vol=0.36)
    sig = agent.on_chain(snap(exps=(10, 30)))
    agent.on_order(fill(sig))
    pos = agent.open_position("AAA")
    assert pos is not None and pos["open_premium"] > 0 and len(pos["legs"]) == 2
    # Vol spikes: straddle reprices far above entry -> profit target
    hot = snap(spot=108.0, sigma=0.60, exps=(10, 30))
    out = agent.on_chain(hot)
    assert out is not None and out.action is SignalAction.CLOSE and "profit target" in out.reason
    assert all(l.quantity == -1 for l in out.legs)
    agent.on_order(fill(out))
    assert agent.open_position("AAA") is None
    closed = ctx.db._one("SELECT * FROM positions WHERE strategy='straddle'")
    assert closed["status"] == "closed" and closed["realized_pnl"] > 0


def test_short_straddle_never_through_earnings():
    ctx = ctx_for(straddle=StraddleSettings(mode="short", short_min_iv_rank=60, min_iv_over_rv=1.1,
                                            min_edge_vs_expected_move=-1.0))
    agent = StraddleAgent(ctx)
    agent.iv["AAA"] = ivm(iv_rank=85.0, realized_vol=0.22, iv_current=0.30)
    agent.events = {"AAA": [TODAY + timedelta(15)]}
    assert agent.on_chain(snap(exps=(30,))) is None
    agent.events = {}
    sig = agent.on_chain(snap(exps=(30,)))
    assert sig is not None and all(l.quantity == -1 for l in sig.legs) and sig.max_loss is None


# ---- risk ------------------------------------------------------------------

async def test_risk_agent_records_signal_and_decision_with_same_correlation_id():
    ctx = ctx_for(risk=RiskSettings(max_capital_at_risk=100_000))
    wheel = WheelAgent(ctx)
    wheel.iv["AAA"] = ivm()
    risk = RiskAgent(ctx)
    risk.spots["AAA"] = 100.0
    sig = wheel.on_chain(snap())
    d = risk.on_signal(sig)
    assert d.verdict is Verdict.APPROVE and d.correlation_id == sig.correlation_id and d.signal_id == sig.id
    row = ctx.db.journal()[0]
    assert row["signal_id"] == sig.id and row["verdict"] == "approve" and row["reason"] == sig.reason


def test_risk_agent_blocks_over_limit_and_uses_book_from_db():
    ctx = ctx_for(risk=RiskSettings(max_capital_at_risk=15_000, max_ticker_concentration=0.5))
    wheel = WheelAgent(ctx)
    wheel.iv["AAA"] = ivm()
    wheel.positions["AAA"].open_csp(95.0, TODAY + timedelta(30), 2.0, 1, TODAY)
    wheel._persist(wheel.positions["AAA"])
    assert [p.symbol for p in build_book(ctx.db)] == ["AAA"]
    risk = RiskAgent(ctx)
    risk.spots["AAA"] = 100.0
    other = snap(symbol="BBB", spot=100.0)
    risk.spots["BBB"] = 100.0
    w2 = WheelAgent(Context(settings=ctx.settings.model_copy(update={"tickers": ["BBB"]}), bus=ctx.bus, db=ctx.db, today=ctx.today))
    w2.iv["BBB"] = ivm(symbol="BBB")
    sig = w2.on_chain(other)  # ~9.3k more on top of ~9.3k already at risk > 15k
    d = risk.on_signal(sig)
    assert d.verdict is Verdict.BLOCK and any("capital" in r for r in d.reasons)


def test_risk_agent_falls_back_to_stored_close_for_unpolled_book_symbol():
    ctx = ctx_for(risk=RiskSettings(max_capital_at_risk=100_000))
    wheel = WheelAgent(ctx)
    wheel.positions["AAA"].open_csp(95.0, TODAY + timedelta(30), 2.0, 1, TODAY)
    wheel._persist(wheel.positions["AAA"])
    ctx.db.upsert_close("AAA", TODAY, 100.0)
    risk = RiskAgent(ctx)  # no live spot for AAA yet
    assert risk._spots_for(build_book(ctx.db)) == {"AAA": 100.0}
