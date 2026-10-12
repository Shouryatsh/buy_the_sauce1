from datetime import date, timedelta

import pytest

from optionsdesk.events import OptionQuote
from optionsdesk.settings import WheelSettings
from optionsdesk.strategies.wheel import (
    CostBasisViolation, InvalidTransition, WheelPosition, WheelState, entry_ok,
    management_actions, select_candidate,
)

D0 = date(2025, 1, 6)


def pos(**kw) -> WheelPosition:
    return WheelPosition(symbol="XYZ", **kw)


def test_full_cycle_with_assignment_and_called_away():
    p = pos()
    p.open_csp(100, D0 + timedelta(30), 2.00, 1, D0)
    assert p.state is WheelState.CSP_OPEN and p.cycle_premium == 200
    p.assign_put(D0 + timedelta(30))
    assert p.state is WheelState.ASSIGNED and p.shares == 100
    assert p.cost_basis == pytest.approx(98.0)  # 100 strike - 2.00 premium
    p.open_cc(102, D0 + timedelta(60), 1.50, D0 + timedelta(31))
    assert p.state is WheelState.CC_OPEN and p.cost_basis == pytest.approx(96.5)
    rec = p.call_away(D0 + timedelta(60))
    assert p.state is WheelState.CALLED_AWAY and p.shares == 0
    # premium 350 + stock (102-100)*100 = 550
    assert rec["pnl"] == pytest.approx(550) and p.realized_pnl == pytest.approx(550)
    assert rec["assigned"] and rec["capital"] == 10_000
    p.restart()
    assert p.state is WheelState.SEARCHING and p.cycles == 1
    assert p.lifetime_premium == pytest.approx(350)


def test_csp_expires_worthless_returns_to_searching():
    p = pos()
    p.open_csp(50, D0 + timedelta(30), 0.80, 2, D0)
    rec = p.expire_leg(D0 + timedelta(30))
    assert p.state is WheelState.SEARCHING and rec["pnl"] == pytest.approx(160)
    assert rec["annualized_return"] == pytest.approx(160 / 10_000 * 365 / 30)


def test_close_csp_early_realizes_net():
    p = pos()
    p.open_csp(50, D0 + timedelta(30), 1.00, 1, D0)
    rec = p.close_leg(0.40, D0 + timedelta(10))
    assert rec["pnl"] == pytest.approx(60) and p.state is WheelState.SEARCHING


def test_close_cc_returns_to_assigned_and_keeps_shares():
    p = pos()
    p.open_csp(100, D0 + timedelta(30), 2.0, 1, D0)
    p.assign_put(D0)
    p.open_cc(101, D0 + timedelta(30), 1.0, D0)
    assert p.close_leg(0.2, D0 + timedelta(5)) is None
    assert p.state is WheelState.ASSIGNED and p.shares == 100
    assert p.cycle_premium == pytest.approx(200 + 100 - 20)


def test_roll_csp_accumulates_net_credit_and_stays_open():
    p = pos()
    p.open_csp(100, D0 + timedelta(30), 2.0, 1, D0)
    p.roll(97, D0 + timedelta(60), new_premium=1.8, close_cost=2.5, asof=D0 + timedelta(20))
    assert p.state is WheelState.CSP_OPEN and p.leg.strike == 97
    assert p.cycle_premium == pytest.approx(200 + (1.8 - 2.5) * 100)
    p.assign_put(D0 + timedelta(60))
    assert p.assignment_price == 97 and p.cost_basis == pytest.approx(97 - 1.30)


def test_cc_below_cost_basis_blocked_unless_enabled():
    p = pos()
    p.open_csp(100, D0 + timedelta(30), 2.0, 1, D0)
    p.assign_put(D0)
    with pytest.raises(CostBasisViolation):
        p.open_cc(97.5, D0 + timedelta(30), 1.0, D0)
    assert p.state is WheelState.ASSIGNED  # failed transition changes nothing
    p.open_cc(98.0, D0 + timedelta(30), 1.0, D0)  # exactly at basis is allowed

    q = pos(allow_cc_below_basis=True)
    q.open_csp(100, D0 + timedelta(30), 2.0, 1, D0)
    q.assign_put(D0)
    q.open_cc(95, D0 + timedelta(30), 1.0, D0)


def test_cc_roll_respects_floor():
    p = pos()
    p.open_csp(100, D0 + timedelta(30), 2.0, 1, D0)
    p.assign_put(D0)
    p.open_cc(102, D0 + timedelta(30), 1.0, D0)
    with pytest.raises(CostBasisViolation):
        p.roll(90, D0 + timedelta(60), 1.0, 0.5, D0 + timedelta(25))


@pytest.mark.parametrize("call,args", [
    ("assign_put", (D0,)), ("open_cc", (100, D0, 1.0, D0)), ("call_away", (D0,)), ("restart", ()),
    ("close_leg", (0.1, D0)), ("roll", (1, D0, 1, 1, D0)),
])
def test_invalid_transitions_from_searching(call, args):
    with pytest.raises(InvalidTransition):
        getattr(pos(), call)(*args)


def test_cannot_open_csp_twice():
    p = pos()
    p.open_csp(50, D0 + timedelta(30), 1.0, 1, D0)
    with pytest.raises(InvalidTransition):
        p.open_csp(50, D0 + timedelta(30), 1.0, 1, D0)


def test_state_roundtrips_through_json():
    p = pos()
    p.open_csp(100, D0 + timedelta(30), 2.0, 1, D0)
    q = WheelPosition.model_validate(p.model_dump(mode="json"))
    assert q == p


def test_total_annualized_return_capital_weighted():
    p = pos()
    p.open_csp(100, D0 + timedelta(30), 2.0, 1, D0)
    p.expire_leg(D0 + timedelta(30))
    assert p.total_annualized_return() == pytest.approx(200 / (10_000 * 30 / 365))


# ---- selection / management -------------------------------------------------
S = WheelSettings()


def q(strike, right="P", delta=-0.27, bid=1.95, ask=2.05, oi=500, dte=35, iv=0.3):
    return OptionQuote(symbol="XYZ", expiry=D0 + timedelta(dte), strike=strike, right=right,
                       bid=bid, ask=ask, delta=delta, iv=iv, open_interest=oi)


def pick(quotes, right="P", spot=100.0, **kw):
    args = dict(delta_range=(0.25, 0.30), dte_range=(30, 45), min_open_interest=100, max_spread_pct=0.10)
    args.update(kw)
    return select_candidate(quotes, right, spot, D0, **args)


def test_select_closest_delta_to_midpoint():
    c = pick([q(95, delta=-0.25), q(96, delta=-0.28), q(97, delta=-0.30)])
    assert c.quote.strike == 96 and c.abs_delta == pytest.approx(0.28)


def test_select_filters():
    assert pick([q(95, delta=-0.40)]) is None  # delta out of range
    assert pick([q(95, dte=10)]) is None  # dte out of range
    assert pick([q(105)]) is None  # ITM put
    assert pick([q(95, bid=1.0, ask=2.0)]) is None  # wide spread
    assert pick([q(95, oi=10)]) is None  # illiquid
    assert pick([q(95, oi=None)]) is None  # unknown OI rejected
    assert pick([q(95, right="C")]) is None  # wrong right


def test_call_selection_honours_min_strike():
    calls = [q(101, "C", delta=0.28), q(104, "C", delta=0.26)]
    assert pick(calls, "C", min_strike=103).quote.strike == 104
    assert pick(calls, "C", min_strike=110) is None


def test_missing_broker_delta_falls_back_to_estimate():
    c = pick([q(95, delta=None, iv=0.35)])
    # 95 put, 35d, 35% IV is ~0.285 delta
    assert c is not None and c.delta_estimated


def test_entry_ok_iv_rank_gate():
    assert entry_ok(10, S)[0] is False and entry_ok(50, S)[0] is True
    assert entry_ok(None, S)[0] is True
    assert entry_ok(None, WheelSettings(require_iv_rank=True))[0] is False


def test_management_profit_take_roll_and_tested():
    p = pos()
    p.open_csp(95, D0 + timedelta(40), 2.00, 1, D0)
    kinds = lambda spot, asof, mid: [a.kind for a in management_actions(p, spot, asof, mid, S)]  # noqa: E731
    assert kinds(105, D0 + timedelta(5), 0.9) == ["close_profit"]  # 55% captured
    assert kinds(105, D0 + timedelta(5), 1.5) == []
    assert kinds(105, D0 + timedelta(20), 1.5) == ["roll_dte"]  # 20 DTE <= 21
    assert kinds(96, D0 + timedelta(5), 3.0) == ["roll_tested"]  # 96 <= 95*1.02
    assert kinds(105, D0 + timedelta(41), 0.1) == ["close_profit"]  # expired: no DTE roll
