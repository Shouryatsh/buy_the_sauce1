from datetime import date, timedelta

import pytest

from optionsdesk.events import Leg, Signal, SignalAction, Verdict
from optionsdesk.risk import RiskPosition, check_signal, compute_metrics, merge_signal, utilization
from optionsdesk.settings import RiskSettings
from optionsdesk.strategies.straddle import PayoffLeg

D0 = date(2025, 3, 3)
EXP = D0 + timedelta(35)
SPOTS = {"AAA": 100.0, "BBB": 50.0}
S = RiskSettings(max_capital_at_risk=50_000, max_ticker_concentration=0.30, max_net_delta=400,
                 max_net_vega=1_000, max_net_theta_abs=500, max_scenario_loss_pct=0.20, warn_utilization=0.8)


def L(right, strike, qty, price, iv=0.3):
    return PayoffLeg(right=right, strike=strike, qty=qty, price=price, expiry=EXP, iv=iv)


def sig(symbol, legs, action=SignalAction.OPEN, strategy="wheel"):
    return Signal(strategy=strategy, symbol=symbol, action=action, reason="t", legs=[
        Leg(right=l.right, strike=l.strike, expiry=l.expiry, quantity=l.qty, price=l.price, iv=l.iv) for l in legs])


def test_short_put_capital_is_strike_less_premium():
    m = compute_metrics([RiskPosition(symbol="AAA", legs=[L("P", 90, -1, 2.0)])], SPOTS, D0, S)
    assert m.capital_by_ticker["AAA"] == pytest.approx(90 * 100 - 200)
    assert not m.unbounded
    assert m.net_delta > 0 and m.net_theta > 0  # short put: long delta, collects theta


def test_covered_call_not_flagged_unbounded_and_capital_is_stock():
    pos = RiskPosition(symbol="AAA", shares=100, legs=[L("C", 105, -1, 1.5)])
    m = compute_metrics([pos], SPOTS, D0, S)
    assert not m.unbounded and m.capital_by_ticker["AAA"] == pytest.approx(10_000)


def test_naked_short_straddle_is_unbounded_with_tail_loss_estimate():
    pos = RiskPosition(symbol="AAA", legs=[L("C", 100, -1, 4), L("P", 100, -1, 4)])
    m = compute_metrics([pos], SPOTS, D0, S)
    assert m.unbounded == ["AAA"] and m.capital_by_ticker["AAA"] > 0


def test_scenario_grid_signs_and_zero_move():
    short_put = RiskPosition(symbol="AAA", legs=[L("P", 90, -1, 2.0)])
    long_straddle = RiskPosition(symbol="BBB", legs=[L("C", 50, 1, 2.5), L("P", 50, 1, 2.5)])
    m = compute_metrics([short_put, long_straddle], SPOTS, D0, S)
    assert m.scenario["AAA"][0.0] == pytest.approx(0, abs=1e-9)
    assert m.scenario["AAA"][-0.2] < 0 < m.scenario["AAA"][0.2]
    assert m.scenario["BBB"][-0.2] > 0 and m.scenario["BBB"][0.2] > 0  # long gamma wins on big moves
    assert m.scenario["TOTAL"][-0.1] == pytest.approx(m.scenario["AAA"][-0.1] + m.scenario["BBB"][-0.1])
    assert m.worst_scenario_pnl == min(m.scenario["TOTAL"].values())


def test_stock_pnl_in_scenario():
    m = compute_metrics([RiskPosition(symbol="AAA", shares=100)], SPOTS, D0, S)
    assert m.scenario["AAA"][-0.1] == pytest.approx(-1000) and m.net_delta == 100


def test_heatmap_groups_by_expiry():
    pos = RiskPosition(symbol="AAA", shares=100, legs=[L("P", 90, -1, 2.0)])
    m = compute_metrics([pos], SPOTS, D0, S)
    assert set(m.heatmap["AAA"]) == {"STK", EXP.isoformat()}


def test_merge_nets_closing_leg_and_adds_new():
    book = [RiskPosition(symbol="AAA", legs=[L("P", 90, -1, 2.0)])]
    closed = merge_signal(book, sig("AAA", [L("P", 90, 1, 0.5)], SignalAction.CLOSE))
    assert closed == []
    opened = merge_signal(book, sig("BBB", [L("P", 45, -1, 1.0)]))
    assert {p.symbol for p in opened} == {"AAA", "BBB"}


def test_small_trade_approved():
    r = check_signal(sig("AAA", [L("P", 90, -1, 2.0)]), [], SPOTS, D0, S)
    assert r.verdict is Verdict.APPROVE and r.utilization["capital"] < 0.5


def test_trade_over_concentration_blocked():
    # 9,800 / 50,000 = 19.6% (ok); two contracts = 39% > 30% limit
    assert check_signal(sig("AAA", [L("P", 90, -1, 2.0)]), [], SPOTS, D0, S).verdict is not Verdict.BLOCK
    r = check_signal(sig("AAA", [L("P", 90, -2, 2.0)]), [], SPOTS, D0, S)
    assert r.verdict is Verdict.BLOCK and any("concentration" in x for x in r.reasons)


def test_trade_over_total_capital_blocked():
    book = [RiskPosition(symbol="BBB", legs=[L("P", 45, -10, 1.0)])]  # 44k; AAA adds 9.8k -> 108%
    r = check_signal(sig("AAA", [L("P", 90, -1, 2.0)]), book, SPOTS, D0, S)
    assert r.verdict is Verdict.BLOCK and any("capital" in x for x in r.reasons)


def test_flag_near_limit():
    s = RiskSettings(max_capital_at_risk=10_000, max_ticker_concentration=1.0, max_net_delta=10_000,
                     max_net_vega=1e6, max_net_theta_abs=1e6, max_scenario_loss_pct=1.0)
    r = check_signal(sig("AAA", [L("P", 90, -1, 2.0)]), [], SPOTS, D0, s)  # 8,800/10,000 = 88%
    assert r.verdict is Verdict.FLAG


def test_risk_reducing_close_never_blocked_even_when_over_limit():
    big = [RiskPosition(symbol="AAA", legs=[L("P", 90, -3, 2.0)])]  # 29,400 > 30% of 50k conc.
    assert utilization(compute_metrics(big, SPOTS, D0, S), S)["concentration"] > 1.0
    close_one = sig("AAA", [L("P", 90, 1, 0.5)], SignalAction.CLOSE)
    r = check_signal(close_one, big, SPOTS, D0, S)
    assert r.verdict is not Verdict.BLOCK and any("still over" in x for x in r.reasons)


def test_missing_price_blocks():
    assert check_signal(sig("ZZZ", [L("P", 10, -1, 1.0)]), [], SPOTS, D0, S).verdict is Verdict.BLOCK
    book = [RiskPosition(symbol="ZZZ", shares=100)]
    assert check_signal(sig("AAA", [L("P", 90, -1, 2.0)]), book, SPOTS, D0, S).verdict is Verdict.BLOCK
