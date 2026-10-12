from datetime import date, timedelta

import numpy as np
import pytest

from optionsdesk.events import OptionQuote
from optionsdesk.settings import StraddleSettings
from optionsdesk.strategies.pricing import bs_price
from optionsdesk.strategies.straddle import (
    PayoffLeg, breakevens, build_legs, choose_expiry, evaluate_entry, exit_decision,
    max_profit_loss, net_greeks, net_value_per_share, payoff_at_expiry, position_pnl,
)

D0 = date(2025, 3, 3)
EXP = D0 + timedelta(30)
S = StraddleSettings()


def leg(right, strike, qty, price, iv=0.3):
    return PayoffLeg(right=right, strike=strike, qty=qty, price=price, expiry=EXP, iv=iv)


LONG = [leg("C", 100, 1, 4.0), leg("P", 100, 1, 4.0)]
SHORT = [leg("C", 100, -1, 4.0), leg("P", 100, -1, 4.0)]
REV_IB = LONG + [leg("C", 105, -1, 2.0), leg("P", 95, -1, 2.0)]  # long straddle, short wings
IB = SHORT + [leg("C", 105, 1, 2.0), leg("P", 95, 1, 2.0)]  # short straddle, long wings


def test_long_straddle_payoff_breakevens_and_limits():
    assert payoff_at_expiry(LONG, [100, 92, 108, 120, 0]).tolist() == [-800, 0, 0, 1200, 9200]
    assert breakevens(LONG) == [92.0, 108.0]
    assert max_profit_loss(LONG) == (None, 800.0)  # unbounded upside, loss = debit


def test_short_straddle_is_mirror_image():
    spots = np.linspace(60, 140, 17)
    assert np.allclose(payoff_at_expiry(SHORT, spots), -payoff_at_expiry(LONG, spots))
    assert breakevens(SHORT) == [92.0, 108.0]
    assert max_profit_loss(SHORT) == (800.0, None)  # unbounded loss


def test_wings_define_risk_both_modes():
    assert net_value_per_share(REV_IB) == pytest.approx(4.0)  # 8 debit - 4 wing credit
    p, l = max_profit_loss(REV_IB)
    assert (p, l) == (100.0, 400.0)  # width 5*100 - 400 debit = 100; loss = debit
    assert breakevens(REV_IB) == [96.0, 104.0]

    p, l = max_profit_loss(IB)  # iron butterfly: credit 4, width 5
    assert (p, l) == (400.0, 100.0)
    assert breakevens(IB) == [96.0, 104.0]


def test_payoff_never_exceeds_reported_extremes():
    spots = np.linspace(0, 400, 4001)
    for legs in (REV_IB, IB, LONG, SHORT):
        v = payoff_at_expiry(legs, spots)
        p, l = max_profit_loss(legs)
        if p is not None:
            assert v.max() <= p + 1e-6
        if l is not None:
            assert v.min() >= -l - 1e-6


def test_net_greeks_signs():
    g = net_greeks(LONG, 100, D0)
    assert g.vega > 0 and g.theta < 0 and g.gamma > 0 and abs(g.delta) < 15
    s = net_greeks(SHORT, 100, D0)
    assert s.vega == pytest.approx(-g.vega) and s.theta > 0 and s.gamma < 0


def test_position_pnl_zero_at_model_price_and_vega_consistent():
    T = 30 / 365
    c, p = bs_price(100, 100, T, 0.045, 0.3, "C"), bs_price(100, 100, T, 0.045, 0.3, "P")
    legs = [leg("C", 100, 1, c), leg("P", 100, 1, p)]
    assert position_pnl(legs, 100, D0) == pytest.approx(0, abs=1e-6)
    up = position_pnl(legs, 100, D0, iv_shift=0.01)
    assert up == pytest.approx(net_greeks(legs, 100, D0).vega, rel=0.02)  # +1 vol point
    assert position_pnl(legs, 100, EXP, iv_shift=0) < 0  # at expiry at-the-money: lose the debit


def _quotes():
    out = []
    for exp in (D0 + timedelta(10), D0 + timedelta(30), D0 + timedelta(60)):
        for k in range(90, 111, 5):
            for r in ("C", "P"):
                out.append(OptionQuote(symbol="X", expiry=exp, strike=float(k), right=r,
                                       bid=3.9, ask=4.1, iv=0.3, open_interest=500))
    return out


def test_choose_expiry_prefers_first_after_event():
    exps = sorted({q.expiry for q in _quotes()})
    assert choose_expiry(exps, D0, (7, 45)) in (exps[0], exps[1])
    assert choose_expiry(exps, D0, (7, 45), event_date=D0 + timedelta(12)) == exps[1]
    assert choose_expiry(exps, D0, (70, 90)) is None


def test_build_legs_atm_and_wings():
    legs = build_legs(_quotes(), 101.0, D0, "long", S, event_date=D0 + timedelta(12))
    assert [(l.right, l.strike, l.qty) for l in legs] == [("C", 100.0, 1), ("P", 100.0, 1)]
    assert all(l.expiry == D0 + timedelta(30) and l.price == pytest.approx(4.0) for l in legs)
    wings = build_legs(_quotes(), 101.0, D0, "short", StraddleSettings(use_wings=True), event_date=None)
    assert sorted((l.right, l.strike, l.qty) for l in wings) == [
        ("C", 100.0, -1), ("C", 105.0, 1), ("P", 95.0, 1), ("P", 100.0, -1)]
    assert build_legs([], 100, D0, "long", S) is None


def entry(mode, **kw):
    base = dict(iv_rank=20, iv_over_rv=0.9, days_to_event=5, days_to_expiry=20, implied_move=5.0, realized_move=6.0)
    if mode == "short":
        base.update(iv_rank=80, iv_over_rv=1.3, days_to_event=None, implied_move=6.0, realized_move=5.0)
    base.update(kw)
    return evaluate_entry(mode, S, **base)


def test_long_entry_rules():
    assert entry("long").ok
    assert not entry("long", iv_rank=50).ok
    assert not entry("long", iv_over_rv=1.2).ok
    assert not entry("long", days_to_event=None).ok  # requires an event
    assert not entry("long", days_to_event=30).ok  # outside window
    assert not entry("long", days_to_event=5, days_to_expiry=3).ok  # event after expiry
    assert not entry("long", implied_move=7.0).ok  # market already prices more than realized
    assert entry("long", iv_rank=None).ok and entry("long", iv_rank=None).flags


def test_short_entry_rules():
    assert entry("short").ok
    assert not entry("short", iv_rank=40).ok
    assert not entry("short", iv_over_rv=1.0).ok
    assert not entry("short", days_to_event=10).ok  # earnings before expiry
    assert not entry("short", implied_move=4.0).ok


def test_exit_rules_long_and_short():
    assert "profit" in exit_decision(8.0, 12.5, 20, S)  # +56%
    assert "stop" in exit_decision(8.0, 3.5, 20, S)  # -56%
    assert "time" in exit_decision(8.0, 8.0, 2, S)
    assert exit_decision(8.0, 8.5, 20, S) is None
    # short: entry value is -credit
    assert "profit" in exit_decision(-8.0, -3.0, 20, S)  # bought back at 3, +62%
    assert "stop" in exit_decision(-8.0, -13.0, 20, S)
