import math
from datetime import date, timedelta

import pytest

from optionsdesk.strategies.pricing import (
    bs_greeks, bs_price, expected_abs_move, implied_vol, iv_percentile, iv_rank,
)


def test_put_call_parity():
    S, K, T, r, s = 100, 105, 0.5, 0.03, 0.3
    c, p = bs_price(S, K, T, r, s, "C"), bs_price(S, K, T, r, s, "P")
    assert c - p == pytest.approx(S - K * math.exp(-r * T), abs=1e-9)


def test_known_atm_call_value():
    # Hull-style reference: S=K=100, T=1, r=5%, sigma=20% -> 10.4506
    assert bs_price(100, 100, 1, 0.05, 0.2, "C") == pytest.approx(10.4506, abs=1e-3)


def test_delta_bounds_and_signs():
    g_c, g_p = bs_greeks(100, 100, 0.25, 0.03, 0.25, "C"), bs_greeks(100, 100, 0.25, 0.03, 0.25, "P")
    assert 0 < g_c.delta < 1 and -1 < g_p.delta < 0
    assert g_c.delta - g_p.delta == pytest.approx(1.0, abs=1e-9)
    assert g_c.gamma == pytest.approx(g_p.gamma) and g_c.vega == pytest.approx(g_p.vega)
    assert g_c.theta < 0


def test_greeks_match_finite_difference():
    S, K, T, r, s = 100, 95, 0.4, 0.04, 0.35
    h = 0.01
    up, dn = bs_price(S + h, K, T, r, s, "P"), bs_price(S - h, K, T, r, s, "P")
    assert (up - dn) / (2 * h) == pytest.approx(bs_greeks(S, K, T, r, s, "P").delta, abs=1e-4)
    v = (bs_price(S, K, T, r, s + 0.005, "P") - bs_price(S, K, T, r, s - 0.005, "P")) / 1.0  # 1 vol point
    assert v == pytest.approx(bs_greeks(S, K, T, r, s, "P").vega, abs=1e-4)


def test_expiry_returns_intrinsic():
    assert bs_price(90, 100, 0, 0.03, 0.3, "P") == 10
    assert bs_greeks(110, 100, 0, 0.03, 0.3, "C").delta == 1.0


def test_implied_vol_roundtrip_and_bounds():
    price = bs_price(100, 110, 0.3, 0.04, 0.42, "C")
    assert implied_vol(price, 100, 110, 0.3, 0.04, "C") == pytest.approx(0.42, abs=1e-6)
    assert implied_vol(0.0, 100, 110, 0.3, 0.04, "C") is None
    assert implied_vol(1.0, 100, 50, 0.3, 0.04, "C") is None  # below intrinsic


def test_iv_rank_and_percentile():
    hist = [0.2 + 0.01 * i for i in range(40)]  # 0.20 .. 0.59
    assert iv_rank(0.395, hist) == pytest.approx(50.0, abs=1e-6)
    assert iv_rank(0.9, hist) == 100.0 and iv_rank(0.1, hist) == 0.0
    assert iv_percentile(0.395, hist) == pytest.approx(50.0)
    assert iv_rank(0.3, hist[:10]) is None  # not enough history
    assert iv_rank(0.3, [0.3] * 40) == 50.0


def test_expected_move_matches_atm_straddle():
    S, T, s = 100, 0.25, 0.3
    straddle = bs_price(S, S, T, 0.0, s, "C") + bs_price(S, S, T, 0.0, s, "P")
    assert straddle == pytest.approx(expected_abs_move(S, s, T), rel=0.01)
