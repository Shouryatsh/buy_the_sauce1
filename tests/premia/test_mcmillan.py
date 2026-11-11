"""McMillan straddle rules (premia/mcmillan.py): probability model, history checks, follow-up actions."""

import numpy as np
import pandas as pd
import pytest

from premia import mcmillan as mm
from .fixtures import panel


@pytest.mark.parametrize("S, up, down, sig, T", [(125, 147, 103, 0.31, 130 / 252), (100, 112, 88, 0.20, 0.25),
                                                  (50, 70, 30, 0.45, 0.5)])
def test_closed_form_ever_probability_matches_monte_carlo(S, up, down, sig, T):
    cf = mm.ever_probability(S, up, down, sig, T)
    mc = mm.ever_probability_mc(S, up, down, sig, T, n_paths=20000, steps=2000)
    assert cf == pytest.approx(mc, abs=0.015)


def test_book_ibm_example_ever_vs_closing():
    """IBM 125, straddle 22, 130 trading days, 31 % vol: the book's calculator says 85 % ever, 48 % closing (p. 226)."""
    p = mm.ever_probability(125, 147, 103, 0.31, 130 / 252)
    assert 0.80 < p < 0.90
    T, s = 130 / 252, 0.31
    closing = mm.norm.cdf(np.log(103 / 125) / (s * np.sqrt(T))) + 1 - mm.norm.cdf(np.log(147 / 125) / (s * np.sqrt(T)))
    assert 0.40 < closing < 0.52 and closing < p


def test_ever_probability_edges():
    assert mm.ever_probability(100, 120, 80, 0.3, 0.0) == 0.0
    assert mm.ever_probability(130, 120, 80, 0.3, 0.5) == 1.0
    one = mm.ever_probability(100, 160, -1, 0.3, 0.5)                 # straddle worth more than the strike: one barrier
    assert 0 < one < mm.ever_probability(100, 160, 40, 0.3, 0.5) + 1e-9
    assert mm.ever_probability(100, 120, 80, 0.4, 0.5) > mm.ever_probability(100, 120, 80, 0.2, 0.5)


def test_iv_percentile_uses_only_past_readings():
    iv = pd.Series(np.r_[np.linspace(0.2, 0.4, 600), 0.1, 0.5])
    p = mm.iv_percentile_all(iv, min_history=504)
    assert np.isnan(p.iloc[100]) and p.iloc[-2] == 0.0 and p.iloc[-1] == 100.0
    q = mm.iv_percentile_all(pd.Series(np.r_[iv.to_numpy(), 0.05]), min_history=504)
    assert q.iloc[:len(iv)].equals(p)                                  # adding a future day leaves the past unchanged
    assert mm.latest_iv_percentile(iv.iloc[:-1]) == 0.0


def test_min_historical_vol_is_the_minimum_of_the_four():
    df = panel(n=400, seed=3)
    hv = mm.historical_vols(df["close"])
    assert np.allclose(mm.min_historical_vol(df["close"]).dropna(), hv.dropna().min(axis=1))


def test_move_frequency_counts_windows_that_reach_the_move():
    n = 1600
    idx = pd.bdate_range("2015-01-02", periods=n)
    flat = pd.DataFrame({"close": 100.0, "high": 100.5, "low": 99.5}, index=idx)
    assert mm.move_frequency(flat, n - 1, 60, 0.10, 0.10) == 0.0
    wave = 100 * np.exp(0.3 * np.sin(np.arange(n) / 20))
    w = pd.DataFrame({"close": wave, "high": wave, "low": wave}, index=idx)
    assert mm.move_frequency(w, n - 1, 60, 0.05, 0.05) > 0.9
    assert mm.move_frequency(w, 300, 60, 0.05, 0.05) is None          # not enough history yet
    # no look-ahead: changing prices after row i does not change the answer at i
    w2 = w.copy()
    w2.iloc[1300:, :] *= 3
    assert mm.move_frequency(w, 1200, 60, 0.05, 0.05) == mm.move_frequency(w2, 1200, 60, 0.05, 0.05)


def test_deal_pinned_detects_a_jump_then_dead_price():
    n = 500
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.02, n)
    r[440] = 0.30
    r[441:] = rng.normal(0, 0.001, n - 441)
    close = pd.Series(100 * np.exp(np.cumsum(r)))
    assert mm.deal_pinned(close, n - 1)
    assert not mm.deal_pinned(pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))), n - 1)


def test_evaluate_reports_every_failed_criterion():
    df = panel(n=1500, seed=5)
    c = mm.evaluate(100, 100, 30.0, 60, 40, 55.0, 0.10, df)
    text = " ".join(c.reasons)
    assert not c.passes and "1: IV percentile" in text and "1: 60 days" in text and "2:" in text
    ok = mm.evaluate(100, 100, 5.0, 182, 125, 5.0, 0.30, df)
    assert ok.ever_prob > 0.8 and ok.hist_freq is not None


def test_follow_up_rules():
    f = mm.follow_up("both", 3, 3, close=125, sma=110, value=26, cost=20, K=100, days_left=90, trading_days_left=60)
    assert (f.sell_calls, f.sell_puts, f.mode) == (1, 3, "call")              # upside breakeven 120 crossed
    f = mm.follow_up("both", 3, 3, close=78, sma=90, value=24, cost=20, K=100, days_left=90, trading_days_left=60)
    assert (f.sell_calls, f.sell_puts, f.mode) == (3, 1, "put")
    f = mm.follow_up("both", 2, 2, close=101, sma=100, value=7.9, cost=20, K=100, days_left=90, trading_days_left=60)
    assert f.mode == "closed" and "60" in f.reason                               # 60 % of the price lost
    f = mm.follow_up("both", 2, 2, close=101, sma=100, value=12, cost=20, K=100, days_left=30, trading_days_left=21)
    assert f.mode == "closed" and "month" in f.reason
    assert mm.follow_up("both", 2, 2, 101, 100, 12, 20, 100, 60, 40).mode == "both"
    assert mm.follow_up("call", 2, 0, 118, 119, 20, 20, 100, 60, 40).sell_calls == 2      # below the 20-day MA
    assert mm.follow_up("call", 2, 0, 125, 119, 25, 20, 100, 60, 40).mode == "call"
    assert mm.follow_up("put", 0, 2, 80, 79, 20, 20, 100, 60, 40).sell_puts == 2
    assert mm.follow_up("call", 2, 0, 125, 119, 25, 20, 100, 7, 4).mode == "closed"       # expiry approaching
    one = mm.follow_up("both", 1, 1, close=125, sma=110, value=26, cost=20, K=100, days_left=90, trading_days_left=60)
    assert (one.sell_calls, one.sell_puts, one.mode) == (0, 1, "call")                   # 1 contract: keep the call


def test_size_shared_by_backtest_and_dashboard():
    # $10/share straddle = $1,000 each; 0.5 % of $1M = $5,000 -> 5
    assert mm.size(10.0, 1e6, 0.0, 0.005, 0.015, 0.10) == (5, None)
    # over budget but under the one-contract ceiling -> 1; over the ceiling -> 0 with a reason
    assert mm.size(10.0, 1e5, 0.0, 0.005, 0.015, 0.10) == (1, None)
    n, why = mm.size(20.0, 1e5, 0.0, 0.005, 0.015, 0.10)
    assert n == 0 and why.startswith("size")
    # near the total cap: cut down to what fits, not skipped
    assert mm.size(10.0, 1e6, 97_500.0, 0.005, 0.015, 0.10) == (2, None)
    n, why = mm.size(10.0, 1e6, 100_000.0, 0.005, 0.015, 0.10)
    assert n == 0 and "cap" in why


def test_straddle_spread_frac():
    assert mm.straddle_spread_frac(9.5, 10.5) == pytest.approx(0.10)
    assert mm.straddle_spread_frac(0.0, 0.0) == float("inf")
