"""Phase 3 / 3B model tests: pricing and Greeks, statistics, RV forecasts, FHS, local vol, smile regimes,
vanna-volga, attribution and the alpha test."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from premia import attribution, greeks as G, physical, pricing, rv, stats, vol
from volsurface.svi import SVI

from .fixtures import SKEWED, panel


# ---------------------------------------------------------------------------- pricing
def test_put_call_parity_and_greeks_match_finite_differences():
    S, K, T, r, q, s = 100.0, 95.0, 0.3, 0.03, 0.01, 0.25
    c, p = pricing.price(S, K, T, r, q, s, "C"), pricing.price(S, K, T, r, q, s, "P")
    assert c - p == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T), abs=1e-10)
    for right in ("C", "P"):
        g = pricing.greeks(S, K, T, r, q, s, right)
        h, v = 0.01, 1e-4
        f = lambda S_, s_: float(pricing.price(S_, K, T, r, q, s_, right))
        assert g.delta == pytest.approx((f(S + h, s) - f(S - h, s)) / (2 * h), rel=1e-5)
        assert g.gamma == pytest.approx((f(S + h, s) - 2 * f(S, s) + f(S - h, s)) / h ** 2, rel=1e-3)
        assert g.vega == pytest.approx((f(S, s + v) - f(S, s - v)) / (2 * v), rel=1e-5)
        assert g.vanna == pytest.approx((f(S + h, s + v) - f(S + h, s - v) - f(S - h, s + v) + f(S - h, s - v)) / (4 * h * v), rel=1e-3)
        assert g.volga == pytest.approx((f(S, s + v) - 2 * f(S, s) + f(S, s - v)) / v ** 2, rel=1e-3)
        dt = 1e-5
        assert g.theta == pytest.approx((float(pricing.price(S, K, T - dt, r, q, s, right)) - f(S, s)) / dt, rel=1e-3)


def test_implied_vol_round_trip_and_strike_for_delta():
    K = np.array([80, 100, 120.0])
    px = pricing.price(100, K, 0.5, 0.02, 0.0, 0.3, "P")
    assert np.allclose(pricing.implied_vol(px, 100, K, 0.5, 0.02, 0.0, "P"), 0.3, atol=1e-6)
    k = pricing.strike_for_delta(100, 0.25, 0.02, 0.01, 0.2, -0.25, "P")
    assert float(pricing.greeks(100, k, 0.25, 0.02, 0.01, 0.2, "P").delta) == pytest.approx(-0.25, abs=1e-8)


# ---------------------------------------------------------------------------- statistics
def test_mean_test_size_and_power():
    rng = np.random.default_rng(0)
    rej0 = np.mean([stats.mean_test(rng.normal(0, 1, 500))["p"] < 0.05 for _ in range(400)])
    rej1 = np.mean([stats.mean_test(rng.normal(0.2, 1, 500))["p"] < 0.05 for _ in range(200)])
    assert 0.02 <= rej0 <= 0.09 and rej1 > 0.95


def test_deflated_sharpe_penalises_many_trials():
    rng = np.random.default_rng(1)
    r = rng.normal(0.0005, 0.01, 1000)
    psr = stats.probabilistic_sharpe(r)
    dsr = stats.deflated_sharpe(r, n_trials=50, var_sr_trials=0.0004)
    assert dsr < psr


def test_ols_hac_recovers_coefficients_and_bootstrap_interval_covers():
    rng = np.random.default_rng(2)
    x = rng.normal(size=800)
    y = 0.5 + 2.0 * x + rng.normal(size=800)
    res = stats.ols_hac(y, x, ["x"], boot_reps=300)
    assert res.coef[1] == pytest.approx(2.0, abs=0.1) and res.ci_boot[1, 0] < 2.0 < res.ci_boot[1, 1]


def test_benjamini_hochberg():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.6])
    assert stats.benjamini_hochberg(p, 0.05).tolist() == [True, True, False, False, False]


# ---------------------------------------------------------------------------- realized vol
def test_garch_fit_recovers_persistence():
    df = panel(n=3000, seed=3)
    r = np.log(df["close"]).diff().dropna().to_numpy()
    fit = rv.fit_garch(r)
    assert 0.85 < fit.persistence < 0.995 and 0.03 < fit.alpha < 0.2


def test_forecasts_never_use_the_future():
    df = panel(n=1300, seed=4)
    fc1 = rv.forecasts(df, h=21, iv=df["iv"], min_train=500)
    df2 = df.copy()
    cut = df.index[1000]
    df2.loc[df2.index > cut, ["close", "open", "high", "low"]] *= np.linspace(1, 3, (df2.index > cut).sum())[:, None]
    df2.loc[df2.index > cut, "iv"] *= 2
    fc2 = rv.forecasts(df2, h=21, iv=df2["iv"], min_train=500)
    cols = ["naive", "ewma", "har", "har_iv", "garch", "gjr"]
    # forecasts at dates whose HAR training targets end before the cut must be identical
    safe = fc1.index <= df.index[1000 - 22]
    pd.testing.assert_frame_equal(fc1.loc[safe, cols], fc2.loc[safe, cols])


def test_forecasts_stay_inside_the_training_range_on_corrupt_histories():
    df = panel(n=1300, seed=9)
    df.iloc[900:905, df.columns.get_loc("high")] = df["close"].iloc[900:905]      # zero-range days
    df.iloc[900:905, df.columns.get_loc("low")] = df["close"].iloc[900:905]
    df.iloc[900:905, df.columns.get_loc("open")] = df["close"].iloc[900:905]
    fc = rv.forecasts(df, h=21, iv=df["iv"], min_train=500)
    cap = fc["target"].max() * 1.5
    for m in ("har", "har_iv", "garch", "gjr"):
        assert fc[m].max() <= cap * 1.0001


def test_qlike_is_minimised_at_the_truth():
    assert rv.qlike(0.04, 0.04) == pytest.approx(0) and rv.qlike(0.04, 0.03) > 0 and rv.qlike(0.04, 0.05) > 0


# ---------------------------------------------------------------------------- physical (FHS)
def test_fhs_forward_is_exact_and_normal_residuals_give_black_scholes():
    rng = np.random.default_rng(5)
    r = rng.normal(0, 0.2 / np.sqrt(252), 3000)
    fhs = physical.FHS.from_returns(r)
    d = physical.distribution(fhs, 100.0, 63, 0.03, 0.01, n=40000, seed=1, target_var_annual=0.04)
    assert d.ST.mean() == pytest.approx(d.forward, rel=1e-9)
    bs = float(pricing.price(100, 95, d.T, 0.03, 0.01, 0.2, "P"))
    assert float(d.fair_price([95], "P")[0]) == pytest.approx(bs, rel=0.06)


def test_signal_is_spread_aware():
    assert physical.signal(0.25, 0.26, 0.22) == "RICH"
    assert physical.signal(0.20, 0.21, 0.24) == "CHEAP"
    assert physical.signal(0.22, 0.24, 0.23) == "FAIR"


# ---------------------------------------------------------------------------- local vol / smile shape
def _svi_slices():
    Ts = np.array([0.1, 0.25, 0.5, 1.0])
    sv = [SVI(0.02 * T, 0.10 * np.sqrt(T), -0.6, 0.0, 0.15) for T in Ts]
    return vol.Slices(Ts, sv)


def test_flat_surface_has_flat_local_vol():
    sl = vol.Slices(np.array([0.2, 1.0]), [SVI(0.04 * t, 0.0, 0.0, 0.0, 0.1) for t in (0.2, 1.0)])
    lv, bad = sl.local_vol(np.array([-0.3, 0.0, 0.3]), 0.5)
    assert np.allclose(lv, 0.2) and not bad.any()


def test_local_vol_pde_reprices_its_own_surface():
    rows = vol.validate_local_vol(_svi_slices(), moneyness=(0.8, 1.2))
    assert rows and max(abs(r["err_volpts"]) for r in rows) < 0.1


def test_smile_shape_reproduces_the_slice_it_came_from():
    p, T = SVI(0.004, 0.03, -0.5, 0.01, 0.1), 0.1
    sh = vol.SmileShape.from_svi(p, T)
    atm = np.sqrt(p.w(0.0) / T)
    k = np.linspace(-0.3, 0.2, 11)
    assert np.allclose(sh.iv(k, atm, T), np.sqrt(p.w(k) / T), atol=2e-4)
    assert np.allclose(sh.iv(k, atm, T, skew_mult=0.0), atm)


# ---------------------------------------------------------------------------- regimes and vanna-volga
def test_sticky_strike_greeks_equal_closed_form_and_regimes_order_delta():
    S, T, r, q = 100.0, 0.25, 0.02, 0.0
    smile = lambda k: 0.2 - 0.25 * np.asarray(k)                # negative skew
    K = np.array([90.0, 100.0])
    ss = G.surface_greeks(S, K, T, r, q, "P", smile, "sticky_strike")
    F = S * np.exp((r - q) * T)
    bs = pricing.greeks(S, K, T, r, q, smile(np.log(K / F)), "P")
    assert np.allclose(ss["delta"], bs.delta, atol=2e-4) and np.allclose(ss["vega"], bs.vega, rtol=1e-3)
    sd = G.surface_greeks(S, K, T, r, q, "P", smile, "sticky_delta")
    lv = G.surface_greeks(S, K, T, r, q, "P", smile, "local_vol")
    # negative skew: under sticky delta the vol at a fixed strike rises with spot, under local vol it falls (Derman)
    assert np.all(lv["delta"] < ss["delta"]) and np.all(ss["delta"] < sd["delta"])


def test_vanna_volga_reprices_pivots_exactly():
    piv, vols = (90.0, 100.0, 110.0), (0.25, 0.20, 0.18)
    assert np.allclose(G.vanna_volga_iv(np.array(piv), piv, vols), vols)
    y = G.vv_weights(np.array([95.0]), *piv)
    assert sum(float(a[0]) for a in y) == pytest.approx(1.0, abs=0.05)


# ---------------------------------------------------------------------------- attribution and alpha test
def test_taylor_attribution_explains_small_daily_moves():
    """Simulated daily paths with vol moves: residual <= 10 % of the P&L variance (phase 3B criterion)."""
    rng = np.random.default_rng(7)
    tot, res = [], []
    S, sig, T = 100.0, 0.25, 45 / 365
    for _ in range(2000):
        S1 = S * np.exp(rng.normal(0, sig / np.sqrt(252)))
        sig1 = max(sig + rng.normal(0, 0.01), 0.05)
        a = attribution.attribute_option(S, S1, sig, sig1, 95.0, T, 1 / 365, 0.02, 0.0, "P", qty=-1)
        tot.append(a["total"])
        res.append(a["residual"])
    assert np.var(res) <= 0.10 * np.var(tot)


def test_alpha_classifier_power_and_false_positive_rate():
    """Size <= 10 % without alpha; >= 80 % power at the minimum detectable annual Sharpe for the sample,
    2.8 / sqrt(years) (two-sided 5 % test). A smaller alpha cannot be detected reliably by any valid test."""
    rng = np.random.default_rng(8)
    n_days, sims = 756, 40
    daily_sd = 1.0
    mde = 2.8 / np.sqrt(n_days / 252) * 1.15           # annual Sharpe, 15 % above the minimum detectable effect
    mu = mde / np.sqrt(252) * daily_sd
    found, false = 0, 0
    for i in range(sims):
        d = pd.DataFrame({"date": np.arange(n_days), "trade_type": "x",
                          "residual_net": rng.normal(mu, daily_sd, n_days)})
        found += attribution.classify(d, boot_reps=200)[0].label == "alpha"
        d0 = d.assign(residual_net=rng.normal(0, daily_sd, n_days))
        false += attribution.classify(d0, boot_reps=200)[0].label == "alpha"
    assert found / sims >= 0.8 and false / sims <= 0.10
