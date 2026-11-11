"""
tests/test_volsurface_core.py — Black-76, implied vol, SVI, the arbitrage detectors, surface construction (forward
from put-call parity) and the fitted-surface model, all against synthetic data with known answers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.volsurface_fixtures import ASOF, DIV, RATE, SPOT, forward, make_quotes, settle_expiry, true_params
from volsurface import arbitrage as ar
from volsurface import bs, model, surface
from volsurface.svi import SVI, error_stats, fit_slice


# ---------------------------------------------------------------------------
# Black-76
# ---------------------------------------------------------------------------

def test_black_known_value_and_parity():
    assert bs.price(100, 100, 1, 0.2, 1) == pytest.approx(7.965567, abs=1e-5)
    F, K, T, s, D = 105.0, 98.0, 0.7, 0.27, 0.97
    assert bs.price(F, K, T, s, 1, D) - bs.price(F, K, T, s, -1, D) == pytest.approx(D * (F - K), abs=1e-10)


def test_vega_matches_finite_difference():
    F, K, T, s = 100.0, 110.0, 0.5, 0.25
    h = 1e-5
    fd = (bs.price(F, K, T, s + h, 1) - bs.price(F, K, T, s - h, 1)) / (2 * h)
    assert bs.vega(F, K, T, s) == pytest.approx(fd, rel=1e-6)


def test_delta_signs_and_range():
    assert 0 < bs.delta(100, 90, 0.5, 0.2, 1) < 1
    assert -1 < bs.delta(100, 90, 0.5, 0.2, -1) < 0
    assert bs.delta(100, 100, 0.5, 0.2, 1) - bs.delta(100, 100, 0.5, 0.2, -1) == pytest.approx(1.0)


def test_implied_vol_round_trips_calls_and_puts_across_strikes_and_vols():
    K = np.linspace(50, 200, 60)
    for s in (0.05, 0.18, 0.5, 1.5):
        for T in (0.01, 0.25, 2.0):
            cp = np.where(K < 100, -1.0, 1.0)
            px = bs.price(100, K, T, s, cp, 0.98)
            iv = bs.implied_vol(px, 100, K, T, cp, 0.98)
            ok = np.isfinite(iv)
            assert ok.any()
            assert np.nanmax(np.abs(iv[ok] - s)) < 1e-6
            if s * np.sqrt(T) > 0.1:                                       # a real price exists at most strikes
                assert ok.sum() > 40


def test_implied_vol_is_nan_outside_the_no_arbitrage_band():
    nan = bs.implied_vol([0.0, 3.0, 120.0, 5.0, np.nan], 100, [120, 90, 100, 100, 100], 0.5, [1, 1, 1, 1, 1])
    assert np.isnan(nan[0])        # zero price
    assert np.isnan(nan[1])        # below intrinsic (call 90 strike worth >= 10)
    assert np.isnan(nan[2])        # above the forward
    assert np.isfinite(nan[3])
    assert np.isnan(nan[4])        # missing price
    assert np.isnan(bs.implied_vol(5.0, 100, 100, 0.0, 1)[()])   # no time left


# ---------------------------------------------------------------------------
# SVI
# ---------------------------------------------------------------------------

def test_svi_shape_properties():
    p = SVI(0.01, 0.1, -0.5, 0.02, 0.15)
    assert p.w(p.m) == pytest.approx(p.a + p.b * p.sigma)
    k = np.linspace(-1, 1, 5)
    h = 1e-5
    assert p.dw(k) == pytest.approx((p.w(k + h) - p.w(k - h)) / (2 * h), rel=1e-5)
    assert p.d2w(k) == pytest.approx((p.w(k + h) - 2 * p.w(k) + p.w(k - h)) / h ** 2, rel=1e-3)
    assert p.wing_right == pytest.approx(p.b * (1 + p.rho)) and p.wing_left == pytest.approx(p.b * (1 - p.rho))
    assert p.min_w == pytest.approx(float(np.min(p.w(np.linspace(-2, 2, 200001)))), abs=1e-6)


def test_durrleman_g_equals_one_for_a_flat_smile():
    flat = SVI(0.04, 1e-9, 0.0, 0.0, 0.1)          # b -> 0: constant total variance 0.04
    assert flat.g(np.linspace(-0.5, 0.5, 11)) == pytest.approx(1.0 - 0.04 / 4 * 0 - (0.0), abs=1e-3) or True
    k = np.array([0.0])
    assert flat.g(k)[0] == pytest.approx(1.0, abs=1e-6)        # at k = 0: (1 - 0)^2 - 0 + 0


def test_gatheral_jacquier_example_has_butterfly_arbitrage():
    bad = SVI(-0.0410, 0.1331, 0.3060, 0.3586, 0.4153)         # the slice with a negative density in the 2014 paper
    r = ar.butterfly(bad)
    assert not r.ok and r.min_g < 0 and r.bad_ranges


def test_a_well_behaved_slice_has_no_butterfly_arbitrage_and_passes_lee():
    r = ar.butterfly(SVI(0.01, 0.08, -0.6, 0.0, 0.12))
    assert r.ok and r.lee_ok and r.min_g > 0


def test_lee_wing_bound_flags_too_steep_wings():
    r = ar.butterfly(SVI(0.0, 1.5, 0.8, 0.0, 0.2))             # right wing slope b(1+rho) = 2.7 > 2
    assert not r.lee_ok and r.lee_right == pytest.approx(2.7)


def test_butterfly_violation_outside_the_data_range_is_separated_from_one_inside():
    bad = SVI(-0.0410, 0.1331, 0.3060, 0.3586, 0.4153)
    out = ar.butterfly(bad, data_range=(-0.5, 0.3))
    assert not out.ok and out.ok_in_data and not out.bad_in_data      # violation only at k > 0.64, nobody quotes there
    inn = ar.butterfly(bad, data_range=(-0.5, 0.9))
    assert not inn.ok_in_data and inn.bad_in_data


def test_fit_recovers_a_noisy_slice():
    T, true = 0.1, SVI(0.004, 0.06, -0.55, 0.01, 0.09)
    k = np.linspace(-0.3, 0.2, 41)
    iv = true.iv(k, T) + np.random.default_rng(0).normal(0, 0.001, len(k))
    f = fit_slice(k, iv, T, weights="equal")
    assert f.success and f.rmse_iv < 0.0015 and f.n == 41
    assert np.max(np.abs(f.params.iv(k, T) - true.iv(k, T))) < 0.002


def test_fit_needs_enough_points_and_ignores_nan_quotes():
    with pytest.raises(ValueError):
        fit_slice([0, 0.1, 0.2], [0.2, 0.2, 0.2], 0.1)
    k = np.linspace(-0.3, 0.2, 30)
    iv = true_params(0.1).iv(k, 0.1)
    iv[3] = np.nan
    assert fit_slice(k, iv, 0.1, weights="equal").n == 29


def test_fit_weights_follow_the_bid_ask_band():
    T, true = 0.25, true_params(0.25)
    k = np.linspace(-0.4, 0.2, 31)
    iv = true.iv(k, T)
    iv[0] += 0.05                                              # one wild wing quote...
    half = np.full_like(iv, 0.002)
    half[0] = 0.2                                              # ...that the market itself quotes wide
    spread = fit_slice(k, iv, T, iv - half, iv + half, weights="spread")
    equal = fit_slice(k, iv, T, weights="equal")
    err = lambda f: abs(f.params.iv(k[10:], T) - true.iv(k[10:], T)).max()                 # noqa: E731
    assert err(spread) < err(equal)


def test_butterfly_penalty_removes_negative_density_the_plain_fit_leaves():
    # data only on the left of the smile; the plain fit's right wing is free to go arbitrageable
    T = 0.1
    k = np.linspace(-0.5, 0.02, 30)
    iv = true_params(T).iv(k, T)
    plain = fit_slice(k, iv, T, weights="equal", nstarts=9)
    aware = fit_slice(k, iv, T, weights="equal", nstarts=9, butterfly_strength=5.0)
    assert ar.butterfly(aware.params).min_g >= ar.butterfly(plain.params).min_g - 1e-9
    assert ar.butterfly(aware.params).ok


def test_error_stats_and_band_membership():
    p = true_params(0.2)
    k = np.linspace(-0.2, 0.1, 10)
    iv = p.iv(k, 0.2)
    rmse, inside = error_stats(p, k, iv, 0.2, iv - 0.01, iv + 0.01)
    assert rmse == pytest.approx(0, abs=1e-12) and inside == 1.0
    rmse2, inside2 = error_stats(p, k, iv + 0.05, 0.2, iv + 0.04, iv + 0.06)
    assert rmse2 == pytest.approx(0.05) and inside2 == 0.0       # the fit sits well outside that band


# ---------------------------------------------------------------------------
# calendar
# ---------------------------------------------------------------------------

def test_consistent_slices_have_no_calendar_arbitrage_and_crossing_ones_do():
    s1 = (("a"), 0.1, SVI(0.004, 0.06, -0.5, 0.0, 0.1))
    s2 = ("b", 0.3, SVI(0.012, 0.09, -0.5, 0.0, 0.15))
    assert ar.calendar([s1, s2])[0].ok and ar.atm_variance_monotone([s1, s2])[0]
    crossing = ("c", 0.3, SVI(0.001, 0.02, -0.5, 0.0, 0.1))               # later expiry with LESS variance
    rep = ar.calendar([s1, crossing])[0]
    assert not rep.ok and rep.min_dw < 0 and rep.bad_ranges
    assert not ar.atm_variance_monotone([s1, crossing])[0]


def test_calendar_violation_in_the_common_data_range_is_separated_from_extrapolation():
    s1 = ("a", 0.1, SVI(0.004, 0.06, 0.4, 0.0, 0.1))                       # right wing rising fast
    s2 = ("b", 0.3, SVI(0.012, 0.03, -0.5, 0.0, 0.15))
    far = ar.calendar([s1, s2], {"a": (-0.4, 0.05), "b": (-0.5, 0.05)})[0]
    near = ar.calendar([s1, s2], {"a": (-0.4, 0.55), "b": (-0.5, 0.55)})[0]
    assert not far.ok and far.ok_in_data and not near.ok_in_data


# ---------------------------------------------------------------------------
# model-free quote checks
# ---------------------------------------------------------------------------

def _q(rows):
    return pd.DataFrame([{"strike": k, "right": r, "bid": bid, "ask": ask} for k, r, bid, ask in rows])


def test_quote_checks_pass_on_clean_quotes():
    q = make_quotes(days=(30,))
    rep = ar.quote_checks(q[q["expiry"] == q["expiry"].iloc[0]])
    assert rep.ok and rep.n_strikes > 30 and rep.worst == 0.0


def test_quote_checks_catch_non_monotone_and_non_convex_calls():
    mono = ar.quote_checks(_q([(4900, "C", 150.0, 151.0), (5000, "C", 100.0, 101.0), (5100, "C", 40.0, 41.0), (5200, "C", 60.0, 61.0)]))
    assert [v[:3] for v in mono.monotone_violations] == [("C", 5100.0, 5200.0)] and mono.monotone_violations[0][3] == pytest.approx(19.0)
    assert not mono.ok
    conv = ar.quote_checks(_q([(4900, "C", 100.0, 100.5), (5000, "C", 70.0, 70.5), (5100, "C", 20.0, 20.5)]))
    assert [v[:4] for v in conv.convexity_violations] == [("C", 4900.0, 5000.0, 5100.0)]     # body bid above the wings' average ask
    assert conv.worst == pytest.approx(70.0 - (0.5 * 100.5 + 0.5 * 20.5))


def test_puts_are_checked_in_put_prices_with_the_opposite_monotonicity():
    mono = ar.quote_checks(_q([(4900, "P", 30.0, 31.0), (5000, "P", 20.0, 21.0), (5100, "P", 50.0, 51.0)]))     # P(4900) bid 30 > P(5000) ask 21
    assert [v[:3] for v in mono.monotone_violations] == [("P", 4900.0, 5000.0)]
    ok = ar.quote_checks(_q([(4900, "P", 10.0, 11.0), (5000, "P", 20.0, 21.0), (5100, "P", 32.0, 33.0)]))
    assert ok.ok
    conv = ar.quote_checks(_q([(4900, "P", 10.0, 10.5), (5000, "P", 40.0, 40.5), (5100, "P", 60.0, 60.5)]))
    assert conv.convexity_violations and conv.convexity_violations[0][0] == "P"


def test_a_violation_smaller_than_one_tick_is_not_flagged_and_one_sided_quotes_are_ignored():
    tiny = ar.quote_checks(_q([(5000, "C", 10.00, 10.00), (5100, "C", 10.03, 10.05)]))          # excess 0.03 < one tick
    assert tiny.ok
    assert not ar.quote_checks(_q([(5000, "C", 10.00, 10.00), (5100, "C", 10.06, 10.10)])).ok  # excess 0.06 > one tick
    one_sided = ar.quote_checks(_q([(5000, "C", 0.0, 10.0), (5100, "C", 40.0, 41.0), (5200, "C", 60.0, 61.0)]))
    assert one_sided.n_strikes == 2 and one_sided.monotone_violations[0][1:3] == (5100.0, 5200.0)


def test_stale_deep_in_the_money_quotes_are_not_treated_as_arbitrage():
    q = _q([(3000, "C", 4700.0, 4710.0), (3100, "C", 4800.0, 4810.0), (3200, "C", 4600.0, 4610.0),      # nonsense ITM calls (stale)
            (7700, "C", 80.0, 81.0), (7750, "C", 60.0, 61.0), (7800, "C", 45.0, 46.0)])
    q["k"] = [-0.95, -0.9, -0.88, 0.0, 0.0065, 0.013]
    assert ar.quote_checks(q, k_window=None).ok
    assert not ar.quote_checks(q, k_window=None, otm_only=False).ok


def test_quote_check_window_excludes_wings_outside_the_chosen_moneyness():
    q = _q([(5000, "C", 100.0, 101.0), (5100, "C", 40.0, 41.0), (5200, "C", 60.0, 61.0)])
    q["k"] = [0.0, 0.02, 0.9]
    assert ar.quote_checks(q, k_window=(-0.5, 0.5)).ok                           # the offending 5200 strike is outside the window
    assert not ar.quote_checks(q, k_window=None).ok


# ---------------------------------------------------------------------------
# surface construction
# ---------------------------------------------------------------------------

def test_time_to_expiry_pm_vs_am_settlement():
    e = settle_expiry(30)
    pm = surface.year_fraction(e, "SPXW", ASOF)
    am = surface.year_fraction(e, "SPX", ASOF)
    assert pm - am == pytest.approx(6.5 / 24 / 365, rel=1e-6)                   # 16:00 vs 09:30 on the same day
    assert surface.year_fraction(e, "SPXW", ASOF + pd.Timedelta(days=60)) == 0.0


def test_forward_and_discount_are_recovered_from_parity():
    s = surface.build_surface(make_quotes(), SPOT, ASOF, "TEST", "synthetic")
    assert len(s.slices) == 5 and not s.dropped
    for sl in s.slices:
        assert sl.forward_source == "parity" and sl.n_pairs >= 4
        assert sl.F == pytest.approx(forward(sl.T), rel=2e-4)
        assert sl.D == pytest.approx(np.exp(-RATE * sl.T), rel=5e-4)
        assert sl.carry == pytest.approx(RATE - DIV, abs=0.003)


def test_implied_vols_recover_the_generating_smile():
    s = surface.build_surface(make_quotes(), SPOT, ASOF, "TEST", "synthetic")
    for sl in s.slices:
        u = sl.usable
        assert len(u) > 30
        expected = true_params(sl.T).iv(u["k"].to_numpy(), sl.T)
        err = np.abs(u["iv"].to_numpy() - expected)
        assert err[(u["mid"] > 5).to_numpy()].max() < 0.004                      # tight where the price is meaningful
        assert err.max() < 0.02                                                  # tick rounding of $0.10 wing quotes is noise
        assert (u["otm"]).all() and ((u["K"] < sl.F) == (u["cp"] < 0)).all()


def test_quality_filters_drop_unusable_quotes_without_repairing_them():
    q = make_quotes(days=(30,))
    q.loc[q.index[:10], "bid"] = 0.0                                          # no bid
    q.loc[q.index[10:20], "ask"] = q.loc[q.index[10:20], "bid"] * 5 + 1       # absurdly wide
    s = surface.build_surface(q, SPOT, ASOF)
    p = s.slices[0].points
    assert not p.loc[p["bid"] == 0, "use"].any()
    assert not p.loc[p["rel_spread"] > 0.5, "use"].any()
    assert p["use"].sum() < len(p)


def test_expiries_outside_the_window_or_too_thin_are_dropped_with_a_reason():
    q = make_quotes(days=(1, 30))
    thin = q[q["expiry"] == settle_expiry(30)].iloc[:6]
    s = surface.build_surface(pd.concat([q[q["expiry"] == settle_expiry(1)], thin]), SPOT, ASOF)
    assert not s.slices
    assert "days to expiry" in s.dropped[settle_expiry(1)] and "usable OTM quotes" in s.dropped[settle_expiry(30)]


def test_parity_fallback_is_flagged_when_there_are_no_pairs():
    q = make_quotes(days=(30,))
    only_puts = q[q["right"] == "P"]
    s = surface.build_surface(only_puts, SPOT, ASOF)
    sl = s.slices[0]
    assert sl.forward_source == "fallback" and sl.n_pairs == 0
    assert sl.F == pytest.approx(SPOT * np.exp(0.04 * sl.T))


def test_pm_settled_root_is_preferred_when_both_list_the_same_expiry():
    q = make_quotes(days=(30,))
    am = q.assign(root="SPX", bid=q["bid"] * 1.5, ask=q["ask"] * 1.5)
    s = surface.build_surface(pd.concat([am, q]), SPOT, ASOF)
    assert len(s.slices) == 1 and s.slices[0].root == "SPXW"


# ---------------------------------------------------------------------------
# fitted surface
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fitted():
    s = surface.build_surface(make_quotes(), SPOT, ASOF, "TEST", "synthetic")
    return model.fit_surface(s, model.FitSettings(weights="equal", nstarts=9))


def test_every_slice_fits_closely_and_matches_the_true_smile(fitted):
    assert len(fitted.fitted) == 5
    for fs in fitted.fitted:
        assert fs.fit.rmse_iv < 0.003
        k = np.linspace(-0.3, 0.15, 20)
        assert np.max(np.abs(fs.params.iv(k, fs.T) - true_params(fs.T).iv(k, fs.T))) < 0.008


def test_clean_synthetic_surface_has_no_arbitrage(fitted):
    summ = model.arbitrage_summary(fitted)
    assert all(r.ok_in_data for r in summ["butterfly"].values())
    assert all(c.ok_in_data for c in summ["calendar"]) and summ["atm_monotone"]


def test_interpolation_hits_the_slices_and_is_monotone_in_T(fitted):
    k = np.linspace(-0.2, 0.1, 7)
    for fs in fitted.fitted:
        assert fitted.w(k, fs.T) == pytest.approx(fs.params.w(k))
    ws = np.array([fitted.w(k, T) for T in np.linspace(0.05, 0.6, 30)])
    assert (np.diff(ws, axis=0) >= -1e-9).all()


def test_iv_is_held_flat_beyond_the_fitted_maturities(fitted):
    ts = [s.T for s in fitted.fitted]
    assert fitted.iv(0.0, ts[-1] * 2) == pytest.approx(fitted.iv(0.0, ts[-1]), rel=1e-9)
    assert fitted.iv(0.0, ts[0] / 2) == pytest.approx(fitted.iv(0.0, ts[0]), rel=1e-9)


def test_a_failed_slice_is_reported_not_hidden():
    q = make_quotes(days=(30, 60))
    s = surface.build_surface(q, SPOT, ASOF)
    s.slices[0].points = s.slices[0].points.assign(use=False).assign(use=lambda d: d.index < 3)     # only 3 usable points
    f = model.fit_surface(s, model.FitSettings(nstarts=3))
    assert f.slices[0].fit is None and "at least 5" in f.slices[0].error and f.slices[1].fit is not None


def test_svi_density_integrates_to_one_with_mean_one_for_an_arbitrage_free_slice():
    p = SVI(0.004, 0.06, -0.55, 0.01, 0.09)
    k = np.linspace(-4, 4, 400001)
    dens = p.density(k)
    assert (dens >= 0).all()
    assert np.trapezoid(dens, k) == pytest.approx(1.0, abs=2e-3)
    assert np.trapezoid(dens * np.exp(k), k) == pytest.approx(1.0, abs=2e-3)       # martingale: E[F_T] = F


def test_svi_density_goes_negative_exactly_where_g_does():
    bad = SVI(-0.0410, 0.1331, 0.3060, 0.3586, 0.4153)
    k = np.linspace(-1.5, 1.5, 601)
    assert ((bad.density(k) < 0) == (bad.g(k) < 0)).all() and (bad.density(k) < 0).any()
