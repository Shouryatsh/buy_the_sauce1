"""
tests/test_volsurface_analysis.py — risk-neutral probabilities from the smile, rich/cheap classification, the put scan,
position check, constant-maturity metrics, surface differences, snapshot storage and the data helpers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import wheel_config as wc
from tests.volsurface_fixtures import ASOF, SPOT, forward, make_quotes, true_params
from volsurface import analysis as an
from volsurface import bs, data, model, store, surface
from volsurface.svi import SVI


@pytest.fixture(scope="module")
def fitted():
    s = surface.build_surface(make_quotes(), SPOT, ASOF, "TEST", "synthetic")
    return model.fit_surface(s, model.FitSettings(weights="equal", nstarts=9))


# ---------------------------------------------------------------------------
# risk-neutral probability read from the smile
# ---------------------------------------------------------------------------

def test_prob_below_equals_the_derivative_of_the_put_price_wrt_strike():
    p, T, F = true_params(0.25), 0.25, 100.0
    for k in (-0.25, -0.1, 0.0, 0.08):
        def put(K):
            s = float(p.iv(np.log(K / F), T))
            return float(bs.price(F, K, T, s, -1.0))
        K, h = F * np.exp(k), 1e-4 * F
        numeric = (put(K + h) - put(K - h)) / (2 * h)
        assert float(an.prob_below(p, k, T)) == pytest.approx(numeric, abs=2e-6)


def test_a_flat_smile_gives_the_black_scholes_probability():
    flat = SVI(0.04, 1e-9, 0.0, 0.0, 0.1)
    T = 0.25
    sigma = float(np.sqrt(0.04 / T))
    for k in (-0.2, 0.0, 0.1):
        assert float(an.prob_below(flat, k, T)) == pytest.approx(float(an.prob_below_flat(sigma, k, T)), abs=1e-6)


def test_the_smile_puts_far_more_probability_on_deep_otm_puts_than_a_flat_atm_vol():
    p, T = true_params(0.25), 0.25
    atm = float(p.iv(0.0, T))
    deep = np.array([-0.45, -0.35, -0.3, -0.25])
    assert (an.prob_below(p, deep, T) > 5 * an.prob_below_flat(atm, deep, T)).all()
    assert float(an.prob_below(p, 0.0, T)) < float(an.prob_below_flat(atm, 0.0, T)) < 0.55   # left-skewed: median above the forward


def test_prob_below_is_a_probability_and_increases_with_strike():
    p, T = true_params(0.5), 0.5
    pb = an.prob_below(p, np.linspace(-0.8, 0.5, 80), T)
    assert ((pb >= 0) & (pb <= 1)).all() and (np.diff(pb) >= -1e-9).all()


# ---------------------------------------------------------------------------
# rich / cheap against the fit
# ---------------------------------------------------------------------------

def test_classification_uses_the_bid_ask_band_not_the_midpoint():
    sig = an.classify_vs_fit([0.22, 0.18, 0.19, np.nan, np.nan], [0.24, 0.195, 0.21, 0.30, np.nan], [0.20, 0.20, 0.20, 0.20, 0.20])
    assert list(sig) == ["RICH", "CHEAP", "FAIR", "FAIR", "UNKNOWN"]
    # mid above the fit, but the fit is still inside the band -> not rich
    assert an.classify_vs_fit(0.19, 0.23, 0.20)[()] == "FAIR"


def test_a_gap_smaller_than_the_materiality_margin_is_fair_even_outside_the_band():
    assert an.classify_vs_fit(0.2010, 0.2020, 0.2000)[()] == "FAIR"              # bid 0.10 vol pts above the fit: real but immaterial
    assert an.classify_vs_fit(0.2030, 0.2040, 0.2000)[()] == "RICH"              # 0.30 vol pts above
    assert an.classify_vs_fit(0.1960, 0.1970, 0.2000)[()] == "CHEAP"             # ask 0.30 below
    assert an.classify_vs_fit(0.2010, 0.2020, 0.2000, min_gap=0.0)[()] == "RICH"  # margin is adjustable


def test_clean_synthetic_quotes_are_mostly_fair(fitted):
    t = an.quote_table(fitted.fitted[1])
    assert (t["signal"] == "FAIR").mean() > 0.6
    assert {"iv_fit", "diff_vp", "signal", "p_below", "p_below_flat", "delta"} <= set(t.columns)


def test_an_inflated_put_is_flagged_rich_and_sorted_first(fitted):
    fsl = fitted.fitted[1]
    puts = fsl.slice.usable[fsl.slice.usable["cp"] < 0]
    row = puts.iloc[len(puts) // 2].name
    fsl.slice.points.loc[row, ["bid", "ask"]] = fsl.slice.points.loc[row, ["bid", "ask"]] * 1.6
    # recompute the bid/ask implied vols for the inflated quote
    p = fsl.slice.points.loc[row]
    fsl.slice.points.loc[row, "iv_bid"] = float(bs.implied_vol(p["bid"], fsl.slice.F, p["K"], fsl.slice.T, p["cp"], fsl.slice.D))
    fsl.slice.points.loc[row, "iv_ask"] = float(bs.implied_vol(p["ask"], fsl.slice.F, p["K"], fsl.slice.T, p["cp"], fsl.slice.D))
    scan = an.put_scan(fsl, SPOT, sort_by="rich")
    assert scan.iloc[0]["signal"] == "RICH" and scan.iloc[0]["K"] == p["K"]
    assert scan.iloc[0]["edge"] > 0


def test_put_scan_default_sort_is_annualised_yield_and_the_others_work(fitted):
    fsl = fitted.fitted[2]
    by_yield = an.put_scan(fsl, SPOT)
    assert by_yield["annual_yield"].is_monotonic_decreasing
    by_strike = an.put_scan(fsl, SPOT, sort_by="strike")
    assert by_strike["K"].is_monotonic_decreasing
    by_rich = an.put_scan(fsl, SPOT, sort_by="rich")
    order = {"RICH": 0, "FAIR": 1, "UNKNOWN": 2, "CHEAP": 3}
    assert by_rich["signal"].map(order).is_monotonic_increasing
    assert set(by_yield["K"]) == set(by_strike["K"]) == set(by_rich["K"])


def test_put_scan_min_bid_removes_dust_premiums_from_the_passing_set(fitted):
    fsl = fitted.fitted[2]
    loose = an.put_scan(fsl, SPOT, 0.5, 1.0, min_bid=0.05)
    strict = an.put_scan(fsl, SPOT, 0.5, 1.0, min_bid=2.0)
    assert strict["passes"].sum() < loose["passes"].sum()
    assert (strict.loc[strict["passes"], "bid"] >= 2.0).all()


def test_put_scan_economics_and_rule_filters(fitted):
    fsl = fitted.fitted[2]
    scan = an.put_scan(fsl, SPOT, max_delta=0.15, max_spread=0.40)
    assert (scan["cp"] < 0).all() and (scan["K"] < fsl.slice.F).all()
    r = scan.iloc[0]
    assert r["yield"] == pytest.approx(r["bid"] / r["K"])
    assert r["annual_yield"] == pytest.approx(r["yield"] * 365 / fsl.slice.days)
    assert r["breakeven"] == pytest.approx(r["K"] - r["bid"])
    assert r["cushion"] == pytest.approx((SPOT - r["breakeven"]) / SPOT)
    assert scan["passes"].equals(((scan["abs_delta"] <= 0.15) & (scan["rel_spread"] <= 0.40) & (scan["bid"] >= 0.05)))
    assert scan["passes"].any() and not scan["passes"].all()
    assert (scan["abs_delta"] < 0.5).all()                                    # OTM puts only


def test_position_check_reports_fit_probability_and_flags_extrapolation(fitted):
    fsl = fitted.fitted[2]
    p = fsl.slice.usable[fsl.slice.usable["cp"] < 0].iloc[5]
    r = an.check_position(fsl, p["K"], "P", SPOT)
    assert r["market_iv"] == pytest.approx(p["iv"]) and r["inside_data"] and r["fit_iv"] == pytest.approx(float(fsl.params.iv(r["k"], fsl.T)))
    assert r["p_itm"] > r["p_itm_flat"] > 0 and r["model_price"] > 0 and r["days"] == pytest.approx(fsl.slice.days)
    far = an.check_position(fsl, forward(fsl.T) * 0.2, "P", SPOT)
    assert far["market_iv"] is None and not far["inside_data"]
    call = an.check_position(fsl, p["K"], "C", SPOT)
    assert call["p_itm"] == pytest.approx(1 - r["p_itm"])


# ---------------------------------------------------------------------------
# constant maturity, snapshot metrics, differences
# ---------------------------------------------------------------------------

def test_constant_maturity_matches_the_generating_surface(fitted):
    cm = an.constant_maturity(fitted, days=(30, 60, 90, 120, 400))
    assert set(cm) == {30, 60, 90, 120}                                      # 400d is beyond the fitted expiries: skipped, not extrapolated
    for d, v in cm.items():
        T = d / 365
        assert v["atm"] == pytest.approx(float(true_params(T).iv(0.0, T)), abs=0.006) or d == 90    # 90d is interpolated between slices
        assert v["skew90"] > v["skew95"] > 0 > v["call105"]                  # puts rich, calls cheap: negative skew


def test_snapshot_metrics_contains_what_the_dynamics_charts_need(fitted):
    m = an.snapshot_metrics(fitted)
    assert {"atm30", "atm60", "atm90", "skew90_30", "term_30_90", "svi_rho", "svi_expiry", "spot", "asof"} <= set(m)
    assert m["term_30_90"] == pytest.approx(m["atm90"] - m["atm30"])


def test_surface_difference_is_zero_against_itself_and_shows_a_parallel_shift(fitted):
    z = an.surface_difference(fitted, fitted)
    assert z.notna().all().all() and np.allclose(z.to_numpy(), 0)
    shifted = model.fit_surface(surface.build_surface(make_quotes(params_fn=lambda T: SVI(true_params(T).a + 0.0006 * T / 0.25, *true_params(T).as_tuple()[1:])),
                                                      SPOT, ASOF), model.FitSettings(weights="equal", nstarts=9))
    d = an.surface_difference(fitted, shifted)
    assert d.notna().any().any() and (d.stack() > 0).mean() > 0.9


def test_percentile_of_last_is_the_own_history_rank():
    s = pd.Series(np.arange(300, dtype=float))
    assert an.percentile_of_last(s, 252, 60) == 100.0
    assert an.percentile_of_last(s.iloc[:30], 252, 60) is None


# ---------------------------------------------------------------------------
# snapshot store
# ---------------------------------------------------------------------------

@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(wc, "WHEEL_DB_PATH", str(tmp_path / "vs.db"))


def quoteset(asof=ASOF, symbol="SPX", source="Yahoo (delayed)", spot=SPOT):
    return data.QuoteSet(symbol, source, asof, spot, make_quotes(days=(30, 60)), ["2026-11-25"], ["a note"])


def test_snapshot_round_trip_preserves_every_quote(db):
    qs = quoteset()
    sid = store.save_snapshot(qs)
    back = store.load_snapshot(sid)
    assert back.symbol == "SPX" and back.source == qs.source and back.spot == SPOT and back.notes == ["a note"]
    assert pd.Timestamp(back.asof) == ASOF
    cols = ["expiry", "root", "strike", "right", "bid", "ask"]
    pd.testing.assert_frame_equal(back.quotes[cols].reset_index(drop=True), qs.quotes[cols].reset_index(drop=True), check_dtype=False)
    s1 = surface.build_surface(qs.quotes, qs.spot, qs.asof)
    s2 = surface.build_surface(back.quotes, back.spot, back.asof)
    assert [round(x.F, 6) for x in s1.slices] == [round(x.F, 6) for x in s2.slices]


def test_a_second_snapshot_within_the_gap_is_not_stored_again(db):
    a = store.save_snapshot(quoteset(ASOF))
    b = store.save_snapshot(quoteset(ASOF + pd.Timedelta(seconds=30)))
    c = store.save_snapshot(quoteset(ASOF + pd.Timedelta(minutes=10)))
    assert a == b and c != a
    assert len(store.list_snapshots("SPX")) == 2
    assert len(store.list_snapshots("SPY")) == 0


def test_metrics_are_cached_and_listed_oldest_first(db):
    store.save_snapshot(quoteset(ASOF + pd.Timedelta(minutes=10)), {"atm30": 0.2, "skew90_30": 0.05})
    first = store.save_snapshot(quoteset(ASOF))
    assert store.snapshot_metrics_df("SPX").shape[0] == 1
    store.update_metrics(first, {"atm30": 0.18, "skew90_30": 0.04})
    df = store.snapshot_metrics_df("SPX")
    assert list(df["atm30"]) == [0.18, 0.2] and df["asof"].is_monotonic_increasing


def test_snapshot_can_be_deleted_and_a_missing_one_is_an_error(db):
    sid = store.save_snapshot(quoteset())
    store.delete_snapshot(sid)
    with pytest.raises(KeyError):
        store.load_snapshot(sid)


# ---------------------------------------------------------------------------
# data helpers
# ---------------------------------------------------------------------------

def test_root_of_parses_option_symbols():
    assert data.root_of("SPXW261125C07700000", "X") == "SPXW"
    assert data.root_of("SPX270416P07000000", "X") == "SPX"
    assert data.root_of("AAPL261120C00250000", "X") == "AAPL"
    assert data.root_of("???", "FALLBACK") == "FALLBACK"


def test_default_expiries_cover_the_term_structure_without_duplicates():
    listed = [(ASOF.tz_localize(None).normalize() + pd.Timedelta(days=d)).strftime("%Y-%m-%d") for d in (1, 2, 5, 9, 16, 30, 33, 47, 63, 100, 130, 190, 280, 380, 700)]
    chosen = data.default_expiries(listed, ASOF)
    assert chosen == sorted(set(chosen)) and 6 <= len(chosen) <= 12
    assert listed[0] not in chosen and listed[1] not in chosen                # nothing under the minimum days
    assert data.default_expiries([], ASOF) == []
