"""
tests/test_stockscreen_technicals.py — price position, percentiles, indicators,
trend and price-zone classification, on synthetic price series with known answers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockscreen import technicals as tc
from stockscreen.config import ScreenConfig

CFG = ScreenConfig()


def series(values, end="2026-10-20"):
    idx = pd.bdate_range(end=end, periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


def ohlcv(close_values, end="2026-10-20"):
    c = series(close_values, end)
    return pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c, "Volume": 1_000.0})


def ramp(n, start, stop):
    return list(np.linspace(start, stop, n))


# --- range position -------------------------------------------------------------------------------

def scenario(final: float, n: int = 400) -> pd.Series:
    """A long history at 150 with a low of 100 and a high of 200 well inside the last year, then *final*."""
    v = [150.0] * n
    v[-200], v[-100] = 100.0, 200.0
    v[-1] = final
    return series(v)


def test_range_position_at_low_midpoint_and_high():
    assert tc.price_position(scenario(100.0), 1)["range_pos"] == pytest.approx(0.0)        # at the 52-week low
    assert tc.price_position(scenario(150.0), 1)["range_pos"] == pytest.approx(50.0)       # midpoint
    assert tc.price_position(scenario(200.0), 1)["range_pos"] == pytest.approx(100.0)      # at the high
    p = tc.price_position(scenario(125.0), 1)
    assert (p["low"], p["high"]) == (100.0, 200.0) and p["range_pos"] == pytest.approx(25.0)


def test_distances_from_high_and_low():
    p = tc.price_position(series(ramp(260, 100, 200) + [160.0]), 1)
    assert p["pct_below_high"] == pytest.approx((200 - 160) / 200 * 100, abs=0.5)
    assert p["pct_above_low"] == pytest.approx((160 - p["low"]) / p["low"] * 100)


def test_zero_high_low_range_is_undefined_not_zero():
    p = tc.price_position(series([50.0] * 300), 1)
    assert p["available"] and p["range_pos"] is None


def test_insufficient_history_marks_longer_lookbacks_unavailable():
    pos = tc.price_positions(series(ramp(400, 80, 120)))             # ~1.6 years of data
    assert pos["52w"]["available"] and not pos["2y"]["available"] and not pos["3y"]["available"]
    assert "history starts" in pos["2y"]["reason"]


# --- percentile --------------------------------------------------------------------------------------------

def test_percentile_excludes_the_current_observation():
    s = series([1, 2, 3, 4, 5] + [5.0] * 0)
    # window is the whole short series only when history covers 1y; build a 1y+ history ending 1,2,3,4,5
    hist = series([3.0] * 300 + [1, 2, 3, 4, 5])
    p = tc.price_position(hist, 1)
    # the latest close (5) is above all but the 3.0 plateau... compare against the PREVIOUS closes only
    prev = hist[hist.index >= tc.window_start(hist.index[-1], 1)].iloc[:-1]
    expected = ((prev < 5).sum() + 0.5 * (prev == 5).sum()) / len(prev) * 100
    assert p["percentile"] == pytest.approx(expected)


def test_percentile_top_and_bottom_and_ties():
    up = tc.price_position(series(ramp(300, 10, 100)), 1)
    assert up["percentile"] == pytest.approx(100.0)                   # above every prior close
    down = tc.price_position(series(ramp(300, 100, 10)), 1)
    assert down["percentile"] == pytest.approx(0.0)                   # below every prior close
    flat = tc.price_position(series([50.0] * 300), 1)
    assert flat["percentile"] == pytest.approx(50.0)                  # ties count half


def test_high_low_range_position_and_percentile_are_different_questions():
    # a spike to 200 then a long stay near 100: price 105 is near the bottom of the range but mid-distribution
    vals = [100.0] * 250 + [200.0] + [100.0] * 100 + [104.0] * 50 + [105.0]
    p = tc.price_position(series(vals), 1)
    assert p["range_pos"] < 10
    assert p["percentile"] > 50


# --- indicators ---------------------------------------------------------------------------------------------

def test_sma_ema():
    s = series([1, 2, 3, 4, 5, 6])
    assert tc.sma(s, 3).iloc[-1] == pytest.approx(5.0) and np.isnan(tc.sma(s, 3).iloc[1])
    assert tc.ema(series([10.0] * 20), 5).iloc[-1] == pytest.approx(10.0)


def test_rsi_matches_hand_calculation_and_extremes():
    closes = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28]
    r = tc.rsi(series(closes))
    # first 14 changes: gains sum 3.34, losses sum 1.40 -> RS = 2.3857, RSI = 70.46
    assert r.iloc[14] == pytest.approx(70.46, abs=0.01)
    assert np.isnan(r.iloc[13])
    assert tc.rsi(series(ramp(60, 10, 70))).iloc[-1] == pytest.approx(100.0)
    assert tc.rsi(series(ramp(60, 70, 10))).iloc[-1] == pytest.approx(0.0, abs=0.01)
    assert tc.rsi(series([50.0] * 60)).iloc[-1] == pytest.approx(50.0)


def test_rsi_equals_an_independent_wilder_implementation():
    rng = np.random.default_rng(7)
    c = 100 + np.cumsum(rng.normal(0, 1, 200))
    diff = np.diff(c)
    gains, losses = np.clip(diff, 0, None), np.clip(-diff, 0, None)
    ag, al = gains[:14].mean(), losses[:14].mean()
    ref = [100 - 100 / (1 + ag / al)]
    for g, l in zip(gains[14:], losses[14:]):
        ag, al = (ag * 13 + g) / 14, (al * 13 + l) / 14
        ref.append(100 - 100 / (1 + ag / al))
    assert tc.rsi(series(c)).iloc[14:].to_numpy() == pytest.approx(np.array(ref))


def test_macd_constant_ramp_and_crossover():
    assert tc.macd(series([100.0] * 80))["macd"].abs().max() < 1e-9
    up = tc.macd(series(ramp(120, 50, 150)))
    assert up["macd"].iloc[-1] > 0
    m = tc.macd(series(ramp(120, 50, 150)))
    assert (m["hist"] == m["macd"] - m["signal"]).all()


def test_atr_and_adx_are_positive_on_a_trending_series():
    df = ohlcv(ramp(200, 50, 150))
    assert tc.atr(df["High"], df["Low"], df["Close"]).iloc[-1] > 0
    assert tc.adx(df["High"], df["Low"], df["Close"]).iloc[-1] > 25           # a clean trend has strong ADX


def test_snapshot_values_none_when_history_is_too_short():
    snap = tc.indicator_snapshot(ohlcv(ramp(60, 50, 60)))
    assert snap["sma200"] is None and snap["sma50"] is not None
    assert tc.classify_trend(snap, CFG)["label"] == "Insufficient data"       # needs a 200-day average


def test_exactly_199_bars_has_no_sma200_and_200_plus_window_does():
    assert tc.indicator_snapshot(ohlcv(ramp(199, 50, 60)))["sma200"] is None
    assert tc.indicator_snapshot(ohlcv(ramp(230, 50, 60)))["sma200"] is not None


# --- trend classification -----------------------------------------------------------------------------------

def classify(values):
    return tc.classify_trend(tc.indicator_snapshot(ohlcv(values), CFG.slope_window), CFG)


def test_strong_uptrend():
    assert classify(ramp(320, 50, 150))["label"] == "Strong uptrend"


def test_downtrend_without_recovery():
    assert classify(ramp(320, 150, 50))["label"] == "Downtrend"


def test_healthy_pullback_long_term_up_short_term_dip():
    vals = ramp(300, 50, 150) + ramp(15, 150, 138)          # rally, then a pullback below SMA20 but far above SMA200
    r = classify(vals)
    assert r["label"] in ("Healthy pullback", "Strong uptrend")
    vals = ramp(300, 50, 150) + ramp(25, 150, 128)
    assert classify(vals)["label"] == "Healthy pullback"    # slipped under SMA50, long-term structure intact


def test_reversal_candidate_after_a_decline_with_recovery_signals():
    vals = ramp(280, 150, 60) + ramp(8, 60, 66) + ramp(6, 66, 70)
    r = classify(vals)
    assert r["label"] == "Trend reversal candidate" and len(r["signals"]) >= 1


def test_sideways_when_flat():
    rng = np.random.default_rng(3)
    assert classify(list(100 + rng.normal(0, 0.4, 320)))["label"] == "Sideways/consolidating"   # noise around a flat level
    assert classify([100.0] * 320)["label"] == "Sideways/consolidating"                        # no direction at all, not a "downtrend"


def test_low_rsi_alone_does_not_make_an_uptrend():
    r = classify(ramp(320, 150, 40))
    assert r["label"] != "Strong uptrend" and tc.rsi_band(15.0, CFG) == "oversold"


def test_rsi_bands_and_macd_status():
    assert [tc.rsi_band(v, CFG) for v in (25, 35, 50, 65, 75)] == ["oversold", "weak momentum", "neutral", "positive momentum", "overbought"]
    assert tc.rsi_band(None, CFG) == "n/a"
    assert tc.macd_status({"macd_hist": 0.5, "macd_hist_prev": 0.2, "macd_cross": None}) == "bullish, strengthening"
    assert "bullish crossover" in tc.macd_status({"macd_hist": 0.5, "macd_hist_prev": -0.1, "macd_cross": "bullish", "macd_cross_days_ago": 2})


# --- price zone --------------------------------------------------------------------------------------------------

def pos(rp52=None, rp2=None, rp3=None, pc52=50.0, pc2=50.0, pc3=50.0, avail=(True, True, True)):
    def one(rp, pc, a):
        return {"available": a, "range_pos": rp, "percentile": pc} if a else {"available": False}
    return {"52w": one(rp52, pc52, avail[0]), "2y": one(rp2, pc2, avail[1]), "3y": one(rp3, pc3, avail[2])}


def test_zone_thresholds_are_inclusive_and_independent():
    assert tc.classify_price_zone(pos(20.0, 50, 50), CFG)["label"] == "Low-price candidate"     # exactly 20 counts
    assert tc.classify_price_zone(pos(20.1, 50, 50), CFG)["label"] == "Mid-range"
    assert tc.classify_price_zone(pos(50, 50, 50, pc2=20.0), CFG)["in_zone"] == ["2y"]          # percentile alone qualifies


def test_zone_deep_needs_two_lookbacks():
    assert tc.classify_price_zone(pos(10, 10, 60), CFG)["label"] == "Deep value-zone candidate"
    assert tc.classify_price_zone(pos(10, 60, 60), CFG)["label"] == "Low-price candidate"


def test_zone_one_lookback_counts_once_even_if_range_and_percentile_both_low():
    z = tc.classify_price_zone(pos(5, 60, 60, pc52=5), CFG)
    assert z["in_zone"] == ["52w"] and z["label"] == "Low-price candidate"


def test_zone_near_highs_and_insufficient_history():
    assert tc.classify_price_zone(pos(95, 80, 70), CFG)["label"] == "Near highs"
    assert tc.classify_price_zone(pos(50, 50, 50, avail=(False, False, False)), CFG)["label"] == "Insufficient history"
    # only the 52-week window is available: it still classifies
    assert tc.classify_price_zone(pos(10, avail=(True, False, False)), CFG)["label"] == "Low-price candidate"


def test_zone_zero_range_is_not_a_false_low():
    z = tc.classify_price_zone(pos(None, None, None, pc52=50, pc2=50, pc3=50), CFG)
    assert z["label"] == "Mid-range"
