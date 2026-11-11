"""
tests/test_ivscreen_metrics.py — percentile maths, the classification bands (every boundary), trends and the
IV-vs-RV relationship rules of the IV screener. Pure functions, synthetic series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import wheel_vol as wv
from ivscreen import metrics as mx
from ivscreen.config import IVConfig


def series(values, start="2023-01-02"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


# ---------------------------------------------------------------------------
# percentile rank
# ---------------------------------------------------------------------------

def test_today_is_the_highest_value_in_the_window_is_the_100th_percentile():
    s = series(list(range(1, 201)))
    assert mx.rank(s, 100, 20) == (100.0, 100)


def test_today_is_the_lowest_value_is_the_0th_percentile():
    s = series(list(range(200, 0, -1)))
    assert mx.rank(s, 100, 20)[0] == 0.0


def test_strictly_below_ties_do_not_count():
    s = series([5.0] * 50)                           # flat: nothing is below today
    assert mx.rank(s, 40, 20)[0] == 0.0


def test_today_is_excluded_from_its_own_window():
    s = series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3 + [5])
    pct, n = mx.rank(s, 10, 5)
    assert n == 10 and pct == 40.0                   # prior 10 are 1..10; four of them (1,2,3,4) are below 5


def test_lookback_limits_the_window():
    s = series([100.0] * 100 + [1.0] * 30 + [50.0])  # the 100s are older than a 30-day lookback
    assert mx.rank(s, 30, 20)[0] == 100.0            # all 30 prior readings (1.0) are below 50
    assert mx.rank(s, 130, 20)[0] == pytest.approx(30 / 130 * 100, abs=1e-4)


@pytest.mark.parametrize("lookback", [60, 126, 252])
def test_matches_the_volatility_page_definition(lookback):
    rng = np.random.default_rng(3)
    s = series(np.abs(rng.normal(0.3, 0.08, 400)))
    assert mx.rank(s, lookback, 20)[0] == pytest.approx(wv.percentile_rank(s, lookback), abs=1e-4)


def test_too_little_history_is_none_never_zero():
    s = series([0.3] * 30)
    assert mx.rank(s, 252, 126) == (None, 29)
    assert mx.rank(series([]), 252, 20) == (None, 0)


def test_nan_readings_are_ignored_not_counted_as_zero():
    vals = list(np.linspace(0.2, 0.4, 100))
    s = series(vals)
    s.iloc[10:20] = np.nan
    pct, n = mx.rank(s, 252, 20)
    assert n == 89 and pct == 100.0


def test_offset_ranks_an_earlier_day_against_the_window_before_it():
    s = series(list(range(1, 101)) + [1.0] * 5)      # a climb then a collapse
    assert mx.rank(s, 252, 20)[0] == 0.0
    assert mx.rank(s, 252, 20, offset=5)[0] == 100.0   # day 100 (value 100) was the highest so far


def test_rolling_rank_agrees_with_rank_at_every_date():
    rng = np.random.default_rng(1)
    s = series(np.abs(rng.normal(0.3, 0.1, 160)))
    roll = mx.rolling_rank(s, 50, 30)
    for off in (0, 7, 40):
        end = len(s) - off
        expected, _ = mx.rank(s.iloc[:end], 50, 30)
        assert roll.iloc[end - 1] == pytest.approx(expected, abs=1e-4)
    assert roll.iloc[:30].isna().all()


# ---------------------------------------------------------------------------
# bands: every boundary
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pct,expected", [
    (0, mx.VERY_LOW), (10.0, mx.VERY_LOW), (9.99, mx.VERY_LOW),
    (10.01, mx.LOW), (25.0, mx.LOW), (24.99, mx.LOW),
    (25.01, mx.NORMAL), (50, mx.NORMAL), (74.99, mx.NORMAL),
    (75.0, mx.HIGH), (89.99, mx.HIGH),
    (90.0, mx.VERY_HIGH), (100, mx.VERY_HIGH),
])
def test_default_band_boundaries(pct, expected):
    assert mx.classify(pct) == expected


def test_missing_percentile_is_na_not_very_low():
    assert mx.classify(None) == mx.NA
    assert mx.classify(float("nan")) == mx.NA


def test_float_noise_on_an_edge_does_not_flip_the_band():
    assert mx.classify(75.0000000001) == mx.HIGH and mx.classify(74.9999999999) == mx.HIGH
    assert mx.classify(10.0000000001) == mx.VERY_LOW


def test_bands_are_configurable():
    assert mx.classify(30, very_low_max=5, low_max=35, high_min=60, very_high_min=80) == mx.LOW
    assert mx.classify(80, very_low_max=5, low_max=35, high_min=60, very_high_min=80) == mx.VERY_HIGH


def test_exact_edge_from_real_counts():
    # 189 of 252 prior readings below today is exactly the 75th percentile -> High
    hist = [0.0] * 189 + [10.0] * 63
    s = series(hist + [5.0])
    pct, n = mx.rank(s, 252, 100)
    assert (pct, n) == (75.0, 252) and mx.classify(pct) == mx.HIGH


# ---------------------------------------------------------------------------
# trend
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("now,before,label,chg", [
    (60, 50, "Rising", 10.0), (50, 60, "Falling", -10.0), (52, 50, "Flat", 2.0),
    (55, 50, "Rising", 5.0), (45, 50, "Falling", -5.0),                  # exactly at the tolerance counts as a move
])
def test_trend(now, before, label, chg):
    assert mx.trend(now, before, 5.0) == (label, chg)


def test_trend_unknown_when_either_end_is_missing():
    assert mx.trend(None, 40, 5.0) == (mx.NA, None) and mx.trend(40, None, 5.0) == (mx.NA, None)


def test_trend_relationship_text():
    assert "following IV up" in mx.trend_relationship("Rising", "Rising")
    assert "not following" in mx.trend_relationship("Rising", "Flat")
    assert "following IV down" in mx.trend_relationship("Falling", "Falling")
    assert mx.trend_relationship("n/a", "Rising") == "n/a"


# ---------------------------------------------------------------------------
# IV vs RV relationship
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("iv,rv,expected", [
    (90, 40, mx.REL_IV_ELEVATED), (80, 55, mx.REL_IV_ELEVATED),            # gap exactly 25 counts
    (30, 80, mx.REL_RV_ELEVATED), (45, 70, mx.REL_RV_ELEVATED),
    (85, 80, mx.REL_BOTH_HIGH), (75, 75, mx.REL_BOTH_HIGH),
    (10, 20, mx.REL_BOTH_LOW), (25, 0, mx.REL_BOTH_LOW),                   # low/low wins over a 25-point gap
    (50, 50, mx.REL_IN_LINE), (60, 45, mx.REL_IN_LINE), (79, 55, mx.REL_IN_LINE),
])
def test_relationship_rules(iv, rv, expected):
    assert mx.relationship(iv, rv) == expected


def test_relationship_unavailable_without_rv():
    assert mx.relationship(80, None) == mx.REL_NA and mx.relationship(None, 80) == mx.REL_NA


def test_relationship_gap_is_configurable():
    assert mx.relationship(70, 50, gap=15) == mx.REL_IV_ELEVATED
    assert mx.relationship(70, 50, gap=25) == mx.REL_IN_LINE


def test_no_relationship_note_calls_options_overpriced():
    for note in mx.REL_NOTE.values():
        assert "overpriced" not in note.lower() or "not by itself evidence" in note.lower()


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def test_config_defaults_match_the_spec():
    c = IVConfig()
    assert (c.very_low_max, c.low_max, c.high_min, c.very_high_min) == (10, 25, 75, 90)
    assert c.problems() == []


def test_config_rejects_inconsistent_bands_and_history():
    assert IVConfig(low_max=5).problems()                          # low must exceed very low
    assert IVConfig(high_min=95).problems()                        # high must be below very high
    assert IVConfig(min_history_days=600, lookback_days=504).problems()


def test_config_roundtrip_ignores_unknown_and_bad_values():
    c = IVConfig.from_dict({"lookback_days": "756", "rv_window": "x", "bogus": 1, "etf_symbols": ["spy", " qqq "]})
    assert c.lookback_days == 756 and c.rv_window == 20 and c.etf_symbols == ["SPY", "QQQ"]
