"""
ivscreen/metrics.py — the maths behind the screener, free of any I/O.

Percentile (McMillan, own history only): the share of the lookback window's PRIOR observations
that were strictly below today's value, 0-100. Today is excluded from the window it is ranked
against. A window with fewer than ``min_obs`` prior observations gives None — a missing or too-short
history is "not available", never zero and never a guess.

Percentiles are rounded to 6 decimals so a value that is exactly on a band edge (e.g. 75) is not
pushed to either side by float noise; every comparison with a threshold uses the rounded value.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

import wheel_vol as wv

DECIMALS = 6

# classification labels
VERY_LOW, LOW, NORMAL, HIGH, VERY_HIGH, NA = "Very low", "Low", "Normal", "High", "Very high", "n/a"
BANDS = (VERY_LOW, LOW, NORMAL, HIGH, VERY_HIGH)

# IV vs RV relationship labels
REL_IV_ELEVATED = "IV elevated vs RV"
REL_RV_ELEVATED = "RV elevated vs IV"
REL_BOTH_HIGH = "Both elevated"
REL_BOTH_LOW = "Both subdued"
REL_IN_LINE = "In line"
REL_NA = "RV unavailable"
RELATIONSHIPS = (REL_IV_ELEVATED, REL_RV_ELEVATED, REL_BOTH_HIGH, REL_BOTH_LOW, REL_IN_LINE, REL_NA)

REL_NOTE = {
    REL_IV_ELEVATED: ("Options are pricing more volatility than recent stock movement implies, each measured against its "
                      "own history. That can reflect a coming event, market stress or expected movement — it is not by "
                      "itself evidence that the options are overpriced."),
    REL_RV_ELEVATED: ("The stock has been moving more than options imply, each measured against its own history. "
                      "Implied volatility may be lagging a change, or may be expecting the movement to fade."),
    REL_BOTH_HIGH: "Both implied and realized volatility are high for this stock: the stock itself is in a volatile phase.",
    REL_BOTH_LOW: "Both implied and realized volatility are low for this stock: a quiet phase for the shares and the options.",
    REL_IN_LINE: "Implied and realized volatility sit at similar points of their own histories.",
    REL_NA: "No usable realized-volatility history, so the two cannot be compared.",
}


def rank(series: pd.Series, lookback: int, min_obs: int, offset: int = 0) -> tuple[Optional[float], int]:
    """(percentile, prior observations used). *offset* ranks the value *offset* observations
    before the last one (used for trends), against the window that preceded it."""
    s = series.dropna()
    end = len(s) - offset
    if end < 1:
        return None, 0
    current = s.iloc[end - 1]
    hist = s.iloc[max(0, end - 1 - lookback):end - 1]
    n = len(hist)
    if n < max(min_obs, 1):
        return None, n
    return round(float((hist.to_numpy() < current).mean() * 100), DECIMALS), n


def rolling_rank(series: pd.Series, lookback: int, min_obs: int) -> pd.Series:
    """The percentile at every date (for charts): each value against the window before it."""
    s = series.dropna()
    v = s.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    for i in range(max(min_obs, 1), len(v)):
        hist = v[max(0, i - lookback):i]
        out[i] = (hist < v[i]).mean() * 100
    return pd.Series(np.round(out, DECIMALS), index=s.index)


def classify(pct: Optional[float], very_low_max=10.0, low_max=25.0, high_min=75.0, very_high_min=90.0) -> str:
    """Very low <= 10 < Low <= 25 < Normal < 75 <= High < 90 <= Very high (defaults)."""
    if pct is None or pd.isna(pct):
        return NA
    p = round(float(pct), DECIMALS)
    if p >= very_high_min:
        return VERY_HIGH
    if p >= high_min:
        return HIGH
    if p <= very_low_max:
        return VERY_LOW
    if p <= low_max:
        return LOW
    return NORMAL


def trend(now: Optional[float], before: Optional[float], tolerance: float) -> tuple[str, Optional[float]]:
    """("Rising" | "Falling" | "Flat" | "n/a", change in percentile points)."""
    if now is None or before is None:
        return NA, None
    change = round(now - before, DECIMALS)
    if change >= tolerance:
        return "Rising", change
    if change <= -tolerance:
        return "Falling", change
    return "Flat", change


_TREND_TEXT = {
    ("Rising", "Rising"): "Both rising — RV is following IV up",
    ("Rising", "Flat"): "IV rising, RV not following",
    ("Rising", "Falling"): "IV rising while RV falls",
    ("Falling", "Falling"): "Both easing — RV is following IV down",
    ("Falling", "Flat"): "IV easing, RV steady",
    ("Falling", "Rising"): "IV easing while RV rises",
    ("Flat", "Rising"): "RV rising, IV steady",
    ("Flat", "Falling"): "RV easing, IV steady",
    ("Flat", "Flat"): "Both steady",
}


def trend_relationship(iv_trend: str, rv_trend: str) -> str:
    return _TREND_TEXT.get((iv_trend, rv_trend), "n/a")


def relationship(iv_pct: Optional[float], rv_pct: Optional[float], low_max=25.0, high_min=75.0, gap=25.0) -> str:
    """How the two percentiles relate. Precedence: both low -> both subdued; a gap of at least
    *gap* points -> the higher one is "elevated" relative to the other; both high -> both elevated."""
    if iv_pct is None or rv_pct is None:
        return REL_NA
    diff = round(iv_pct - rv_pct, DECIMALS)
    if iv_pct <= low_max and rv_pct <= low_max:
        return REL_BOTH_LOW
    if diff >= gap:
        return REL_IV_ELEVATED
    if diff <= -gap:
        return REL_RV_ELEVATED
    if iv_pct >= high_min and rv_pct >= high_min:
        return REL_BOTH_HIGH
    return REL_IN_LINE


def realized_vol(close: pd.Series, window: int) -> pd.Series:
    """Annualized close-to-close realized volatility (same definition as the Volatility page)."""
    return wv.realized_vol(close, window).dropna()
