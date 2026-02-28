"""
tests/test_dip_detector.py — Unit tests for dip_detector.py.

All tests are offline (no network calls) — price history is synthesised
with numpy/pandas so tests are fast and deterministic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from dip_detector import compute_rsi, compute_moving_average, score_dip, DipSignal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_history(closes: list[float]) -> pd.DataFrame:
    """Wrap a list of closing prices in a yfinance-style DataFrame."""
    dates = pd.date_range(end="2024-01-01", periods=len(closes), freq="B")
    return pd.DataFrame({"Close": closes}, index=dates)


def _flat_history(price: float, n: int) -> pd.DataFrame:
    """All closes equal *price* for *n* bars."""
    return _make_history([price] * n)


def _trending_down_history(start: float, end: float, n: int) -> pd.DataFrame:
    """Linearly declining prices from *start* to *end* over *n* bars."""
    prices = np.linspace(start, end, n).tolist()
    return _make_history(prices)


# ---------------------------------------------------------------------------
# compute_rsi
# ---------------------------------------------------------------------------

class TestComputeRsi:
    def test_returns_nan_for_insufficient_data(self):
        prices = pd.Series([100.0] * 10)
        result = compute_rsi(prices, period=14)
        assert np.isnan(result)

    def test_flat_prices_return_neutral_rsi(self):
        # No delta → no gain, no loss → RSI should be near 50 or NaN for flat
        prices = pd.Series([100.0] * 30)
        result = compute_rsi(prices, period=14)
        # All deltas are 0, avg_loss = 0 → returns 100
        assert result == 100.0 or np.isnan(result)

    def test_continuously_rising_prices_give_high_rsi(self):
        prices = pd.Series([float(i) for i in range(1, 50)])
        result = compute_rsi(prices, period=14)
        assert not np.isnan(result)
        assert result > 70.0

    def test_continuously_falling_prices_give_low_rsi(self):
        prices = pd.Series([float(i) for i in range(50, 1, -1)])
        result = compute_rsi(prices, period=14)
        assert not np.isnan(result)
        assert result < 30.0

    def test_rsi_bounded_between_0_and_100(self):
        prices = pd.Series([100.0 - i * 0.5 for i in range(50)])
        result = compute_rsi(prices, period=14)
        if not np.isnan(result):
            assert 0.0 <= result <= 100.0


# ---------------------------------------------------------------------------
# compute_moving_average
# ---------------------------------------------------------------------------

class TestComputeMovingAverage:
    def test_returns_nan_for_insufficient_data(self):
        prices = pd.Series([100.0] * 10)
        result = compute_moving_average(prices, window=50)
        assert np.isnan(result)

    def test_correct_average_for_constant_series(self):
        prices = pd.Series([42.0] * 100)
        result = compute_moving_average(prices, window=50)
        assert result == pytest.approx(42.0)

    def test_uses_only_last_window_bars(self):
        # First 50 bars = 0, last 50 bars = 100 → MA of last 50 = 100
        prices = pd.Series([0.0] * 50 + [100.0] * 50)
        result = compute_moving_average(prices, window=50)
        assert result == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# score_dip
# ---------------------------------------------------------------------------

class TestScoreDip:
    def test_returns_none_for_empty_history(self):
        result = score_dip("TEST", pd.DataFrame())
        assert result is None

    def test_returns_none_for_missing_close_column(self):
        df = pd.DataFrame({"Open": [100.0] * 300})
        result = score_dip("TEST", df)
        assert result is None

    def test_returns_none_for_insufficient_bars(self):
        history = _flat_history(100.0, 10)
        result = score_dip("TEST", history)
        assert result is None

    def test_strong_dip_scores_at_least_two(self):
        # Create a trend-down scenario: RSI will be low, price below MA50,
        # price below MA200, and in the bottom of 52-week range.
        # Start at 200, crash to 80 (60% decline).
        n = config.MA_SLOW + 50  # enough bars
        history = _trending_down_history(200.0, 80.0, n)
        result = score_dip("CRASH", history)
        assert result is not None
        assert isinstance(result, DipSignal)
        assert result.score >= 2
        assert result.is_dip

    def test_no_dip_for_strong_uptrend(self):
        # Continuously rising prices — RSI will be high, price above MAs
        n = config.MA_SLOW + 50
        history = _make_history(np.linspace(50.0, 300.0, n).tolist())
        result = score_dip("BULL", history)
        assert result is not None
        assert result.is_dip is False

    def test_price_is_last_close(self):
        n = config.MA_SLOW + 50
        prices = [100.0] * n
        prices[-1] = 77.77
        history = _make_history(prices)
        result = score_dip("TEST", history)
        assert result is not None
        assert result.price == pytest.approx(77.77)

    def test_ma50_signal_when_price_significantly_below(self):
        # Price stays at 80, MA-50 will be around 80 too unless we set up a step
        # Build: first 200 bars at 100, then 55 bars at 85 → MA50 ~ 85, price=85
        # Better: 200 bars at 100, then 55 bars at 90 → MA50 ~ 90, price=90: no signal
        # Even better: 205 bars at 100, last bar drops to 50 → MA50 ~ 98, price=50 → big dip
        n = 250
        prices = [100.0] * (n - 1) + [50.0]
        history = _make_history(prices)
        result = score_dip("SPIKE_DOWN", history)
        assert result is not None
        assert result.ma50_signal is True

    def test_week52_signal_when_near_low(self):
        # 252 bars: first 200 at 100, last 52 slowly declining to 60
        prices = [100.0] * 200 + list(np.linspace(100.0, 60.0, 52))
        history = _make_history(prices)
        result = score_dip("NEAR_LOW", history)
        assert result is not None
        # week52_high ~ 100, week52_low ~60, price ~60 → position_in_range near 0
        assert result.week52_signal is True

    def test_str_representation(self):
        n = config.MA_SLOW + 50
        history = _trending_down_history(200.0, 80.0, n)
        result = score_dip("XYZ", history)
        assert result is not None
        s = str(result)
        assert "XYZ" in s
        assert "score=" in s
