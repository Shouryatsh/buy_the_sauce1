"""
dip_detector.py — Identifies dip-buying opportunities using a multi-signal
scoring approach.

Each signal contributes 1 point to a score.  A stock must reach
MIN_DIP_SCORE (from config) before it is considered a dip opportunity.

Signals
-------
1. RSI oversold   : 14-day RSI < RSI_OVERSOLD (default 35)
2. Below MA-50    : price ≥ DIP_FROM_MA50_PCT below the 50-day MA
3. Below MA-200   : price is below the 200-day MA (longer-term weakness)
4. 52-week range  : price is in the bottom WEEK52_LOWER_BAND of its
                    52-week high-low range
5. ML prediction  : Random Forest predicts UP in the next ML_PREDICT_HORIZON
                    days with at least MEDIUM confidence (optional, controlled
                    by config.ML_ENABLED and config.ML_GATE_BUY_SIGNAL)

A stock that triggers all four technical signals has a perfect score of 4.
The ML signal does NOT add to the numeric score — instead it gates whether
a dip is promoted to a full BUY SIGNAL (when config.ML_GATE_BUY_SIGNAL=True).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
class DipSignal:
    """Result returned for each ticker after scoring."""
    symbol: str
    score: int
    rsi: float
    rsi_signal: bool
    price: float
    ma50: float
    ma50_signal: bool
    ma200: float
    ma200_signal: bool
    week52_high: float
    week52_low: float
    week52_signal: bool
    is_dip: bool          # score >= MIN_DIP_SCORE
    # ML prediction fields (None when ML is disabled or has insufficient data)
    ml_direction: str | None = None     # "UP" | "DOWN" | None
    ml_probability: float | None = None # probability of predicted direction
    ml_confidence: str | None = None    # "HIGH" | "MEDIUM" | "LOW" | None
    ml_buy_confirmed: bool = False      # True when ML predicts UP with >= MEDIUM confidence

    def __str__(self) -> str:
        signals = []
        if self.rsi_signal:
            signals.append(f"RSI={self.rsi:.1f}")
        if self.ma50_signal:
            signals.append(f"below MA50 by {((self.ma50 - self.price)/self.ma50*100):.1f}%")
        if self.ma200_signal:
            signals.append(f"below MA200")
        if self.week52_signal:
            pct = (self.price - self.week52_low) / max(self.week52_high - self.week52_low, 1e-6)
            signals.append(f"52wk-range {pct*100:.0f}%")
        return (
            f"{self.symbol}: score={self.score} "
            f"[{', '.join(signals) if signals else 'no signals'}] "
            f"is_dip={self.is_dip}"
        )


# ---------------------------------------------------------------------------
# Technical helpers
# ---------------------------------------------------------------------------

def compute_rsi(prices: pd.Series, period: int = config.RSI_PERIOD) -> float:
    """Return the most recent RSI value for a price series.

    Uses Wilder's smoothing (same as TradingView / most platforms).
    Returns NaN if there is insufficient data.
    """
    if len(prices) < period + 1:
        return float("nan")

    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Seed with simple averages for the first period
    avg_gain = gain.iloc[1 : period + 1].mean()
    avg_loss = loss.iloc[1 : period + 1].mean()

    # Wilder smoothing for subsequent bars
    for g, l in zip(gain.iloc[period + 1 :], loss.iloc[period + 1 :]):
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_moving_average(prices: pd.Series, window: int) -> float:
    """Return the most recent simple moving average value.

    Returns NaN if there is insufficient data.
    """
    if len(prices) < window:
        return float("nan")
    return float(prices.iloc[-window:].mean())


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def score_dip(symbol: str, history: pd.DataFrame) -> DipSignal | None:
    """Score a stock for dip-buying potential.

    Parameters
    ----------
    symbol:
        Ticker symbol.
    history:
        DataFrame with at minimum a ``Close`` column, sorted oldest → newest.
        Typically 1 year of daily OHLCV from yfinance.

    Returns
    -------
    DipSignal or None if there is insufficient data.
    """
    if history is None or history.empty or "Close" not in history.columns:
        logger.warning("%s: insufficient history, skipping", symbol)
        return None

    closes = history["Close"].dropna()
    if len(closes) < config.MA_SLOW + 5:
        logger.warning("%s: not enough bars (%d), need %d+", symbol, len(closes), config.MA_SLOW + 5)
        return None

    price = float(closes.iloc[-1])
    rsi = compute_rsi(closes)
    ma50 = compute_moving_average(closes, config.MA_FAST)
    ma200 = compute_moving_average(closes, config.MA_SLOW)
    week52_high = float(closes.tail(252).max())
    week52_low = float(closes.tail(252).min())

    # --- Individual signals ---
    rsi_signal = (not np.isnan(rsi)) and (rsi < config.RSI_OVERSOLD)

    ma50_signal = (
        (not np.isnan(ma50))
        and ma50 > 0
        and (ma50 - price) / ma50 >= config.DIP_FROM_MA50_PCT
    )

    ma200_signal = (not np.isnan(ma200)) and (price < ma200)

    range52 = week52_high - week52_low
    if range52 > 0:
        position_in_range = (price - week52_low) / range52
        week52_signal = position_in_range <= config.WEEK52_LOWER_BAND
    else:
        week52_signal = False

    score = sum([rsi_signal, ma50_signal, ma200_signal, week52_signal])
    is_dip = score >= config.MIN_DIP_SCORE

    # --- ML prediction (optional) ---
    ml_direction    = None
    ml_probability  = None
    ml_confidence   = None
    ml_buy_confirmed = False

    if config.ML_ENABLED:
        try:
            from ml_predictor import predict as ml_predict
            ml_pred = ml_predict(symbol, history)
            if ml_pred is not None:
                ml_direction   = ml_pred.direction
                ml_probability = ml_pred.probability
                ml_confidence  = ml_pred.confidence
                # Confirmed = ML predicts UP with at least MEDIUM confidence
                ml_buy_confirmed = (
                    ml_pred.direction == "UP"
                    and ml_pred.confidence in ("MEDIUM", "HIGH")
                )
        except Exception as exc:
            logger.debug("%s: ML prediction failed — %s", symbol, exc)

    result = DipSignal(
        symbol=symbol,
        score=score,
        rsi=rsi if not np.isnan(rsi) else -1.0,
        rsi_signal=rsi_signal,
        price=price,
        ma50=ma50 if not np.isnan(ma50) else 0.0,
        ma50_signal=ma50_signal,
        ma200=ma200 if not np.isnan(ma200) else 0.0,
        ma200_signal=ma200_signal,
        week52_high=week52_high,
        week52_low=week52_low,
        week52_signal=week52_signal,
        is_dip=is_dip,
        ml_direction=ml_direction,
        ml_probability=ml_probability,
        ml_confidence=ml_confidence,
        ml_buy_confirmed=ml_buy_confirmed,
    )
    logger.info(str(result))
    return result
