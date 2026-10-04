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
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

import config

if TYPE_CHECKING:
    from ml_predictor import MultiHorizonOutlook, SwingMetrics

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
    ml_auroc_cv: float | None = None    # walk-forward CV AUROC (if computed)
    ml_ks_cv: float | None = None       # walk-forward CV KS statistic (if computed)
    ml_elapsed_s: float | None = None   # seconds taken by ML predict()
    # Multi-horizon outlook (None when ML disabled or data insufficient)
    multi_horizon: Optional["MultiHorizonOutlook"] = None
    # Swing-trading sell metrics (None when data insufficient)
    swing_metrics: Optional["SwingMetrics"] = None

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
        ml_str = ""
        if self.ml_direction is not None:
            ml_str = (
                f" | ML={self.ml_direction} p={self.ml_probability:.0%}"
                f" [{self.ml_confidence}]"
            )
            if self.ml_auroc_cv is not None:
                ml_str += f" AUROC={self.ml_auroc_cv:.3f}"
        swing_str = ""
        if self.swing_metrics is not None:
            sm = self.swing_metrics
            swing_str = (
                f" | Swing:{sm.sell_recommendation}"
                f"(score={sm.composite_sell_score:.0f},"
                f"stop=${sm.atr_stop_price:.2f},"
                f"target=${sm.atr_target_price:.2f})"
            )
        horizon_str = ""
        if self.multi_horizon is not None:
            mh = self.multi_horizon
            parts = []
            for label, pred in [("1W", mh.week1), ("1M", mh.month1), ("1Y", mh.year1)]:
                if pred is not None:
                    parts.append(f"{label}:{pred.direction}({pred.probability:.0%})")
            if parts:
                horizon_str = " | " + " ".join(parts)
        return (
            f"{self.symbol}: score={self.score} "
            f"[{', '.join(signals) if signals else 'no signals'}] "
            f"is_dip={self.is_dip}"
            f"{ml_str}{horizon_str}{swing_str}"
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
    ml_auroc_cv     = None
    ml_ks_cv        = None
    ml_elapsed_s    = None
    multi_horizon   = None
    swing_metrics   = None

    if config.ML_ENABLED:
        try:
            from ml_predictor import predict as ml_predict, predict_multi_horizon, compute_swing_metrics

            # ── 5-day (default-horizon) prediction for buy gate ──────────────
            ml_pred = ml_predict(
                symbol,
                history,
                compute_cv_auroc=getattr(config, "ML_COMPUTE_CV_AUROC", True),
            )
            if ml_pred is not None:
                ml_direction     = ml_pred.direction
                ml_probability   = ml_pred.probability
                ml_confidence    = ml_pred.confidence
                ml_auroc_cv      = getattr(ml_pred, "auroc_cv",       None)
                ml_ks_cv         = getattr(ml_pred, "ks_cv",          None)
                ml_elapsed_s     = getattr(ml_pred, "elapsed_seconds", None)
                ml_buy_confirmed = (
                    ml_pred.direction == "UP"
                    and ml_pred.confidence in ("MEDIUM", "HIGH")
                )
            else:
                logger.debug(
                    "%s: ML prediction suppressed (insufficient data or AUROC < threshold)",
                    symbol,
                )

            # ── Multi-horizon outlook (1W / 1M / 1Y) + swing metrics ─────────
            multi_horizon = predict_multi_horizon(
                symbol,
                history,
                compute_cv_auroc=getattr(config, "ML_COMPUTE_CV_AUROC", True),
            )
            # Swing metrics are embedded inside multi_horizon but also exposed
            # directly on DipSignal for easy access by trader.py
            if multi_horizon is not None:
                swing_metrics = multi_horizon.swing_metrics

        except Exception as exc:
            logger.debug("%s: ML/multi-horizon prediction failed — %s", symbol, exc)

    # Always compute swing metrics even when ML is disabled (they are pure
    # technical indicators and require no ML libraries).
    if swing_metrics is None:
        try:
            from ml_predictor import compute_swing_metrics
            swing_metrics = compute_swing_metrics(symbol, history)
        except Exception as exc:
            logger.debug("%s: swing metrics computation failed — %s", symbol, exc)

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
        ml_auroc_cv=ml_auroc_cv,
        ml_ks_cv=ml_ks_cv,
        ml_elapsed_s=ml_elapsed_s,
        multi_horizon=multi_horizon,
        swing_metrics=swing_metrics,
    )
    logger.info(str(result))
    return result
