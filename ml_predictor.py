"""
ml_predictor.py — 5-day price-direction predictor using a Random Forest.

How it works
------------
1.  For each row of price history, compute a set of technical features:
      • RSI (14-day)
      • Price vs MA20 ratio
      • Price vs MA50 ratio
      • Price vs MA100 ratio
      • 5-day momentum (%)
      • 10-day momentum (%)
      • 20-day momentum (%)
      • Historical volatility — 10-day rolling std of log-returns
      • Bollinger Band position  (price vs upper/lower band)
      • Volume trend — 5-day vs 20-day volume ratio  (if Volume column present)
      • 52-week range position

2.  The label is 1 (UP) if Close[t+PREDICT_HORIZON] > Close[t], else 0 (DOWN).

3.  A Random Forest classifier is trained on the older 80% of the history
    (walk-forward split — NO look-ahead bias).

4.  The model predicts on the most recent bar to give the current signal.

Public API
----------
    predict(symbol, history) -> MLPrediction | None

MLPrediction fields
-------------------
    symbol          str
    direction       "UP" | "DOWN"
    probability     float   0-1  (probability of the predicted class)
    confidence      "HIGH" | "MEDIUM" | "LOW"   (based on config thresholds)
    feature_importances  dict[str, float]   top features driving the prediction
    n_train_samples int
    horizon_days    int     (always config.ML_PREDICT_HORIZON)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class MLPrediction:
    symbol: str
    direction: str                        # "UP" or "DOWN"
    probability: float                    # probability of predicted direction
    confidence: str                       # "HIGH" | "MEDIUM" | "LOW"
    feature_importances: dict = field(default_factory=dict)
    n_train_samples: int = 0
    horizon_days: int = 0

    def __str__(self) -> str:
        return (
            f"{self.symbol}: ML={self.direction} "
            f"p={self.probability:.0%} [{self.confidence}] "
            f"(trained on {self.n_train_samples} samples, {self.horizon_days}d horizon)"
        )


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Vectorised RSI using Wilder's smoothing."""
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_l = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_g / avg_l.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def build_features(history: pd.DataFrame) -> pd.DataFrame:
    """
    Build a feature DataFrame from OHLCV price history.

    All features are normalised/ratio-based so they are scale-invariant
    across different price levels and time periods.
    """
    df = history.copy()
    c  = df["Close"]
    v  = df.get("Volume", pd.Series(dtype=float, name="Volume"))

    feat = pd.DataFrame(index=df.index)

    # ── momentum ─────────────────────────────────────────────────────────────
    for n in [5, 10, 20]:
        feat[f"mom_{n}d"] = c.pct_change(n)

    # ── RSI ──────────────────────────────────────────────────────────────────
    feat["rsi_14"] = _rsi(c, 14) / 100.0   # scale to [0, 1]

    # ── price vs moving averages ──────────────────────────────────────────────
    for w in [20, 50, 100]:
        ma = _sma(c, w)
        feat[f"price_vs_ma{w}"] = (c - ma) / ma.replace(0, np.nan)

    # ── Bollinger Band position ───────────────────────────────────────────────
    ma20  = _sma(c, 20)
    std20 = c.rolling(20, min_periods=20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    band_width = (upper - lower).replace(0, np.nan)
    feat["bb_position"] = (c - lower) / band_width   # 0 = at lower band, 1 = at upper

    # ── historical volatility ─────────────────────────────────────────────────
    log_ret = np.log(c / c.shift(1))
    feat["volatility_10d"] = log_ret.rolling(10, min_periods=10).std()

    # ── 52-week range position ────────────────────────────────────────────────
    hi52 = c.rolling(252, min_periods=50).max()
    lo52 = c.rolling(252, min_periods=50).min()
    rng  = (hi52 - lo52).replace(0, np.nan)
    feat["week52_pos"] = (c - lo52) / rng

    # ── volume trend (ratio of short-term to long-term avg volume) ────────────
    if not v.empty and v.notna().sum() > 20:
        vol_s = v.rolling(5,  min_periods=5).mean()
        vol_l = v.rolling(20, min_periods=20).mean()
        feat["volume_ratio"] = vol_s / vol_l.replace(0, np.nan)
    else:
        feat["volume_ratio"] = np.nan

    return feat


# ---------------------------------------------------------------------------
# Main predictor
# ---------------------------------------------------------------------------

def predict(
    symbol: str,
    history: pd.DataFrame,
    horizon: int | None = None,
    train_split: float | None = None,
    min_confidence: float | None = None,
) -> Optional[MLPrediction]:
    """
    Train a Random Forest on *history* and predict the 5-day direction for
    the most recent bar.

    Parameters
    ----------
    symbol        : ticker (for logging / result labelling only)
    history       : DataFrame with at least a 'Close' column, oldest→newest
    horizon       : days ahead to predict (default: config.ML_PREDICT_HORIZON)
    train_split   : fraction of data used for training (default: config.ML_TRAIN_SPLIT)
    min_confidence: probability above which confidence = HIGH (default: config.ML_HIGH_CONFIDENCE_THRESHOLD)

    Returns None if there is insufficient data or sklearn is not installed.
    """
    horizon       = horizon       or config.ML_PREDICT_HORIZON
    train_split   = train_split   or config.ML_TRAIN_SPLIT
    min_confidence = min_confidence or config.ML_HIGH_CONFIDENCE_THRESHOLD

    # ── guard: sklearn optional ───────────────────────────────────────────────
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        logger.warning(
            "scikit-learn not installed — ML predictor disabled. "
            "Run: pip install scikit-learn"
        )
        return None

    if history is None or history.empty or "Close" not in history.columns:
        return None

    # ── build features + labels ───────────────────────────────────────────────
    features = build_features(history)
    closes   = history["Close"]

    # Label: 1 if price is higher in `horizon` days, else 0
    future_ret = closes.shift(-horizon) / closes - 1
    labels     = (future_ret > 0).astype(int)

    # Align and drop NaN rows
    combined = features.copy()
    combined["_label"] = labels
    combined = combined.dropna()

    # We can't use the last `horizon` rows for training (no future label)
    combined = combined.iloc[:-horizon] if len(combined) > horizon else combined

    min_samples = config.ML_MIN_TRAIN_SAMPLES
    if len(combined) < min_samples:
        logger.warning(
            "%s: only %d clean samples after feature engineering (need %d) — skipping ML",
            symbol, len(combined), min_samples,
        )
        return None

    X = combined.drop(columns=["_label"]).values
    y = combined["_label"].values

    # ── walk-forward train / predict split ───────────────────────────────────
    split_idx   = int(len(X) * train_split)
    X_train     = X[:split_idx]
    y_train     = y[:split_idx]

    if len(X_train) < min_samples // 2:
        logger.warning("%s: not enough training samples (%d) — skipping ML", symbol, len(X_train))
        return None

    # ── train ─────────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)

    clf = RandomForestClassifier(
        n_estimators      = config.ML_N_ESTIMATORS,
        max_depth         = config.ML_MAX_DEPTH,
        min_samples_leaf  = config.ML_MIN_SAMPLES_LEAF,
        class_weight      = "balanced",   # handles imbalanced UP/DOWN labels
        random_state      = 42,
        n_jobs            = -1,
    )
    clf.fit(X_train_sc, y_train)

    # ── predict on the LATEST bar (the one we actually care about) ────────────
    # Rebuild features on the full history (including the most recent row)
    full_features = build_features(history).dropna()
    if full_features.empty:
        return None

    latest_X = full_features.iloc[[-1]].values   # shape (1, n_features)
    latest_X_sc = scaler.transform(latest_X)

    proba   = clf.predict_proba(latest_X_sc)[0]   # [p_down, p_up]
    pred    = int(clf.predict(latest_X_sc)[0])
    prob    = float(proba[pred])
    direction = "UP" if pred == 1 else "DOWN"

    # ── confidence tier ───────────────────────────────────────────────────────
    if prob >= min_confidence:
        confidence = "HIGH"
    elif prob >= config.ML_MEDIUM_CONFIDENCE_THRESHOLD:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # ── feature importances (top 5) ───────────────────────────────────────────
    feat_names = list(full_features.columns)
    importances = dict(zip(feat_names, clf.feature_importances_))
    top_features = dict(
        sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
    )

    result = MLPrediction(
        symbol              = symbol,
        direction           = direction,
        probability         = prob,
        confidence          = confidence,
        feature_importances = top_features,
        n_train_samples     = len(X_train),
        horizon_days        = horizon,
    )
    logger.info(str(result))
    return result
