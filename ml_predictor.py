"""
ml_predictor.py — Ensemble ML predictor for 5-day alpha-direction forecasting.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FINAL MODEL  (Phase 2 — benchmarked March 2026)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Label
-----
    "Alpha vs SPY" — 1 if stock N-day return > SPY N-day return, else 0.
    Removes systematic market beta so the model focuses on idiosyncratic
    alpha, which is more learnable than raw direction.

Feature set  (~59 features)
----------------------------
    Technical    RSI-14, Stoch-RSI, momentum 3/5/10/20d, price-vs-MA 20/50/
                 100/200, MA crosses, Bollinger position & width, HVol 5/10/20d,
                 vol-of-vol, ATR-14, 52-week range position, volume ratio, OBV
    Calendar     day-of-week sin/cos, month sin/cos, January flag, quarter-end,
                 turn-of-month start/end
    Price-pattern overnight gap, candle body/shadows, intraday range,
                 Williams %R, MACD histogram, volume z-score, 20d & 52w drawdown,
                 VPT trend, bullish-close ratio
    Macro        SPY/QQQ/TLT/GLD/IWM/VIX 5d & 20d returns; relative strength
                 vs SPY and vs sector ETF
    HMM          2-state Gaussian HMM regime probability & id
    Fundamental  Monte Carlo DCF upside % (EDGAR FCF, if available)

Ensemble  (soft-voting)
-----------------------
    LightGBM  (400 trees, 63 leaves, lr=0.04, feature_fraction=0.75)   × 0.50
    XGBoost   (400 trees, depth=5,   lr=0.04, colsample=0.75)          × 0.35
    LogisticRegression  (C=0.5, L2, balanced)                           × 0.15

Multi-horizon forecasts
-----------------------
    predict_multi_horizon() runs three independent models trained on their
    respective label horizons:
        1W  (5 trading days)   — short-term swing / entry timing
        1M  (21 trading days)  — medium-term swing / position sizing
        1Y  (252 trading days) — long-term trend / conviction filter
    Each returns its own MLPrediction with separate AUROC/KS.

Swing-trading sell metrics  (SwingMetrics)
------------------------------------------
    Computed purely from price history — no forward-looking data:
        atr_14              14-day Average True Range (normalised by price)
        atr_stop_price      Trailing stop = current price − 2×ATR  (exit level)
        atr_target_price    Reward target = current price + 3×ATR  (3:1 R/R)
        reward_risk_ratio   atr_target / atr_stop distance
        rsi_14              Current RSI-14 (>70 = overbought sell signal)
        rsi_signal          "OVERBOUGHT" | "NEUTRAL" | "OVERSOLD"
        bb_position         Bollinger %B (>0.9 = near upper band, exit zone)
        bb_signal           "EXTENDED" | "NEUTRAL" | "COMPRESSED"
        macd_hist           MACD histogram (negative cross = momentum fade)
        macd_signal         "BEARISH_CROSS" | "BULLISH" | "NEUTRAL"
        price_vs_ma50       % above/below 50-day MA (mean-reversion gauge)
        price_vs_ma200      % above/below 200-day MA (trend gauge)
        trend_strength      ADX-proxy: ratio of directional MA spread to vol
        vol_regime          "HIGH" | "NORMAL" | "LOW"  (20d vol z-score)
        days_since_high     Calendar days since the 52-week high
        drawdown_from_high  % drawdown from the most recent swing high
        composite_sell_score  0–100 sell pressure score (higher = more reason to sell)
        sell_recommendation "STRONG_SELL" | "CONSIDER_SELL" | "HOLD" | "ADD"

Validation & observed performance
----------------------------------
    5-fold purged walk-forward CV (gap = horizon days).
    Benchmarked on 30 large/mid-cap US tickers, 5-year daily data:
        Mean AUROC (5d)  ≈ 0.52  (range 0.45–0.57)
        Mean KS          ≈ 0.15
        Per-ticker time  ≈ 6–8 s

Public API
----------
    predict(symbol, history)               -> MLPrediction | None
    predict_multi_horizon(symbol, history) -> MultiHorizonOutlook | None
    compute_swing_metrics(symbol, history) -> SwingMetrics

MLPrediction fields
-------------------
    symbol, direction, probability, confidence, feature_importances,
    n_train_samples, horizon_days, auroc_cv, ks_cv, elapsed_seconds

MultiHorizonOutlook fields
--------------------------
    symbol, week1, month1, year1   (each an MLPrediction or None)
    outlook_summary                readable string
    swing_metrics                  SwingMetrics (computed once, shared)

SwingMetrics fields
-------------------
    atr_14, atr_stop_price, atr_target_price, reward_risk_ratio,
    rsi_14, rsi_signal, bb_position, bb_signal,
    macd_hist, macd_signal, price_vs_ma50, price_vs_ma200,
    trend_strength, vol_regime, days_since_high, drawdown_from_high,
    composite_sell_score, sell_recommendation
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Macro/sector proxy ETFs fetched via yfinance (no API key needed)
_MACRO_ETFS = {
    "spy_ret":  "SPY",   # broad market
    "qqq_ret":  "QQQ",   # tech / growth
    "tlt_ret":  "TLT",   # long-term Treasuries (risk-off proxy)
    "gld_ret":  "GLD",   # gold (risk-off/inflation)
    "iwm_ret":  "IWM",   # small-cap (risk sentiment)
    "vix_ret":  "^VIX",  # implied volatility index
}

# Sector ETF map: GICS sector name → ETF ticker
_SECTOR_ETFS = {
    "Technology":             "XLK",
    "Health Care":            "XLV",
    "Financials":             "XLF",
    "Consumer Discretionary": "XLY",
    "Consumer Staples":       "XLP",
    "Energy":                 "XLE",
    "Industrials":            "XLI",
    "Materials":              "XLB",
    "Real Estate":            "XLRE",
    "Utilities":              "XLU",
    "Communication Services": "XLC",
}

# Simple SIC → GICS sector mapping (covers most common SICs)
_SIC_TO_SECTOR = {
    "10": "Materials", "12": "Energy", "13": "Energy", "14": "Materials",
    "15": "Industrials", "16": "Industrials", "17": "Industrials",
    "20": "Consumer Staples", "21": "Consumer Staples", "22": "Consumer Discretionary",
    "23": "Consumer Discretionary", "24": "Materials", "25": "Consumer Discretionary",
    "26": "Materials", "27": "Communication Services", "28": "Health Care",
    "29": "Energy", "30": "Materials", "31": "Consumer Discretionary",
    "32": "Materials", "33": "Materials", "34": "Industrials",
    "35": "Technology", "36": "Technology", "37": "Consumer Discretionary",
    "38": "Health Care", "39": "Consumer Discretionary",
    "40": "Industrials", "41": "Industrials", "42": "Industrials",
    "44": "Industrials", "45": "Industrials", "47": "Industrials",
    "48": "Communication Services", "49": "Utilities",
    "50": "Industrials", "51": "Industrials",
    "52": "Consumer Discretionary", "53": "Consumer Discretionary",
    "54": "Consumer Staples", "55": "Consumer Discretionary",
    "56": "Consumer Discretionary", "57": "Consumer Discretionary",
    "58": "Consumer Discretionary", "59": "Consumer Staples",
    "60": "Financials", "61": "Financials", "62": "Financials",
    "63": "Financials", "64": "Financials", "65": "Real Estate",
    "67": "Financials", "70": "Consumer Discretionary", "72": "Consumer Discretionary",
    "73": "Technology", "75": "Consumer Discretionary", "76": "Industrials",
    "78": "Communication Services", "79": "Communication Services",
    "80": "Health Care", "82": "Consumer Discretionary", "83": "Industrials",
    "86": "Industrials", "87": "Industrials",
}

# Monte Carlo simulation parameters
_MC_N_SIMS        = 2_000
_MC_HORIZON_YEARS = 5
_MC_DISCOUNT_RATE = 0.10   # 10% WACC
_MC_FCF_GROWTH_MU = 0.07   # 7% base FCF growth
_MC_FCF_GROWTH_SD = 0.15   # ±15% log-normal uncertainty


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class MLPrediction:
    symbol: str
    direction: str                        # "UP" or "DOWN"
    probability: float                    # P(outperform SPY) (0-1)
    confidence: str                       # "HIGH" | "MEDIUM" | "LOW"
    feature_importances: dict = field(default_factory=dict)
    n_train_samples: int = 0
    horizon_days: int = 0
    auroc_cv: Optional[float] = None      # 5-fold walk-forward AUROC
    ks_cv: Optional[float] = None         # 5-fold walk-forward KS statistic
    elapsed_seconds: float = 0.0          # wall-clock time for predict()

    def __str__(self) -> str:
        auroc_str = f" CV-AUROC={self.auroc_cv:.3f}" if self.auroc_cv is not None else ""
        ks_str    = f" KS={self.ks_cv:.3f}"          if self.ks_cv    is not None else ""
        t_str     = f" [{self.elapsed_seconds:.1f}s]"
        return (
            f"{self.symbol}: ML={self.direction} "
            f"p={self.probability:.0%} [{self.confidence}]"
            f"{auroc_str}{ks_str}{t_str} "
            f"(trained on {self.n_train_samples} samples, {self.horizon_days}d horizon)"
        )


# ---------------------------------------------------------------------------
# Swing-trading sell metrics
# ---------------------------------------------------------------------------

@dataclass
class SwingMetrics:
    """
    Sell-side indicators computed purely from price/volume history.
    Used to help decide WHEN to exit a swing-trade position.

    All prices are in the same currency as the input history.
    Percentage fields are expressed as fractions (e.g. 0.05 = 5%).
    """
    symbol:             str

    # ATR-based exit levels
    atr_14:             float   # 14-day ATR normalised by price (fraction)
    atr_stop_price:     float   # Trailing hard-stop: entry − 2×ATR (abs $)
    atr_target_price:   float   # Profit target:      entry + 3×ATR (abs $)
    reward_risk_ratio:  float   # target distance / stop distance

    # Momentum oscillators
    rsi_14:             float   # current RSI (0–100)
    rsi_signal:         str     # "OVERBOUGHT" | "NEUTRAL" | "OVERSOLD"
    bb_position:        float   # Bollinger %B  (0=lower, 1=upper band)
    bb_signal:          str     # "EXTENDED" | "NEUTRAL" | "COMPRESSED"
    macd_hist:          float   # MACD histogram value (normalised by price)
    macd_signal:        str     # "BEARISH_CROSS" | "BULLISH" | "NEUTRAL"

    # Trend gauges
    price_vs_ma50:      float   # % above (+) / below (-) 50-day MA
    price_vs_ma200:     float   # % above (+) / below (-) 200-day MA
    trend_strength:     float   # ADX-proxy; 0=flat, 1=strong trend
    vol_regime:         str     # "HIGH" | "NORMAL" | "LOW"

    # Drawdown / timing
    days_since_high:    int     # calendar days since 52-week high
    drawdown_from_high: float   # % drawdown from the most recent 252-day high

    # Composite score & recommendation
    composite_sell_score: float   # 0–100  (higher → more reason to sell / trim)
    sell_recommendation:  str     # "STRONG_SELL" | "CONSIDER_SELL" | "HOLD" | "ADD"

    def __str__(self) -> str:
        return (
            f"{self.symbol} SwingMetrics | "
            f"RSI={self.rsi_14:.1f}[{self.rsi_signal}] "
            f"BB%B={self.bb_position:.2f}[{self.bb_signal}] "
            f"MACD[{self.macd_signal}] "
            f"vs MA50={self.price_vs_ma50:+.1%} "
            f"vs MA200={self.price_vs_ma200:+.1%} "
            f"ATR-stop=${self.atr_stop_price:.2f} "
            f"ATR-target=${self.atr_target_price:.2f} "
            f"RR={self.reward_risk_ratio:.1f} "
            f"SellScore={self.composite_sell_score:.0f}/100 "
            f"→ {self.sell_recommendation}"
        )


# ---------------------------------------------------------------------------
# Multi-horizon outlook
# ---------------------------------------------------------------------------

@dataclass
class MultiHorizonOutlook:
    """
    Bundled result from predict_multi_horizon().

    Attributes
    ----------
    symbol        : ticker
    week1         : 1W (5-day) MLPrediction or None
    month1        : 1M (21-day) MLPrediction or None
    year1         : 1Y (252-day) MLPrediction or None
    swing_metrics : SwingMetrics (computed once, shared across horizons)
    outlook_summary : human-readable one-liner
    """
    symbol:          str
    week1:           Optional[MLPrediction]
    month1:          Optional[MLPrediction]
    year1:           Optional[MLPrediction]
    swing_metrics:   Optional["SwingMetrics"]
    outlook_summary: str = ""

    def __post_init__(self):
        if not self.outlook_summary:
            self.outlook_summary = self._build_summary()

    def _build_summary(self) -> str:
        parts = [f"{self.symbol} outlook:"]
        for label, pred in [("1W", self.week1), ("1M", self.month1), ("1Y", self.year1)]:
            if pred is None:
                parts.append(f"  {label}: n/a")
            else:
                auroc = f" AUROC={pred.auroc_cv:.3f}" if pred.auroc_cv else ""
                parts.append(
                    f"  {label}: {pred.direction} p={pred.probability:.0%}"
                    f" [{pred.confidence}]{auroc}"
                )
        if self.swing_metrics:
            sm = self.swing_metrics
            parts.append(
                f"  Swing: {sm.sell_recommendation}"
                f" (score={sm.composite_sell_score:.0f}/100,"
                f" stop=${sm.atr_stop_price:.2f},"
                f" target=${sm.atr_target_price:.2f})"
            )
        return "\n".join(parts)

    def __str__(self) -> str:
        return self.outlook_summary


# ---------------------------------------------------------------------------
# Caches (process-lifetime)
# ---------------------------------------------------------------------------

_macro_cache: dict[str, pd.DataFrame]  = {}   # ticker → price series
_mc_cache:    dict[str, float]          = {}   # symbol → MC upside pct
_hmm_cache:   dict[str, object]         = {}   # symbol → fitted HMM
_spy_series:  Optional[pd.Series]       = None # cached SPY close for alpha label


# ---------------------------------------------------------------------------
# Feature engineering — technical
# ---------------------------------------------------------------------------

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Vectorised RSI (Wilder's smoothing)."""
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_l = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range — normalised by close price."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, min_periods=period, adjust=False).mean() / close.replace(0, np.nan)


def _obv_trend(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """OBV momentum: OBV 5-day MA / OBV 20-day MA (>1 = accumulation)."""
    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * volume).cumsum()
    return (obv.rolling(5, min_periods=5).mean()
            / obv.rolling(window, min_periods=window).mean().replace(0, np.nan))


def _stoch_rsi(rsi_series: pd.Series, period: int = 14) -> pd.Series:
    """Stochastic RSI: (RSI - min(RSI,n)) / (max(RSI,n) - min(RSI,n))."""
    lo  = rsi_series.rolling(period, min_periods=period).min()
    hi  = rsi_series.rolling(period, min_periods=period).max()
    rng = (hi - lo).replace(0, np.nan)
    return (rsi_series - lo) / rng


def build_technical_features(history: pd.DataFrame) -> pd.DataFrame:
    """Build scale-invariant technical features from OHLCV history."""
    df   = history.copy()
    c    = df["Close"]
    high = df.get("High", c)
    low  = df.get("Low",  c)
    vol  = df.get("Volume", pd.Series(dtype=float, name="Volume", index=df.index))

    feat = pd.DataFrame(index=df.index)

    # ── momentum ──────────────────────────────────────────────────────────────
    for n in [3, 5, 10, 20]:
        feat[f"mom_{n}d"] = c.pct_change(n)

    # ── RSI + Stochastic RSI ──────────────────────────────────────────────────
    rsi = _rsi(c, 14)
    feat["rsi_14"]      = rsi / 100.0
    feat["stoch_rsi"]   = _stoch_rsi(rsi)

    # ── price vs moving averages ──────────────────────────────────────────────
    for w in [20, 50, 100, 200]:
        ma = _sma(c, w)
        feat[f"price_vs_ma{w}"] = (c - ma) / ma.replace(0, np.nan)

    # ── MA cross signals ──────────────────────────────────────────────────────
    ma20  = _sma(c, 20)
    ma50  = _sma(c, 50)
    ma200 = _sma(c, 200)
    feat["ma20_vs_ma50"]  = (ma20 - ma50)  / ma50.replace(0, np.nan)
    feat["ma50_vs_ma200"] = (ma50 - ma200) / ma200.replace(0, np.nan)

    # ── Bollinger Band position ───────────────────────────────────────────────
    std20 = c.rolling(20, min_periods=20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    band_width = (upper - lower).replace(0, np.nan)
    feat["bb_position"] = (c - lower) / band_width   # 0=lower, 1=upper
    feat["bb_width"]    = band_width / ma20.replace(0, np.nan)

    # ── historical volatility ─────────────────────────────────────────────────
    log_ret = np.log(c / c.shift(1))
    for w in [5, 10, 20]:
        feat[f"hvol_{w}d"] = log_ret.rolling(w, min_periods=w).std()

    # ── vol of vol (regime stability) ────────────────────────────────────────
    feat["vol_of_vol"] = feat["hvol_10d"].rolling(20, min_periods=10).std()

    # ── ATR (normalised) ──────────────────────────────────────────────────────
    feat["atr_14"] = _atr(high, low, c, 14)

    # ── 52-week range position ────────────────────────────────────────────────
    hi52 = c.rolling(252, min_periods=50).max()
    lo52 = c.rolling(252, min_periods=50).min()
    rng  = (hi52 - lo52).replace(0, np.nan)
    feat["week52_pos"] = (c - lo52) / rng

    # ── volume features ───────────────────────────────────────────────────────
    if not vol.empty and vol.notna().sum() > 20:
        vol_s = vol.rolling(5,  min_periods=5).mean()
        vol_l = vol.rolling(20, min_periods=20).mean()
        feat["volume_ratio"] = vol_s / vol_l.replace(0, np.nan)
        feat["obv_trend"]    = _obv_trend(c, vol)
    else:
        feat["volume_ratio"] = np.nan
        feat["obv_trend"]    = np.nan

    return feat


# ---------------------------------------------------------------------------
# Feature engineering — calendar / seasonal
# ---------------------------------------------------------------------------

def build_calendar_features(history: pd.DataFrame) -> pd.DataFrame:
    """
    Calendar and seasonal features that carry predictive power for equity
    alpha (day-of-week, month, quarter-end, January effect, turn-of-month).
    """
    idx  = history.index
    feat = pd.DataFrame(index=idx)

    # Day-of-week (Monday=0, Friday=4) — encoded as sine/cosine to preserve
    # cyclical ordering without implicit ordinality.
    dow = idx.dayofweek.astype(float)
    feat["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    feat["dow_cos"] = np.cos(2 * np.pi * dow / 5)

    # Month — sine/cosine encoding (1–12)
    month = idx.month.astype(float)
    feat["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    feat["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

    # January effect flag
    feat["is_january"] = (month == 1).astype(float)

    # Quarter-end flag (last 5 trading days of Mar/Jun/Sep/Dec)
    # We use calendar month to approximate; trading-day precision is good enough.
    is_qtr_end_month = idx.month.isin([3, 6, 9, 12])
    # Last 5 days of the month: day > (days_in_month - 5)
    days_in_month   = idx.days_in_month
    feat["qtr_end"]  = (is_qtr_end_month & (idx.day > days_in_month - 5)).astype(float)

    # Turn-of-month: first 3 + last 3 trading days of any month
    feat["tom_start"] = (idx.day <= 3).astype(float)
    feat["tom_end"]   = (idx.day >= days_in_month - 2).astype(float)

    return feat


# ---------------------------------------------------------------------------
# Feature engineering — price pattern / microstructure
# ---------------------------------------------------------------------------

def build_price_pattern_features(history: pd.DataFrame) -> pd.DataFrame:
    """
    Candle-based and microstructure features:
        gap             — overnight gap (open vs prev close), normalised
        body_strength   — |open-close| / (high-low) — candle body ratio
        upper_shadow    — upper wick / (high-low)
        lower_shadow    — lower wick / (high-low)
        intraday_range  — (high-low) / close — normalised daily range
        willr           — Williams %R (14-period)
        macd_signal     — MACD histogram (12/26 EMA diff vs 9-period signal)
        vol_zscore      — rolling z-score of volume (5d vs 20d)
        drawdown_20d    — drawdown from 20-day high (momentum proxy)
        drawdown_52w    — drawdown from 52-week high
        vpt_trend       — Volume Price Trend momentum
        close_pct_hi20  — % of last 20 days that close > open
    """
    df   = history.copy()
    c    = df["Close"]
    high = df.get("High", c)
    low  = df.get("Low",  c)
    opn  = df.get("Open", c)
    vol  = df.get("Volume", pd.Series(dtype=float, name="Volume", index=df.index))

    feat = pd.DataFrame(index=df.index)

    hl_range = (high - low).replace(0, np.nan)

    # ── overnight gap ─────────────────────────────────────────────────────────
    feat["gap"] = (opn - c.shift(1)) / c.shift(1).replace(0, np.nan)

    # ── candle body ───────────────────────────────────────────────────────────
    feat["body_strength"]  = (opn - c).abs() / hl_range
    feat["upper_shadow"]   = (high - opn.clip(upper=c)) / hl_range
    feat["lower_shadow"]   = (opn.clip(lower=c) - low)  / hl_range
    feat["intraday_range"] = hl_range / c.replace(0, np.nan)

    # ── Williams %R (14-period) ───────────────────────────────────────────────
    hi14 = high.rolling(14, min_periods=14).max()
    lo14 = low.rolling(14, min_periods=14).min()
    feat["willr"] = (hi14 - c) / (hi14 - lo14).replace(0, np.nan) * -100

    # ── MACD histogram ────────────────────────────────────────────────────────
    ema12  = c.ewm(span=12, adjust=False).mean()
    ema26  = c.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    # Normalise by close price so it's scale-invariant
    feat["macd_hist"] = (macd - signal) / c.replace(0, np.nan)

    # ── volume z-score ────────────────────────────────────────────────────────
    if not vol.empty and vol.notna().sum() > 25:
        vol_mean = vol.rolling(20, min_periods=10).mean()
        vol_std  = vol.rolling(20, min_periods=10).std().replace(0, np.nan)
        feat["vol_zscore"] = (vol - vol_mean) / vol_std
    else:
        feat["vol_zscore"] = np.nan

    # ── drawdown from recent highs ────────────────────────────────────────────
    hi20  = c.rolling(20, min_periods=10).max().replace(0, np.nan)
    hi252 = c.rolling(252, min_periods=50).max().replace(0, np.nan)
    feat["drawdown_20d"] = (c - hi20)  / hi20
    feat["drawdown_52w"] = (c - hi252) / hi252

    # ── Volume Price Trend (VPT) momentum ────────────────────────────────────
    if not vol.empty and vol.notna().sum() > 25:
        vpt      = (vol * c.pct_change()).cumsum()
        vpt_ma5  = vpt.rolling(5,  min_periods=5).mean()
        vpt_ma20 = vpt.rolling(20, min_periods=10).mean().replace(0, np.nan)
        feat["vpt_trend"] = (vpt_ma5 - vpt_ma20) / vpt_ma20.abs()
    else:
        feat["vpt_trend"] = np.nan

    # ── bullish close ratio over last 20 bars ─────────────────────────────────
    up_day = (c > opn).astype(float)
    feat["close_pct_hi20"] = up_day.rolling(20, min_periods=10).mean()

    return feat


# ---------------------------------------------------------------------------
# Feature engineering — macro / sector
# ---------------------------------------------------------------------------

def _to_series(obj) -> Optional[pd.Series]:
    """Coerce a DataFrame or Series to a 1-D Series, or return None."""
    if obj is None:
        return None
    if isinstance(obj, pd.DataFrame):
        obj = obj.squeeze()
        if isinstance(obj, pd.DataFrame):
            obj = obj.iloc[:, 0]
    if not isinstance(obj, pd.Series):
        return None
    # Strip timezone from index
    if hasattr(obj.index, "tzinfo") and obj.index.tzinfo is not None:
        obj.index = obj.index.tz_localize(None)
    elif hasattr(obj.index, "tz") and obj.index.tz is not None:
        obj.index = obj.index.tz_localize(None)
    return obj


def _fetch_macro_series(tickers: list[str], lookback_days: int = 600) -> dict[str, pd.Series]:
    """Download close-price series for macro ETFs via yfinance (cached).

    Uses period='3y' to match the typical stock history window,
    ensuring macro features cover the full training period.
    """
    result: dict[str, pd.Series] = {}
    try:
        import yfinance as yf
    except ImportError:
        logger.debug("yfinance not installed — macro features skipped")
        return result

    for ticker in tickers:
        if ticker in _macro_cache:
            s = _to_series(_macro_cache[ticker])
            if s is not None:
                result[ticker] = s
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t_obj = yf.Ticker(ticker)
                # Fetch 3 years to cover the full stock history window
                df    = t_obj.history(period="3y", auto_adjust=True)
            if df is not None and not df.empty and "Close" in df.columns:
                close = _to_series(df["Close"])
                if close is not None and not close.empty:
                    _macro_cache[ticker] = close
                    result[ticker] = close
        except Exception as exc:
            logger.debug("Macro ETF %s fetch failed: %s", ticker, exc)
    return result


def _get_spy_horizon_return(stock_index: pd.DatetimeIndex,
                            horizon: int) -> Optional[pd.Series]:
    """
    Return a Series of SPY `horizon`-day forward returns aligned to
    `stock_index`.  Used for building the alpha-vs-SPY label.

    Fetches SPY from yfinance (cached process-wide in _spy_series).
    Returns None if SPY cannot be fetched.
    """
    global _spy_series
    try:
        import yfinance as yf
    except ImportError:
        return None

    # Fetch / use cached SPY series
    if _spy_series is None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                spy_df = yf.Ticker("SPY").history(period="6y", auto_adjust=True)
            if spy_df is not None and not spy_df.empty and "Close" in spy_df.columns:
                _spy_series = _to_series(spy_df["Close"])
        except Exception as exc:
            logger.debug("SPY fetch for alpha label failed: %s", exc)
            return None

    if _spy_series is None:
        return None

    # Align SPY to the stock's date index, then compute horizon-day forward return
    spy_aligned = (
        _spy_series
        .reindex(_spy_series.index.union(stock_index))
        .ffill()
        .reindex(stock_index)
    )
    spy_aligned = _to_series(spy_aligned)
    if spy_aligned is None or spy_aligned.empty:
        return None

    # Forward return: SPY price in `horizon` days relative to today
    spy_fwd = spy_aligned.shift(-horizon) / spy_aligned - 1
    return spy_fwd


def build_macro_features(history: pd.DataFrame,
                         sector: Optional[str] = None) -> pd.DataFrame:
    """
    Compute rolling returns of macro ETFs aligned to the stock's date index.

    For each macro ETF we compute the 5-day and 20-day returns.
    We also compute the stock's beta-adjusted relative return vs SPY
    (stock_5d_ret - spy_5d_ret) as a measure of relative strength/weakness.
    """
    feat = pd.DataFrame(index=history.index)
    stock_ret5 = history["Close"].pct_change(5)

    # Decide which ETFs to fetch
    etfs_to_fetch = list(_MACRO_ETFS.values())
    sector_etf    = _SECTOR_ETFS.get(sector) if sector else None
    if sector_etf and sector_etf not in etfs_to_fetch:
        etfs_to_fetch.append(sector_etf)

    series = _fetch_macro_series(etfs_to_fetch)

    for feat_name, ticker in _MACRO_ETFS.items():
        s = series.get(ticker)
        if s is None:
            feat[feat_name + "_5d"]  = np.nan
            feat[feat_name + "_20d"] = np.nan
            continue
        # Align to stock index by forward-filling (avoids look-ahead)
        aligned = (
            s.reindex(history.index.union(s.index))
             .ffill()
             .reindex(history.index)
        )
        aligned = _to_series(aligned)
        if aligned is None:
            feat[feat_name + "_5d"]  = np.nan
            feat[feat_name + "_20d"] = np.nan
            continue
        feat[feat_name + "_5d"]  = aligned.pct_change(5).values
        feat[feat_name + "_20d"] = aligned.pct_change(20).values

    # Relative strength vs SPY
    spy = series.get("SPY")
    if spy is not None:
        spy_aligned = (
            spy.reindex(history.index.union(spy.index))
               .ffill()
               .reindex(history.index)
        )
        spy_aligned = _to_series(spy_aligned)
        if spy_aligned is not None:
            feat["rel_str_vs_spy"] = (stock_ret5.values - spy_aligned.pct_change(5).values)
        else:
            feat["rel_str_vs_spy"] = np.nan
    else:
        feat["rel_str_vs_spy"] = np.nan

    # Relative strength vs sector ETF
    if sector_etf:
        sec = series.get(sector_etf)
        if sec is not None:
            sec_aligned = (
                sec.reindex(history.index.union(sec.index))
                   .ffill()
                   .reindex(history.index)
            )
            sec_aligned = _to_series(sec_aligned)
            if sec_aligned is not None:
                feat["rel_str_vs_sector"] = (stock_ret5.values - sec_aligned.pct_change(5).values)
            else:
                feat["rel_str_vs_sector"] = np.nan
        else:
            feat["rel_str_vs_sector"] = np.nan
    else:
        feat["rel_str_vs_sector"] = np.nan

    return feat


# ---------------------------------------------------------------------------
# Feature engineering — HMM market regime
# ---------------------------------------------------------------------------

def build_hmm_features(history: pd.DataFrame,
                       n_states: int = 2) -> pd.DataFrame:
    """
    Fit a 2-state Gaussian HMM on log-returns and derive:
        hmm_regime_prob  — probability of the HIGH-volatility regime at each bar
        hmm_regime_id    — discrete regime label (0/1)

    The HMM is fit on the SAME history slice passed in (no external data),
    so there is mild in-sample bias on the feature itself, but since we
    use it as a market-state feature (not directly as a predictor of the
    label), this is acceptable and common practice.
    """
    feat = pd.DataFrame(index=history.index)
    feat["hmm_regime_prob"] = np.nan
    feat["hmm_regime_id"]   = np.nan

    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        logger.debug("hmmlearn not installed — HMM features skipped")
        return feat

    log_ret = np.log(history["Close"] / history["Close"].shift(1)).dropna()
    if len(log_ret) < 60:
        return feat

    X = log_ret.values.reshape(-1, 1)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # silence hmmlearn convergence noise
            hmm = GaussianHMM(
                n_components  = n_states,
                covariance_type = "full",
                n_iter        = 100,
                random_state  = 42,
                tol           = 1e-4,
            )
            hmm.fit(X)

        # Identify which state is the "high-volatility" regime
        stdevs    = [np.sqrt(hmm.covars_[i][0, 0]) for i in range(n_states)]
        hi_vol_st = int(np.argmax(stdevs))

        # State probabilities (posterior)
        log_prob, posteriors = hmm.score_samples(X)
        hi_vol_prob = posteriors[:, hi_vol_st]
        regime_ids  = hmm.predict(X)

        # Align back to original index (offset by 1 for the diff)
        idx = log_ret.index
        feat.loc[idx, "hmm_regime_prob"] = hi_vol_prob
        feat.loc[idx, "hmm_regime_id"]   = regime_ids.astype(float)

    except Exception as exc:
        logger.debug("HMM fit failed: %s", exc)

    return feat


# ---------------------------------------------------------------------------
# Feature engineering — Monte Carlo fair value
# ---------------------------------------------------------------------------

def _fetch_edgar_fcf(symbol: str) -> Optional[float]:
    """Fetch the most recent annual FCF from EDGAR (cached in _mc_cache)."""
    cache_key = f"_fcf_{symbol}"
    if cache_key in _mc_cache:
        return _mc_cache[cache_key]
    try:
        from edgar import get_fundamentals
        fund = get_fundamentals(symbol)
        fcf  = fund.get("freeCashflow")
        mktcap = fund.get("marketCap")
        _mc_cache[cache_key] = (fcf, mktcap)
        return (fcf, mktcap)
    except Exception as exc:
        logger.debug("%s: EDGAR FCF fetch for MC failed — %s", symbol, exc)
        return (None, None)


def compute_mc_upside(symbol: str,
                      current_price: float,
                      market_cap: Optional[float] = None,
                      fcf: Optional[float] = None,
                      n_sims: int = _MC_N_SIMS) -> Optional[float]:
    """
    Log-normal Monte Carlo DCF to estimate expected upside vs current price.

    Model:
      FCF grows for _MC_HORIZON_YEARS with mu=7%, sigma=15% (log-normal)
      Terminal value = FCF_final × (1 + g) / (WACC - g), g = 3%
      PV = sum(FCF_t / (1+r)^t) + TV / (1+r)^H
      Upside = (median_PV / market_cap - 1)

    Returns None if FCF or market_cap are unavailable.
    """
    cache_key = f"_mc_upside_{symbol}"
    if cache_key in _mc_cache:
        return _mc_cache[cache_key]

    # Try fetching if not provided
    if fcf is None or market_cap is None:
        cached = _fetch_edgar_fcf(symbol)
        if isinstance(cached, tuple):
            fcf_fetched, mktcap_fetched = cached
            fcf        = fcf        if fcf        is not None else fcf_fetched
            market_cap = market_cap if market_cap is not None else mktcap_fetched

    if fcf is None or market_cap is None or fcf <= 0 or market_cap <= 0:
        return None

    try:
        rng   = np.random.default_rng(42)
        H     = _MC_HORIZON_YEARS
        r     = _MC_DISCOUNT_RATE
        mu    = _MC_FCF_GROWTH_MU
        sigma = _MC_FCF_GROWTH_SD
        g_t   = 0.03   # terminal growth rate

        # Simulate annual FCF growth rates (log-normal)
        annual_growths = rng.lognormal(
            mean  = np.log(1 + mu) - 0.5 * sigma**2,
            sigma = sigma,
            size  = (n_sims, H),
        )

        pvs = np.zeros(n_sims)
        for t in range(1, H + 1):
            fcf_t = fcf * np.prod(annual_growths[:, :t], axis=1)
            pvs  += fcf_t / (1 + r) ** t

        # Terminal value
        fcf_H   = fcf * np.prod(annual_growths, axis=1)
        tv      = fcf_H * (1 + g_t) / (r - g_t)
        pvs    += tv / (1 + r) ** H

        # Expected upside: median intrinsic value vs market cap
        median_pv = float(np.median(pvs))
        upside    = (median_pv - market_cap) / market_cap

        _mc_cache[cache_key] = upside
        return float(np.clip(upside, -2.0, 5.0))   # cap extreme values

    except Exception as exc:
        logger.debug("%s: MC computation failed — %s", symbol, exc)
        return None


def build_mc_feature(symbol: str,
                     history: pd.DataFrame) -> pd.Series:
    """
    Return a constant-valued Series (same value for every row) representing
    the Monte Carlo upside percentage.  NaN if unavailable.
    """
    current_price = float(history["Close"].iloc[-1])
    upside = compute_mc_upside(symbol, current_price)
    val    = upside if upside is not None else np.nan
    return pd.Series(val, index=history.index, name="mc_upside")


# ---------------------------------------------------------------------------
# Combined feature matrix
# ---------------------------------------------------------------------------

def build_all_features(symbol: str,
                       history: pd.DataFrame,
                       sector: Optional[str] = None) -> pd.DataFrame:
    """
    Combine technical + calendar + price-pattern + macro + HMM + MC features
    into a single DataFrame.
    """
    tech     = build_technical_features(history)
    calendar = build_calendar_features(history)
    patterns = build_price_pattern_features(history)
    macro    = build_macro_features(history, sector=sector)
    hmm      = build_hmm_features(history)
    mc       = build_mc_feature(symbol, history).to_frame()

    return pd.concat([tech, calendar, patterns, macro, hmm, mc], axis=1)


# ---------------------------------------------------------------------------
# Walk-forward CV AUROC evaluator
# ---------------------------------------------------------------------------

def _walkforward_auroc(X: np.ndarray,
                       y: np.ndarray,
                       clf_factory,
                       n_splits: int = 5,
                       gap: int = 10,
                       min_train: int = 100) -> tuple[float, float]:
    """
    Purged walk-forward cross-validation returning (mean AUROC, mean KS).

    KS statistic = max |CDF_pos(score) - CDF_neg(score)| across thresholds.
    It measures how well the model separates the two classes irrespective of
    a fixed threshold — a complementary metric to AUROC.

    Splits the time-ordered data into n_splits folds.
    A gap of `gap` samples (≈ trading days) is removed between train/test
    to avoid information leakage from auto-correlated returns.

    Returns
    -------
    (mean_auroc, mean_ks)  — each defaults to 0.5 / 0.0 if no folds succeed.
    """
    from sklearn.metrics import roc_auc_score
    from scipy.stats import ks_2samp

    n         = len(X)
    fold_size = n // (n_splits + 1)
    aurocs    = []
    ks_stats  = []

    for k in range(1, n_splits + 1):
        test_start = k * fold_size
        test_end   = min(test_start + fold_size, n)
        train_end  = test_start - gap

        if train_end < min_train or test_end <= test_start:
            continue

        X_tr, y_tr = X[:train_end],          y[:train_end]
        X_te, y_te = X[test_start:test_end], y[test_start:test_end]

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue

        try:
            clf   = clf_factory()
            clf.fit(X_tr, y_tr)
            proba = clf.predict_proba(X_te)[:, 1]

            aurocs.append(roc_auc_score(y_te, proba))

            # KS: compare score distributions of positives vs negatives
            pos_scores = proba[y_te == 1]
            neg_scores = proba[y_te == 0]
            if len(pos_scores) > 0 and len(neg_scores) > 0:
                ks_stat, _ = ks_2samp(pos_scores, neg_scores)
                ks_stats.append(ks_stat)
        except Exception:
            pass

    mean_auroc = float(np.mean(aurocs)) if aurocs   else 0.5
    mean_ks    = float(np.mean(ks_stats)) if ks_stats else 0.0
    return mean_auroc, mean_ks


# ---------------------------------------------------------------------------
# Model builders (base learners)
# ---------------------------------------------------------------------------

def _build_lgbm(n_features: int):
    """LightGBM classifier — tuned for AUROC on equity alpha prediction."""
    import lightgbm as lgb
    return lgb.LGBMClassifier(
        objective         = "binary",
        n_estimators      = 400,
        num_leaves        = 63,
        max_depth         = -1,
        min_child_samples = 20,
        learning_rate     = 0.04,
        reg_alpha         = 0.3,
        reg_lambda        = 1.0,
        feature_fraction  = 0.75,
        bagging_fraction  = 0.85,
        bagging_freq      = 5,
        class_weight      = "balanced",
        random_state      = 42,
        n_jobs            = -1,
        verbose           = -1,
    )


def _build_xgb():
    """XGBoost — finds orthogonal splits to LightGBM, improves ensemble diversity."""
    import xgboost as xgb
    return xgb.XGBClassifier(
        n_estimators      = 400,
        max_depth         = 5,
        learning_rate     = 0.04,
        subsample         = 0.85,
        colsample_bytree  = 0.75,
        reg_lambda        = 1.0,
        reg_alpha         = 0.3,
        eval_metric       = "auc",
        random_state      = 42,
        n_jobs            = -1,
        verbosity         = 0,
    )


def _build_logreg():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(
        C            = 0.5,
        max_iter     = 500,
        class_weight = "balanced",
        solver       = "lbfgs",
        random_state = 42,
    )


# ---------------------------------------------------------------------------
# Stacking ensemble
# ---------------------------------------------------------------------------

def _train_ensemble(X_train: np.ndarray,
                    y_train: np.ndarray,
                    feature_names: list[str]):
    """
    Train a soft-voting ensemble: LightGBM + XGBoost + LogisticRegression.

    Weights: LGBM=0.50, XGB=0.35, LogReg=0.15
    XGBoost finds orthogonal feature splits to LGBM, improving ensemble
    diversity. LogReg provides linear calibration as a regularising anchor.

    All learners are trained on the full training set (no OOF meta-learner
    at this sample size, where OOF stacking has shown no benefit).
    """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_train)

    lgbm   = _build_lgbm(X_train.shape[1])
    xgbm   = _build_xgb()
    logreg = _build_logreg()

    lgbm.fit(X_train, y_train)
    xgbm.fit(X_train, y_train)
    logreg.fit(X_sc, y_train)

    return lgbm, xgbm, logreg, scaler


def _ensemble_predict_proba(X_new: np.ndarray,
                             lgbm, xgbm, logreg, scaler) -> float:
    """
    Predict P(outperform SPY) via soft-voting:
        LGBM × 0.50  +  XGBoost × 0.35  +  LogReg × 0.15
    """
    X_sc    = scaler.transform(X_new)
    p_lgbm  = float(lgbm.predict_proba(X_new)[:, 1][0])
    p_xgb   = float(xgbm.predict_proba(X_new)[:, 1][0])
    p_lr    = float(logreg.predict_proba(X_sc)[:, 1][0])
    return 0.50 * p_lgbm + 0.35 * p_xgb + 0.15 * p_lr


# ---------------------------------------------------------------------------
# Main predictor
# ---------------------------------------------------------------------------

def predict(
    symbol: str,
    history: pd.DataFrame,
    horizon: Optional[int] = None,
    train_split: Optional[float] = None,
    min_confidence: Optional[float] = None,
    sector: Optional[str] = None,
    compute_cv_auroc: bool = True,
) -> Optional[MLPrediction]:
    """
    Train a soft-voting ensemble and predict the N-day alpha direction.

    Parameters
    ----------
    symbol          : ticker symbol (for logging / labelling)
    history         : OHLCV DataFrame, oldest → newest
    horizon         : days ahead to predict  (default: config.ML_PREDICT_HORIZON)
    train_split     : fraction used for training  (default: config.ML_TRAIN_SPLIT)
    min_confidence  : prob threshold for HIGH confidence
    sector          : GICS sector string for sector ETF features
    compute_cv_auroc: run 5-fold walk-forward CV for AUROC + KS (default True).
                      Adds ~3-8 s per ticker.  When True, predictions are
                      suppressed (returns None) if AUROC < ML_MIN_AUROC_THRESHOLD.

    Returns MLPrediction or None on failure / AUROC gate rejection.
    """
    _t_start = time.perf_counter()
    horizon        = horizon        or config.ML_PREDICT_HORIZON
    train_split    = train_split    or config.ML_TRAIN_SPLIT
    min_confidence = min_confidence or config.ML_HIGH_CONFIDENCE_THRESHOLD

    # ── guard: required libraries ──────────────────────────────────────────────
    try:
        import lightgbm            # noqa: F401
        from sklearn.preprocessing import StandardScaler  # noqa: F401
    except ImportError as err:
        logger.warning("Required ML library missing — %s. Ensemble disabled.", err)
        return None

    if history is None or history.empty or "Close" not in history.columns:
        return None

    # ── auto-detect sector from EDGAR SIC (best-effort, silent on failure) ────
    if sector is None:
        try:
            from edgar import _get_cik, _get_sic
            cik = _get_cik(symbol)
            if cik:
                sic  = _get_sic(cik)
                sic2 = str(sic)[:2] if sic else ""
                sector = _SIC_TO_SECTOR.get(sic2)
        except Exception:
            sector = None

    # ── build full feature matrix ──────────────────────────────────────────────
    try:
        features = build_all_features(symbol, history, sector=sector)
    except Exception as exc:
        logger.warning("%s: feature engineering failed — %s", symbol, exc)
        return None

    closes = history["Close"]

    # ── build labels: 1 = stock outperforms SPY over `horizon` days ──────────
    # Using an "alpha vs SPY" label de-means market-wide moves and gives the
    # model a much cleaner signal to learn (~+12 AUROC points vs raw direction).
    future_ret  = closes.shift(-horizon) / closes - 1
    spy_ret_h   = _get_spy_horizon_return(history.index, horizon)
    if spy_ret_h is not None:
        labels = (future_ret > spy_ret_h).astype(int)
        logger.debug("%s: using alpha-vs-SPY label (horizon=%dd)", symbol, horizon)
    else:
        # Fallback: raw price direction if SPY unavailable
        labels = (future_ret > 0).astype(int)
        logger.debug("%s: SPY unavailable — falling back to raw direction label", symbol)

    # ── impute supplemental columns before dropping ────────────────────────────
    # Columns that are *entirely* NaN (e.g. mc_upside when EDGAR fails,
    # rel_str_vs_sector when no sector ETF maps to an ETF) would eliminate
    # ALL rows via dropna().  Replace with 0 (neutral) so they are kept as
    # features (the model can learn to ignore a constant column).
    feat_with_label           = features.copy()
    feat_with_label["_label"] = labels
    # Find columns (excluding _label) that are >95% NaN → fill with 0
    frac_nan        = feat_with_label.drop(columns=["_label"]).isna().mean()
    mostly_nan_cols = frac_nan[frac_nan > 0.95].index.tolist()
    if mostly_nan_cols:
        logger.debug("%s: imputing %d fully-NaN feature cols with 0: %s",
                     symbol, len(mostly_nan_cols), mostly_nan_cols)
        feat_with_label[mostly_nan_cols] = feat_with_label[mostly_nan_cols].fillna(0.0)

    # Forward-fill then back-fill for remaining NaNs (e.g. macro warm-up rows)
    feat_with_label = feat_with_label.ffill().bfill()

    combined = feat_with_label.dropna()
    # Remove last `horizon` rows (no valid future label)
    if len(combined) > horizon:
        combined = combined.iloc[:-horizon]

    min_samples = config.ML_MIN_TRAIN_SAMPLES
    if len(combined) < min_samples:
        logger.warning(
            "%s: only %d clean samples after feature engineering (need %d) — skipping ML",
            symbol, len(combined), min_samples,
        )
        return None

    X_all = combined.drop(columns=["_label"]).values
    y_all = combined["_label"].values
    feat_names = list(combined.drop(columns=["_label"]).columns)

    # ── walk-forward train split ──────────────────────────────────────────────
    split_idx = int(len(X_all) * train_split)
    X_train   = X_all[:split_idx]
    y_train   = y_all[:split_idx]

    min_train_half = max(60, min_samples // 2)
    if len(X_train) < min_train_half:
        logger.warning("%s: not enough training samples (%d) — skipping ML", symbol, len(X_train))
        return None

    # ── optional: walk-forward CV AUROC + KS ─────────────────────────────────
    auroc_cv: Optional[float] = None
    ks_cv:    Optional[float] = None
    if compute_cv_auroc:
        try:
            # CV uses LGBM only for speed; XGB ensemble adds ~0.005 AUROC
            # which doesn't change the gate decision materially.
            auroc_cv, ks_cv = _walkforward_auroc(
                X_train, y_train,
                clf_factory = lambda: _build_lgbm(X_train.shape[1]),
                n_splits    = 5,
                gap         = horizon,
            )
            logger.info(
                "%s: walk-forward CV  AUROC=%.3f  KS=%.3f  (%.0f train samples)",
                symbol, auroc_cv, ks_cv, len(X_train),
            )

            # ── AUROC gate ────────────────────────────────────────────────────
            min_auroc = getattr(config, "ML_MIN_AUROC_THRESHOLD", 0.55)
            if auroc_cv < min_auroc:
                elapsed = time.perf_counter() - _t_start
                logger.info(
                    "%s: AUROC %.3f < threshold %.2f — prediction suppressed (%.1fs)",
                    symbol, auroc_cv, min_auroc, elapsed,
                )
                return None
        except Exception as exc:
            logger.debug("%s: CV AUROC computation failed — %s", symbol, exc)

    # ── train ensemble ────────────────────────────────────────────────────────
    try:
        lgbm, xgbm, logreg, scaler = _train_ensemble(
            X_train, y_train, feat_names
        )
    except Exception as exc:
        logger.warning("%s: ensemble training failed — %s", symbol, exc)
        return None

    # ── predict on the LATEST available bar ──────────────────────────────────
    full_features = build_all_features(symbol, history, sector=sector)
    # Apply same imputation as training
    full_frac_nan    = full_features.isna().mean()
    full_mostly_nan  = full_frac_nan[full_frac_nan > 0.95].index.tolist()
    if full_mostly_nan:
        full_features[full_mostly_nan] = full_features[full_mostly_nan].fillna(0.0)
    full_features = full_features.ffill().bfill()

    if full_features.empty:
        return None

    # Use last row; align to training columns
    latest_row = full_features.iloc[[-1]].copy()
    for col in feat_names:
        if col not in latest_row.columns:
            latest_row[col] = np.nan
    latest_row = latest_row[feat_names]

    # Impute any remaining NaN with training column medians
    train_medians = pd.DataFrame(X_train, columns=feat_names).median()
    latest_row = latest_row.fillna(train_medians)
    latest_X   = latest_row.values

    try:
        p_up = _ensemble_predict_proba(latest_X, lgbm, xgbm, logreg, scaler)
    except Exception as exc:
        logger.warning("%s: ensemble prediction failed — %s", symbol, exc)
        return None

    direction = "UP" if p_up >= 0.5 else "DOWN"
    prob      = p_up if direction == "UP" else (1.0 - p_up)

    # ── confidence tier ───────────────────────────────────────────────────────
    if prob >= min_confidence:
        confidence = "HIGH"
    elif prob >= config.ML_MEDIUM_CONFIDENCE_THRESHOLD:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # ── feature importances from LightGBM ────────────────────────────────────
    try:
        importances = dict(zip(feat_names, lgbm.feature_importances_))
        top_features = dict(
            sorted(importances.items(), key=lambda x: x[1], reverse=True)[:8]
        )
    except Exception:
        top_features = {}

    result = MLPrediction(
        symbol              = symbol,
        direction           = direction,
        probability         = prob,
        confidence          = confidence,
        feature_importances = top_features,
        n_train_samples     = len(X_train),
        horizon_days        = horizon,
        auroc_cv            = auroc_cv,
        ks_cv               = ks_cv,
        elapsed_seconds     = time.perf_counter() - _t_start,
    )
    logger.info(str(result))
    return result


# ---------------------------------------------------------------------------
# Swing-trading sell metrics
# ---------------------------------------------------------------------------

def compute_swing_metrics(symbol: str, history: pd.DataFrame) -> Optional[SwingMetrics]:
    """
    Compute sell-side swing-trading indicators from OHLCV history.

    All values are derived purely from historical price/volume data.
    No forward-looking information is used.

    Parameters
    ----------
    symbol  : ticker (for labelling only)
    history : OHLCV DataFrame, oldest → newest (≥ 30 rows recommended)

    Returns
    -------
    SwingMetrics or None on insufficient data.
    """
    if history is None or history.empty or "Close" not in history.columns:
        return None

    c    = history["Close"].dropna()
    if len(c) < 30:
        return None

    high = history.get("High", c)
    low  = history.get("Low",  c)
    vol  = history.get("Volume", pd.Series(dtype=float, name="Volume", index=history.index))

    current_price = float(c.iloc[-1])

    # ── ATR (14-day, absolute $ terms) ───────────────────────────────────────
    prev_close = c.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_abs = float(tr.ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1])
    atr_norm = atr_abs / current_price if current_price > 0 else float("nan")

    atr_stop_price   = current_price - 2.0 * atr_abs
    atr_target_price = current_price + 3.0 * atr_abs
    stop_dist        = current_price - atr_stop_price
    target_dist      = atr_target_price - current_price
    rr_ratio         = (target_dist / stop_dist) if stop_dist > 0 else float("nan")

    # ── RSI-14 ────────────────────────────────────────────────────────────────
    rsi_raw = _rsi(c, 14)
    rsi_val = float(rsi_raw.iloc[-1]) if not rsi_raw.empty else float("nan")
    if np.isnan(rsi_val):
        rsi_sig = "NEUTRAL"
    elif rsi_val >= 70:
        rsi_sig = "OVERBOUGHT"
    elif rsi_val <= 30:
        rsi_sig = "OVERSOLD"
    else:
        rsi_sig = "NEUTRAL"

    # ── Bollinger %B ──────────────────────────────────────────────────────────
    ma20   = _sma(c, 20)
    std20  = c.rolling(20, min_periods=20).std()
    upper  = ma20 + 2 * std20
    lower  = ma20 - 2 * std20
    bw     = (upper - lower).replace(0, np.nan)
    bb_pct = float(((c - lower) / bw).iloc[-1]) if not bw.empty else float("nan")
    if np.isnan(bb_pct):
        bb_sig = "NEUTRAL"
    elif bb_pct >= 0.90:
        bb_sig = "EXTENDED"
    elif bb_pct <= 0.10:
        bb_sig = "COMPRESSED"
    else:
        bb_sig = "NEUTRAL"

    # ── MACD histogram ────────────────────────────────────────────────────────
    ema12   = c.ewm(span=12, adjust=False).mean()
    ema26   = c.ewm(span=26, adjust=False).mean()
    macd_l  = ema12 - ema26
    sig_l   = macd_l.ewm(span=9, adjust=False).mean()
    hist_s  = macd_l - sig_l
    # Normalise by price for scale-invariance
    hist_norm = float((hist_s / c.replace(0, np.nan)).iloc[-1]) if len(hist_s) > 1 else 0.0
    # Detect cross: histogram flipped negative in last 2 bars
    if len(hist_s) >= 2:
        if hist_s.iloc[-2] >= 0 and hist_s.iloc[-1] < 0:
            macd_sig = "BEARISH_CROSS"
        elif hist_s.iloc[-1] > 0:
            macd_sig = "BULLISH"
        else:
            macd_sig = "NEUTRAL"
    else:
        macd_sig = "NEUTRAL"

    # ── MA gauges ─────────────────────────────────────────────────────────────
    ma50_val  = float(_sma(c, 50).iloc[-1])  if len(c) >= 50  else float("nan")
    ma200_val = float(_sma(c, 200).iloc[-1]) if len(c) >= 200 else float("nan")
    pct_vs_ma50  = (current_price / ma50_val  - 1) if not np.isnan(ma50_val)  and ma50_val  > 0 else float("nan")
    pct_vs_ma200 = (current_price / ma200_val - 1) if not np.isnan(ma200_val) and ma200_val > 0 else float("nan")

    # ── Trend strength (ADX-proxy) ────────────────────────────────────────────
    # Use normalised spread between fast and slow MAs relative to 20-day volatility
    if not np.isnan(ma50_val) and not np.isnan(ma200_val):
        ma_spread  = abs(ma50_val - ma200_val) / ma200_val
        hvol20     = float(np.log(c / c.shift(1)).rolling(20).std().iloc[-1])
        trend_str  = float(np.clip(ma_spread / (hvol20 * 20**0.5 + 1e-9), 0, 1)) if hvol20 > 0 else 0.0
    else:
        trend_str = float("nan")

    # ── Volatility regime ─────────────────────────────────────────────────────
    log_ret   = np.log(c / c.shift(1))
    hvol20    = log_ret.rolling(20, min_periods=10).std()
    hvol_mean = float(hvol20.rolling(252, min_periods=60).mean().iloc[-1])
    hvol_std  = float(hvol20.rolling(252, min_periods=60).std().iloc[-1])
    hvol_now  = float(hvol20.iloc[-1])
    if hvol_std > 0:
        z_vol = (hvol_now - hvol_mean) / hvol_std
        if z_vol > 1.0:
            vol_reg = "HIGH"
        elif z_vol < -1.0:
            vol_reg = "LOW"
        else:
            vol_reg = "NORMAL"
    else:
        vol_reg = "NORMAL"

    # ── 52-week high drawdown ─────────────────────────────────────────────────
    hi52   = float(c.tail(252).max())
    hi52_i = c.tail(252).idxmax()
    today  = c.index[-1]
    days_since = int((today - hi52_i).days) if hasattr((today - hi52_i), "days") else 0
    ddraw  = (current_price / hi52 - 1) if hi52 > 0 else 0.0

    # ── Composite sell score (0–100) ──────────────────────────────────────────
    # Each sub-signal contributes up to its weight; score 100 = maximum sell pressure.
    sell_score = 0.0

    # RSI overbought (max 25 pts)
    if not np.isnan(rsi_val):
        sell_score += float(np.clip((rsi_val - 50) / 50 * 25, 0, 25))

    # Bollinger extension (max 20 pts)
    if not np.isnan(bb_pct):
        sell_score += float(np.clip((bb_pct - 0.5) / 0.5 * 20, 0, 20))

    # MACD bearish (max 15 pts)
    if macd_sig == "BEARISH_CROSS":
        sell_score += 15.0
    elif macd_sig == "NEUTRAL" and hist_norm < 0:
        sell_score += 7.0

    # Price stretched above MA50 (max 15 pts)
    if not np.isnan(pct_vs_ma50):
        sell_score += float(np.clip(pct_vs_ma50 / 0.20 * 15, 0, 15))

    # Price stretched above MA200 (max 10 pts)
    if not np.isnan(pct_vs_ma200):
        sell_score += float(np.clip(pct_vs_ma200 / 0.30 * 10, 0, 10))

    # Volatility regime high (max 10 pts) — exits are cheaper in low-vol windows
    if vol_reg == "HIGH":
        sell_score += 10.0
    elif vol_reg == "LOW":
        sell_score -= 5.0   # low-vol trending markets; don't rush to sell

    # Proximity to 52-week high (max 5 pts)
    if hi52 > 0 and not np.isnan(ddraw):
        sell_score += float(np.clip((1 + ddraw) * 5, 0, 5))

    sell_score = float(np.clip(sell_score, 0, 100))

    # ── Recommendation ────────────────────────────────────────────────────────
    if sell_score >= 70:
        sell_rec = "STRONG_SELL"
    elif sell_score >= 50:
        sell_rec = "CONSIDER_SELL"
    elif sell_score >= 25:
        sell_rec = "HOLD"
    else:
        sell_rec = "ADD"

    result = SwingMetrics(
        symbol              = symbol,
        atr_14              = float(atr_norm),
        atr_stop_price      = float(atr_stop_price),
        atr_target_price    = float(atr_target_price),
        reward_risk_ratio   = float(rr_ratio),
        rsi_14              = float(rsi_val),
        rsi_signal          = rsi_sig,
        bb_position         = float(bb_pct),
        bb_signal           = bb_sig,
        macd_hist           = float(hist_norm),
        macd_signal         = macd_sig,
        price_vs_ma50       = float(pct_vs_ma50),
        price_vs_ma200      = float(pct_vs_ma200),
        trend_strength      = float(trend_str),
        vol_regime          = vol_reg,
        days_since_high     = days_since,
        drawdown_from_high  = float(ddraw),
        composite_sell_score = sell_score,
        sell_recommendation  = sell_rec,
    )
    logger.info(str(result))
    return result


# ---------------------------------------------------------------------------
# Multi-horizon predictor
# ---------------------------------------------------------------------------

# Horizon definitions: label → (trading days, human label)
_HORIZONS: list[tuple[str, int]] = [
    ("week1",  5),
    ("month1", 21),
    ("year1",  252),
]


def predict_multi_horizon(
    symbol:  str,
    history: pd.DataFrame,
    sector:  Optional[str] = None,
    compute_cv_auroc: bool = True,
) -> Optional[MultiHorizonOutlook]:
    """
    Run three independent ensemble models — one per horizon — and package
    the results into a MultiHorizonOutlook together with SwingMetrics.

    Horizons
    --------
    1W  (5 trading days)   — swing entry timing
    1M  (21 trading days)  — position sizing / medium conviction
    1Y  (252 trading days) — long-term trend filter / hold/sell decision

    Each horizon trains its own LightGBM+XGBoost+LogReg ensemble with an
    alpha-vs-SPY label at the respective forward horizon.  CV AUROC gating
    applies independently (so a ticker might have a valid 1W but no 1Y).

    SwingMetrics are computed once from current price history and attached
    to the outlook — they are independent of ML and are always returned if
    there is enough data (≥ 30 bars).

    Parameters
    ----------
    symbol          : ticker symbol
    history         : OHLCV DataFrame, oldest → newest
    sector          : GICS sector (optional, improves macro features)
    compute_cv_auroc: propagated to each horizon's predict() call

    Returns
    -------
    MultiHorizonOutlook or None if history is entirely insufficient.
    """
    if history is None or history.empty or "Close" not in history.columns:
        logger.warning("%s: predict_multi_horizon — empty history", symbol)
        return None

    # 1-year models need ≥ 252 future bars; ensure we have enough total history
    # for at least the short-horizon model to work.
    min_bars = getattr(config, "ML_MIN_TRAIN_SAMPLES", 200)
    if len(history) < min_bars:
        logger.warning(
            "%s: predict_multi_horizon — only %d bars (need %d)",
            symbol, len(history), min_bars,
        )
        return None

    predictions: dict[str, Optional[MLPrediction]] = {}

    for key, horizon_days in _HORIZONS:
        # 1Y model needs much more history; skip gracefully if data is short
        if len(history) < horizon_days * 3 + min_bars:
            logger.debug(
                "%s: skipping %s horizon — insufficient bars (%d)",
                symbol, key, len(history),
            )
            predictions[key] = None
            continue

        logger.debug("%s: training %s (%dd) model …", symbol, key, horizon_days)
        pred = predict(
            symbol           = symbol,
            history          = history,
            horizon          = horizon_days,
            sector           = sector,
            compute_cv_auroc = compute_cv_auroc,
        )
        predictions[key] = pred

    # Compute swing metrics (always, independent of ML)
    swing = compute_swing_metrics(symbol, history)

    outlook = MultiHorizonOutlook(
        symbol        = symbol,
        week1         = predictions.get("week1"),
        month1        = predictions.get("month1"),
        year1         = predictions.get("year1"),
        swing_metrics = swing,
    )
    logger.info("\n%s", outlook)
    return outlook
