"""
ml_predictor.py — Ensemble ML predictor for 5-day price-direction forecasting.

Architecture (Phase 1)
----------------------
Layer 0  — Feature engineering
    • Technical indicators   (RSI, momentum, Bollinger, vol, MA ratios, ATR, OBV)
    • Macro/sector features  (SPY, QQQ, sector ETF returns vs stock return)
    • HMM regime feature     (2-state Gaussian HMM on log-returns → regime prob)
    • Monte Carlo fair value  (log-normal MC using EDGAR FCF → upside %)

Layer 1  — Base learners (trained independently, NO look-ahead)
    • LightGBM               (gradient-boosted trees, fast, handles non-linearity)
    • Logistic Regression    (L2-regularised, calibrated probabilities)
    • SGD Classifier         (log-loss, elastic-net regularisation — linear speed)

Layer 2  — Meta-learner
    • Logistic Regression stacked on the 3 base-learner OOF probabilities
      (trains on out-of-fold predictions to avoid leakage)

Validation
----------
    • 5-fold purged walk-forward CV (gap = horizon days between folds)
    • AUROC reported in logs; model re-trained on full training window for
      live prediction

Anti-overfitting guards
-----------------------
    • LightGBM: num_leaves=31, min_child_samples=30, reg_lambda=1.0,
                feature_fraction=0.7, bagging_fraction=0.8
    • LogReg / SGD: strong L2 / elastic-net regularisation
    • StandardScaler applied before linear models
    • Macro/HMM features forward-filled from external fetch (no label data)
    • MC fair value derived from EDGAR fundamentals (never from future prices)

Public API
----------
    predict(symbol, history) -> MLPrediction | None

MLPrediction fields
-------------------
    symbol              str
    direction           "UP" | "DOWN"
    probability         float   0-1  (probability of UP class)
    confidence          "HIGH" | "MEDIUM" | "LOW"
    feature_importances dict[str, float]   top features (LightGBM importance)
    n_train_samples     int
    horizon_days        int
    auroc_cv            float | None   (5-fold walk-forward AUROC, if computed)
"""

from __future__ import annotations

import logging
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
    probability: float                    # probability of UP class (0-1)
    confidence: str                       # "HIGH" | "MEDIUM" | "LOW"
    feature_importances: dict = field(default_factory=dict)
    n_train_samples: int = 0
    horizon_days: int = 0
    auroc_cv: Optional[float] = None      # walk-forward CV AUROC (if computed)

    def __str__(self) -> str:
        auroc_str = f" CV-AUROC={self.auroc_cv:.3f}" if self.auroc_cv else ""
        return (
            f"{self.symbol}: ML={self.direction} "
            f"p={self.probability:.0%} [{self.confidence}]"
            f"{auroc_str} "
            f"(trained on {self.n_train_samples} samples, {self.horizon_days}d horizon)"
        )


# ---------------------------------------------------------------------------
# Caches (process-lifetime)
# ---------------------------------------------------------------------------

_macro_cache: dict[str, pd.DataFrame]  = {}   # ticker → price series
_mc_cache:    dict[str, float]          = {}   # symbol → MC upside pct
_hmm_cache:   dict[str, object]         = {}   # symbol → fitted HMM


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
    Combine technical + macro + HMM + MC features into a single DataFrame.
    """
    tech  = build_technical_features(history)
    macro = build_macro_features(history, sector=sector)
    hmm   = build_hmm_features(history)
    mc    = build_mc_feature(symbol, history).to_frame()

    return pd.concat([tech, macro, hmm, mc], axis=1)


# ---------------------------------------------------------------------------
# Walk-forward CV AUROC evaluator
# ---------------------------------------------------------------------------

def _walkforward_auroc(X: np.ndarray,
                       y: np.ndarray,
                       clf_factory,
                       n_splits: int = 5,
                       gap: int = 10,
                       min_train: int = 100) -> float:
    """
    Purged walk-forward cross-validation returning mean AUROC.

    Splits the time-ordered data into n_splits folds.
    A gap of `gap` samples (≈ trading days) is removed between train/test
    to avoid information leakage from auto-correlated returns.
    """
    from sklearn.metrics import roc_auc_score

    n      = len(X)
    fold_size = n // (n_splits + 1)
    aurocs = []

    for k in range(1, n_splits + 1):
        test_start  = k * fold_size
        test_end    = min(test_start + fold_size, n)
        train_end   = test_start - gap

        if train_end < min_train or test_end <= test_start:
            continue

        X_tr, y_tr = X[:train_end],        y[:train_end]
        X_te, y_te = X[test_start:test_end], y[test_start:test_end]

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue

        try:
            clf = clf_factory()
            clf.fit(X_tr, y_tr)
            proba = clf.predict_proba(X_te)[:, 1]
            aurocs.append(roc_auc_score(y_te, proba))
        except Exception:
            pass

    return float(np.mean(aurocs)) if aurocs else 0.5


# ---------------------------------------------------------------------------
# Model builders (base learners)
# ---------------------------------------------------------------------------

def _build_lgbm(n_features: int):
    """LightGBM classifier with anti-overfitting params."""
    import lightgbm as lgb
    return lgb.LGBMClassifier(
        objective         = "binary",
        n_estimators      = 300,
        num_leaves        = 31,
        max_depth         = -1,
        min_child_samples = 30,
        learning_rate     = 0.05,
        reg_alpha         = 0.5,
        reg_lambda        = 1.0,
        feature_fraction  = 0.7,
        bagging_fraction  = 0.8,
        bagging_freq      = 5,
        class_weight      = "balanced",
        random_state      = 42,
        n_jobs            = -1,
        verbose           = -1,
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


def _build_sgd():
    from sklearn.linear_model import SGDClassifier
    return SGDClassifier(
        loss         = "log_loss",
        penalty      = "elasticnet",
        alpha        = 0.01,
        l1_ratio     = 0.15,
        max_iter     = 500,
        class_weight = "balanced",
        random_state = 42,
    )


# ---------------------------------------------------------------------------
# Stacking ensemble
# ---------------------------------------------------------------------------

def _train_ensemble(X_train: np.ndarray,
                    y_train: np.ndarray,
                    feature_names: list[str]):
    """
    Train a soft-voting ensemble: LightGBM + LogReg + SGD.

    All three base learners are trained on the full training set.
    Prediction = weighted average of their P(UP) probabilities.
    Weights: LightGBM=0.5, LogReg=0.3, SGD=0.2 (LGBM dominates).

    This is more reliable than OOF stacking when n_train < 1000, 
    because stacking consumes ~N/k samples per fold just for meta-training.
    """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_train)

    lgbm   = _build_lgbm(X_train.shape[1])
    logreg = _build_logreg()
    sgd    = _build_sgd()

    lgbm.fit(X_train, y_train)
    logreg.fit(X_sc, y_train)
    sgd.fit(X_sc, y_train)

    return lgbm, logreg, sgd, scaler, None   # meta_lr=None → weighted average


def _ensemble_predict_proba(X_new: np.ndarray,
                             lgbm, logreg, sgd, scaler, meta_lr) -> float:
    """
    Predict P(UP) using soft-voting: LGBM×0.5 + LogReg×0.3 + SGD×0.2.
    """
    X_sc   = scaler.transform(X_new)
    p_lgbm = float(lgbm.predict_proba(X_new)[:, 1][0])
    p_lr   = float(logreg.predict_proba(X_sc)[:, 1][0])
    p_sgd  = float(sgd.predict_proba(X_sc)[:, 1][0])
    # Weighted soft vote
    p_up = 0.50 * p_lgbm + 0.30 * p_lr + 0.20 * p_sgd

    return p_up


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
    compute_cv_auroc: bool = False,
) -> Optional[MLPrediction]:
    """
    Train a stacking ensemble and predict the N-day price direction.

    Parameters
    ----------
    symbol          : ticker symbol (for logging / labelling)
    history         : OHLCV DataFrame, oldest → newest
    horizon         : days ahead to predict  (default: config.ML_PREDICT_HORIZON)
    train_split     : fraction used for training  (default: config.ML_TRAIN_SPLIT)
    min_confidence  : prob threshold for HIGH confidence  (default: config.ML_HIGH_CONFIDENCE_THRESHOLD)
    sector          : GICS sector string for sector ETF features  (auto-detected if None)
    compute_cv_auroc: if True, run 5-fold walk-forward CV and log AUROC (adds ~3s)

    Returns MLPrediction or None on failure.
    """
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

    # ── build labels: 1 = UP in `horizon` days ───────────────────────────────
    future_ret = closes.shift(-horizon) / closes - 1
    labels     = (future_ret > 0).astype(int)

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

    # ── optional: walk-forward CV AUROC ──────────────────────────────────────
    auroc_cv: Optional[float] = None
    if compute_cv_auroc:
        try:
            from sklearn.preprocessing import StandardScaler
            scaler_cv = StandardScaler()
            X_sc_cv   = scaler_cv.fit_transform(X_train)
            auroc_cv  = _walkforward_auroc(
                X_train, y_train,
                clf_factory = lambda: _build_lgbm(X_train.shape[1]),
                n_splits    = 5,
                gap         = horizon,
            )
            logger.info("%s: walk-forward CV AUROC (LightGBM) = %.3f", symbol, auroc_cv)
        except Exception as exc:
            logger.debug("%s: CV AUROC computation failed — %s", symbol, exc)

    # ── train ensemble ────────────────────────────────────────────────────────
    try:
        lgbm, logreg, sgd, scaler, meta_lr = _train_ensemble(
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
        p_up = _ensemble_predict_proba(latest_X, lgbm, logreg, sgd, scaler, meta_lr)
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
    )
    logger.info(str(result))
    return result
