"""
stockscreen/technicals.py — price position, historical percentiles, and
trend indicators. Pure functions over a daily OHLCV DataFrame (split-adjusted
closes, ascending dates); nothing here touches the network or the database.

Definitions
-----------
Range position (%)   = (P - low) / (high - low) x 100 over the lookback window of
                       closing prices, P included. 0 = at the low, 100 = at the high.
                       Undefined (None) when high == low.
Closing-price        = share of the window's *previous* closes below P, ties counted
percentile             half (mean rank): (below + 0.5 x equal) / n x 100. The current
                       close is excluded from the reference distribution so it is
                       never counted against itself.
Lookback windows     are calendar-based (1, 2, 3 years back from the last bar) and
                       are "available" only if the history begins within a grace period
                       of the window start — a 2.5-year-old stock has no 3-year numbers.
RSI                  Wilder's 14-day RSI (seeded with the first 14-day simple average).
MACD                 12/26 EMAs (adjust=False), 9-EMA signal, histogram = MACD - signal.
ATR / ADX            Wilder smoothing, 14 days.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

LOOKBACKS = {"52w": 1, "2y": 2, "3y": 3}     # label -> years


# ---------------------------------------------------------------------------
# indicators
# ---------------------------------------------------------------------------

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _wilder(x: pd.Series, n: int) -> pd.Series:
    """Wilder smoothing: first value = simple mean of the first n, then
    avg_t = (avg_{t-1} x (n-1) + x_t) / n."""
    x = x.astype(float)
    out = pd.Series(np.nan, index=x.index)
    vals = x.to_numpy()
    valid = np.flatnonzero(~np.isnan(vals))
    if len(valid) < n:
        return out
    first = valid[0] + n - 1
    avg = np.nanmean(vals[valid[0]:first + 1])
    res = np.full(len(vals), np.nan)
    res[first] = avg
    for i in range(first + 1, len(vals)):
        avg = (avg * (n - 1) + vals[i]) / n
        res[i] = avg
    return pd.Series(res, index=x.index)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.astype(float).diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = _wilder(gain.iloc[1:], n).reindex(close.index)
    avg_loss = _wilder(loss.iloc[1:], n).reindex(close.index)
    rs = avg_gain / avg_loss
    out = 100 - 100 / (1 + rs)
    out[(avg_loss == 0) & (avg_gain > 0)] = 100.0
    out[(avg_loss == 0) & (avg_gain == 0)] = 50.0
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return _wilder(tr.iloc[1:], n).reindex(close.index)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    up, down = high.diff(), -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    a = atr(high, low, close, n)
    plus_di = 100 * _wilder(plus_dm.iloc[1:], n).reindex(close.index) / a
    minus_di = 100 * _wilder(minus_dm.iloc[1:], n).reindex(close.index) / a
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return _wilder(dx.dropna(), n).reindex(close.index)


# ---------------------------------------------------------------------------
# price position
# ---------------------------------------------------------------------------

def window_start(last: pd.Timestamp, years: int) -> pd.Timestamp:
    try:
        return last.replace(year=last.year - years)
    except ValueError:                       # Feb 29
        return (last - pd.Timedelta(days=1)).replace(year=last.year - years)


def price_position(close: pd.Series, years: int, grace_days: int = 7) -> dict:
    """Range position, percentile and distances for one lookback. Returns
    {"available": False, ...} when the history doesn't cover the window."""
    if close is None or close.empty:
        return {"available": False, "reason": "no price history"}
    close = close.dropna()
    last = close.index[-1]
    start = window_start(last, years)
    if close.index[0] > start + pd.Timedelta(days=grace_days):
        return {"available": False, "reason": f"history starts {close.index[0].date()}, needs {start.date()}"}
    win = close[close.index >= start]
    p = float(win.iloc[-1])
    lo, hi = float(win.min()), float(win.max())
    prev = win.iloc[:-1].to_numpy()                 # the current close is not its own reference
    pct = None
    if len(prev):
        pct = float(((prev < p).sum() + 0.5 * (prev == p).sum()) / len(prev) * 100)
    return {
        "available": True, "window_start": start.strftime("%Y-%m-%d"), "n": int(len(win)),
        "low": lo, "high": hi, "price": p,
        "range_pos": None if hi == lo else float((p - lo) / (hi - lo) * 100),
        "percentile": pct,
        "pct_below_high": float((hi - p) / hi * 100),
        "pct_above_low": float((p - lo) / lo * 100),
    }


def price_positions(close: pd.Series, grace_days: int = 7) -> dict:
    return {label: price_position(close, yrs, grace_days) for label, yrs in LOOKBACKS.items()}


# ---------------------------------------------------------------------------
# indicator snapshot
# ---------------------------------------------------------------------------

def _f(x) -> Optional[float]:
    return None if x is None or pd.isna(x) else float(x)


def indicator_snapshot(df: pd.DataFrame, slope_window: int = 20) -> dict:
    """Latest indicator values plus the few derived facts the classifier needs.
    Missing indicators are None, never defaulted."""
    close = df["Close"].astype(float)
    n = len(close)
    out: dict = {"n_bars": n, "last_date": df.index[-1].strftime("%Y-%m-%d"), "price": float(close.iloc[-1])}
    s20, s50, s200 = sma(close, 20), sma(close, 50), sma(close, 200)
    out.update(sma20=_f(s20.iloc[-1]), sma50=_f(s50.iloc[-1]), sma200=_f(s200.iloc[-1]))

    def slope(series: pd.Series) -> Optional[float]:
        if len(series) <= slope_window or pd.isna(series.iloc[-1]) or pd.isna(series.iloc[-1 - slope_window]):
            return None
        return float(series.iloc[-1] / series.iloc[-1 - slope_window] - 1)

    out.update(sma20_slope=slope(s20), sma50_slope=slope(s50), sma200_slope=slope(s200))
    out["dev_sma50_pct"] = None if out["sma50"] is None else (out["price"] / out["sma50"] - 1) * 100
    out["dev_sma200_pct"] = None if out["sma200"] is None else (out["price"] / out["sma200"] - 1) * 100
    out["sma50_vs_sma200_pct"] = (None if out["sma50"] is None or out["sma200"] is None
                                  else (out["sma50"] / out["sma200"] - 1) * 100)

    r = rsi(close)
    out["rsi"] = _f(r.iloc[-1])
    out["rsi_5d_ago"] = _f(r.iloc[-6]) if n > 6 else None

    m = macd(close)
    hist = m["hist"]
    out.update(macd=_f(m["macd"].iloc[-1]), macd_signal=_f(m["signal"].iloc[-1]), macd_hist=_f(hist.iloc[-1]),
               macd_hist_prev=_f(hist.iloc[-2]) if n > 2 else None)
    # most recent signal-line crossover within the last 10 sessions
    cross, cross_days = None, None
    if n >= 35:
        sign = np.sign(hist.iloc[-11:].to_numpy())
        for k in range(len(sign) - 1, 0, -1):
            if sign[k] != sign[k - 1] and sign[k] != 0:
                cross = "bullish" if sign[k] > 0 else "bearish"
                cross_days = len(sign) - 1 - k
                break
    out.update(macd_cross=cross, macd_cross_days_ago=cross_days)

    out["high_60d"] = _f(close.iloc[-60:].max())
    out["pct_below_60d_high"] = None if out["high_60d"] in (None, 0) else (out["high_60d"] - out["price"]) / out["high_60d"] * 100

    if {"High", "Low"} <= set(df.columns):
        a = atr(df["High"].astype(float), df["Low"].astype(float), close)
        out["atr_pct"] = None if pd.isna(a.iloc[-1]) else float(a.iloc[-1] / out["price"] * 100)
        ad = adx(df["High"].astype(float), df["Low"].astype(float), close)
        out["adx"] = _f(ad.iloc[-1])
    if "Volume" in df.columns:
        vol = df["Volume"].astype(float)
        base = vol.iloc[-21:-1].mean() if n > 21 else np.nan
        out["rel_volume"] = None if pd.isna(base) or base == 0 else float(vol.iloc[-1] / base)
    return out


# ---------------------------------------------------------------------------
# classification (thresholds come from ScreenConfig)
# ---------------------------------------------------------------------------

TREND_LABELS = ["Strong uptrend", "Healthy pullback", "Trend reversal candidate", "Downtrend",
                "Sideways/consolidating", "Insufficient data"]


def recovery_signals(t: dict, cfg) -> list[str]:
    """Transparent, individually-listed signs that a decline is stabilising."""
    sig = []
    if t.get("sma20") is not None and t["price"] > t["sma20"]:
        sig.append("price reclaimed SMA20")
    if t.get("rsi") is not None and t.get("rsi_5d_ago") is not None and t["rsi"] > cfg.rsi_weak and t["rsi"] > t["rsi_5d_ago"]:
        sig.append("RSI above 40 and rising")
    if (t.get("macd_hist") is not None and t["macd_hist"] > 0) or t.get("macd_cross") == "bullish":
        sig.append("MACD bullish")
    if t.get("sma20_slope") is not None and t["sma20_slope"] > 0:
        sig.append("SMA20 rising")
    return sig


def classify_trend(t: dict, cfg) -> dict:
    """Explicit rules (see README). Returns {label, signals, reasons}."""
    need = ("sma50", "sma200", "sma200_slope", "sma50_slope")
    if t.get("n_bars", 0) < 200 + cfg.slope_window or any(t.get(k) is None for k in need):
        return {"label": "Insufficient data", "signals": [], "reasons": ["fewer than 200 + slope-window daily closes"]}
    p, s20, s50, s200 = t["price"], t["sma20"], t["sma50"], t["sma200"]
    a, b = p > s200, s50 > s200
    c, d = t["sma200_slope"] > 0, t["sma50_slope"] > 0
    e = p > s50
    reasons = [f"price {'>' if a else '<'} SMA200", f"SMA50 {'>' if b else '<'} SMA200",
               f"SMA200 slope {'up' if c else 'down/flat'}", f"SMA50 slope {'up' if d else 'down/flat'}"]
    sig = recovery_signals(t, cfg)
    band = cfg.sideways_band_pct
    flat = abs(t["dev_sma200_pct"]) < band and abs(t["sma50_vs_sma200_pct"]) < band
    if flat:                                       # price and both averages bunched together: no directional structure
        return {"label": "Sideways/consolidating", "signals": sig,
                "reasons": reasons + [f"price and SMA50 within {band:g}% of SMA200 (no directional structure)"]}
    if a and b and c and d and e:
        return {"label": "Strong uptrend", "signals": sig, "reasons": reasons}
    if a and b and t["sma200_slope"] >= 0:
        return {"label": "Healthy pullback", "signals": sig,
                "reasons": reasons + ["long-term structure intact; short-term weakness"]}
    if (not a) and (not b):                                        # price and SMA50 both under SMA200
        if len(sig) >= cfg.reversal_min_signals:
            return {"label": "Trend reversal candidate", "signals": sig, "reasons": reasons + [f"{len(sig)} recovery signal(s)"]}
        return {"label": "Downtrend", "signals": sig, "reasons": reasons}
    if a and not b:                                                # recovering through SMA200, SMA50 still below
        if len(sig) >= cfg.reversal_min_signals:
            return {"label": "Trend reversal candidate", "signals": sig, "reasons": reasons + [f"{len(sig)} recovery signal(s)"]}
        return {"label": "Sideways/consolidating", "signals": sig, "reasons": reasons}
    # price below SMA200 while SMA50 is still above it: mixed
    return {"label": "Sideways/consolidating", "signals": sig, "reasons": reasons + ["mixed structure"]}


def rsi_band(rsi_value: Optional[float], cfg) -> str:
    if rsi_value is None:
        return "n/a"
    if rsi_value < cfg.rsi_oversold:
        return "oversold"
    if rsi_value < cfg.rsi_weak:
        return "weak momentum"
    if rsi_value <= cfg.rsi_positive:
        return "neutral"
    if rsi_value <= cfg.rsi_overbought:
        return "positive momentum"
    return "overbought"


def macd_status(t: dict) -> str:
    h, hp = t.get("macd_hist"), t.get("macd_hist_prev")
    if h is None:
        return "n/a"
    if t.get("macd_cross") == "bullish":
        return f"bullish crossover ({t['macd_cross_days_ago']}d ago)"
    if t.get("macd_cross") == "bearish":
        return f"bearish crossover ({t['macd_cross_days_ago']}d ago)"
    if h > 0:
        return "bullish, strengthening" if hp is not None and h > hp else "bullish, fading"
    return "bearish, recovering" if hp is not None and h > hp else "bearish"


PRICE_ZONES = ["Deep value-zone candidate", "Low-price candidate", "Near highs", "Mid-range", "Insufficient history"]


def classify_price_zone(pos: dict, cfg) -> dict:
    """Price location (not valuation). A lookback window is "in the low zone" if
    its range position OR its closing-price percentile is at or below the threshold."""
    thr_range = {"52w": cfg.low_range_pos_52w, "2y": cfg.low_range_pos_2y, "3y": cfg.low_range_pos_3y}
    in_zone, available = [], 0
    for label, p in pos.items():
        if not p.get("available"):
            continue
        available += 1
        rp, pc = p.get("range_pos"), p.get("percentile")
        if (rp is not None and rp <= thr_range[label]) or (pc is not None and pc <= cfg.low_percentile):
            in_zone.append(label)
    if available == 0:
        return {"label": "Insufficient history", "in_zone": [], "available": 0}
    if len(in_zone) >= cfg.deep_zone_min_lookbacks:
        label = "Deep value-zone candidate"
    elif in_zone:
        label = "Low-price candidate"
    else:
        rp52 = pos.get("52w", {}).get("range_pos")
        label = "Near highs" if rp52 is not None and rp52 >= cfg.near_high_range_pos else "Mid-range"
    return {"label": label, "in_zone": in_zone, "available": available}
