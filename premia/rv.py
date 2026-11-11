"""
premia/rv.py — realized volatility: estimators, forecasts, and their out-of-sample evaluation (phase 3).

Estimators (daily variance proxies, from OHLC; no intraday data is available):
  close-to-close r^2; Parkinson (1980); Garman-Klass (1980); Rogers-Satchell (1991) + overnight^2, the building block
  of Yang-Zhang (2000), which is drift-robust and keeps overnight gaps.

Forecasts of the annualised realized variance over the next h trading days (target: close-to-close, because daily
delta-hedge P&L depends on close-to-close moves):
  naive    trailing-h close-to-close RV                          (benchmark the models must beat)
  ewma     RiskMetrics, lambda = 0.94
  har      log-HAR on 1/5/22-day range-based variance (Corsi 2009), refit every 21 days on data known at t
  har_iv   log-HAR plus log IV^2 as a regressor (Busch, Christensen & Nielsen 2011)
  garch    GARCH(1,1) (Bollerslev 1986), Gaussian QMLE, refit yearly, filtered daily
  gjr      GJR-GARCH(1,1) (Glosten, Jagannathan & Runkle 1993): adds the leverage term

Evaluation: QLIKE loss L = RV/F - ln(RV/F) - 1 (Patton 2011, robust to noise in the RV proxy) and Diebold-Mariano
tests against the naive forecast. Training rows for a fit at t only use targets fully realised by t (no overlap leak).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter

from .config import TRADING_DAYS

LN2 = np.log(2.0)


# ---------------------------------------------------------------------------
# estimators (daily variances, not annualised)
# ---------------------------------------------------------------------------

def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff()


def parkinson(df: pd.DataFrame) -> pd.Series:
    return np.log(df["high"] / df["low"]) ** 2 / (4 * LN2)


def garman_klass(df: pd.DataFrame) -> pd.Series:
    return 0.5 * np.log(df["high"] / df["low"]) ** 2 - (2 * LN2 - 1) * np.log(df["close"] / df["open"]) ** 2


def rogers_satchell(df: pd.DataFrame) -> pd.Series:
    h, l, o, c = (np.log(df[x]) for x in ("high", "low", "open", "close"))
    return (h - c) * (h - o) + (l - c) * (l - o)


def range_variance(df: pd.DataFrame) -> pd.Series:
    """Daily variance proxy: overnight^2 + Rogers-Satchell intraday. Falls back to r^2 where OHLC is unusable."""
    ok = (df["high"] >= df["low"]) & (df["low"] > 0) & (df["open"] > 0)
    overnight = np.log(df["open"] / df["close"].shift(1)) ** 2
    v = overnight + rogers_satchell(df).clip(lower=0)
    r2 = log_returns(df["close"]) ** 2
    return v.where(ok & v.notna() & (v > 0), r2)


def yang_zhang(df: pd.DataFrame, n: int) -> pd.Series:
    """Yang-Zhang (2000) n-day variance estimate (daily units)."""
    o = np.log(df["open"] / df["close"].shift(1))
    c = np.log(df["close"] / df["open"])
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    return o.rolling(n).var() + k * c.rolling(n).var() + (1 - k) * rogers_satchell(df).rolling(n).mean()


def future_rv(close: pd.Series, h: int) -> pd.Series:
    """Annualised close-to-close realized variance over (t, t+h]: the target. NaN where the future is unknown."""
    r2 = log_returns(close) ** 2
    return (r2[::-1].rolling(h).sum()[::-1].shift(-1)) * TRADING_DAYS / h


def trailing_rv(close: pd.Series, h: int) -> pd.Series:
    return (log_returns(close) ** 2).rolling(h).sum() * TRADING_DAYS / h


# ---------------------------------------------------------------------------
# GARCH family (variance recursion via a linear filter: fast enough for hundreds of names)
# ---------------------------------------------------------------------------

@dataclass
class GarchFit:
    omega: float
    alpha: float
    beta: float
    gamma: float          # 0 for plain GARCH
    llf: float

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta + 0.5 * self.gamma

    @property
    def uncond(self) -> float:
        return self.omega / max(1 - self.persistence, 1e-6)


def garch_filter(r: np.ndarray, p: GarchFit, init: float | None = None) -> np.ndarray:
    """sigma^2_t for t = 0..n (one longer than r: the last entry is the one-step-ahead forecast)."""
    r2 = r * r
    x = p.alpha * r2 + p.gamma * r2 * (r < 0)
    s0 = init if init is not None else np.var(r)
    y, _ = lfilter([1.0], [1.0, -p.beta], np.concatenate([[0.0], x]) + p.omega, zi=[p.beta * s0])
    return np.maximum(y, 1e-12)


def fit_garch(r: np.ndarray, leverage: bool = False) -> GarchFit:
    r = np.asarray(r, float)
    r = r[np.isfinite(r)] - np.nanmean(r)
    v0 = np.var(r)

    def unpack(th):
        a, b = 1 / (1 + np.exp(-th[1])), 1 / (1 + np.exp(-th[2]))
        g = 1 / (1 + np.exp(-th[3])) if leverage else 0.0
        # keep persistence < 1: scale a, b, g jointly
        tot = a + b + 0.5 * g
        if tot >= 0.999:
            s = 0.999 / tot
            a, b, g = a * s, b * s, g * s
        return GarchFit(np.exp(th[0]), a, b, g, 0.0)

    def nll(th):
        p = unpack(th)
        s2 = garch_filter(r, p, v0)[:-1]
        return 0.5 * np.sum(np.log(s2) + r * r / s2)

    th0 = np.array([np.log(v0 * 0.05), np.log(0.08 / 0.92), np.log(0.88 / 0.12), np.log(0.05 / 0.95)])
    res = minimize(nll, th0 if leverage else th0[:3].tolist() + [0.0], method="Nelder-Mead",
                   options={"maxiter": 2000, "xatol": 1e-5, "fatol": 1e-6})
    p = unpack(res.x)
    p.llf = -float(res.fun)
    return p


def garch_h_forecast(sig2_next: np.ndarray, p: GarchFit, h: int) -> np.ndarray:
    """Average expected daily variance over the next h days, given the one-step variance."""
    phi = p.persistence
    vbar = p.uncond
    geo = h if abs(1 - phi) < 1e-9 else (1 - phi ** h) / (1 - phi)
    return vbar + (sig2_next - vbar) * geo / h


# ---------------------------------------------------------------------------
# walk-forward forecasts
# ---------------------------------------------------------------------------

def _har_design(v: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"d": v, "w": v.rolling(5).mean(), "m": v.rolling(22).mean()})


def forecasts(df: pd.DataFrame, h: int = 21, start: str | None = None, refit_har: int = 21,
              refit_garch: int = 252, min_train: int = 500, iv: pd.Series | None = None,
              models: tuple = ("naive", "ewma", "har", "har_iv", "garch", "gjr")) -> pd.DataFrame:
    """Out-of-sample forecasts (annualised variance) of the next-h-day close-to-close RV, made at each date's close."""
    close = df["close"]
    r = log_returns(close)
    v = range_variance(df)
    out = pd.DataFrame(index=df.index)
    out["target"] = future_rv(close, h)
    if "naive" in models:
        out["naive"] = trailing_rv(close, h)
    if "ewma" in models:
        out["ewma"] = (r ** 2).ewm(alpha=0.06, adjust=False).mean() * TRADING_DAYS

    n = len(df)
    first = max(min_train, 30)
    if start is not None:
        first = max(first, int(np.searchsorted(df.index.values, np.datetime64(pd.Timestamp(start)))))

    X = _har_design(v)
    logX = np.log(X.clip(lower=1e-10))
    ylog = np.log(out["target"].clip(lower=1e-10) / TRADING_DAYS)
    ivlog = np.log((iv.reindex(df.index) ** 2 / TRADING_DAYS).clip(lower=1e-10)) if iv is not None else None

    for name in ("har", "har_iv"):
        if name not in models or (name == "har_iv" and ivlog is None):
            continue
        Z = logX.copy()
        if name == "har_iv":
            Z["iv"] = ivlog
        Zc = np.column_stack([np.ones(n), Z.values])
        pred = np.full(n, np.nan)
        for t0 in range(first, n, refit_har):
            train_end = t0 - h                                   # rows whose h-day target is fully known at t0
            if train_end < 100:
                continue
            rows = np.arange(22, train_end + 1)
            A, y = Zc[rows], ylog.values[rows]
            ok = np.all(np.isfinite(A), axis=1) & np.isfinite(y)
            if ok.sum() < 100:
                continue
            beta, *_ = np.linalg.lstsq(A[ok], y[ok], rcond=None)
            s2 = np.var(y[ok] - A[ok] @ beta)
            t1 = min(t0 + refit_har, n)
            # sanity bound: never outside the range of realised targets in the training window (x1.5 on top);
            # a log-linear fit extrapolates explosively on short or corrupted histories otherwise
            lo, hi = np.exp(y[ok].min()), np.exp(y[ok].max()) * 1.5
            pred[t0:t1] = np.clip(np.exp(Zc[t0:t1] @ beta + 0.5 * s2), lo, hi) * TRADING_DAYS
        out[name] = pred

    for name, lev in (("garch", False), ("gjr", True)):
        if name not in models:
            continue
        pred = np.full(n, np.nan)
        rv_ = r.values
        for t0 in range(first, n, refit_garch):
            lo = max(1, t0 - 2520)
            fit = fit_garch(rv_[lo:t0 + 1], leverage=lev)
            t1 = min(t0 + refit_garch, n)
            seg = rv_[lo:t1]
            seg = np.where(np.isfinite(seg), seg, 0.0) - np.nanmean(rv_[lo:t0 + 1])
            s2 = garch_filter(seg, fit, np.nanvar(rv_[lo:t0 + 1]))     # s2[i] = variance for day lo+i (from info to lo+i-1)
            nxt = s2[1:]                                                # one-step-ahead made at the close of day lo+i
            idx = np.arange(lo, t1)
            sel = idx >= t0
            tr = out["target"].values[max(lo, 0):max(t0 - h, lo + 1)] / TRADING_DAYS
            tr = tr[np.isfinite(tr) & (tr > 0)]
            f = garch_h_forecast(nxt[sel], fit, h)
            if len(tr):
                f = np.clip(f, tr.min(), tr.max() * 1.5)               # same sanity bound as HAR
            pred[idx[sel]] = f * TRADING_DAYS
        out[name] = pred
    return out


def qlike(rv: np.ndarray, f: np.ndarray) -> np.ndarray:
    x = np.asarray(rv, float) / np.asarray(f, float)
    return x - np.log(x) - 1


def evaluate(fc: pd.DataFrame, models: list[str], benchmark: str = "naive", h: int = 21) -> pd.DataFrame:
    """Per model: mean QLIKE, MSE (vol units), DM t-stat against the benchmark (negative = better)."""
    from .stats import diebold_mariano
    d = fc.dropna(subset=["target", benchmark] + models)
    d = d[d["target"] > 0]
    rows = []
    for m in [benchmark] + models:
        q = qlike(d["target"], d[m])
        row = {"model": m, "qlike": float(np.mean(q)), "rmse_vol": float(np.sqrt(np.mean((np.sqrt(d["target"]) - np.sqrt(d[m])) ** 2))),
               "n": len(d)}
        if m != benchmark:
            dm = diebold_mariano(q, qlike(d["target"], d[benchmark]), h=h)
            row["dm_t"], row["dm_p"] = dm["t"], dm["p"]
        rows.append(row)
    return pd.DataFrame(rows)
