"""
premia/stats.py — the statistics used to decide whether anything is real.

  * stationary bootstrap (Politis & Romano 1994): resamples blocks of geometric length, keeping serial dependence
  * Newey-West HAC standard errors (Newey & West 1987) for regressions on overlapping / autocorrelated P&L
  * OLS with HAC t-stats and block-bootstrap confidence intervals
  * Sharpe ratio, probabilistic and deflated Sharpe ratio (Bailey & Lopez de Prado 2014)
  * Benjamini-Hochberg false-discovery control (1995)
  * Diebold-Mariano test of equal forecast accuracy (1995) with HAC variance
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy import stats as sps

EULER = 0.5772156649015329


def stationary_bootstrap_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    p = 1.0 / max(mean_block, 1.0)
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(n)
    for t in range(1, n):
        idx[t] = rng.integers(n) if rng.random() < p else (idx[t - 1] + 1) % n
    return idx


def bootstrap(stat: Callable[[np.ndarray], float], x: np.ndarray, reps: int = 2000, mean_block: float = 20.0,
              seed: int = 0) -> np.ndarray:
    """Distribution of stat(x*) over stationary-bootstrap resamples of the rows of x."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    return np.array([stat(x[stationary_bootstrap_indices(len(x), mean_block, rng)]) for _ in range(reps)])


def newey_west_var(u: np.ndarray, lags: Optional[int] = None) -> float:
    """Long-run variance of a (demeaned) series with Bartlett weights."""
    u = np.asarray(u, float) - np.mean(u)
    n = len(u)
    L = lags if lags is not None else int(np.floor(4 * (n / 100.0) ** (2 / 9)))
    v = u @ u / n
    for l in range(1, L + 1):
        v += 2 * (1 - l / (L + 1)) * (u[l:] @ u[:-l]) / n
    return float(v)


def mean_test(x: np.ndarray, lags: Optional[int] = None) -> dict:
    """HAC t-test that the mean of x is zero."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    se = np.sqrt(newey_west_var(x, lags) / n) if n > 2 else np.nan
    t = float(np.mean(x) / se) if se and se > 0 else np.nan
    return {"mean": float(np.mean(x)) if n else np.nan, "se": float(se), "t": t, "n": n,
            "p": float(2 * sps.norm.sf(abs(t))) if np.isfinite(t) else np.nan}


@dataclass
class OLSResult:
    coef: np.ndarray
    se_hac: np.ndarray
    t: np.ndarray
    p: np.ndarray
    r2: float
    n: int
    names: list
    ci_boot: Optional[np.ndarray] = None       # (k, 2) percentile intervals

    def table(self) -> list[dict]:
        rows = []
        for i, nm in enumerate(self.names):
            row = {"term": nm, "coef": float(self.coef[i]), "se_hac": float(self.se_hac[i]), "t": float(self.t[i]),
                   "p": float(self.p[i])}
            if self.ci_boot is not None:
                row["ci_lo"], row["ci_hi"] = float(self.ci_boot[i, 0]), float(self.ci_boot[i, 1])
            rows.append(row)
        return rows


def ols_hac(y: np.ndarray, X: np.ndarray, names: list, lags: Optional[int] = None, boot_reps: int = 0,
            mean_block: float = 20.0, seed: int = 0, add_const: bool = True) -> OLSResult:
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    if add_const:
        X = np.column_stack([np.ones(len(y)), X])
        names = ["const"] + list(names)
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y, X = y[ok], X[ok]
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    L = lags if lags is not None else int(np.floor(4 * (n / 100.0) ** (2 / 9)))
    Xe = X * e[:, None]
    S = Xe.T @ Xe / n
    for l in range(1, L + 1):
        G = Xe[l:].T @ Xe[:-l] / n
        S += (1 - l / (L + 1)) * (G + G.T)
    V = n * XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(V), 0))
    t = b / np.where(se > 0, se, np.nan)
    p = 2 * sps.norm.sf(np.abs(t))
    r2 = 1 - (e @ e) / max(((y - y.mean()) ** 2).sum(), 1e-300)
    ci = None
    if boot_reps:
        data = np.column_stack([y, X])

        def coef(d):
            return np.linalg.lstsq(d[:, 1:], d[:, 0], rcond=None)[0]
        rng = np.random.default_rng(seed)
        draws = np.array([coef(data[stationary_bootstrap_indices(n, mean_block, rng)]) for _ in range(boot_reps)])
        ci = np.percentile(draws, [2.5, 97.5], axis=0).T
    return OLSResult(b, se, t, p, float(r2), n, list(names), ci)


def sharpe(r: np.ndarray, periods: int = 252) -> float:
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    sd = r.std(ddof=1) if len(r) > 1 else np.nan
    return float(np.sqrt(periods) * r.mean() / sd) if sd and sd > 0 else np.nan


def sortino(r: np.ndarray, periods: int = 252) -> float:
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    dd = np.sqrt(np.mean(np.minimum(r, 0) ** 2)) if len(r) else np.nan
    return float(np.sqrt(periods) * r.mean() / dd) if dd and dd > 0 else np.nan


def probabilistic_sharpe(r: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """P(true per-period Sharpe > sr_benchmark) given skew and kurtosis of r (Bailey & Lopez de Prado 2012)."""
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 10 or r.std(ddof=1) == 0:
        return np.nan
    sr = r.mean() / r.std(ddof=1)
    g3 = sps.skew(r)
    g4 = sps.kurtosis(r, fisher=False)
    denom = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr * sr, 1e-12))
    return float(sps.norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / denom))


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """Expected maximum of n_trials per-period Sharpe estimates under the null (false strategy theorem)."""
    if n_trials <= 1:
        return 0.0
    z1 = sps.norm.ppf(1 - 1.0 / n_trials)
    z2 = sps.norm.ppf(1 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(var_sr) * ((1 - EULER) * z1 + EULER * z2))


def deflated_sharpe(r: np.ndarray, n_trials: int, var_sr_trials: float) -> float:
    """Deflated Sharpe ratio: PSR against the Sharpe the best of *n_trials* would show by luck alone."""
    return probabilistic_sharpe(r, expected_max_sharpe(n_trials, var_sr_trials))


def benjamini_hochberg(p: np.ndarray, q: float = 0.10) -> np.ndarray:
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    thresh = q * (np.arange(1, n + 1) / n)
    passed = p[order] <= thresh
    k = np.max(np.nonzero(passed)[0]) + 1 if passed.any() else 0
    out = np.zeros(n, bool)
    out[order[:k]] = True
    return out


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1) -> dict:
    """DM test of E[loss_a - loss_b] = 0. Negative t: model a is more accurate. HAC with h-1 lags (overlap)."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[np.isfinite(d)]
    return mean_test(d, lags=max(h - 1, int(np.floor(4 * (len(d) / 100.0) ** (2 / 9)))))
