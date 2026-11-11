"""
volsurface/svi.py — Gatheral's raw SVI slice and its fit.

    w(k) = a + b ( rho (k - m) + sqrt( (k - m)^2 + sigma^2 ) )          total variance, k = ln(K/F)

a  vertical level          b  angle between the wings (>= 0)       rho  rotation / skew (|rho| < 1)
m  horizontal translation  sigma  smoothness of the vertex (> 0)

Durrleman's condition: the slice admits no butterfly arbitrage iff the density
    g(k) = (1 - k w'/(2w))^2 - (w'^2 / 4)(1/w + 1/4) + w''/2   is >= 0 for all k (and w > 0).
Roger Lee's moment formula: the wing slopes b(1 + rho) and b(1 - rho) must not exceed 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import least_squares

LEE_MAX_SLOPE = 2.0
G_MARGIN = 0.004          # the soft penalty aims for g >= this, so the fitted density clears zero with room to spare


@dataclass(frozen=True)
class SVI:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def _s(self, k):
        return np.sqrt((np.asarray(k, dtype=float) - self.m) ** 2 + self.sigma ** 2)

    def w(self, k):
        k = np.asarray(k, dtype=float)
        return self.a + self.b * (self.rho * (k - self.m) + self._s(k))

    def dw(self, k):
        k = np.asarray(k, dtype=float)
        return self.b * (self.rho + (k - self.m) / self._s(k))

    def d2w(self, k):
        return self.b * self.sigma ** 2 / self._s(k) ** 3

    def g(self, k):
        """Durrleman's butterfly density; negative where the slice implies a negative risk-neutral density."""
        k = np.asarray(k, dtype=float)
        w, dw, d2w = self.w(k), self.dw(k), self.d2w(k)
        with np.errstate(divide="ignore", invalid="ignore"):
            g = (1.0 - k * dw / (2.0 * w)) ** 2 - dw ** 2 / 4.0 * (1.0 / w + 0.25) + d2w / 2.0
        return np.where(w > 0, g, -np.inf)

    def density(self, k):
        """Risk-neutral density of the log-moneyness k = ln(F_T / F) implied by the slice:
        p(k) = g(k) / sqrt(2 pi w) * exp(-d2^2 / 2), d2 = -k/sqrt(w) - sqrt(w)/2. Negative wherever g(k) < 0."""
        k = np.asarray(k, dtype=float)
        w = np.maximum(self.w(k), 1e-12)
        d2 = -k / np.sqrt(w) - np.sqrt(w) / 2.0
        g = self.g(k)
        return np.where(np.isfinite(g), g, 0.0) / np.sqrt(2.0 * np.pi * w) * np.exp(-0.5 * d2 ** 2)

    def iv(self, k, T):
        return np.sqrt(np.maximum(self.w(k), 0.0) / T)

    @property
    def min_w(self) -> float:
        """Lowest total variance of the slice (at the vertex region)."""
        return float(self.a + self.b * self.sigma * np.sqrt(1.0 - self.rho ** 2))

    @property
    def wing_right(self) -> float:
        return float(self.b * (1.0 + self.rho))

    @property
    def wing_left(self) -> float:
        return float(self.b * (1.0 - self.rho))

    def as_tuple(self) -> tuple:
        return (self.a, self.b, self.rho, self.m, self.sigma)

    @classmethod
    def from_tuple(cls, t) -> "SVI":
        return cls(*(float(x) for x in t))


PARAM_NAMES = ("a", "b", "rho", "m", "sigma")


@dataclass
class FitResult:
    params: SVI
    rmse_iv: float                 # root-mean-square implied-vol error, in vol units (0.01 = one vol point)
    within_band: Optional[float]   # share of points whose fitted IV lies inside the bid-ask IV band
    n: int
    success: bool
    message: str = ""


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------

def _weights(kind: str, iv: np.ndarray, iv_bid: np.ndarray, iv_ask: np.ndarray, vega: Optional[np.ndarray]) -> np.ndarray:
    if kind == "equal":
        return np.ones_like(iv)
    if kind == "vega" and vega is not None:
        v = np.where(np.isfinite(vega), vega, 0.0)
        return v / v.max() if v.max() > 0 else np.ones_like(iv)
    half = 0.5 * (iv_ask - iv_bid)                     # "spread": precision = 1 / half-spread in vol terms
    floor = max(np.nanmedian(half[np.isfinite(half)]) * 0.25, 1e-4) if np.isfinite(half).any() else 1e-3
    half = np.where(np.isfinite(half), np.maximum(half, floor), np.nanmedian(half) if np.isfinite(half).any() else 1e-2)
    w = 1.0 / half
    return w / w.max()


def _starts(k: np.ndarray, w: np.ndarray, n: int) -> list[np.ndarray]:
    span = max(k.max() - k.min(), 0.05)
    k_min_w = float(k[np.argmin(w)])
    out = []
    for m0 in (k_min_w, 0.0):
        for s0 in (0.05, 0.15, 0.4):
            for r0 in (-0.6, -0.2, 0.3):
                a0 = 0.5 * float(w.min())
                b0 = max((float(w.max()) - float(w.min())) / span, 1e-3) / (1.0 + abs(r0)) / 1.5
                out.append(np.array([a0, b0, r0, m0, s0]))
    return out[:max(n, 1)] if n < len(out) else out


def fit_slice(k, iv, T: float, iv_bid=None, iv_ask=None, vega=None, weights: str = "spread", lee: bool = True,
              butterfly_strength: float = 0.0, calendar_prev: Optional[SVI] = None, calendar_strength: float = 0.0,
              warm: Optional[SVI] = None, nstarts: int = 18) -> FitResult:
    """Least-squares raw-SVI fit to one expiry's implied vols (residuals in vol units, weighted).

    Soft penalties, switched on by their strength (0 = off): Lee's wing bound and non-negative variance are
    always enforced when ``lee`` is True; ``butterfly_strength`` punishes g(k) < 0 on a grid;
    ``calendar_strength`` punishes w(k) falling below the previous (shorter) expiry's fitted slice."""
    k, iv = np.asarray(k, dtype=float), np.asarray(iv, dtype=float)
    iv_bid = np.full_like(iv, np.nan) if iv_bid is None else np.asarray(iv_bid, dtype=float)
    iv_ask = np.full_like(iv, np.nan) if iv_ask is None else np.asarray(iv_ask, dtype=float)
    good = np.isfinite(k) & np.isfinite(iv) & (iv > 0)
    k, iv, iv_bid, iv_ask = k[good], iv[good], iv_bid[good], iv_ask[good]
    vega = None if vega is None else np.asarray(vega, dtype=float)[good]
    if len(k) < 5:
        raise ValueError(f"need at least 5 usable quotes to fit SVI, have {len(k)}")
    w = iv ** 2 * T
    wt = _weights(weights, iv, iv_bid, iv_ask, vega)
    kgrid = np.linspace(k.min() - 0.30, k.max() + 0.80, 111)        # penalties also police the extrapolated wings

    lo = np.array([-max(w.max(), 1e-3), 1e-5, -0.999, k.min() - 0.3, 1e-3])
    hi = np.array([max(w.max(), 1e-3), 2.0, 0.999, k.max() + 0.3, 1.5])

    def resid(x):
        p = SVI(*x)
        model_iv = np.sqrt(np.maximum(p.w(k), 1e-12) / T)
        r = [wt * (model_iv - iv)]
        if lee:
            r.append(np.array([5.0 * max(0.0, p.wing_right - LEE_MAX_SLOPE), 5.0 * max(0.0, p.wing_left - LEE_MAX_SLOPE),
                               5.0 * max(0.0, -p.min_w)]))
        if butterfly_strength > 0:
            g = p.g(kgrid)
            r.append(butterfly_strength * np.minimum(np.where(np.isfinite(g), g, -1.0) - G_MARGIN, 0.0))
        if calendar_prev is not None and calendar_strength > 0:
            r.append(calendar_strength * np.maximum(calendar_prev.w(kgrid) - p.w(kgrid), 0.0))
        return np.concatenate(r)

    starts = _starts(k, w, nstarts)
    if warm is not None:
        starts.insert(0, np.array(warm.as_tuple()))
    best = None
    for x0 in starts:
        x0 = np.clip(x0, lo + 1e-9, hi - 1e-9)
        try:
            sol = least_squares(resid, x0, bounds=(lo, hi), method="trf", x_scale=np.maximum(np.abs(x0), 0.05), max_nfev=200)
        except Exception:                                                    # a start can fail numerically; others will do
            continue
        if best is None or sol.cost < best.cost:
            best = sol
    if best is None:
        raise RuntimeError("SVI fit failed from every starting point")
    params = SVI(*best.x)
    fit_iv = params.iv(k, T)
    inside = None
    if np.isfinite(iv_bid).any() and np.isfinite(iv_ask).any():
        m = np.isfinite(iv_bid) & np.isfinite(iv_ask)
        inside = float(np.mean((fit_iv[m] >= iv_bid[m]) & (fit_iv[m] <= iv_ask[m]))) if m.any() else None
    return FitResult(params=params, rmse_iv=float(np.sqrt(np.mean((fit_iv - iv) ** 2))), within_band=inside,
                     n=int(len(k)), success=bool(best.success), message=str(best.message))


def error_stats(params: SVI, k, iv, T: float, iv_bid=None, iv_ask=None) -> tuple[float, Optional[float]]:
    """(rmse in vol units, share inside the bid-ask band) of any parameter set against the quotes."""
    k, iv = np.asarray(k, dtype=float), np.asarray(iv, dtype=float)
    ok = np.isfinite(k) & np.isfinite(iv)
    f = params.iv(k[ok], T)
    rmse = float(np.sqrt(np.mean((f - iv[ok]) ** 2))) if ok.any() else float("nan")
    inside = None
    if iv_bid is not None and iv_ask is not None:
        b, a = np.asarray(iv_bid, dtype=float)[ok], np.asarray(iv_ask, dtype=float)[ok]
        m = np.isfinite(b) & np.isfinite(a)
        inside = float(np.mean((f[m] >= b[m]) & (f[m] <= a[m]))) if m.any() else None
    return rmse, inside
