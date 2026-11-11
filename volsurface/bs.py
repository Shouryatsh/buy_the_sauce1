"""
volsurface/bs.py — Black-76 in the forward measure, vectorised.

Everything is expressed through the forward F (so rates and dividends are absorbed in F and the
discount factor D): call = D [F N(d1) - K N(d2)], put = D [K N(-d2) - F N(-d1)],
d1 = (ln(F/K) + sigma^2 T / 2) / (sigma sqrt T), d2 = d1 - sigma sqrt T.
``cp`` is +1 for calls and -1 for puts (arrays broadcast).
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

SQRT_2PI = np.sqrt(2.0 * np.pi)


def _arrays(*xs):
    return np.broadcast_arrays(*[np.asarray(x, dtype=float) for x in xs])


def d1_d2(F, K, T, sigma):
    F, K, T, sigma = _arrays(F, K, T, sigma)
    sd = sigma * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(F / K) + 0.5 * sd ** 2) / sd
    return d1, d1 - sd


def price(F, K, T, sigma, cp, D=1.0):
    F, K, T, sigma, cp, D = _arrays(F, K, T, sigma, cp, D)
    d1, d2 = d1_d2(F, K, T, sigma)
    call = F * ndtr(d1) - K * ndtr(d2)
    put = K * ndtr(-d2) - F * ndtr(-d1)
    return D * np.where(cp > 0, call, put)


def vega(F, K, T, sigma, D=1.0):
    """d price / d sigma (per 1.00 of volatility, not per vol point)."""
    F, K, T, sigma, D = _arrays(F, K, T, sigma, D)
    d1, _ = d1_d2(F, K, T, sigma)
    return D * F * np.exp(-0.5 * d1 ** 2) / SQRT_2PI * np.sqrt(T)


def delta(F, K, T, sigma, cp):
    """Forward delta: N(d1) for a call, N(d1) - 1 for a put."""
    d1, _ = d1_d2(F, K, T, sigma)
    cp = np.asarray(cp, dtype=float)
    return np.where(cp > 0, ndtr(d1), ndtr(d1) - 1.0)


def implied_vol(px, F, K, T, cp, D=1.0, lo=1e-4, hi=8.0, iters=64):
    """Implied volatility by vectorised bisection (monotone in sigma, so it cannot diverge).
    NaN where the price is outside the no-arbitrage band (at or below intrinsic, at or above the
    upper bound), where T <= 0, or where the root falls on the search boundary."""
    px, F, K, T, cp, D = _arrays(px, F, K, T, cp, D)
    p = px / D
    intrinsic = np.maximum(cp * (F - K), 0.0)
    upper = np.where(cp > 0, F, K)
    ok = np.isfinite(p) & np.isfinite(F) & np.isfinite(K) & (T > 0) & (K > 0) & (F > 0) & (p > intrinsic + 1e-12) & (p < upper)
    a = np.full(p.shape, lo)
    b = np.full(p.shape, hi)
    Fs, Ks, Ts = np.where(ok, F, 1.0), np.where(ok, K, 1.0), np.where(ok, T, 1.0)
    pp = np.where(ok, p, 0.5)
    for _ in range(iters):
        m = 0.5 * (a + b)
        high = price(Fs, Ks, Ts, m, cp) > pp
        b = np.where(high, m, b)
        a = np.where(high, a, m)
    iv = 0.5 * (a + b)
    return np.where(ok & (iv > lo * 1.001) & (iv < hi * 0.999), iv, np.nan)
