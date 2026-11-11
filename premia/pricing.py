"""
premia/pricing.py — Black-Scholes-Merton with a continuous dividend yield (equivalently Black-76 on the forward,
Black 1976), vectorised, with first- and second-order Greeks in closed form.

Conventions: sigma and rates are annual decimals, T in years (ACT/365), vega/vanna/volga per 1.00 of vol
(divide by 100 for "per vol point"), theta per year (divide by 365 for per calendar day). `right` is "C" or "P"
(arrays of either are accepted).

  d1 = (ln(S/K) + (r - q + sigma^2/2) T) / (sigma sqrt T),  d2 = d1 - sigma sqrt T
  vanna = d delta / d sigma = -e^{-qT} phi(d1) d2 / sigma          (Taleb's "DdeltaDvol")
  volga = d vega / d sigma  = vega d1 d2 / sigma                   (Taleb's "DvegaDvol", a.k.a. vomma)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

SQRT2PI = np.sqrt(2.0 * np.pi)


def _phi(x):
    return np.exp(-0.5 * x * x) / SQRT2PI


def _arr(*xs):
    return [np.asarray(x, dtype=float) for x in xs]


def _is_call(right) -> np.ndarray:
    r = np.asarray(right)
    return np.char.upper(r.astype(str)) == "C"


def d1d2(S, K, T, r, q, sigma):
    S, K, T, r, q, sigma = _arr(S, K, T, r, q, sigma)
    vt = sigma * np.sqrt(np.maximum(T, 1e-12))
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vt
    return d1, d1 - vt


def price(S, K, T, r, q, sigma, right):
    S, K, T, r, q, sigma = _arr(S, K, T, r, q, sigma)
    call = _is_call(right)
    d1, d2 = d1d2(S, K, T, r, q, sigma)
    dq, dr = np.exp(-q * T), np.exp(-r * T)
    c = S * dq * ndtr(d1) - K * dr * ndtr(d2)
    p = K * dr * ndtr(-d2) - S * dq * ndtr(-d1)
    out = np.where(call, c, p)
    intrinsic = np.where(call, np.maximum(S * dq - K * dr, 0.0), np.maximum(K * dr - S * dq, 0.0))
    return np.where(T <= 0, np.where(call, np.maximum(S - K, 0.0), np.maximum(K - S, 0.0)),
                    np.where(sigma <= 0, intrinsic, out))


@dataclass
class Greeks:
    price: np.ndarray
    delta: np.ndarray
    gamma: np.ndarray
    vega: np.ndarray
    theta: np.ndarray
    vanna: np.ndarray
    volga: np.ndarray


def greeks(S, K, T, r, q, sigma, right) -> Greeks:
    S, K, T, r, q, sigma = _arr(S, K, T, r, q, sigma)
    T = np.maximum(T, 1e-8)
    sigma = np.maximum(sigma, 1e-6)
    call = _is_call(right)
    d1, d2 = d1d2(S, K, T, r, q, sigma)
    dq, dr = np.exp(-q * T), np.exp(-r * T)
    pdf = _phi(d1)
    sq = np.sqrt(T)
    px = price(S, K, T, r, q, sigma, right)
    delta = np.where(call, dq * ndtr(d1), -dq * ndtr(-d1))
    gamma = dq * pdf / (S * sigma * sq)
    vega = S * dq * pdf * sq
    common = -S * dq * pdf * sigma / (2 * sq)
    theta = np.where(call, common - r * K * dr * ndtr(d2) + q * S * dq * ndtr(d1),
                     common + r * K * dr * ndtr(-d2) - q * S * dq * ndtr(-d1))
    vanna = -dq * pdf * d2 / sigma
    volga = vega * d1 * d2 / sigma
    return Greeks(px, delta, gamma, vega, theta, vanna, volga)


def implied_vol(px, S, K, T, r, q, right, lo=1e-4, hi=5.0, iters=100):
    """Vectorised bisection (cannot diverge); NaN outside the no-arbitrage band."""
    px, S, K, T, r, q = _arr(px, S, K, T, r, q)
    shape = np.broadcast(px, S, K, T, r, q).shape
    a, b = np.full(shape, lo), np.full(shape, hi)
    pa = price(S, K, T, r, q, a, right)
    pb = price(S, K, T, r, q, b, right)
    ok = (px >= pa - 1e-10) & (px <= pb + 1e-10)
    for _ in range(iters):
        m = 0.5 * (a + b)
        pm = price(S, K, T, r, q, m, right)
        hi_side = pm > px
        b = np.where(hi_side, m, b)
        a = np.where(hi_side, a, m)
    return np.where(ok, 0.5 * (a + b), np.nan)


def strike_for_delta(S, T, r, q, sigma, delta, right):
    """Strike whose BSM delta equals *delta* (puts: negative delta), at volatility *sigma*."""
    S, T, r, q, sigma, delta = _arr(S, T, r, q, sigma, delta)
    from scipy.special import ndtri
    call = _is_call(right)
    x = np.where(call, delta * np.exp(q * T), 1.0 + delta * np.exp(q * T))      # N(d1)
    d1 = ndtri(np.clip(x, 1e-12, 1 - 1e-12))
    return S * np.exp(-(d1 * sigma * np.sqrt(T)) + (r - q + 0.5 * sigma * sigma) * T)


def prob_itm(S, K, T, mu, q, sigma, right):
    """Probability of finishing in the money under a lognormal with drift *mu* (r for risk-neutral)."""
    _, d2 = d1d2(S, K, T, mu, q, sigma)
    return np.where(_is_call(right), ndtr(d2), ndtr(-d2))
