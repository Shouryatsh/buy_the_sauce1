"""Pure option math (no I/O): Black-Scholes price/Greeks, implied vol, IV rank, moves."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

from scipy.optimize import brentq
from scipy.stats import norm

Right = Literal["C", "P"]

MIN_IV_HISTORY_DAYS = 30


@dataclass(frozen=True)
class Greeks:
    price: float
    delta: float
    gamma: float
    theta: float  # per calendar day, per share
    vega: float  # per 1 vol point (0.01), per share


def _intrinsic(S: float, K: float, right: Right) -> float:
    return max(0.0, S - K) if right == "C" else max(0.0, K - S)


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float, right: Right) -> Greeks:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        intrinsic = _intrinsic(S, K, right)
        if right == "C":
            delta = 1.0 if S > K else 0.0
        else:
            delta = -1.0 if S < K else 0.0
        return Greeks(intrinsic, delta, 0.0, 0.0, 0.0)
    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    disc = math.exp(-r * T)
    pdf = norm.pdf(d1)
    if right == "C":
        price = S * norm.cdf(d1) - K * disc * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = -S * pdf * sigma / (2 * sqrt_t) - r * K * disc * norm.cdf(d2)
    else:
        price = K * disc * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1.0
        theta = -S * pdf * sigma / (2 * sqrt_t) + r * K * disc * norm.cdf(-d2)
    gamma = pdf / (S * sigma * sqrt_t)
    vega = S * pdf * sqrt_t / 100.0
    return Greeks(price, delta, gamma, theta / 365.0, vega)


def bs_price(S: float, K: float, T: float, r: float, sigma: float, right: Right) -> float:
    return bs_greeks(S, K, T, r, sigma, right).price


def implied_vol(price: float, S: float, K: float, T: float, r: float, right: Right) -> Optional[float]:
    """Solve for sigma; None if the price is outside no-arbitrage bounds."""
    if price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return None
    if price < _intrinsic(S, K * math.exp(-r * T), right) - 1e-9:
        return None
    lo, hi = 1e-4, 5.0
    f = lambda s: bs_price(S, K, T, r, s, right) - price  # noqa: E731
    if f(lo) > 0 or f(hi) < 0:
        return None
    return float(brentq(f, lo, hi, xtol=1e-8))


def iv_rank(current: float, history: Sequence[float], min_days: int = MIN_IV_HISTORY_DAYS) -> Optional[float]:
    """0-100: where `current` sits between the min and max of `history`."""
    if len(history) < min_days:
        return None
    lo, hi = min(history), max(history)
    if hi - lo < 1e-12:
        return 50.0
    return max(0.0, min(100.0, (current - lo) / (hi - lo) * 100.0))


def iv_percentile(current: float, history: Sequence[float], min_days: int = MIN_IV_HISTORY_DAYS) -> Optional[float]:
    """0-100: percent of history days with IV below `current`."""
    if len(history) < min_days:
        return None
    return 100.0 * sum(1 for v in history if v < current) / len(history)


def sigma_move(S: float, sigma: float, T: float) -> float:
    """1-standard-deviation move in dollars."""
    return S * sigma * math.sqrt(max(T, 0.0))


def expected_abs_move(S: float, sigma: float, T: float) -> float:
    """Mean absolute move implied by `sigma`; an ATM straddle prices ~this."""
    return sigma_move(S, sigma, T) * math.sqrt(2.0 / math.pi)
