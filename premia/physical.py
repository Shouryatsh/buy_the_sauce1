"""
premia/physical.py — the real-world (P) distribution used for fair value, expected edge, assignment probability and
tail loss (phases 3 and 4).

Filtered historical simulation (Barone-Adesi, Engle & Mancini 2008): GJR-GARCH(1,1) is fitted to the underlying's
daily log returns up to the decision date; paths are simulated by re-drawing the *standardised residuals* of that fit,
so the fat tails and the leverage effect are the stock's own, while the variance follows the GARCH recursion.

Two choices keep the model direction-neutral and honest:
  * Drift: paths are rescaled so that E_P[S_T] equals the forward F = S e^{(r-q)T}. No equity premium or other view on
    direction enters any decision (the strategy must not depend on one).
  * Level: the simulated mean realized variance is anchored to the best out-of-sample RV forecast available at the date
    (phase 3 picks which), so the GARCH supplies the *shape* (tails, clustering) and the forecaster supplies the *level*.

With drift = r - q, the expected gain of any delta hedge is zero, so the expected P&L of a hedged and an unhedged option
is the same: premium received minus D * E_P[payoff] (Ahmad & Wilmott 2005). Hedging changes the variance, not the mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import pricing
from .rv import GarchFit, fit_garch, garch_filter


@dataclass
class FHS:
    fit: GarchFit
    z: np.ndarray            # standardised residuals (mean 0, var 1)
    sig2_next: float         # one-step-ahead daily variance at the decision date

    @classmethod
    def from_returns(cls, r: np.ndarray, max_obs: int = 2520) -> "FHS":
        r = np.asarray(r, float)
        r = r[np.isfinite(r)][-max_obs:]
        mu = r.mean()
        fit = fit_garch(r, leverage=True)
        s2 = garch_filter(r - mu, fit, np.var(r))
        z = (r - mu) / np.sqrt(s2[:-1])
        z = (z - z.mean()) / z.std()
        return cls(fit, z, float(s2[-1]))

    def simulate_logret(self, h: int, n: int, rng: np.random.Generator, target_var_daily: Optional[float] = None,
                        sig2_next: Optional[float] = None) -> np.ndarray:
        """(n, h) daily log returns (zero mean before the drift correction)."""
        p = self.fit
        s2 = np.full(n, sig2_next if sig2_next is not None else self.sig2_next)
        out = np.empty((n, h))
        for t in range(h):
            e = self.z[rng.integers(0, len(self.z), n)]
            ret = np.sqrt(s2) * e
            out[:, t] = ret
            s2 = p.omega + (p.alpha + p.gamma * (ret < 0)) * ret * ret + p.beta * s2
        if target_var_daily is not None:                      # anchor the level, keep the shape
            cur = np.mean(out * out)
            if cur > 0:
                out *= np.sqrt(target_var_daily / cur)
        return out


@dataclass
class Distribution:
    S0: float
    T: float
    r: float
    q: float
    ST: np.ndarray           # terminal prices, E[ST] = forward
    paths: Optional[np.ndarray] = None    # (n, h+1) price paths when kept

    @property
    def forward(self) -> float:
        return self.S0 * np.exp((self.r - self.q) * self.T)

    def payoff(self, K, right) -> np.ndarray:
        K = np.atleast_1d(np.asarray(K, float))[:, None]
        call = np.atleast_1d(np.char.upper(np.asarray(right).astype(str)) == "C")
        call = np.broadcast_to(call, K.shape[:1])[:, None]
        return np.where(call, np.maximum(self.ST[None, :] - K, 0), np.maximum(K - self.ST[None, :], 0))

    def fair_price(self, K, right) -> np.ndarray:
        return np.exp(-self.r * self.T) * self.payoff(K, right).mean(axis=1)

    def prob_below(self, K) -> np.ndarray:
        K = np.atleast_1d(np.asarray(K, float))
        return (self.ST[None, :] < K[:, None]).mean(axis=1)

    def fair_iv(self, K, right) -> np.ndarray:
        """The Black-Scholes vol that reproduces the P-fair price ("physical-fair IV")."""
        return pricing.implied_vol(self.fair_price(K, right), self.S0, K, self.T, self.r, self.q, right)

    def short_loss_stats(self, K, right, premium, level: float = 0.95) -> dict:
        """Loss distribution at expiry of one short option (per share): premium kept minus payoff, no hedge."""
        pay = self.payoff(K, right)
        pnl = np.atleast_1d(premium)[:, None] - np.exp(-self.r * self.T) * pay
        q = np.quantile(pnl, 1 - level, axis=1)
        cvar = np.array([row[row <= qq].mean() for row, qq in zip(pnl, q)])
        return {"mean": pnl.mean(axis=1), "var": -q, "cvar": -cvar, "p_loss": (pnl < 0).mean(axis=1)}


def distribution(fhs: FHS, S0: float, days: int, r: float, q: float, n: int = 4000, seed: int = 0,
                 target_var_annual: Optional[float] = None, keep_paths: bool = False,
                 trading_days: int = 252) -> Distribution:
    """Terminal distribution after *days* trading days (T = days/252 years for pricing consistency)."""
    rng = np.random.default_rng(seed)
    T = days / trading_days
    lr = fhs.simulate_logret(days, n, rng, None if target_var_annual is None else target_var_annual / trading_days)
    cum = np.cumsum(lr, axis=1)
    growth = np.exp(cum[:, -1])
    fwd = S0 * np.exp((r - q) * T)
    scale = fwd / (S0 * growth.mean())                         # E[S_T] = forward: no directional drift
    ST = S0 * growth * scale
    paths = None
    if keep_paths:
        tgrid = np.arange(1, days + 1) / days
        paths = np.column_stack([np.full(n, S0), S0 * np.exp(cum) * scale ** tgrid[None, :]])
    return Distribution(S0, T, r, q, ST, paths)


def signal(market_iv_bid: float, market_iv_ask: float, fair_iv: float, min_gap: float = 0.0025) -> str:
    """RICH if even the bid IV is above the physical-fair IV by min_gap (selling has positive edge before costs);
    CHEAP if even the ask IV is below it by min_gap (buying has positive edge); else FAIR."""
    if np.isfinite(market_iv_bid) and market_iv_bid - fair_iv >= min_gap:
        return "RICH"
    if np.isfinite(market_iv_ask) and fair_iv - market_iv_ask >= min_gap:
        return "CHEAP"
    return "FAIR"
