"""Synthetic underlyings for the premia tests: GARCH-like price paths with an implied vol that carries a premium."""

from __future__ import annotations

import numpy as np
import pandas as pd

from premia import market
from premia.vol import SmileShape

SKEWED = SmileShape(tuple(np.linspace(-5, 4, 19).round(4)),
                    tuple((1 - 0.12 * np.linspace(-5, 4, 19) + 0.02 * np.linspace(-5, 4, 19) ** 2).clip(0.7, 2.2).round(4)),
                    "test skew")


def panel(n: int = 1400, seed: int = 0, vol: float = 0.20, iv_premium: float = 0.03, start: str = "2015-01-02",
          S0: float = 100.0, drift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    s2 = np.empty(n)
    s2[0] = vol ** 2 / 252
    r = np.empty(n)
    omega, a, b = (vol ** 2 / 252) * 0.05, 0.08, 0.87
    for t in range(n):
        if t:
            s2[t] = omega + a * r[t - 1] ** 2 + b * s2[t - 1]
        r[t] = np.sqrt(s2[t]) * rng.standard_t(6) / np.sqrt(1.5) + drift / 252
    close = S0 * np.exp(np.cumsum(r))
    iv = np.sqrt(s2 * 252) + iv_premium
    df = pd.DataFrame({"close": close, "open": close * np.exp(rng.normal(0, 0.002, n)),
                       "iv": iv, "r": 0.02, "q": 0.01, "div_cash": 0.0, "volume": 2e6}, index=idx)
    df["high"] = df[["open", "close"]].max(axis=1) * 1.004
    df["low"] = df[["open", "close"]].min(axis=1) * 0.996
    df["adj_close"] = df["close"]
    df["iv_age"] = 0
    return df


def underlying(symbol: str = "TST", seed: int = 0, shape: SmileShape = SKEWED, **kw) -> market.Underlying:
    return market.prepare(symbol, panel(seed=seed, **kw), [], shape, min_train=500)
