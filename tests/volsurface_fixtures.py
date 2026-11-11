"""
tests/volsurface_fixtures.py — a synthetic option chain generated from a KNOWN SVI surface, so every stage of the
pipeline (forward from parity, implied vols, fits, arbitrage checks, scans) can be tested against ground truth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from volsurface import bs
from volsurface.svi import SVI

ASOF = pd.Timestamp("2026-10-26 12:00", tz="America/New_York")
SPOT = 5000.0
RATE, DIV = 0.04, 0.013


def true_params(T: float) -> SVI:
    """Arbitrage-free, calendar-consistent smiles whose total variance grows with T: negative skew, ~16-20% ATM."""
    return SVI(a=0.002 * T, b=0.09 * np.sqrt(T) + 0.01, rho=-0.55, m=0.01, sigma=0.12 * np.sqrt(T) + 0.03)


def forward(T: float) -> float:
    return SPOT * np.exp((RATE - DIV) * T)


def make_quotes(days=(14, 30, 60, 120, 240), root="SPXW", half_spread=0.02, strikes_per_expiry=61, drop_far=False,
                params_fn=true_params, tick=0.05) -> pd.DataFrame:
    rows = []
    for d in days:
        T = d / 365.0
        expiry = (ASOF.tz_localize(None).normalize() + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        F, D = forward(T), float(np.exp(-RATE * T))
        p = params_fn(T)
        ks = np.linspace(-0.45, 0.25, strikes_per_expiry)
        for k in ks:
            K = round(F * np.exp(k) / 5) * 5.0
            iv = float(p.iv(np.log(K / F), T))
            for right, cp in (("C", 1.0), ("P", -1.0)):
                mid = float(bs.price(F, K, T, iv, cp, D))
                spread = max(tick, half_spread * mid * 0.1 + tick)
                bid = max(round((mid - spread / 2) / tick) * tick, 0.0)
                ask = max(round((mid + spread / 2) / tick) * tick, bid + tick)
                rows.append({"expiry": expiry, "root": root, "strike": K, "right": right, "bid": bid, "ask": ask,
                             "volume": 10.0, "open_interest": 100.0})
    q = pd.DataFrame(rows).drop_duplicates(["expiry", "strike", "right"])
    return q


def settle_expiry(d: int) -> str:
    return (ASOF.tz_localize(None).normalize() + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
