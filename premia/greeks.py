"""
premia/greeks.py — phase 3B steps 1 and 8: Greeks from the fitted surface (not broker Greeks) under three smile
regimes, and the vanna-volga price.

Smile regimes (Derman 1999, "Regimes of Volatility"; Derman & Miller 2016). After a spot move that takes the forward
from F to F' (x = ln F'/F), the implied vol at a fixed strike K with old log-moneyness k0 = ln(K/F) becomes

    sigma_new(K) = sigma_old(k0 + c x) + dv        c =  0  sticky-strike
                                                   c = -1  sticky-delta (sticky-moneyness)
                                                   c = +1  sticky-implied-tree (local-vol dynamics, to first order)

dv is a parallel shift of the whole smile (the "vol" in vega, vanna, volga). All Greeks are central finite
differences of the full revaluation V(S', dv) = BSM(S', K, T, r, q, sigma_new(K)), so they include the smile's
response to spot. Under sticky-strike they equal the closed-form BSM Greeks at the strike's own vol (tested).

Vanna-volga (Castagna & Mercurio 2007; Castagna 2010; Wystup 2006): the first-order approximation
    sigma_VV(K) ~ y1(K) sigma(K1) + y2(K) sigma(K2) + y3(K) sigma(K3),
    y1 = ln(K2/K) ln(K3/K) / (ln(K2/K1) ln(K3/K1)),  y2 = ln(K/K1) ln(K3/K) / (ln(K2/K1) ln(K3/K2)),
    y3 = ln(K/K1) ln(K/K2) / (ln(K3/K1) ln(K3/K2)),
with pivots at the 25-delta put, ATM and 25-delta call. It reprices the pivots exactly and is an FX-market convention:
here only an optional mispricing signal (SPEC.md conflict 8).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from . import pricing

REGIMES = {"sticky_strike": 0.0, "sticky_delta": -1.0, "local_vol": 1.0}


def surface_greeks(S: float, K, T: float, r: float, q: float, right, smile: Callable, regime: str = "sticky_strike",
                   h_s: float = 0.005, h_v: float = 0.005) -> dict:
    """Greeks of options on one expiry from the smile function sigma(k) (k = ln(K/F)).

    Returns per-share delta, gamma, vega, vanna, volga (vega-family per 1.00 vol), and price."""
    c = REGIMES[regime]
    K = np.atleast_1d(np.asarray(K, float))
    F = S * np.exp((r - q) * T)
    k0 = np.log(K / F)

    def V(ds: float, dv: float):
        x = np.log1p(ds)
        sig = np.maximum(smile(k0 + c * x) + dv, 1e-4)
        return pricing.price(S * (1 + ds), K, T, r, q, sig, right)

    v00 = V(0, 0)
    vp0, vm0 = V(h_s, 0), V(-h_s, 0)
    v0p, v0m = V(0, h_v), V(0, -h_v)
    vpp, vpm, vmp, vmm = V(h_s, h_v), V(h_s, -h_v), V(-h_s, h_v), V(-h_s, -h_v)
    dS = h_s * S
    return {
        "price": v00,
        "delta": (vp0 - vm0) / (2 * dS),
        "gamma": (vp0 - 2 * v00 + vm0) / (dS * dS),
        "vega": (v0p - v0m) / (2 * h_v),
        "vanna": (vpp - vpm - vmp + vmm) / (4 * dS * h_v),
        "volga": (v0p - 2 * v00 + v0m) / (h_v * h_v),
    }


def regime_table(S, K, T, r, q, right, smile) -> dict:
    """The same Greeks under every regime, plus the spread between them (sensitivity to the regime choice)."""
    out = {name: surface_greeks(S, K, T, r, q, right, smile, name) for name in REGIMES}
    spread = {g: np.max([out[n][g] for n in REGIMES], axis=0) - np.min([out[n][g] for n in REGIMES], axis=0)
              for g in ("delta", "gamma", "vega", "vanna", "volga")}
    out["spread"] = spread
    return out


def vv_weights(K, K1: float, K2: float, K3: float):
    K = np.asarray(K, float)
    y1 = np.log(K2 / K) * np.log(K3 / K) / (np.log(K2 / K1) * np.log(K3 / K1))
    y2 = np.log(K / K1) * np.log(K3 / K) / (np.log(K2 / K1) * np.log(K3 / K2))
    y3 = np.log(K / K1) * np.log(K / K2) / (np.log(K3 / K1) * np.log(K3 / K2))
    return y1, y2, y3


def vanna_volga_iv(K, pivots: tuple, vols: tuple):
    """First-order vanna-volga implied vol at strikes K from three pivot (strike, vol) pairs."""
    (K1, K2, K3), (s1, s2, s3) = pivots, vols
    y1, y2, y3 = vv_weights(K, K1, K2, K3)
    return y1 * s1 + y2 * s2 + y3 * s3


def vv_pivots(S: float, T: float, r: float, q: float, smile: Callable) -> tuple[tuple, tuple]:
    """25-delta put, ATM-forward and 25-delta call strikes, each solved at its own smile vol (fixed point)."""
    F = S * np.exp((r - q) * T)
    out_k, out_v = [], []
    for delta, right in ((-0.25, "P"), (None, "C"), (0.25, "C")):
        if delta is None:
            out_k.append(F)
            out_v.append(float(smile(np.array([0.0]))[0]))
            continue
        sig = float(smile(np.array([0.0]))[0])
        for _ in range(30):
            K = float(pricing.strike_for_delta(S, T, r, q, sig, delta, right))
            sig_new = float(smile(np.array([np.log(K / F)]))[0])
            if abs(sig_new - sig) < 1e-7:
                break
            sig = sig_new
        out_k.append(K)
        out_v.append(sig)
    return tuple(out_k), tuple(out_v)
