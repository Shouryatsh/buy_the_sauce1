"""
premia/attribution.py — phase 3B steps 2-6: daily P&L decomposition and the alpha / risk-premium classification.

Second-order Taylor expansion of each option's value over one day, with Greeks taken at the start of the day:

    dV = delta dS + 1/2 gamma dS^2 + vega dsigma + theta dt + vanna dS dsigma + 1/2 volga dsigma^2 + residual

dsigma is the change of the contract's *own* implied vol (at its strike). Hedge trades (stock) are booked under
delta; costs (commissions, spread, borrow) are a separate line. Only the residual can be alpha
(SPEC.md section 3 explains why it cannot be established on model-priced history).

Classification (step 6): a trade type is "alpha" only if its mean daily residual P&L after costs is positive and
significant out of sample: HAC t-stat, block-bootstrap interval excluding zero, and Benjamini-Hochberg across
the trade types tested. Otherwise it is "risk premium": kept, hedged and size-limited.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import pricing, stats

COMPONENTS = ["delta", "gamma", "vega", "theta", "vanna", "volga"]


def attribute_option(S0, S1, sig0, sig1, K, T0, dt, r, q, right, qty: float = 1.0, mult: float = 100.0) -> dict:
    """One day of P&L for *qty* contracts (negative = short), Greeks from Black-Scholes-Merton at the strike's vol."""
    g = pricing.greeks(S0, K, T0, r, q, sig0, right)
    v1 = pricing.price(S1, K, max(T0 - dt, 0.0), r, q, sig1, right)
    dS, ds = S1 - S0, sig1 - sig0
    parts = {
        "delta": g.delta * dS,
        "gamma": 0.5 * g.gamma * dS * dS,
        "vega": g.vega * ds,
        "theta": g.theta * dt,
        "vanna": g.vanna * dS * ds,
        "volga": 0.5 * g.volga * ds * ds,
    }
    total = v1 - g.price
    scale = qty * mult
    out = {k: float(np.sum(v * scale)) for k, v in parts.items()}
    out["total"] = float(np.sum(total * scale))
    out["residual"] = out["total"] - sum(out[c] for c in COMPONENTS)
    return out


def attribute_frame(marks: pd.DataFrame) -> pd.DataFrame:
    """marks: one row per (date, contract) with S, iv, K, T, r, q, right, qty. Returns one row per (date, contract)
    with the P&L components from the previous mark to this one."""
    rows = []
    for cid, g in marks.sort_values("date").groupby("contract"):
        g = g.reset_index(drop=True)
        for i in range(1, len(g)):
            a, b = g.iloc[i - 1].to_dict(), g.iloc[i].to_dict()          # dicts: a Series' .T is its transpose
            dt = (pd.Timestamp(b["date"]) - pd.Timestamp(a["date"])).days / 365.0
            rec = attribute_option(a["S"], b["S"], a["iv"], b["iv"], a["K"], a["T"], dt, a["r"], a["q"], a["right"], a["qty"])
            rec.update({"date": b["date"], "contract": cid, "trade_type": a.get("trade_type", "")})
            rows.append(rec)
    return pd.DataFrame(rows)


@dataclass
class Classification:
    trade_type: str
    label: str              # "alpha" | "risk premium" | "insufficient data"
    mean_residual: float
    t_hac: float
    p: float
    ci: tuple
    n_days: int
    note: str = ""


def classify(daily: pd.DataFrame, value_col: str = "residual_net", group_col: str = "trade_type", min_days: int = 120,
             fdr: float = 0.10, boot_reps: int = 1000) -> list[Classification]:
    """daily: one row per (date, trade_type) with the residual P&L after costs. Out-of-sample rows only."""
    out, ps, idx = [], [], []
    for tt, g in daily.groupby(group_col):
        x = g.sort_values("date")[value_col].to_numpy(float)
        if len(x) < min_days:
            out.append(Classification(tt, "insufficient data", float(np.mean(x)) if len(x) else np.nan, np.nan, np.nan,
                                      (np.nan, np.nan), len(x), f"fewer than {min_days} days"))
            continue
        mt = stats.mean_test(x)
        boots = stats.bootstrap(np.mean, x, reps=boot_reps, mean_block=20)
        ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
        out.append(Classification(tt, "pending", mt["mean"], mt["t"], mt["p"], ci, len(x)))
        ps.append(mt["p"] / 2 if mt["mean"] > 0 else 1 - mt["p"] / 2)       # one-sided: residual > 0
        idx.append(len(out) - 1)
    passed = stats.benjamini_hochberg(np.array(ps), fdr) if ps else np.array([], bool)
    for j, i in enumerate(idx):
        c = out[i]
        alpha = bool(passed[j]) and c.ci[0] > 0 and c.mean_residual > 0
        c.label = "alpha" if alpha else "risk premium"
        c.note = "residual > 0, significant after BH and bootstrap" if alpha else "residual not significantly > 0"
    return out
