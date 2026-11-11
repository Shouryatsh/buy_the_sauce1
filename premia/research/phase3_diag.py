"""
Phase 3 diagnostics, run after the pre-registered FHS calibration test failed (3.7 pts > 3.0).

1. Is the miss the mean (drift) or the shape? Re-score the same FHS forecasts with each path shifted by a
   point-in-time drift estimate (trailing 5-year mean daily log return of that name, known at t). If the shifted
   forecasts are calibrated, the distribution's shape (vol level, tails) is right and only the drift-neutral mean
   (a deliberate choice: no view on direction) misses.
2. Forecast bias: mean forecast vs mean realized variance, per model (the edge model needs an unbiased level).

This is a diagnostic, not a re-test: the phase-3 verdict stays as recorded.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .. import physical, pricing
from ..config import TRADING_DAYS
from .build import symbols_with_iv
from .common import save_json
from .phase3 import _load_u


def _one(sym: str) -> tuple[list[dict], dict]:
    u = _load_u(sym)
    if u is None:
        return [], {}
    df = u.df
    lr = np.log(df["close"]).diff()
    mu5 = lr.rolling(1260, min_periods=500).mean()            # trailing 5-year drift, daily, known at t
    out = []
    h = 21
    for t in df.index[(df.index >= "2010-01-01")][::21]:
        i = df.index.get_loc(t)
        if i + h >= len(df):
            break
        row = df.iloc[i]
        if not np.isfinite(row["iv"]) or not np.isfinite(row.get("fc_har_iv", np.nan)) or not np.isfinite(mu5.iloc[i]):
            continue
        fhs = u.fhs_at(t)
        if fhs is None:
            continue
        S, r, q, iv = float(row["close"]), float(row["r"]), float(row["q"]), float(row["iv"])
        T = h / TRADING_DAYS
        d = physical.distribution(fhs, S, h, r, q, n=2000, seed=i, target_var_annual=float(row["fc_har_iv"]))
        shift = np.exp(mu5.iloc[i] * h - (r - q) * T)
        ST_real = float(df["close"].iloc[i + h])
        for dl in (0.10, 0.25, 0.50):
            K = float(pricing.strike_for_delta(S, T, r, q, iv, -dl, "P"))
            out.append({"delta": dl, "p_fhs": float((d.ST < K).mean()), "p_fhs_drift": float((d.ST * shift < K).mean()),
                        "hit": float(ST_real < K)})
    bias = {}
    v = df[df.index >= "2010-01-01"]
    for m in ("naive", "har", "har_iv", "gjr"):
        x = v[[f"fc_{m}", "rv_target"]].dropna()
        bias[m] = float(np.sqrt(x[f"fc_{m}"]).mean() - np.sqrt(x["rv_target"]).mean())
    x = pd.DataFrame({"iv": v["iv"], "rv": v["rv_target"]}).dropna()
    bias["iv"] = float(x["iv"].mean() - np.sqrt(x["rv"]).mean())
    return out, bias


def ece(df: pd.DataFrame, col: str) -> float:
    bins = pd.cut(df[col], [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0], include_lowest=True)
    g = df.groupby(bins, observed=True).agg(pred=(col, "mean"), real=("hit", "mean"), n=("hit", "size"))
    return float((g["n"] * (g["pred"] - g["real"]).abs()).sum() / g["n"].sum() * 100)


def main() -> dict:
    syms = symbols_with_iv()
    with ProcessPoolExecutor(8) as ex:
        res = list(ex.map(_one, syms))
    rows = pd.DataFrame([r for rr, _ in res for r in rr])
    biases = pd.DataFrame([b for _, b in res if b])
    out = {"n": len(rows), "ece_fhs_pts": ece(rows, "p_fhs"), "ece_fhs_pit_drift_pts": ece(rows, "p_fhs_drift"),
           "by_delta": rows.groupby("delta")[["p_fhs", "p_fhs_drift", "hit"]].mean().reset_index().to_dict(orient="records"),
           "forecast_bias_volpts_mean_over_names": (biases.mean() * 100).to_dict()}
    save_json("phase3_diagnostics", out)
    print(out)
    return out


if __name__ == "__main__":
    main()
