"""
Phase 3B on real quotes: second-order Greeks from the fitted surface under three smile regimes (step 1) and the
vanna-volga smile against SVI and the market (step 8). The backtest-based steps (2-7) are in research/analysis.py.

    python -m premia.research.phase3b
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from volsurface import model as vmodel
from volsurface import store as vstore
from volsurface import surface as vsurface

from .. import greeks as G
from .common import save_csv, save_json


def _slice(snapshot_id: int, days: float = 30):
    qs = vstore.load_snapshot(snapshot_id)
    surf = vsurface.build_surface(qs.quotes, qs.spot, qs.asof, qs.symbol, qs.source)
    fs = vmodel.fit_surface(surf, vmodel.FitSettings(butterfly_strength=1.0, calendar_strength=1.0, nstarts=10))
    s = min(fs.fitted, key=lambda x: abs(x.T * 365 - days))
    return qs, s


def regime_report(snapshot_id: int = 5) -> pd.DataFrame:
    qs, s = _slice(snapshot_id)
    p, T = s.params, s.T
    F, D = s.slice.F, s.slice.D
    r = -np.log(D) / T
    S = qs.spot or F * D
    q = r - np.log(F / S) / T
    smile = lambda k: np.sqrt(np.maximum(p.w(np.asarray(k)), 1e-10) / T)
    rows = []
    for mny in (0.85, 0.90, 0.95, 0.975, 1.0, 1.025, 1.05):
        K = np.array([F * mny])
        right = "P" if mny <= 1.0 else "C"
        tab = G.regime_table(S, K, T, r, q, right, smile)
        for reg in G.REGIMES:
            g = tab[reg]
            rows.append({"K/F": mny, "right": right, "regime": reg, "iv": float(smile(np.log(mny))),
                         **{k: float(g[k][0]) for k in ("delta", "gamma", "vega", "vanna", "volga")}})
    df = pd.DataFrame(rows)
    df.attrs["meta"] = {"snapshot": snapshot_id, "expiry": s.expiry, "days": round(T * 365), "spot": S}
    return df


def vv_report(snapshot_id: int = 5) -> dict:
    qs, s = _slice(snapshot_id)
    p, T = s.params, s.T
    F, D = s.slice.F, s.slice.D
    r = -np.log(D) / T
    S = qs.spot or F * D
    q = r - np.log(F / S) / T
    smile = lambda k: np.sqrt(np.maximum(p.w(np.asarray(k)), 1e-10) / T)
    piv, vols = G.vv_pivots(S, T, r, q, smile)
    u = s.slice.usable
    K = F * np.exp(u["k"].to_numpy())
    mkt = u["iv"].to_numpy()
    vv = G.vanna_volga_iv(K, piv, vols)
    sv = smile(u["k"].to_numpy())
    inside = (K >= piv[0] * 0.97) & (K <= piv[2] * 1.03)
    rm = lambda a, b, m: float(np.sqrt(np.nanmean((a[m] - b[m]) ** 2)) * 100)
    return {"snapshot": snapshot_id, "expiry": s.expiry, "pivots_K": piv, "pivot_vols": vols, "n_quotes": int(len(K)),
            "rmse_vv_vs_market_volpts_between_pivots": rm(vv, mkt, inside),
            "rmse_svi_vs_market_volpts_between_pivots": rm(sv, mkt, inside),
            "rmse_vv_vs_market_volpts_all": rm(vv, mkt, np.isfinite(mkt)),
            "rmse_svi_vs_market_volpts_all": rm(sv, mkt, np.isfinite(mkt)),
            "note": "vanna-volga is a 3-point interpolation: good between the pivots, poor in the wings; as a mispricing "
                    "signal it cannot be tested without historical chains (kept off)"}


def main() -> dict:
    reg = regime_report()
    save_csv("phase3b_regime_greeks", reg)
    piv = reg.pivot_table(index="K/F", columns="regime", values=["delta", "vanna", "volga"])
    spread = {}
    for g in ("delta", "vanna", "volga"):
        ss = piv[(g, "sticky_strike")]
        spread[g] = {str(k): {"sticky_delta_vs_strike": float(piv.loc[k, (g, "sticky_delta")] - ss[k]),
                              "local_vol_vs_strike": float(piv.loc[k, (g, "local_vol")] - ss[k]),
                              "sticky_strike": float(ss[k])} for k in piv.index}
    vv = vv_report()
    out = {"meta": reg.attrs.get("meta"), "regime_differences": spread, "vanna_volga": vv}
    save_json("phase3b_real", out)
    print(reg.round(4).to_string(index=False))
    print(vv)
    return out


if __name__ == "__main__":
    main()
