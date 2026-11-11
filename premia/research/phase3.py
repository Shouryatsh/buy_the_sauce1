"""
Phase 3 runner: SVI fit errors and static arbitrage on real chains, local-vol repricing, realized-vol forecast
evaluation, the volatility risk premium, FHS calibration, and today's per-option fair value / edge / signal table.

    python -m premia.research.phase3
"""

from __future__ import annotations

import json
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from volsurface import arbitrage as ar
from volsurface import model as vmodel
from volsurface import store as vstore
from volsurface import surface as vsurface

from .. import market, physical, pricing, rv, stats, store
from ..config import TRADING_DAYS
from ..vol import Slices, validate_local_vol
from .build import SHAPES_FILE, symbols_with_iv
from .common import save_csv, save_json, verdict


# ---------------------------------------------------------------------------- SVI on real chains
def svi_report() -> pd.DataFrame:
    rows = []
    for sid in vstore.list_snapshots("SPX")["id"]:
        qs = vstore.load_snapshot(int(sid))
        surf = vsurface.build_surface(qs.quotes, qs.spot, qs.asof, qs.symbol, qs.source)
        fs = vmodel.fit_surface(surf, vmodel.FitSettings(butterfly_strength=1.0, calendar_strength=1.0, nstarts=10))
        ranges = {s.expiry: vmodel.data_range(s.slice) for s in fs.fitted}
        cal = {c.label_long: c.ok_in_data for c in ar.calendar(fs.labelled(), ranges)}
        for s in fs.fitted:
            bf = ar.butterfly(s.params, ranges[s.expiry])
            rows.append({"underlying": "SPX", "snapshot": int(sid), "source": qs.source, "asof": str(qs.asof)[:16],
                         "expiry": s.expiry, "days": round(s.T * 365), "n_quotes": len(s.slice.usable),
                         "rmse_volpts": s.fit.rmse_iv * 100, "within_band": s.fit.within_band,
                         "butterfly_ok_in_data": bf.ok_in_data, "calendar_ok_in_data": cal.get(s.expiry, True)})
    d = json.loads(SHAPES_FILE.read_text())
    for f in d["fits"]:
        rows.append({"underlying": f["symbol"], "snapshot": None, "source": f["source"], "asof": d["asof"][:16],
                     "expiry": f["expiry"], "days": round(f["T"] * 365), "n_quotes": f["n_quotes"],
                     "rmse_volpts": f["rmse_volpts"], "within_band": f["inside_band"],
                     "butterfly_ok_in_data": f["butterfly_ok_in_data"],
                     "calendar_ok_in_data": True if f["calendar_ok_in_data"] is None else f["calendar_ok_in_data"]})
    return pd.DataFrame(rows)


def local_vol_report() -> pd.DataFrame:
    """PDE repricing at the quoted strikes with 0.8 <= K/F <= 1.2, for the latest IBKR and Yahoo SPX snapshots.
    Slices whose local variance is undefined (calendar arbitrage with a neighbour) are reported, not repriced."""
    snaps = vstore.list_snapshots("SPX")
    picks = [int(snaps[snaps["source"].str.startswith("IBKR")]["id"].iloc[0]), int(snaps["id"].iloc[0])]
    rows = []
    for sid in picks:
        qs = vstore.load_snapshot(sid)
        surf = vsurface.build_surface(qs.quotes, qs.spot, qs.asof, qs.symbol, qs.source)
        fs = vmodel.fit_surface(surf, vmodel.FitSettings(butterfly_strength=1.0, calendar_strength=1.0, nstarts=10))
        ranges = {s.expiry: vmodel.data_range(s.slice) for s in fs.fitted}
        cal_bad = {c.label_long for c in ar.calendar(fs.labelled(), ranges) if not c.ok}
        sl = Slices.from_fitted(fs)
        pts = {}
        for s in fs.fitted:
            k = s.slice.usable["k"].to_numpy()
            pts[s.T] = k[(np.exp(k) >= 0.8) & (np.exp(k) <= 1.2)]
        res = validate_local_vol(sl, points=pts)
        exp_of = {s.T: s.expiry for s in fs.fitted}
        for r in res:
            e = exp_of[r["T"]]
            r.update({"snapshot": sid, "source": qs.source, "expiry": e, "calendar_arbitrage_before_this_slice": e in cal_bad})
            rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- RV forecasts and VRP
def _load_u(sym: str):
    p = market.CACHE_DIR / f"u_{sym}.pkl"
    return pickle.loads(p.read_bytes()) if p.exists() else None


def _rv_one(sym: str) -> dict | None:
    u = _load_u(sym)
    if u is None:
        return None
    df = u.df
    fc = rv.forecasts(df, h=21, iv=df["iv"], min_train=750)
    fc = fc[fc.index >= "2010-01-01"]
    fc["iv2"] = df["iv"].reindex(fc.index) ** 2
    models = ["ewma", "har", "har_iv", "garch", "gjr", "iv2"]
    ev = rv.evaluate(fc, models)
    d = fc.dropna(subset=["target", "naive"] + models)
    d = d[d["target"] > 0]
    diff = {m: (rv.qlike(d["target"], d[m]) - rv.qlike(d["target"], d["naive"])) for m in models}
    out = {"symbol": sym, "n": len(d), **{f"qlike_{r.model}": r.qlike for r in ev.itertuples()},
           **{f"dm_t_{r.model}": getattr(r, "dm_t", np.nan) for r in ev.itertuples()}}
    out["diff"] = pd.DataFrame(diff, index=d.index)
    # VRP: IV^2 at t minus realized variance over the next 21 trading days (~30 calendar days)
    v = pd.DataFrame({"iv": df["iv"], "rv": np.sqrt(df["rv_target"])}).dropna()
    v = v[v.index >= "2010-01-01"]
    out["vrp_vol_mean"] = float((v["iv"] - v["rv"]).mean())
    out["vrp_var_mean"] = float((v["iv"] ** 2 - v["rv"] ** 2).mean())
    out["vrp_share_pos"] = float((v["iv"] > v["rv"]).mean())
    out["vrp_by_year"] = (v["iv"] - v["rv"]).groupby(v.index.year).mean().to_dict()
    return out


def rv_vrp_report(symbols: list[str]) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    with ProcessPoolExecutor(8) as ex:
        res = [r for r in ex.map(_rv_one, symbols) if r is not None]
    per = pd.DataFrame([{k: v for k, v in r.items() if k not in ("diff", "vrp_by_year")} for r in res])
    models = ["ewma", "har", "har_iv", "garch", "gjr", "iv2"]
    pooled = {}
    for m in models:
        panel = pd.concat([r["diff"][m].rename(r["symbol"]) for r in res], axis=1)
        x = panel.mean(axis=1).dropna().to_numpy()                       # cross-sectional mean loss differential per day
        mt = stats.mean_test(x, lags=21)
        pooled[m] = {"mean_qlike_diff": mt["mean"], "t_hac": mt["t"], "p": mt["p"],
                     "share_names_better": float((per[f"qlike_{m}"] < per["qlike_naive"]).mean())}
    years = pd.DataFrame({r["symbol"]: r["vrp_by_year"] for r in res}).T
    return per, pooled, years


def vrp_by_tenor_spx() -> pd.DataFrame:
    """SPX: implied vol by tenor (VIX9D, VIX, VIX3M, VIX6M) against the realized vol of the S&P 500 over that tenor."""
    spx = store.load_series("^GSPC")
    r2 = np.log(spx).diff() ** 2
    rows = []
    for name, days, td in (("^VIX9D", 9, 6), ("^VIX", 30, 21), ("^VIX3M", 93, 63), ("^VIX6M", 183, 126)):
        iv = store.load_series(name) / 100
        fut = (r2[::-1].rolling(td).sum()[::-1].shift(-1) * TRADING_DAYS / td).pipe(np.sqrt)
        d = pd.DataFrame({"iv": iv, "rv": fut}).dropna()
        d = d[d.index >= "2010-01-01"]
        rows.append({"tenor": name, "days": days, "n": len(d), "iv_mean": d["iv"].mean(), "rv_mean": d["rv"].mean(),
                     "vrp_volpts": (d["iv"] - d["rv"]).mean() * 100, "share_iv_above_rv": (d["iv"] > d["rv"]).mean()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- FHS calibration
def _calib_one(sym: str) -> list[dict]:
    u = _load_u(sym)
    if u is None:
        return []
    df = u.df
    out = []
    dates = df.index[(df.index >= "2010-01-01")][::21]
    h = 21
    for t in dates:
        i = df.index.get_loc(t)
        if i + h >= len(df):
            break
        row = df.iloc[i]
        if not np.isfinite(row["iv"]) or not np.isfinite(row.get("fc_har_iv", np.nan)):
            continue
        fhs = u.fhs_at(t)
        if fhs is None:
            continue
        S, r, q, iv = float(row["close"]), float(row["r"]), float(row["q"]), float(row["iv"])
        T = h / TRADING_DAYS
        d = physical.distribution(fhs, S, h, r, q, n=2000, seed=i, target_var_annual=float(row["fc_har_iv"]))
        ST_real = float(df["close"].iloc[i + h])
        for dl in (0.10, 0.25, 0.50):
            K = float(pricing.strike_for_delta(S, T, r, q, iv, -dl, "P"))
            p_fhs = float(d.prob_below([K])[0])
            p_bs = float(1 - pricing.prob_itm(S, K, T, r, q, iv, "C"))           # risk-neutral N(-d2) at IV
            out.append({"symbol": sym, "date": t, "delta": dl, "p_fhs": p_fhs, "p_bs_iv": p_bs, "hit": float(ST_real < K)})
    return out


def calibration(symbols: list[str]) -> tuple[pd.DataFrame, dict]:
    with ProcessPoolExecutor(8) as ex:
        rows = [r for chunk in ex.map(_calib_one, symbols) for r in chunk]
    df = pd.DataFrame(rows)
    res = {}
    for col in ("p_fhs", "p_bs_iv"):
        bins = pd.cut(df[col], [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0], include_lowest=True)
        g = df.groupby(bins, observed=True).agg(pred=(col, "mean"), real=("hit", "mean"), n=("hit", "size"))
        ece = float((g["n"] * (g["pred"] - g["real"]).abs()).sum() / g["n"].sum())
        res[col] = {"ece_pts": ece * 100, "brier": float(((df[col] - df["hit"]) ** 2).mean()),
                    "table": g.reset_index().astype({col: str}).to_dict(orient="records")}
    res["by_delta"] = df.groupby("delta").agg(p_fhs=("p_fhs", "mean"), p_bs_iv=("p_bs_iv", "mean"), hit=("hit", "mean"),
                                              n=("hit", "size")).reset_index().to_dict(orient="records")
    res["n"] = len(df)
    # time-clustered uncertainty: per-month calibration gaps
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
    m = df.groupby("month").agg(gap=("p_fhs", "mean"), hit=("hit", "mean"))
    mt = stats.mean_test((m["gap"] - m["hit"]).to_numpy(), lags=3)
    res["fhs_mean_gap_monthly"] = {"mean_pts": mt["mean"] * 100, "t_hac": mt["t"]}
    return df, res


# ---------------------------------------------------------------------------- per-option output today
def option_table(symbols: list[str]) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        u = _load_u(sym)
        if u is None:
            continue
        t = u.df.index[-1]
        row = u.df.iloc[-1]
        fhs = u.fhs_at(t)
        if fhs is None or not np.isfinite(row["iv"]):
            continue
        S, r, q = float(row["close"]), float(row["r"]), float(row["q"])
        from ..markets_calendar import third_friday
        from ..strategies import forecast_var, trading_days
        exp = market.monthly_expiries(t, 3)
        expiry = min(exp, key=lambda e: abs((e - t).days - 35))
        T = market.year_frac(t, expiry)
        fv = forecast_var(u, t, T, expiry)
        d = physical.distribution(fhs, S, trading_days(t, expiry), r, q, n=8000, seed=0, target_var_annual=fv)
        d.T = T
        Ks = u.strikes_near(t, 0.80, 1.05)
        qt = u.quote(t, Ks, expiry, "P")
        fair = d.fair_price(Ks, "P")
        fiv = pricing.implied_vol(fair, S, Ks, T, r, q, "P")
        for i, K in enumerate(Ks):
            delta = float(qt["greeks"].delta[i])
            if not -0.45 <= delta <= -0.05:
                continue
            bid, ask = float(qt["bid"][i]), float(qt["ask"][i])
            ivb = float(pricing.implied_vol(bid, S, K, T, r, q, "P")) if bid > 0 else np.nan
            iva = float(pricing.implied_vol(ask, S, K, T, r, q, "P"))
            rows.append({"symbol": sym, "asof": t.date(), "expiry": expiry.date(), "K": K, "delta": delta,
                         "model_iv": float(qt["iv"][i]), "bid": bid, "ask": ask, "fair_value_P": float(fair[i]),
                         "fair_iv_P": float(fiv[i]), "edge_sell_per_share": bid - float(fair[i]),
                         "signal": physical.signal(ivb, iva, float(fiv[i])), "forecast_vol": float(np.sqrt(fv))})
    return pd.DataFrame(rows)


def main() -> dict:
    syms = [s for s in symbols_with_iv() if (market.CACHE_DIR / f"u_{s}.pkl").exists()]
    print(f"{len(syms)} cached underlyings")
    svi = svi_report()
    save_csv("phase3_svi_fits", svi)
    lv = local_vol_report()
    save_csv("phase3_local_vol", lv)
    per, pooled, vrp_years = rv_vrp_report(syms)
    save_csv("phase3_rv_per_name", per)
    save_csv("phase3_vrp_by_year", vrp_years.reset_index().rename(columns={"index": "symbol"}))
    tenor = vrp_by_tenor_spx()
    save_csv("phase3_vrp_spx_tenor", tenor)
    cal_df, cal = calibration(syms)
    opt = option_table(["SPY", "MSFT", "JPM", "KO" if "KO" in syms else "PG", "V", "HD"])
    save_csv("phase3_option_signals_today", opt)

    med_rmse = float(svi["rmse_volpts"].median())
    clean = float((svi["butterfly_ok_in_data"] & svi["calendar_ok_in_data"]).mean())
    lv_ok = lv[~lv["calendar_arbitrage_before_this_slice"]]
    lv_max = float(lv_ok["err_volpts"].abs().max())
    best = max(("har", "har_iv", "garch", "gjr"), key=lambda m: pooled[m]["share_names_better"])
    share = pooled[best]["share_names_better"]
    dm_sig = pooled[best]["t_hac"] < -1.96
    ece_f, ece_b = cal["p_fhs"]["ece_pts"], cal["p_bs_iv"]["ece_pts"]
    summary = {"svi_median_rmse_volpts": med_rmse, "svi_share_clean": clean, "svi_slices": len(svi),
               "local_vol_max_err_volpts": lv_max, "local_vol_points": len(lv_ok),
               "local_vol_slices_excluded_calendar_arb": int(lv["calendar_arbitrage_before_this_slice"].sum()),
               "rv_pooled": pooled, "rv_best_model": best, "vrp_spx_tenor": tenor.to_dict(orient="records"),
               "vrp_names_mean_volpts": float(per["vrp_vol_mean"].mean() * 100),
               "vrp_names_share_positive_mean": float(per["vrp_share_pos"].mean()),
               "calibration": cal, "n_names": len(per)}
    save_json("phase3_summary", summary)
    checks = [
        ("SVI median IV RMSE <= 0.5 vol pt on real chains", med_rmse <= 0.5, f"median {med_rmse:.3f} over {len(svi)} slices"),
        (">= 90 % of slices butterfly- and calendar-clean inside the quoted range", clean >= 0.9, f"{clean:.1%}"),
        ("local-vol PDE reprices quoted vanillas (80-120 %) within 0.5 vol pt", lv_max <= 0.5,
         f"max |error| {lv_max:.3f} vol pt over {len(lv_ok)} quotes; {summary['local_vol_slices_excluded_calendar_arb']} quotes on slices with calendar arbitrage excluded"),
        ("HAR/GARCH beats trailing RV on QLIKE for >= 60 % of names", share >= 0.6, f"{best}: {share:.1%} of {len(per)} names"),
        ("pooled Diebold-Mariano significant", dm_sig, f"{best}: t = {pooled[best]['t_hac']:.2f}"),
        ("FHS calibration error <= 3 pts and <= Black-Scholes at IV", ece_f <= 3 and ece_f <= ece_b,
         f"FHS {ece_f:.2f} pts vs BS(IV) {ece_b:.2f} pts over {cal['n']} forecasts"),
    ]
    v = verdict("phase3", checks)
    print(json.dumps(v, indent=1))
    return v


if __name__ == "__main__":
    main()


def real_spx_table(snapshot_id: int = 5, days: float = 30) -> pd.DataFrame:
    """Per-option fair value / edge / signal on REAL quotes: the saved IBKR SPX chain, with the physical distribution
    of the S&P 500 fitted on ^GSPC returns up to the snapshot date and its level anchored to SPY's walk-forward HAR-IV
    forecast at that date (SPY and SPX share the index)."""
    from ..rv import fit_garch, garch_filter
    qs = vstore.load_snapshot(snapshot_id)
    surf = vsurface.build_surface(qs.quotes, qs.spot, qs.asof, qs.symbol, qs.source)
    sl = min(surf.slices, key=lambda s: abs(s.T * 365 - days))
    asof = pd.Timestamp(qs.asof).tz_localize(None).normalize()
    spx = store.load_series("^GSPC")
    r = np.log(spx[spx.index < asof]).diff().dropna().to_numpy()[-2520:]
    fhs = physical.FHS.from_returns(r)
    u = _load_u("SPY")
    t = u.df.index[u.df.index < asof][-1]
    fv = float(u.df.at[t, "fc_har_iv"])
    F, D, T = sl.F, sl.D, sl.T
    rate = -np.log(D) / T
    S = float(qs.spot)
    q = rate - np.log(F / S) / T
    days_td = max(int(round(T * 252)), 1)
    d = physical.distribution(fhs, S, days_td, rate, q, n=20000, seed=0, target_var_annual=fv)
    d.T = T
    pts = sl.usable
    puts = pts[(pts["right"] == "P")] if "right" in pts.columns else pts
    rows = []
    for x in puts.itertuples():
        K = float(x.strike)
        if not 0.80 <= K / F <= 1.0:
            continue
        fair = float(d.fair_price([K], "P")[0])
        fiv = float(pricing.implied_vol(fair, S, K, T, rate, q, "P"))
        rows.append({"underlying": "SPX", "asof": str(qs.asof)[:16], "expiry": sl.expiry, "K": K, "K/F": K / F,
                     "bid": x.bid, "ask": x.ask, "iv_bid": getattr(x, "iv_bid", np.nan), "iv_ask": getattr(x, "iv_ask", np.nan),
                     "fair_value_P": fair, "fair_iv_P": fiv, "edge_sell_per_share": float(x.bid) - fair,
                     "signal": physical.signal(getattr(x, "iv_bid", np.nan), getattr(x, "iv_ask", np.nan), fiv),
                     "forecast_vol_30d": float(np.sqrt(fv))})
    return pd.DataFrame(rows)
