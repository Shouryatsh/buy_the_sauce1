"""
Phases 3B (backtest part), 4, 5 and 6: walk-forward selection, baselines, hedging frontier, direction independence,
P&L attribution and classification, sensitivity runs, verdicts.

    python -m premia.research.analysis            # after walkforward run (and walkforward sensitivity)
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import attribution, stats, store, validation as V
from ..config import RESULTS_DIR
from .common import save_csv, save_json, verdict
from .walkforward import RUN_DIR, load_run, walk_forward

OOS_START = "2013-01-01"


def _navs(prefix: str) -> dict[str, pd.Series]:
    out = {}
    for p in RUN_DIR.glob(f"{prefix}*.pkl"):
        if "__" in p.stem:
            continue
        out[p.stem] = load_run(p.stem)["daily"]["nav"]
    return out


def _grid(prefix: str) -> dict[str, pd.Series]:
    import re
    pat = re.compile(rf"^{prefix}_[lt][0-9.]+_b[0-9.]+$")
    return {k: v for k, v in _navs(prefix).items() if pat.match(k)}


def _oos(nav: pd.Series) -> pd.Series:
    return nav[nav.index >= OOS_START]


def nav_from_returns(r: pd.Series, nav0: float = 1e6) -> pd.Series:
    return nav0 * (1 + r.fillna(0)).cumprod()


def stitched_frames(choices: pd.DataFrame, col: str, frame: str = "daily") -> pd.DataFrame:
    """The OOS rows of a per-run frame (daily / sym_rows / attrib) following the walk-forward choices."""
    parts = []
    bounds = list(pd.to_datetime(choices["boundary"])) + [pd.Timestamp("2100-01-01")]
    cache = {}
    for i, c in enumerate(choices.itertuples()):
        lab = c.chosen
        if lab not in cache:
            cache[lab] = load_run(lab)
        f = cache[lab][frame]
        if f is None or len(f) == 0:
            continue
        idx = f.index if frame == "daily" else pd.to_datetime(f[col])
        m = (idx >= bounds[i]) & (idx < bounds[i + 1])
        parts.append(f[m])
    return pd.concat(parts) if parts else pd.DataFrame()


def sharpe_of(nav: pd.Series) -> float:
    return stats.sharpe(V.returns(_oos(nav)).to_numpy())


def main() -> dict:
    res: dict = {}
    wheel_grid, strad_grid = _grid("wheel"), _grid("straddle")
    w_ret, w_choice = walk_forward(wheel_grid, OOS_START)
    s_ret, s_choice = walk_forward(strad_grid, OOS_START)
    save_csv("wf_wheel_choices", w_choice)
    save_csv("wf_straddle_choices", s_choice)
    w_nav, s_nav = nav_from_returns(w_ret), nav_from_returns(s_ret)
    comb = (w_ret.reindex(w_ret.index.union(s_ret.index)).fillna(0) + s_ret.reindex(w_ret.index.union(s_ret.index)).fillna(0))
    c_nav = nav_from_returns(comb)
    pd.DataFrame({"wheel": w_nav, "straddle": s_nav, "combined": c_nav}).to_csv(RESULTS_DIR / "wf_oos_nav.csv")
    sel = {"wheel": w_choice["chosen"].mode().iloc[0], "straddle": s_choice["chosen"].mode().iloc[0]}
    (RESULTS_DIR / "selected_params.json").write_text(json.dumps(sel))

    # ---- performance (phase 6 report) ----------------------------------------------------------------
    perf = {"wheel_wf": V.performance(w_nav), "straddle_wf": V.performance(s_nav), "combined_wf": V.performance(c_nav)}
    others = {k: v for k, v in _navs("").items()}
    for k, nav in others.items():
        perf[k] = V.performance(_oos(nav))
    res["performance_oos"] = perf

    # deflated Sharpe of each walk-forward sleeve against its number of trials
    for name, grid, nav in (("wheel", wheel_grid, w_nav), ("straddle", strad_grid, s_nav)):
        trial_sr = [stats.sharpe(V.returns(_oos(v)).to_numpy()) / np.sqrt(252) for v in grid.values()]
        r = V.returns(nav).to_numpy()
        res[f"dsr_{name}"] = {"n_trials": len(grid), "psr": stats.probabilistic_sharpe(r),
                              "dsr": stats.deflated_sharpe(r, len(grid), float(np.nanvar(trial_sr)) or 1e-6)}

    # ---- phase 4: models vs fixed-rule baselines, hedging frontier ----------------------------------------
    base_w = others.get("wheel_base_b1")
    base_s = others.get("straddle_base_b1")
    res["phase4"] = {
        "wheel_model_sharpe": perf["wheel_wf"].get("sharpe"), "wheel_baseline_sharpe": sharpe_of(base_w) if base_w is not None else np.nan,
        "straddle_model_sharpe": perf["straddle_wf"].get("sharpe"), "straddle_baseline_sharpe": sharpe_of(base_s) if base_s is not None else np.nan}
    frontier = []
    for lab in ("wheel_l0.1_b0.5", "wheel_l0.1_b1", "wheel_l0.1_b2", "wheel_l0.1_time1", "wheel_l0.1_time5", "wheel_l0.1_none"):
        if not (RUN_DIR / f"{lab}.pkl").exists():
            continue
        d = load_run(lab)["daily"]
        d = d[d.index >= OOS_START]
        hc = V.hedge_cost(d)
        frontier.append({"run": lab, "hedge_cost_pct_nav_per_year": hc["total_hedge_cost_per_year_pct_nav"],
                         "daily_pnl_vol_pct_nav": float((d["nav"].diff() / d["nav"].shift()).std()),
                         "mean_abs_delta_pct_nav": float((d["net_delta"] / d["nav"]).abs().mean()),
                         "sharpe": stats.sharpe(V.returns(d["nav"]).to_numpy())})
    fr = pd.DataFrame(frontier)
    save_csv("phase4_hedge_frontier", fr)
    band = fr[fr["run"].str.contains("_b")]
    timep = fr[fr["run"].str.contains("time")]
    dominated = []
    for b in band.itertuples():
        if ((timep["hedge_cost_pct_nav_per_year"] <= b.hedge_cost_pct_nav_per_year) &
                (timep["daily_pnl_vol_pct_nav"] <= b.daily_pnl_vol_pct_nav)).any():
            dominated.append(b.run)
    res["phase4"]["band_points_dominated_by_time_hedging"] = dominated
    res["phase4"]["frontier"] = frontier

    # ---- phase 5: direction independence ---------------------------------------------------------------
    m = V.market_returns(c_nav.index)
    p5 = {}
    for name, nav in (("combined", c_nav), ("wheel", w_nav), ("straddle", s_nav)):
        p5[name] = {"beta": V.beta_test(V.returns(nav), m), "regimes": V.regimes(nav)}
    for lab in ("wheel_base_none", "wheel_l0.1_none", "wheel_base_b1"):
        if lab in others:
            nav = _oos(others[lab])
            p5[lab] = {"beta": V.beta_test(V.returns(nav), V.market_returns(nav.index))}
    wd = stitched_frames(w_choice, "date", "daily")
    sd = stitched_frames(s_choice, "date", "daily")
    p5["wheel_delta_profile"] = V.delta_profile(wd)
    p5["wheel_hedge_cost"] = V.hedge_cost(wd)
    p5["straddle_hedge_cost"] = V.hedge_cost(sd)
    closes = {}
    syms_w = stitched_frames(w_choice, "date", "sym_rows")
    syms_s = stitched_frames(s_choice, "date", "sym_rows")
    for s in set(syms_w.get("symbol", [])) | set(syms_s.get("symbol", [])):
        px = store.load_prices(s)
        if not px.empty:
            closes[s] = px["close"]
    uw = V.underlying_correlation(syms_w, closes)
    us = V.underlying_correlation(syms_s, closes)
    if "table" in uw:
        save_csv("phase5_underlying_corr_wheel", uw.pop("table"))
    if "table" in us:
        save_csv("phase5_underlying_corr_straddle", us.pop("table"))
    p5["underlying_corr_wheel"], p5["underlying_corr_straddle"] = uw, us
    res["phase5"] = p5

    # ---- phase 3B: attribution, exposures, classification, regression ------------------------------------
    att_w = stitched_frames(w_choice, "date", "attrib")
    att_s = stitched_frames(s_choice, "date", "attrib")
    att = pd.concat([att_w, att_s])
    comp = attribution.COMPONENTS + ["residual", "total"]
    by_kind = att.groupby("kind")[comp].sum()
    costs = {}
    for name, d in (("wheel", wd), ("straddle", sd)):
        for c in [c for c in d.columns if c.startswith("cost_")]:
            costs[f"{name}:{c}"] = float(d[c].sum())
    res["attribution_by_kind"] = by_kind.reset_index().to_dict(orient="records")
    res["costs_total"] = costs
    # exposure by strike and DTE
    att["mny_bucket"] = pd.cut(att["moneyness"], [0, 0.85, 0.9, 0.95, 0.98, 1.02, 1.05, 1.1, 10])
    att["dte_bucket"] = pd.cut(att["dte"], [0, 7, 14, 21, 30, 45, 90, 400])
    exp = att.groupby(["kind", "mny_bucket", "dte_bucket"], observed=True)[["vanna_usd", "volga_usd", "vega_usd"]].mean().reset_index()
    save_csv("phase3b_exposure_by_strike_dte", exp.astype({"mny_bucket": str, "dte_bucket": str}))
    # realised premium (theta + gamma + vega + vanna + volga: the delta-hedged option P&L) per $ of |vega| per day
    att["hedged_pnl"] = att[["gamma", "vega", "theta", "vanna", "volga"]].sum(axis=1)
    prem = att.groupby(["kind", "mny_bucket", "dte_bucket"], observed=True).apply(
        lambda g: pd.Series({"pnl_per_abs_vega": g["hedged_pnl"].sum() / max(g["vega_usd"].abs().sum(), 1e-9),
                             "position_days": len(g)}), include_groups=False).reset_index()
    save_csv("phase3b_premium_by_strike_dte", prem.astype({"mny_bucket": str, "dte_bucket": str}))
    # classification: residual after option costs of that trade type, per day
    daily_kind = att.groupby(["date", "kind"])["residual"].sum().reset_index()
    kc = []
    for name, d in (("wheel", wd), ("straddle", sd)):
        for c in [c for c in d.columns if c.startswith("cost_") and c[5:] in ("csp", "cc", "straddle", "tail")]:
            kc.append(pd.DataFrame({"date": d.index, "kind": c[5:], "cost": d[c].fillna(0).to_numpy()}))
    kcost = pd.concat(kc) if kc else pd.DataFrame(columns=["date", "kind", "cost"])
    daily_kind = daily_kind.merge(kcost, on=["date", "kind"], how="left").fillna({"cost": 0.0})
    daily_kind["residual_net"] = daily_kind["residual"] - daily_kind["cost"]
    cls = attribution.classify(daily_kind.rename(columns={"kind": "trade_type"}))
    # SPEC section 3: on model-priced history the residual is the model's own higher-order terms, so a significant
    # residual cannot be alpha. Keep the statistics, cap the label until real quotes (paper marks) exist.
    for c in cls:
        c.raw_label = c.label
        if c.label == "alpha":
            c.label = "risk premium"
            c.note += "; capped: model-priced backtest cannot establish alpha (SPEC section 3)"
    res["classification"] = [c.__dict__ for c in cls]
    # residual regressed on market factors (HAC + bootstrap)
    port = att.groupby("date")["residual"].sum()
    spy = V.market_returns(port.index)
    vix = store.load_series("^VIX").reindex(port.index).ffill()
    dv = vix.diff() / 100
    X = pd.concat([spy.rename("spy"), dv.rename("dvix"), (dv ** 2).rename("dvix2")], axis=1)
    d = pd.concat([port.rename("res"), X], axis=1).dropna()
    reg = stats.ols_hac(d["res"].to_numpy(), d[["spy", "dvix", "dvix2"]].to_numpy(), ["spy", "dvix", "dvix2"], boot_reps=500)
    res["residual_regression"] = reg.table()

    # ---- sensitivity runs --------------------------------------------------------------------------------
    sens = {}
    for p in RUN_DIR.glob("*__*.pkl"):
        sens[p.stem] = V.performance(_oos(load_run(p.stem)["daily"]["nav"]))
    for sleeve, lab in sel.items():
        if lab in others:
            sens[lab] = V.performance(_oos(others[lab]))
    res["sensitivity"] = sens

    save_json("analysis", res)

    # ---- verdicts ----------------------------------------------------------------------------------------
    cb = p5["combined"]["beta"]
    perf_c = perf["combined_wf"]
    checks5 = [
        ("|beta| to SPY < 0.10 and Newey-West |t| < 2", abs(cb["beta"]) < 0.10 and abs(cb["t_hac"]) < 2,
         f"beta {cb['beta']:.3f}, t {cb['t_hac']:.2f}"),
        ("|correlation| to SPY < 0.20", abs(cb["corr"]) < 0.20, f"corr {cb['corr']:.3f}"),
        ("|correlation| of per-name P&L to its underlying < 0.20 (held-days weighted)",
         abs(uw.get("weighted_mean_corr", 1)) < 0.2 and abs(us.get("weighted_mean_corr", 1)) < 0.2,
         f"wheel {uw.get('weighted_mean_corr', float('nan')):.3f} (median |corr| {uw.get('median_abs_corr', float('nan')):.3f}), "
         f"straddle {us.get('weighted_mean_corr', float('nan')):.3f}"),
        ("net annualised return after costs > 0", perf_c.get("cagr", -1) > 0,
         f"CAGR {perf_c.get('cagr', float('nan')):.2%}, Sharpe {perf_c.get('sharpe', float('nan')):.2f}, "
         f"mean-return HAC t {perf_c.get('mean_t_hac', float('nan')):.2f}"),
    ]
    v5 = verdict("phase5", checks5)
    ph4 = res["phase4"]
    checks4 = [
        ("engines run to specification (unit tests)", True, "tests/premia/test_engine.py"),
        ("model decisions recorded vs fixed-rule baselines (keep / drop)", True,
         f"wheel model {ph4['wheel_model_sharpe']:.2f} vs baseline {ph4['wheel_baseline_sharpe']:.2f} -> "
         f"{'KEEP' if ph4['wheel_model_sharpe'] > ph4['wheel_baseline_sharpe'] else 'DROP'}; straddle model "
         f"{ph4['straddle_model_sharpe']:.2f} vs baseline {ph4['straddle_baseline_sharpe']:.2f} -> "
         f"{'KEEP' if ph4['straddle_model_sharpe'] > ph4['straddle_baseline_sharpe'] else 'DROP'}"),
        ("band hedge decision recorded (on/inside time-hedge frontier = keep)", True,
         f"band points dominated by time hedging: {dominated or 'none'}"),
    ]
    v4 = verdict("phase4", checks4)
    alpha_types = [c.trade_type for c in cls if c.raw_label == "alpha"]
    checks3b = [
        ("Taylor residual <= 10 % of P&L variance on simulated paths", True, "tests/premia/test_models.py::test_taylor_attribution_explains_small_daily_moves"),
        ("alpha test size <= 10 %, power >= 80 % at the minimum detectable effect", True,
         "tests/premia/test_models.py::test_alpha_classifier_power_and_false_positive_rate"),
        ("as pre-registered: detects annual Sharpe 0.5 in >= 80 % of simulations", False,
         "impossible at realistic sample lengths: 80 % power at Sharpe 0.5 needs ~31 years of daily data (2.8/sqrt(years))"),
        ("every backtested trade type classified", len(cls) > 0,
         f"{[(c.trade_type, c.label) for c in cls]}; statistically significant residual (not alpha, model-priced): "
         f"{alpha_types or 'none'}"),
    ]
    v3b = verdict("phase3b", checks3b)
    print(json.dumps({"phase3b": v3b["verdict"], "phase4": v4["verdict"], "phase5": v5["verdict"]}, indent=1))
    return res


if __name__ == "__main__":
    main()
