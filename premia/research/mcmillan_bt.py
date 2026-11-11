"""
Backtest of McMillan's straddle-buying strategy (premia/mcmillan.py) on the dashboard's wheel universe.

    python -m premia.research.mcmillan_bt term          # how much the SPX term structure lifts 3-6 month IV
    python -m premia.research.mcmillan_bt run           # the base run and the sensitivity runs (processes)
    python -m premia.research.mcmillan_bt report        # performance, direction tests, exits, criterion-2 calibration, verdict

The book fixes every parameter, so nothing is selected on the data: the whole period after the warm-up is out of
sample. 3-6 month options are priced off each name's 30-day IV times that day's SPX term-structure ratios (VIX3M/VIX,
VIX6M/VIX): observed data, no fitted parameter. Pass criteria are pre-registered in premia/MCMILLAN.md.

Universe: the wheel watchlist (wheel_watchlist.WHEEL_WATCHLIST) minus the manual exclusions, every name with at least
500 days of IV history. Today's list applied to the past: names on it survived to today (look-ahead bias, stated in
the report).
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
import pandas as pd

from .. import backtest as bt, market, mcmillan as mm, stats, store, validation as V
from ..config import RESULTS_DIR
from ..hedging import HedgePolicy
from .build import load_shapes, shape_for
from .common import save_csv, save_json, verdict
from .walkforward import RUN_DIR, _sectors

TERM_FILE = RESULTS_DIR / "mcmillan_term_check.json"
START, OOS_START = "2009-01-02", "2013-01-01"
VARIANTS = {
    "mcm_base": {},
    "mcm_hist80": {"mm_hist_freq": 0.80},
    "mcm_3month": {"mm_target_days": 100},
    "mcm_fullspread": {"slip": 1.0},
    "mcm_no_followup": {"mm_follow_up": False},
    "mcm_flat_term": {"_term": False},
}


# ---------------------------------------------------------------------------
# term structure: the SPX VIX family
# ---------------------------------------------------------------------------

def _vix_frame() -> pd.DataFrame:
    d = pd.DataFrame({k: store.load_series(s) for k, s in (("v30", "^VIX"), ("v93", "^VIX3M"), ("v183", "^VIX6M"))})
    return d.dropna() / 100


def term_check() -> dict:
    """How much the SPX term structure lifts 3- and 6-month IV above 30-day IV, overall and in the regime the strategy
    buys in (30-day IV at or below the 10th percentile of its past). These are the ratios the backtest applies to
    every name's 30-day IV, each day (no fitted parameters). A fitted mean-reversion model was tried first: calibrated
    on 2008-2012, it still under-priced 6-month SPX vol by ~1.9 vol pts out of sample in low-IV regimes, so the observed
    ratios replaced it."""
    d = _vix_frame()
    low = d["v30"] <= d["v30"].expanding(min_periods=504).quantile(0.10)
    out = {}
    for col in ("v93", "v183"):
        out[f"{col}_minus_v30_volpts_all"] = float((d[col] - d["v30"]).mean() * 100)
        out[f"{col}_minus_v30_volpts_low_iv"] = float((d[col] - d["v30"])[low].mean() * 100)
        out[f"{col}_over_v30_low_iv"] = float((d[col] / d["v30"])[low].mean())
    out["low_iv_days"] = int(low.sum())
    TERM_FILE.write_text(json.dumps(out, indent=1))
    print(out)
    return out


# ---------------------------------------------------------------------------
# universe and the per-name inputs of the rules
# ---------------------------------------------------------------------------

def wheel_universe() -> list[str]:
    from wheel_watchlist import WHEEL_WATCHLIST, load_exclusions
    ex = load_exclusions()
    return [s for s in dict.fromkeys(s.upper() for s in WHEEL_WATCHLIST) if s not in ex]


def prepare(u: market.Underlying, term: bool) -> market.Underlying:
    df = u.df
    df["mm_ivpct"] = mm.iv_percentile_all(df["iv"])
    df["mm_minhv"] = mm.min_historical_vol(df["close"])
    df["mm_sma"] = df["close"].rolling(mm.TRAIL_MA).mean()
    if term and not u.etf:
        market.add_index_term(u)
    return u


def load_world(term: bool = True):
    shapes = load_shapes()
    U = {}
    for s in wheel_universe() + ["SPY"]:
        u = market.build_underlying(s, shape_for(s, shapes))
        if u is not None:
            U[s] = prepare(u, term)
    names = sorted(s for s in U if s != "SPY")
    return U, (lambda t: names), _sectors(list(U))


def run_one(args) -> tuple[str, float, str]:
    label, overrides = args
    t0 = time.time()
    overrides = dict(overrides)
    term = overrides.pop("_term", True)
    U, sched, sectors = load_world(term)
    betas = bt.rolling_betas(U)
    dates = U["SPY"].df.index
    dates = dates[dates >= pd.Timestamp(START)]
    cfg = replace(bt.RunConfig(sleeve="mcmillan", policy=HedgePolicy("none"), tail_hedge=False, label=label), **overrides)
    b = bt.Backtest(cfg, U, sched, dates, sectors)
    bmap = {t: row.dropna().to_dict() for t, row in betas.iterrows()}
    daily = b.run(lambda t: bmap.get(t, {}))
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    out = {"label": label, "spec": {"sleeve": "mcmillan", "term": term}, "overrides": overrides, "cfg": cfg,
           "daily": daily, "trades": pd.DataFrame(b.trades), "attrib": pd.DataFrame(b.attrib),
           "sym_rows": pd.DataFrame(b.sym_rows), "blocked": pd.DataFrame(b.blocked), "universe": sorted(U),
           "signals": pd.DataFrame(b.mm_signals)}
    (RUN_DIR / f"{label}.pkl").write_bytes(pickle.dumps(out))
    return label, time.time() - t0, f"nav {daily['nav'].iloc[-1]:,.0f}, {len(b.trades)} trade events, {len(U) - 1} names"


def run(only: list[str] | None = None, workers: int = 6) -> None:
    jobs = [(k, v) for k, v in VARIANTS.items() if not only or k in only]
    with ProcessPoolExecutor(min(workers, len(jobs))) as ex:
        for label, secs, msg in ex.map(run_one, jobs):
            print(f"{label}: {secs / 60:.1f} min, {msg}", flush=True)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def straddle_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """One row per straddle: entry cost, every exit, P&L (before commissions), how it ended."""
    t = trades[trades["kind"] == "mcm"].copy()
    if t.empty:
        return t
    opens = t[t["action"] == "open"]
    rows = []
    for (sym, K, exp, d0), g in opens.groupby(["symbol", "K", "expiry", "date"]):
        pids = set(g["pid"])
        ex = t[t["pid"].isin(pids) & (t["action"] != "open")]
        n = int(g["qty"].iloc[0])
        cost = float(g["px"].sum()) * n * 100
        pnl = float(ex["pnl"].sum()) if len(ex) else np.nan
        reasons = list(dict.fromkeys(ex.sort_values("date")["reason"].fillna(ex["action"]).tolist()))
        rows.append({"symbol": sym, "entry": d0, "expiry": exp, "K": K, "contracts": n, "premium": cost,
                     "exit": ex["date"].max() if len(ex) else pd.NaT, "pnl": pnl, "ret_on_premium": pnl / cost if cost else np.nan,
                     "first_exit": reasons[0] if reasons else "open", "path": " > ".join(reasons),
                     "ever_prob": g["m_ever_prob"].iloc[0] if "m_ever_prob" in g else np.nan,
                     "hist_freq": g["m_hist_freq"].iloc[0] if "m_hist_freq" in g else np.nan,
                     "iv_pct": g["m_iv_pct"].iloc[0] if "m_iv_pct" in g else np.nan})
    return pd.DataFrame(rows).sort_values("entry")


def touch_calibration(sig: pd.DataFrame) -> pd.DataFrame:
    """Criterion 2 tested against what happened: for every name-day that passed criterion 1, did the price (daily high
    or low) actually touch a breakeven before expiry? Bucketed by the predicted 'ever' probability."""
    rows = []
    for sym, g in sig.groupby("symbol"):
        px = store.load_prices(sym)
        if px.empty:
            continue
        hi, lo = px["high"], px["low"]
        for r in g.itertuples():
            w = (px.index > r.date) & (px.index <= r.expiry)
            if px.index[-1] < r.expiry or not w.any() or not np.isfinite(r.ever_prob):
                continue
            up, dn = r.K + r.cost, r.K - r.cost
            rows.append({"ever_prob": r.ever_prob, "touched": bool((hi[w] >= up).any() or (lo[w] <= dn).any()),
                         "passes": r.passes, "taken": r.taken})
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    d["bucket"] = pd.cut(d["ever_prob"], [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], include_lowest=True)
    out = d.groupby("bucket", observed=True).agg(predicted=("ever_prob", "mean"), realized=("touched", "mean"),
                                                 n=("touched", "size")).reset_index()
    out["bucket"] = out["bucket"].astype(str)
    return out


def report() -> dict:
    res: dict = {"term_check": json.loads(TERM_FILE.read_text()) if TERM_FILE.exists() else {}}
    base = None
    for label in VARIANTS:
        p = RUN_DIR / f"{label}.pkl"
        if not p.exists():
            continue
        r = pickle.loads(p.read_bytes())
        d = r["daily"]
        nav_oos = d["nav"][d.index >= OOS_START]
        entry = {"oos": V.performance(nav_oos), "full": V.performance(d["nav"]),
                 "beta": V.beta_test(V.returns(nav_oos), V.market_returns(nav_oos.index))}
        tr = straddle_trades(r["trades"])
        tr_oos = tr[tr["entry"] >= OOS_START] if len(tr) else tr
        if len(tr_oos):
            entry["trades"] = {"n": len(tr_oos), "win_rate": float((tr_oos["pnl"] > 0).mean()),
                               "mean_ret_on_premium": float(tr_oos["ret_on_premium"].mean()),
                               "median_ret_on_premium": float(tr_oos["ret_on_premium"].median()),
                               "by_first_exit": tr_oos.groupby("first_exit")["ret_on_premium"].agg(["size", "mean"]).reset_index().to_dict(orient="records"),
                               "avg_premium": float(tr_oos["premium"].mean()), "names": int(tr_oos["symbol"].nunique())}
        entry["costs_oos"] = {c: float(d.loc[d.index >= OOS_START, c].sum()) for c in ("cost_opt_comm", "cost_opt_spread", "cost_stock")}
        entry["mean_open_straddles"] = float((d.loc[d.index >= OOS_START, "n_opts"] / 2).mean())
        entry["blocked_by_limits"] = int(len(r["blocked"]))
        res[label] = entry
        if label == "mcm_base":
            base = r
            save_csv("mcmillan_trades", tr)
            entry["regimes"] = V.regimes(nav_oos)
            entry["delta_profile"] = V.delta_profile(d[d.index >= OOS_START])
            y = d["nav"].resample("YE").last()
            y.loc[pd.Timestamp(d.index[0].year - 1, 12, 31)] = bt.RunConfig().nav0
            entry["calendar_years"] = {str(k.year): float(v) for k, v in y.sort_index().pct_change().dropna().items()}
            sig = r["signals"]
            entry["signals"] = {"name_days_criterion1": len(sig), "pass_all": int(sig["passes"].sum()) if len(sig) else 0,
                                "taken": int(sig["taken"].sum()) if len(sig) else 0}
            cal = touch_calibration(sig)
            save_csv("mcmillan_touch_calibration", cal)
            entry["touch_calibration"] = cal.to_dict(orient="records")
            pd.DataFrame({"nav": d["nav"]}).to_csv(RESULTS_DIR / "mcmillan_nav.csv")
            entry["psr"] = stats.probabilistic_sharpe(V.returns(nav_oos).to_numpy())
    if base is not None:
        b = res["mcm_base"]
        rr = V.returns(base["daily"]["nav"][base["daily"].index >= OOS_START]).to_numpy()
        srs = [res[k]["oos"]["sharpe"] / np.sqrt(252) for k in VARIANTS if k in res]
        b["dsr"] = stats.deflated_sharpe(rr, len(srs), float(np.nanvar(srs)) or 1e-6)
    save_json("mcmillan_report", res)
    if base is not None:
        b = res["mcm_base"]
        prev = json.loads((RESULTS_DIR / "analysis.json").read_text())["performance_oos"]["straddle_wf"]["sharpe"]
        checks = [
            ("net annualised return after costs > 0 (2013 onward)", b["oos"]["cagr"] > 0,
             f"CAGR {b['oos']['cagr']:.2%}, Sharpe {b['oos']['sharpe']:.2f}"),
            ("mean daily return significant (Newey-West t >= 2)", b["oos"]["mean_t_hac"] >= 2,
             f"t {b['oos']['mean_t_hac']:.2f}"),
            ("beats the earlier model straddle sleeve (walk-forward Sharpe)", b["oos"]["sharpe"] > prev,
             f"{b['oos']['sharpe']:.2f} vs {prev:.2f}"),
            ("result holds with fills at the full half-spread", res.get("mcm_fullspread", {}).get("oos", {}).get("cagr", -1) > 0,
             f"CAGR {res.get('mcm_fullspread', {}).get('oos', {}).get('cagr', float('nan')):.2%}"),
        ]
        v = verdict("mcmillan_straddle", checks)
        print(json.dumps(v, indent=1, default=str))
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["term", "run", "report"])
    ap.add_argument("--only", default="")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    if a.cmd == "term":
        term_check()
    elif a.cmd == "run":
        run([s for s in a.only.split(",") if s], a.workers)
    else:
        report()


if __name__ == "__main__":
    main()
