"""
Backtest runs for phases 4-6 and the walk-forward selection.

    python -m premia.research.walkforward run            # every run in RUNS (8 processes), results/premia/runs/*.pkl
    python -m premia.research.walkforward run --only wheel_l0.1_b1
    python -m premia.research.walkforward sensitivity     # skew x0.5 / x1.5 and full-spread fills for the selected params

Walk-forward: every run covers 2010-2026 with fixed parameters. At each 6-month boundary from 2013-01-01 the
parameter set with the best Sharpe over the trailing 3 years (training window, data known at the boundary) is chosen
and its next 6 months are the out-of-sample segment. The stitched segments are the only performance evidence
reported. Approximation: each parameter set runs as its own book, so a switch between sets is modelled as an
instantaneous change of book (the positions of the old book are not carried over).
"""

from __future__ import annotations

import argparse
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
import pandas as pd

from .. import backtest as bt, market, store
from ..config import BACKTEST_START, PARAMS, RESULTS_DIR
from ..hedging import HedgePolicy
from .build import load_shapes, shape_for

RUN_DIR = RESULTS_DIR / "runs"
SCREEN_FILE = RESULTS_DIR / "phase2_screen.csv"


def run_specs() -> dict[str, dict]:
    specs = {}
    for lam in PARAMS.risk_aversion_grid:
        for m in PARAMS.hedge_band_mult_grid:
            specs[f"wheel_l{lam:g}_b{m:g}"] = {"sleeve": "wheel", "lam": lam, "policy": ("band", 1, m)}
    specs["wheel_l0.1_time1"] = {"sleeve": "wheel", "lam": 0.1, "policy": ("time", 1, 1.0)}
    specs["wheel_l0.1_time5"] = {"sleeve": "wheel", "lam": 0.1, "policy": ("time", 5, 1.0)}
    specs["wheel_base_b1"] = {"sleeve": "wheel", "use_model": False, "policy": ("band", 1, 1.0)}
    specs["wheel_base_none"] = {"sleeve": "wheel", "use_model": False, "policy": ("none", 1, 1.0), "tail_hedge": False}
    specs["wheel_l0.1_none"] = {"sleeve": "wheel", "lam": 0.1, "policy": ("none", 1, 1.0), "tail_hedge": False}
    for th in PARAMS.straddle_edge_grid:
        for m in PARAMS.hedge_band_mult_grid:
            specs[f"straddle_t{th:g}_b{m:g}"] = {"sleeve": "straddle", "straddle_threshold": th, "policy": ("band", 1, m)}
    specs["straddle_base_b1"] = {"sleeve": "straddle", "use_model": False, "policy": ("band", 1, 1.0)}
    return specs


def _sectors(symbols: list[str]) -> dict[str, str]:
    with store.connect() as c:
        rows = dict(c.execute("SELECT symbol, sic FROM facts").fetchall())
    return {s: (str(rows.get(s) or "")[:2] or "etf") for s in symbols}


def load_world(n_names: int = 15):
    screen = pd.read_csv(SCREEN_FILE)
    shapes = load_shapes()
    schedule_all = bt.schedule_from_screen(screen, use_score=False)
    cached = {p.stem[2:] for p in market.CACHE_DIR.glob("u_*.pkl")}
    names = set()
    for q in pd.to_datetime(screen.loc[screen["pass_all"], "asof"]).unique():
        names |= {s for s in schedule_all(q + pd.Timedelta(days=1)) if s in cached}
    names |= {s for s in ("SPY", "QQQ", "IWM") if s in cached}
    U = {}
    for s in sorted(names):
        u = market.build_underlying(s, shape_for(s, shapes))
        if u is not None:
            U[s] = u
    sched = lambda t: [s for s in schedule_all(t) if s in U]
    return U, sched, _sectors(list(U))


def run_one(args) -> tuple[str, float, str]:
    label, spec, start, end, overrides = args
    t0 = time.time()
    U, sched, sectors = load_world()
    betas = bt.rolling_betas(U)
    dates = U["SPY"].df.index
    dates = dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
    kind, every, mult = spec["policy"]
    cfg = bt.RunConfig(sleeve=spec["sleeve"], policy=HedgePolicy(kind, every, mult), lam=spec.get("lam", 0.1),
                       straddle_threshold=spec.get("straddle_threshold", 0.02), use_model=spec.get("use_model", True),
                       tail_hedge=spec.get("tail_hedge", True), label=label)
    if overrides:
        cfg = replace(cfg, **overrides)
    b = bt.Backtest(cfg, U, sched, dates, sectors)
    bmap = {t: row.dropna().to_dict() for t, row in betas.iterrows()}
    daily = b.run(lambda t: bmap.get(t, {}))
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    out = {"label": label, "spec": spec, "overrides": overrides, "cfg": cfg, "daily": daily,
           "trades": pd.DataFrame(b.trades), "attrib": pd.DataFrame(b.attrib), "sym_rows": pd.DataFrame(b.sym_rows),
           "blocked": pd.DataFrame(b.blocked), "universe": sorted(U)}
    (RUN_DIR / f"{label}.pkl").write_bytes(pickle.dumps(out))
    return label, time.time() - t0, f"nav {daily['nav'].iloc[-1]:,.0f}, {len(b.trades)} trade events"


def load_run(label: str) -> dict:
    return pickle.loads((RUN_DIR / f"{label}.pkl").read_bytes())


def walk_forward(navs: dict[str, pd.Series], start: str = "2013-01-01", train_years: int = PARAMS.train_years,
                 step_months: int = PARAMS.test_months) -> tuple[pd.Series, pd.DataFrame]:
    """Stitched out-of-sample daily returns and the choice made at each boundary."""
    rets = pd.DataFrame({k: v.pct_change() for k, v in navs.items()}).dropna(how="all")
    bounds = pd.date_range(start, rets.index[-1], freq=f"{step_months}MS")
    pieces, choices = [], []
    for i, b in enumerate(bounds):
        train = rets[(rets.index < b) & (rets.index >= b - pd.DateOffset(years=train_years))]
        nxt = bounds[i + 1] if i + 1 < len(bounds) else rets.index[-1] + pd.Timedelta(days=1)
        test = rets[(rets.index >= b) & (rets.index < nxt)]
        if train.empty or test.empty:
            continue
        sh = train.mean() / train.std().replace(0, np.nan) * np.sqrt(252)
        pick = sh.idxmax() if sh.notna().any() else train.columns[0]
        choices.append({"boundary": b.date(), "chosen": pick, "train_sharpe": float(sh.max()) if sh.notna().any() else np.nan})
        pieces.append(test[pick].fillna(0.0))
    return pd.concat(pieces), pd.DataFrame(choices)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "sensitivity"])
    ap.add_argument("--only", default="")
    ap.add_argument("--start", default=BACKTEST_START)
    ap.add_argument("--end", default="2026-12-31")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    specs = run_specs()
    jobs = []
    if args.cmd == "run":
        labels = [s for s in args.only.split(",") if s] or list(specs)
        jobs = [(l, specs[l], args.start, args.end, {}) for l in labels]
    else:
        import json
        sel = json.loads((RESULTS_DIR / "selected_params.json").read_text())
        for sleeve, label in sel.items():
            for tag, ov in (("skew0.5", {"skew_mult": 0.5}), ("skew1.5", {"skew_mult": 1.5}), ("slip1.0", {"slip": 1.0})):
                jobs.append((f"{label}__{tag}", specs[label], args.start, args.end, ov))
    print(f"{len(jobs)} runs")
    with ProcessPoolExecutor(args.workers) as ex:
        for label, secs, msg in ex.map(run_one, jobs):
            print(f"{label}: {secs / 60:.1f} min, {msg}", flush=True)


if __name__ == "__main__":
    main()
