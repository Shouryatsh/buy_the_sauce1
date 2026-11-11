"""
Phase 2 runner: point-in-time screen over every S&P 500 member since 2010, pass counts per filter per quarter,
and the out-of-sample test of the model-layer safety score.

    python -m premia.research.phase2
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .. import data, screener, store
from ..config import load_exclusions
from .common import save_csv, save_json, verdict


def main() -> dict:
    t0 = time.time()
    membership = store.load_membership()
    excl = load_exclusions()
    syms = sorted(set(store.symbols_with("facts")) - excl)
    dates = screener.quarter_dates("2010-01-01")
    dates = [d for d in dates if d <= pd.Timestamp.today() - pd.Timedelta(days=1)]
    df = screener.run_pit_screen(syms, dates)
    print(f"evaluated {len(df)} symbol-quarters for {df['symbol'].nunique()} symbols in {time.time() - t0:.0f}s")

    panels = {}
    for s in df["symbol"].unique():
        px = store.load_prices(s)
        if not px.empty:
            panels[s] = {"px": px, "earn": store.load_earnings(s)}
    df["member"] = screener.member_flags(df, membership)
    df = screener.liquidity_flags(df, panels)
    df["pass_all"] = (df["status"] == "pass") & df["pass_liquidity"] & df["member"]
    df = screener.add_scores(df, panels)
    save_csv("phase2_screen", df)

    counts = screener.pass_counts(df)
    save_csv("phase2_pass_counts", counts)

    passing = df[df["pass_all"]]
    val_pass = screener.validate_score(passing, min_names=10)
    val_all = screener.validate_score(df[df["member"] & df["pass_liquidity"]], min_names=30)
    for v in (val_pass, val_all):
        if "ic" in v:
            v["ic_series"] = v.pop("ic")
    save_json("phase2_score_validation", {"passing_names": val_pass, "all_liquid_members": val_all})

    # PIT sanity: no evaluation used a filing dated after its as-of date (checked inside evaluate by construction;
    # here we assert the latest quarter always precedes the as-of date)
    pit_ok = bool((pd.to_datetime(df["latest_quarter"].replace("", np.nan)).dropna()
                   <= pd.to_datetime(df.loc[df["latest_quarter"] != "", "asof"])).all())

    kept = (val_pass.get("n_quarters", 0) >= 12 and val_pass.get("mean_ic", 0) < 0
            and val_pass.get("p_boot_one_sided", 1) < 0.05)
    checks = [
        ("hard filters use only data filed on or before each date", pit_ok, "latest quarter end <= as-of for every row; filings filtered by filed date (unit-tested)"),
        ("pass counts per filter per quarter reported", len(counts) > 0, f"{len(counts)} quarters in phase2_pass_counts.csv"),
        ("model-layer decision recorded (kept or dropped)", True,
         f"safety score {'KEPT' if kept else 'DROPPED'}: mean IC {val_pass.get('mean_ic', float('nan')):.3f} over "
         f"{val_pass.get('n_quarters', 0)} quarters, one-sided bootstrap p={val_pass.get('p_boot_one_sided', float('nan')):.3f}; "
         f"all liquid members: mean IC {val_all.get('mean_ic', float('nan')):.3f}, p={val_all.get('p_boot_one_sided', float('nan')):.3f}"),
    ]
    v = verdict("phase2", checks)
    v["score_kept"] = kept
    save_json("verdict_phase2", v)
    print(counts.tail(8).to_string(index=False))
    print(v)
    return v


if __name__ == "__main__":
    main()
