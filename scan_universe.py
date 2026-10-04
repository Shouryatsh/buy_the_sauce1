"""
scan_universe.py — One-time broad universe scan to build the options wheel
dashboard's shortlist (see wheel_universe.py for the pipeline).

This is a slow, occasional batch job, not something to run on every
dashboard load: screening ~1000+ tickers against SEC EDGAR takes several
minutes minimum (SEC's own ~10 req/s rate guidance is the bottleneck, not
this script). Run it once to build/refresh the shortlist, then use the
dashboard normally -- wheel_watchlist.py picks up wheel_shortlist.json
automatically once it exists.

Usage
-----
    python scan_universe.py                  # default universe size (see wheel_config.UNIVERSE_SIZE)
    python scan_universe.py --size 500        # smaller/faster run
    python scan_universe.py --min-margin 0.15 # stricter net-margin bar for this run
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys
import time

logging.basicConfig(level=logging.WARNING)

import wheel_config as wc
from wheel_universe import (
    build_shortlist,
    get_universe_tickers,
    save_shortlist,
    scan_universe,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _parse_args():
    parser = argparse.ArgumentParser(description="One-time broad universe scan for the wheel dashboard")
    parser.add_argument("--size", type=int, default=wc.UNIVERSE_SIZE, help=f"Universe size to pull from SEC (default: {wc.UNIVERSE_SIZE})")
    parser.add_argument("--min-margin", type=float, default=wc.UNIVERSE_MIN_NET_MARGIN, help=f"Minimum net margin (default: {wc.UNIVERSE_MIN_NET_MARGIN:.0%})")
    parser.add_argument("--workers", type=int, default=wc.UNIVERSE_SCAN_WORKERS, help=f"Concurrent EDGAR workers (default: {wc.UNIVERSE_SCAN_WORKERS})")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    wc.UNIVERSE_MIN_NET_MARGIN = args.min_margin   # honor the CLI override for this run

    print(f"Fetching top {args.size} tickers from SEC EDGAR (roughly market-cap ordered)...")
    tickers = get_universe_tickers(args.size)
    print(f"{len(tickers)} tickers after filtering out non-common-stock symbols.\n")

    print(
        f"Scanning {len(tickers)} tickers ({args.workers} concurrent workers, "
        f"SEC-limited to ~8 req/s aggregate) -- this will take several minutes...\n"
    )

    start = time.time()
    last_print = [0.0]

    def _progress(done: int, total: int) -> None:
        now = time.time()
        if now - last_print[0] > 2.0 or done == total:
            elapsed = now - start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            print(f"\r  {done}/{total} scanned  ({rate:.1f}/s, ETA {eta:.0f}s)   ", end="", flush=True)
            last_print[0] = now

    profiles = scan_universe(tickers, max_workers=args.workers, progress_callback=_progress)
    print()

    elapsed = time.time() - start
    shortlist = build_shortlist(profiles)
    save_shortlist(shortlist)

    unreliable = [p for p in profiles if not p.data_reliable]
    print(f"\nDone in {elapsed:.0f}s.")
    print(f"  Scanned      : {len(profiles)}")
    print(f"  Shortlisted  : {len(shortlist)}  (net margin >= {wc.UNIVERSE_MIN_NET_MARGIN:.0%}, FCF > 0)")
    print(f"  Data unclear : {len(unreliable)} (scored anyway, flagged data_reliable=False)")
    print(f"  Saved to     : {wc.SHORTLIST_PATH}")

    date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    csv_path = os.path.join(RESULTS_DIR, f"universe_scan_{date_str}.csv")
    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["symbol", "shortlisted", "quality_score", "price", "net_margin", "fcf", "fcf_yield", "roe", "data_reliable", "fail_reasons"])
        shortlist_set = set(shortlist)
        for p in sorted(profiles, key=lambda p: p.quality_score or 0, reverse=True):
            writer.writerow([
                p.symbol, p.symbol in shortlist_set, p.quality_score, p.price,
                p.net_margin, p.free_cash_flow, p.fcf_yield, p.return_on_equity,
                p.data_reliable, "; ".join(p.fail_reasons),
            ])
    print(f"  Full detail  : {csv_path}")

    if shortlist:
        preview = ", ".join(shortlist[:20])
        more = f" (+{len(shortlist) - 20} more)" if len(shortlist) > 20 else ""
        print(f"\n  Top of shortlist: {preview}{more}")


if __name__ == "__main__":
    main()
