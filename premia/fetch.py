"""
premia/fetch.py — populate premia_data.db. Resumable: a symbol fetched successfully in the last `--fresh-days` is skipped.

    python -m premia.fetch all                 # membership, Yahoo prices, index series, IBKR IV (TWS 7496, read-only), SEC
    python -m premia.fetch iv --symbols SPY,QQQ
    python -m premia.fetch report              # coverage / survivorship table

IBKR is used only for historical-data requests (connect(readonly=True)); no order can be sent from this module.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta, timezone

from . import data, store

log = logging.getLogger("premia.fetch")


def _fresh(kind: str, days: float) -> set[str]:
    st = store.fetch_status(kind)
    if st.empty:
        return set()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    ok = st[(st["status"].isin(["ok", "no_data", "unknown", "no_cik"])) & (st["fetched_at"] >= cutoff)]
    return set(ok["symbol"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m premia.fetch")
    ap.add_argument("step", choices=["all", "membership", "prices", "series", "iv", "sec", "report"])
    ap.add_argument("--symbols", default="")
    ap.add_argument("--port", type=int, default=7496)
    ap.add_argument("--fresh-days", type=float, default=5.0)
    ap.add_argument("--concurrency", type=int, default=3)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("ib_insync", "yfinance", "peewee"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    if args.step in ("all", "membership") or store.load_membership().empty:
        m = data.fetch_membership()
        log.info("membership rows: %d", len(m))
    membership = store.load_membership().rename(columns={"start": "start", "end": "end"})
    universe = [data.norm(s) for s in args.symbols.split(",") if s] or data.candidate_universe(membership)
    log.info("universe: %d symbols", len(universe))

    if args.step in ("all", "series"):
        log.info("index series: %s", data.fetch_index_series())
    if args.step in ("all", "prices"):
        todo = [s for s in universe if s not in _fresh("price", args.fresh_days)]
        res = data.fetch_yahoo_prices(todo)
        log.info("prices: %d fetched, %d ok", len(res), sum(v == "ok" for v in res.values()))
    if args.step in ("all", "sec"):
        todo = [s for s in universe if s not in _fresh("sec", args.fresh_days)]
        res = data.fetch_sec(todo)
        log.info("sec: %d fetched, %d ok", len(res), sum(v == "ok" for v in res.values()))
    if args.step in ("all", "iv"):
        todo = [s for s in universe if s not in _fresh("iv", args.fresh_days)]
        # only symbols Yahoo knows: a ticker with no price history cannot be traded in the backtest anyway
        have_px = set(store.symbols_with("price"))
        todo = [s for s in todo if s in have_px or args.symbols]
        log.info("iv: %d to fetch", len(todo))
        res = data.fetch_ibkr_iv(todo, port=args.port, concurrency=args.concurrency)
        log.info("iv: %s", {k: sum(v == k for v in res.values()) for k in set(res.values())})
    if args.step in ("all", "report"):
        rep = data.coverage_report(membership)
        print(rep.iloc[::4].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
