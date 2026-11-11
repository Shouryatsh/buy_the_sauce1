"""
volsurface/record.py — save a surface snapshot (and its summary metrics) from the command line, so surface dynamics
build up over time even when the dashboard is closed. Schedule it (e.g. Windows Task Scheduler, every 30-60 minutes
during market hours):

    python -m volsurface.record --symbol SPX [--source yahoo|ibkr]

yahoo needs no TWS; ibkr needs TWS running (read-only connection, client id 79 or $WHEEL_IBKR_CLIENT_ID).
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Optional

from . import analysis, data, model, store, surface


def record(symbol: str, source: str = "yahoo", client=None, fit: Optional[model.FitSettings] = None) -> int:
    qs = data.fetch_ibkr(client, symbol) if source == "ibkr" else data.fetch_yahoo(symbol)
    sid = store.save_snapshot(qs)
    surf = surface.build_surface(qs.quotes, qs.spot, qs.asof, qs.symbol, qs.source)
    fs = model.fit_surface(surf, fit or model.FitSettings(butterfly_strength=3.0, calendar_strength=3.0))
    store.update_metrics(sid, analysis.snapshot_metrics(fs))
    return sid


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Record a volatility-surface snapshot.")
    ap.add_argument("--symbol", default="SPX")
    ap.add_argument("--source", default="yahoo", choices=["yahoo", "ibkr"])
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    client = None
    if args.source == "ibkr":
        from wheel_ibkr import WheelIBKRClient
        client = WheelIBKRClient(client_id=int(os.environ.get("WHEEL_IBKR_CLIENT_ID", "79")))
        client.connect()
    try:
        sid = record(args.symbol, args.source, client)
        print(f"saved snapshot {sid} for {args.symbol} ({args.source})")
        return 0
    finally:
        if client is not None:
            client.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
