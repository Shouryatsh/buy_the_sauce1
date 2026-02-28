"""
run.py — CLI entry point for the buy-the-dip scanner.

Usage
-----
    # Live run (connects to IBKR and places real orders):
    python run.py

    # Dry run (no IBKR connection, prints signals and order sizes):
    python run.py --dry-run

    # Run continuously on a daily schedule at a specific time:
    python run.py --schedule 09:45

Options
-------
--dry-run           Evaluate signals and sizes but do not place orders.
--schedule HH:MM    Run every day at the given time (24-hour clock, US Eastern).
                    If omitted, the scan runs once immediately and exits.
--log-level LEVEL   Override log level (DEBUG, INFO, WARNING). Default: INFO.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import schedule

import config
from trader import run_scan


def _setup_logging(level: str) -> None:
    fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(config.LOG_FILE),
        ],
    )


def _run_once(dry_run: bool) -> None:
    specs = run_scan(dry_run=dry_run)
    if dry_run and specs:
        print("\n── Dry-run order preview ──")
        for s in specs:
            print(f"  {s}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Buy-the-dip scanner (IBKR / US stocks)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate signals and sizes only — no orders placed",
    )
    parser.add_argument(
        "--schedule",
        metavar="HH:MM",
        help="Run every day at this time (24-hour, US Eastern). "
             "If omitted, runs once immediately.",
    )
    parser.add_argument(
        "--log-level",
        default=config.LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: %(default)s)",
    )
    args = parser.parse_args()

    _setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    if args.schedule:
        logger.info("Scheduling daily scan at %s US-Eastern", args.schedule)
        schedule.every().day.at(args.schedule).do(_run_once, dry_run=args.dry_run)
        logger.info("Scheduler running — press Ctrl+C to stop")
        try:
            while True:
                schedule.run_pending()
                time.sleep(30)
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
    else:
        _run_once(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
