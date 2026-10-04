"""
run.py — Production automation entry point for the buy-the-sauce trading system.

Modes
-----
  python run.py                        Full auto: buy scans + position management
  python run.py --dry-run              Same schedule, no real orders (audit mode)
  python run.py --manage-only          Position management loop only (no buy scans)
  python run.py --once                 Single buy scan + manage, then exit
  python run.py --once --manage-only   Single manage cycle, then exit

Schedule (configurable in config.py)
--------
  Buy scans:           09:45, 12:45, 14:45 US/Eastern  (3× per day)
  Position management: every 15 min during market hours (09:30–16:00 ET)
  End-of-day sweep:    once after market close
  Heartbeat log:       every 60 min

All times are US/Eastern.  The loop automatically skips weekends and NYSE
holidays.  IBKR TWS or IB Gateway must be running on the configured port
(default 7497 = paper trading).

Safety
------
  • IBKR hard stops protect positions even if this process crashes
  • Position state persisted to disk  (results/position_state.json)
  • On IBKR connection failure during management: logs error, retries next
    cycle — broker-side stops remain active
  • Graceful shutdown:  Ctrl-C or SIGTERM → finishes current cycle then exits
  • Kill switch:
      python -c "from broker import IBKRBroker; b=IBKRBroker(); b.connect(); b.kill_switch()"
"""

from __future__ import annotations

import argparse
import datetime
import io
import logging
import os
import signal
import sys
import time

import config
from trader import run_scan, run_manage_only

# ---------------------------------------------------------------------------
# Timezone handling
# ---------------------------------------------------------------------------

try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except ImportError:
    import pytz  # type: ignore
    _ET = pytz.timezone("America/New_York")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(level: str) -> None:
    fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        stream = sys.stdout
    except Exception:
        stream = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    stream_handler = logging.StreamHandler(stream)
    file_handler = logging.FileHandler(config.LOG_FILE, encoding="utf-8")

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        handlers=[stream_handler, file_handler],
    )


logger = logging.getLogger("run")

# ---------------------------------------------------------------------------
# Market hours
# ---------------------------------------------------------------------------

_NYSE_HOLIDAYS_2026 = {
    datetime.date(2026, 1, 1),   # New Year
    datetime.date(2026, 1, 19),  # MLK Day
    datetime.date(2026, 2, 16),  # Presidents' Day
    datetime.date(2026, 4, 3),   # Good Friday
    datetime.date(2026, 5, 25),  # Memorial Day
    datetime.date(2026, 7, 3),   # Independence Day (observed)
    datetime.date(2026, 9, 7),   # Labor Day
    datetime.date(2026, 11, 26), # Thanksgiving
    datetime.date(2026, 12, 25), # Christmas
}


def _get_holidays() -> set[datetime.date]:
    """Try pandas_market_calendars for accurate holidays, else hardcoded fallback."""
    try:
        import pandas_market_calendars as mcal
        import pandas as pd
        nyse = mcal.get_calendar("NYSE")
        year = datetime.datetime.now(_ET).year
        sched = nyse.schedule(start_date=f"{year}-01-01", end_date=f"{year}-12-31")
        all_days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="B")
        trading_days = set(sched.index.date)
        return {d.date() for d in all_days if d.date() not in trading_days}
    except Exception:
        return _NYSE_HOLIDAYS_2026


def _now_et() -> datetime.datetime:
    return datetime.datetime.now(_ET)


def _parse_time(t_str: str) -> datetime.time:
    h, m = (int(x) for x in t_str.split(":"))
    return datetime.time(h, m)


def _is_market_hours() -> bool:
    """True if Mon–Fri, inside market hours, non-holiday."""
    now = _now_et()
    if now.weekday() >= 5:
        return False
    if now.date() in _get_holidays():
        return False
    return _parse_time(config.MARKET_OPEN_TIME) <= now.time() < _parse_time(config.MARKET_CLOSE_TIME)


def _is_buy_scan_time() -> bool:
    """True if current time is within ±2 minutes of a configured buy scan time."""
    now = _now_et()
    now_min = now.hour * 60 + now.minute
    for t_str in config.BUY_SCAN_TIMES:
        h, m = (int(x) for x in t_str.split(":"))
        if abs(now_min - (h * 60 + m)) <= 2:
            return True
    return False


def _next_market_open_str() -> str:
    """Human-readable string for when the market next opens."""
    now = _now_et()
    holidays = _get_holidays()
    candidate = now.date()
    for _ in range(10):
        if candidate.weekday() < 5 and candidate not in holidays:
            open_t = _parse_time(config.MARKET_OPEN_TIME)
            open_dt = datetime.datetime.combine(candidate, open_t).replace(tzinfo=_ET)
            if open_dt > now:
                return open_dt.strftime("%a %b %d at %H:%M ET")
        candidate += datetime.timedelta(days=1)
    return "next trading day"


def _sleep_label(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

_shutdown_requested = False


def _handle_signal(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    logger.warning("Shutdown signal received (sig=%s) — finishing current cycle then exiting", signum)


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ---------------------------------------------------------------------------
# Interruptible sleep
# ---------------------------------------------------------------------------

def _sleep_interruptible(seconds: float) -> None:
    """Sleep in 1-second chunks so we can respond to shutdown signals."""
    end = time.time() + seconds
    while time.time() < end and not _shutdown_requested:
        time.sleep(1)


# ---------------------------------------------------------------------------
# Core automation loop
# ---------------------------------------------------------------------------

def _run_automation_loop(dry_run: bool, manage_only: bool) -> None:
    """Main loop.  Runs until Ctrl-C / SIGTERM.

    Every MANAGE_INTERVAL_MINUTES during market hours:
      • If it's a buy scan time → full scan (fundamentals + dip + ML + orders)
      • Otherwise → position management only (trailing stops, partials, time stops)

    Outside market hours:
      • One end-of-day management sweep after close
      • Sleep until next market open
    """
    interval_secs = max(config.MANAGE_INTERVAL_MINUTES, 1) * 60
    last_buy_minute = -1       # dedup buy scans within the ±2 min window
    last_heartbeat = time.time()
    last_eod_date = None       # one EOD sweep per day

    logger.info("═" * 68)
    logger.info("🍅 Buy The Sauce — Automation started")
    logger.info(
        "  Mode:           %s%s",
        "MANAGE-ONLY" if manage_only else "FULL (buy + manage)",
        "  [DRY RUN]" if dry_run else "",
    )
    logger.info("  IBKR:           %s:%d  (clientId %d)", config.IBKR_HOST, config.IBKR_PORT, config.IBKR_CLIENT_ID)
    if not manage_only:
        logger.info("  Buy scans:      %s ET", ", ".join(config.BUY_SCAN_TIMES))
    logger.info(
        "  Manage every:   %d min during %s–%s ET",
        config.MANAGE_INTERVAL_MINUTES, config.MARKET_OPEN_TIME, config.MARKET_CLOSE_TIME,
    )
    logger.info(
        "  Position state: %s",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "position_state.json"),
    )
    logger.info("  Shutdown:       Ctrl-C or SIGTERM → graceful exit")
    logger.info("═" * 68)

    while not _shutdown_requested:
        now = _now_et()

        # ── Heartbeat ─────────────────────────────────────────────────────
        if time.time() - last_heartbeat >= config.HEARTBEAT_INTERVAL_MINUTES * 60:
            from trader import get_open_positions_snapshot
            n_pos = len(get_open_positions_snapshot())
            logger.info(
                "♥ Heartbeat — %s ET | %d tracked position(s) | market %s",
                now.strftime("%H:%M"), n_pos,
                "OPEN" if _is_market_hours() else "CLOSED",
            )
            last_heartbeat = time.time()

        # ── Outside market hours ──────────────────────────────────────────
        if not _is_market_hours():
            today = now.date()
            close_t = _parse_time(config.MARKET_CLOSE_TIME)

            # End-of-day sweep: once after close on a trading day
            just_closed = (
                now.time() >= close_t
                and now.hour < close_t.hour + 1
                and now.weekday() < 5
                and today not in _get_holidays()
            )
            if just_closed and last_eod_date != today:
                logger.info("📊 End-of-day management sweep")
                try:
                    run_manage_only(dry_run=dry_run)
                except Exception as exc:
                    logger.error("EOD management failed: %s", exc, exc_info=True)
                last_eod_date = today

            next_open = _next_market_open_str()
            logger.debug("Market closed — next open: %s", next_open)
            _sleep_interruptible(60)
            continue

        # ── Inside market hours ───────────────────────────────────────────
        current_minute = now.hour * 60 + now.minute

        # Check if it's a buy scan slot
        should_buy = (
            not manage_only
            and _is_buy_scan_time()
            and current_minute != last_buy_minute
        )

        if should_buy:
            last_buy_minute = current_minute
            logger.info("🔍 ═══ BUY SCAN (%s ET) ═══", now.strftime("%H:%M"))
            try:
                specs = run_scan(dry_run=dry_run)
                logger.info(
                    "🔍 Buy scan complete — %d order(s) %s",
                    len(specs), "would be placed" if dry_run else "submitted",
                )
            except Exception as exc:
                logger.error("Buy scan FAILED: %s", exc, exc_info=True)
        else:
            # Lightweight position management
            logger.info("🔄 Position management (%s ET)", now.strftime("%H:%M"))
            try:
                result = run_manage_only(dry_run=dry_run)
                n_actions = sum(
                    len(result[k]) for k in ("trailing_updates", "partial_exits", "time_stops", "full_exits")
                )
                if n_actions > 0:
                    logger.info("🔄 %d action(s): %s", n_actions, result)
                else:
                    logger.info("🔄 No actions needed")
            except Exception as exc:
                logger.error("Position management FAILED: %s", exc, exc_info=True)

        # ── Sleep until next cycle ────────────────────────────────────────
        elapsed = (_now_et() - now).total_seconds()
        sleep_secs = max(interval_secs - elapsed, 30)
        logger.debug("Next cycle in %s", _sleep_label(sleep_secs))
        _sleep_interruptible(sleep_secs)

    logger.warning("🛑 Automation stopped (shutdown requested)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Buy-the-sauce automated trading system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                        Full automation (buy scans @ 09:45/12:45/14:45 + management every 15m)
  python run.py --dry-run              Audit mode — no real orders, same schedule
  python run.py --manage-only          Manage existing positions only (no new buys)
  python run.py --once                 Single buy scan + manage, then exit
  python run.py --once --dry-run       Single dry-run scan, then exit
  python run.py --once --manage-only   Single management cycle, then exit
        """,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate signals and sizes but don't place real orders",
    )
    parser.add_argument(
        "--manage-only",
        action="store_true",
        help="Position management loop only — skip all buy scans",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single cycle and exit (don't loop)",
    )
    parser.add_argument(
        "--log-level",
        default=config.LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _setup_logging(args.log_level)

    if args.once:
        # ── Single execution mode ─────────────────────────────────────────
        if args.manage_only:
            logger.info("=== Single position management cycle ===")
            result = run_manage_only(dry_run=args.dry_run)
            logger.info("Result: %s", result)
        else:
            logger.info("=== Single buy scan + manage ===")
            specs = run_scan(dry_run=args.dry_run)
            if args.dry_run and specs:
                print("\n── Dry-run order preview ──")
                for s in specs:
                    print(f"  {s}")
        return

    # ── Continuous automation loop ────────────────────────────────────────
    _run_automation_loop(dry_run=args.dry_run, manage_only=args.manage_only)


if __name__ == "__main__":
    main()
