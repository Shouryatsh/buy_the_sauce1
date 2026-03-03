"""
run_screen.py — Screener runner with market-hours-aware scheduling.

Shows TWO tables:
  1. DIP DETECTOR  — all stocks ranked by dip score (best opportunities first)
  2. FUNDAMENTALS  — pass/fail with reasons (for your own judgement)

Saves to:
  results/screen_YYYY-MM-DD_HH-MM.txt   (snapshot per run)
  results/screen_log.csv                 (cumulative history)

Usage
-----
  # Run once immediately (manual):
  python3 run_screen.py

  # Run every 15 min during market hours automatically (9:30–16:00 ET, Mon–Fri):
  python3 run_screen.py --loop

  # Same but with a custom interval (e.g. every 30 minutes):
  python3 run_screen.py --loop --interval 30

Safety guarantees built into --loop mode
-----------------------------------------
  ✅ Only runs Mon–Fri, 9:30 AM – 4:00 PM US/Eastern (NYSE hours)
  ✅ Skips US market holidays automatically (uses pandas_market_calendars if
     installed, otherwise falls back to a hardcoded 2026 holiday list)
  ✅ Rate-limited: a minimum of 10 minutes between runs is enforced regardless
     of --interval, so you can't accidentally spam SEC EDGAR or Stooq
  ✅ This script is READ-ONLY — it fetches data and prints results.
     It never connects to IBKR and never places orders.
     (Orders are only placed by run.py → trader.py → broker.py)
  ✅ SEC EDGAR fair-use: edgar.py already sleeps 120 ms between requests
     (~8 req/s), well under their 10 req/s limit.  Running every 15 min
     on a ~13-stock watchlist sends ~130 requests per run — perfectly fine.
"""
import sys, os, csv, warnings, logging, datetime, argparse, time
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import edgar
from screener import screen_fundamental
from dip_detector import score_dip, DipSignal
import config

# ---------------------------------------------------------------------------
# Market-hours guard
# ---------------------------------------------------------------------------

# US Eastern timezone (works on macOS/Linux without extra packages)
try:
    from zoneinfo import ZoneInfo          # Python 3.9+
    _ET = ZoneInfo("America/New_York")
except ImportError:
    from datetime import timezone
    import subprocess, re
    _ET = timezone.utc                     # safe fallback — loop will warn

# NYSE 2026 holidays (hardcoded fallback if pandas_market_calendars not installed)
_NYSE_HOLIDAYS_2026 = {
    datetime.date(2026, 1, 1),   # New Year's Day
    datetime.date(2026, 1, 19),  # MLK Jr. Day
    datetime.date(2026, 2, 16),  # Presidents' Day
    datetime.date(2026, 4, 3),   # Good Friday
    datetime.date(2026, 5, 25),  # Memorial Day
    datetime.date(2026, 6, 19),  # Juneteenth
    datetime.date(2026, 7, 3),   # Independence Day (observed)
    datetime.date(2026, 9, 7),   # Labor Day
    datetime.date(2026, 11, 26), # Thanksgiving
    datetime.date(2026, 11, 27), # Black Friday (early close, skip to be safe)
    datetime.date(2026, 12, 24), # Christmas Eve (early close, skip to be safe)
    datetime.date(2026, 12, 25), # Christmas Day
}

def _get_holidays() -> set:
    """Return NYSE holiday dates for the current year.

    Uses pandas_market_calendars if available (accurate for any year),
    otherwise falls back to the hardcoded 2026 list above.
    """
    try:
        import pandas_market_calendars as mcal
        import pandas as pd
        year = datetime.date.today().year
        nyse = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(
            start_date=f"{year}-01-01",
            end_date=f"{year}-12-31",
        )
        all_days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="B")
        trading_days = set(schedule.index.date)
        holidays = {d.date() for d in all_days if d.date() not in trading_days}
        return holidays
    except Exception:
        return _NYSE_HOLIDAYS_2026


def _is_market_open() -> bool:
    """Return True if the NYSE is currently open (Mon–Fri, 9:30–16:00 ET, non-holiday)."""
    now_et = datetime.datetime.now(_ET)
    today  = now_et.date()

    # Weekend check
    if now_et.weekday() >= 5:       # 5=Saturday, 6=Sunday
        return False

    # Holiday check
    if today in _get_holidays():
        return False

    # Time-of-day check (9:30 AM – 4:00 PM ET)
    open_time  = datetime.time(9, 30)
    close_time = datetime.time(16, 0)
    return open_time <= now_et.time() < close_time


def _next_open_str() -> str:
    """Human-readable string for when the market next opens."""
    now_et = datetime.datetime.now(_ET)
    today  = now_et.date()
    holidays = _get_holidays()

    # Find the next trading day
    candidate = today
    for _ in range(10):
        if candidate.weekday() < 5 and candidate not in holidays:
            open_dt = datetime.datetime.combine(candidate, datetime.time(9, 30))
            open_dt = open_dt.replace(tzinfo=_ET)
            if open_dt > now_et:
                return open_dt.strftime("%a %b %d at 9:30 AM ET")
        candidate += datetime.timedelta(days=1)

    return "next trading day at 9:30 AM ET"


# ---------------------------------------------------------------------------
# Argument parsing (must come before the single-run body)
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Buy-the-sauce screener — single run or market-hours loop"
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run repeatedly every --interval minutes during NYSE market hours (Mon–Fri 9:30–16:00 ET)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=15,
        metavar="MINUTES",
        help="Minutes between scans in --loop mode (default: 15, minimum enforced: 10)",
    )
    return parser.parse_args()


WATCHLIST = [
    "AAPL","MSFT","GOOGL","META","AMZN","NVDA",
    "JPM","V","BRK-B","UNH","COST","HD","HIMS",
    # ← add new tickers here, one per line or comma-separated:
    # "TSLA","PLTR","RDDT",
]

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEP  = "─" * 100
SEP2 = "═" * 100

# ── signal notes from DipSignal fields ───────────────────────────────────────
def _signal_notes(s: DipSignal) -> str:
    parts = []
    if s.rsi_signal:
        parts.append(f"RSI={s.rsi:.1f} oversold")
    if s.ma50_signal:
        pct = (s.ma50 - s.price) / s.ma50 * 100
        parts.append(f"{pct:.1f}% below MA50")
    if s.ma200_signal:
        pct = (s.ma200 - s.price) / s.ma200 * 100
        parts.append(f"{pct:.1f}% below MA200")
    if s.week52_signal:
        rng = max(s.week52_high - s.week52_low, 1e-6)
        pct = (s.price - s.week52_low) / rng * 100
        parts.append(f"52wk bottom {pct:.0f}%")
    return " | ".join(parts) if parts else "—"

# ── score bar e.g. "████░░" ──────────────────────────────────────────────────
def _bar(score, max_score=4):
    filled = "█" * score
    empty  = "░" * (max_score - score)
    return filled + empty


# ---------------------------------------------------------------------------
# Core single-run function
# ---------------------------------------------------------------------------

def run_screen() -> None:
    """Fetch data, build tables, print output, save .txt and append .csv."""
    now       = datetime.datetime.now()
    date_str  = now.strftime("%Y-%m-%d")
    time_str  = now.strftime("%H:%M:%S")
    file_slug = now.strftime("%Y-%m-%d_%H-%M")
    txt_path  = os.path.join(RESULTS_DIR, f"screen_{file_slug}.txt")
    csv_path  = os.path.join(RESULTS_DIR, "screen_log.csv")

    # ── collect all data ──────────────────────────────────────────────────────
    print(f"\nFetching data for {len(WATCHLIST)} tickers — {date_str} {time_str}\n")

    records = []
    for symbol in WATCHLIST:
        info    = edgar.get_fundamentals(symbol)
        profile = screen_fundamental(symbol, info=info)

        hist, price_source = edgar.get_price_history(symbol, period_years=2)
        signal = None
        if hist is not None and not hist.empty:
            signal = score_dip(symbol, hist)

        records.append({
            "symbol":       symbol,
            "profile":      profile,
            "signal":       signal,
            "info":         info,
            "price_source": price_source,
        })

    # ═══════════════════════════════════════════════════════════════════════════
    # TABLE 1 — DIP DETECTOR
    # ═══════════════════════════════════════════════════════════════════════════
    dip_hdr = (
        f"{'SYMBOL':<8} {'SCORE':<8} {'BAR':<8} {'PRICE':>8} "
        f"{'MA50':>8} {'MA200':>8} {'RSI':>6} {'52WK%':>7}  "
        f"{'FUND':>5}  {'SRC':<12}  SIGNALS"
    )

    dip_rows = []
    for r in sorted(records, key=lambda x: x["signal"].score if x["signal"] else -1, reverse=True):
        s       = r["signal"]
        profile = r["profile"]
        symbol  = r["symbol"]
        fund_ok = "PASS" if profile.passes else "FAIL"
        src     = r.get("price_source", "unknown")

        if s is None:
            dip_rows.append(
                f"{symbol:<8} {'N/A':<8} {'░░░░':<8} {'N/A':>8} "
                f"{'—':>8} {'—':>8} {'—':>6} {'—':>7}  "
                f"{fund_ok:>5}  {'no price data':<12}  —"
            )
            continue

        rng_pct = (s.price - s.week52_low) / max(s.week52_high - s.week52_low, 1e-6) * 100
        tag = ""
        if s.is_dip and not profile.passes:
            tag = " ◄ DIP (fund FAIL — manual call)"
        elif s.is_dip and profile.passes:
            tag = " ◄◄ BUY SIGNAL (fund PASS)"

        dip_rows.append(
            f"{symbol:<8} {s.score}/4{'':<4} {_bar(s.score):<8} "
            f"{s.price:>8.2f} {s.ma50:>8.2f} {s.ma200:>8.2f} "
            f"{s.rsi:>6.1f} {rng_pct:>6.0f}%  {fund_ok:>5}  {src:<12}  "
            f"{_signal_notes(s)}{tag}"
        )

    sources_str = " + ".join(set(r.get("price_source", "unknown") for r in records))
    table1 = "\n".join([
        SEP2,
        f"  TABLE 1 — DIP DETECTOR  (sorted by dip score, best first)",
        f"  Price source : {sources_str}",
        f"  Thresholds   : RSI < {config.RSI_OVERSOLD} | price > {config.DIP_FROM_MA50_PCT:.0%} below MA50 | below MA200 | 52wk bottom {config.WEEK52_LOWER_BAND:.0%}",
        SEP2,
        dip_hdr,
        SEP,
    ] + dip_rows + [SEP])

    # ═══════════════════════════════════════════════════════════════════════════
    # TABLE 2 — FUNDAMENTALS
    # ═══════════════════════════════════════════════════════════════════════════
    fund_hdr = (
        f"{'SYMBOL':<8} {'STATUS':<6} {'FCF':>10} {'YIELD':>7} "
        f"{'MARGIN':>8} {'D/E':>7} {'ROE':>7} {'PE':>7}  REASON / NOTES"
    )

    fund_rows = []
    for r in records:
        profile = r["profile"]
        symbol  = r["symbol"]
        status  = "PASS" if profile.passes else "FAIL"
        pe_val  = r["info"].get("trailingPE") or r["info"].get("forwardPE")
        fcf_s   = f"${profile.free_cash_flow/1e9:.1f}B"  if profile.free_cash_flow  is not None else "N/A"
        yld_s   = f"{profile.fcf_yield:.2%}"             if profile.fcf_yield        is not None else "N/A"
        mgn_s   = f"{profile.profit_margin:.1%}"         if profile.profit_margin    is not None else "N/A"
        de_s    = f"{profile.debt_to_equity:.1f}x"       if profile.debt_to_equity   is not None else "N/A"
        roe_s   = f"{profile.return_on_equity:.0%}"      if profile.return_on_equity is not None else "N/A"
        pe_s    = f"{pe_val:.1f}"                        if pe_val                   is not None else "N/A"
        reason  = "; ".join(profile.fail_reasons) if not profile.passes else "All filters passed"

        fund_rows.append(
            f"{symbol:<8} {status:<6} {fcf_s:>10} {yld_s:>7} "
            f"{mgn_s:>8} {de_s:>7} {roe_s:>7} {pe_s:>7}  {reason}"
        )

    table2 = "\n".join([
        SEP2,
        f"  TABLE 2 — FUNDAMENTAL SCREEN",
        f"  Thresholds: PE<{config.MAX_PE_RATIO} | margin>{config.MIN_PROFIT_MARGIN:.0%} | "
        f"D/E<{config.MAX_DEBT_TO_EQUITY} | FCF>0 | FCFyield>{config.MIN_FCF_YIELD:.1%} | "
        f"ROE>{config.MIN_RETURN_ON_EQUITY:.0%} | capex/FCF<{config.MAX_CAPEX_TO_FCF:.0%}",
        SEP2,
        fund_hdr,
        SEP,
    ] + fund_rows + [SEP])

    # ── summary ───────────────────────────────────────────────────────────────
    buy_both    = [r["symbol"] for r in records if r["signal"] and r["signal"].is_dip and r["profile"].passes]
    buy_diponly = [r["symbol"] for r in records if r["signal"] and r["signal"].is_dip and not r["profile"].passes]

    summary = "\n".join([
        SEP2,
        f"  SUMMARY — {date_str}  {time_str}",
        SEP2,
        f"  🟢 BUY SIGNAL  (dip ✅  +  fundamentals ✅) : {', '.join(buy_both)    if buy_both    else 'None today'}",
        f"  🟡 DIP ONLY    (dip ✅  +  fundamentals ❌) : {', '.join(buy_diponly)  if buy_diponly else 'None'}  ← manual call",
        SEP,
    ])

    header = (
        f"BUY-THE-SAUCE SCREENER\n"
        f"  Date   : {date_str}  {time_str}\n"
        f"  Source : SEC EDGAR (fundamentals)  +  {sources_str} (prices)\n"
        f"  Tickers: {', '.join(WATCHLIST)}\n"
    )
    output = f"{header}\n{table1}\n\n{table2}\n\n{summary}\n"

    print(output)

    with open(txt_path, "w") as f:
        f.write(output)
    print(f"  Saved : {txt_path}")

    # ── append CSV ────────────────────────────────────────────────────────────
    CSV_FIELDS = [
        "date","time","symbol","dip_score","is_dip","fund_pass",
        "price","ma50","ma200","rsi","week52_pct",
        "fcf_usd","fcf_yield_pct","profit_margin_pct","debt_to_equity","roe_pct","pe",
        "price_source","fail_reasons",
    ]
    csv_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not csv_exists:
            writer.writeheader()
        for r in records:
            s       = r["signal"]
            profile = r["profile"]
            pe_val  = r["info"].get("trailingPE") or r["info"].get("forwardPE")
            rng_pct = (
                (s.price - s.week52_low) / max(s.week52_high - s.week52_low, 1e-6) * 100
                if s else ""
            )
            writer.writerow({
                "date":               date_str,
                "time":               time_str,
                "symbol":             r["symbol"],
                "dip_score":          s.score          if s else "",
                "is_dip":             s.is_dip         if s else "",
                "fund_pass":          profile.passes,
                "price":              f"{s.price:.2f}" if s else "",
                "ma50":               f"{s.ma50:.2f}"  if s else "",
                "ma200":              f"{s.ma200:.2f}" if s else "",
                "rsi":                f"{s.rsi:.1f}"   if s else "",
                "week52_pct":         f"{rng_pct:.0f}" if s else "",
                "fcf_usd":            f"{profile.free_cash_flow:.0f}"       if profile.free_cash_flow      is not None else "",
                "fcf_yield_pct":      f"{profile.fcf_yield*100:.2f}"        if profile.fcf_yield           is not None else "",
                "profit_margin_pct":  f"{profile.profit_margin*100:.2f}"    if profile.profit_margin       is not None else "",
                "debt_to_equity":     f"{profile.debt_to_equity:.2f}"       if profile.debt_to_equity      is not None else "",
                "roe_pct":            f"{profile.return_on_equity*100:.1f}" if profile.return_on_equity    is not None else "",
                "pe":                 f"{pe_val:.1f}"                       if pe_val                      is not None else "",
                "price_source":       r.get("price_source", ""),
                "fail_reasons":       "; ".join(profile.fail_reasons),
            })
    print(f"  Logged: {csv_path}\n")


# ---------------------------------------------------------------------------
# Market-hours loop (--loop mode)
# ---------------------------------------------------------------------------

def main() -> None:
    args     = _parse_args()
    interval = max(10, args.interval)   # enforce 10-minute minimum — safety floor

    if not args.loop:
        # Single run — no market-hours check (user asked for it explicitly)
        run_screen()
        return

    print(f"\n{'═'*60}")
    print(f"  LOOP MODE — every {interval} min during NYSE hours (Mon–Fri 9:30–16:00 ET)")
    print(f"  Press Ctrl+C to stop.")
    print(f"{'═'*60}\n")

    import schedule as _sched

    def _guarded_run():
        if _is_market_open():
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Market open — running screen …")
            run_screen()
        else:
            now_et  = datetime.datetime.now(_ET)
            weekday = now_et.strftime("%A")
            today   = now_et.date()
            if today in _get_holidays():
                reason = f"market holiday ({today})"
            elif now_et.weekday() >= 5:
                reason = "weekend"
            elif now_et.time() < datetime.time(9, 30):
                reason = f"pre-market (opens at 9:30 AM ET)"
            else:
                reason = "market closed (after 4:00 PM ET)"
            print(
                f"[{now_et.strftime('%H:%M:%S')} ET] Skipping — {reason}. "
                f"Next open: {_next_open_str()}"
            )

    # Schedule on the chosen interval and fire once immediately
    _sched.every(interval).minutes.do(_guarded_run)
    _guarded_run()   # run right away so you don't wait interval minutes on startup

    try:
        while True:
            _sched.run_pending()
            time.sleep(15)
    except KeyboardInterrupt:
        print("\nScheduler stopped. Goodbye.")


if __name__ == "__main__":
    main()

