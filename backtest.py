"""
backtest.py — Historical back-test for the buy-the-sauce dip-buying strategy.

Methodology
-----------
Walk forward through each ticker's price history day by day and apply the
SAME signals used in production (RSI, MA50, MA200, 52-week range).  When a
dip is detected and no position is open, simulate a buy at the NEXT day's
open.  The position is closed at the first of:
  • TAKE_PROFIT_PCT above entry  (default +15%)
  • STOP_LOSS_PCT below entry    (default −7%)
  • End of the test window

One position per ticker at a time (mirrors MAX_POSITIONS logic at the
per-ticker level).  Position sizing is flat-dollar (equal weight) for
simplicity — portfolio-level sizing is handled by risk_manager.py in live
trading.

When --fundamentals is passed the screener's fundamental filters (FCF, P/E,
D/E, ROE, margin …) are also applied per ticker.  A ticker that fails the
fundamental screen is excluded from the simulation entirely — mirroring what
the live system does.  Results are shown side-by-side so you can see the
impact of the filter.

Usage
-----
  python3 backtest.py                       # dip signals only, 3-year window
  python3 backtest.py --fundamentals        # also apply fundamental filters
  python3 backtest.py --years 5             # extend look-back
  python3 backtest.py --tickers AAPL MSFT   # override watchlist
  python3 backtest.py --min-score 2         # relax dip threshold

Outputs
-------
  • Printed per-trade log and summary table to stdout
  • results/backtest_YYYY-MM-DD.txt   (snapshot)
  • results/backtest_trades.csv       (all trades, appendable)
"""

from __future__ import annotations

import argparse
import csv
import datetime
import logging
import os
import sys
import warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import config
from dip_detector import compute_rsi, compute_moving_average

# ── optional: colour terminal output if colorama is installed ─────────────
try:
    from colorama import Fore, Style, init as _colorama_init
    _colorama_init(autoreset=True)
    _GREEN  = Fore.GREEN
    _RED    = Fore.RED
    _YELLOW = Fore.YELLOW
    _RESET  = Style.RESET_ALL
except ImportError:
    _GREEN = _RED = _YELLOW = _RESET = ""

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEP  = "─" * 110
SEP2 = "═" * 110


# ---------------------------------------------------------------------------
# Signal detection (mirrors dip_detector.score_dip but purely on history slice)
# ---------------------------------------------------------------------------

def _detect_dip(closes: pd.Series, min_score: int) -> tuple[bool, int]:
    """Return (is_dip, score) using the same thresholds as dip_detector.py."""
    if len(closes) < config.MA_SLOW + 5:
        return False, 0

    price  = float(closes.iloc[-1])
    rsi    = compute_rsi(closes)
    ma50   = compute_moving_average(closes, config.MA_FAST)
    ma200  = compute_moving_average(closes, config.MA_SLOW)
    hi52   = float(closes.tail(252).max())
    lo52   = float(closes.tail(252).min())

    rsi_sig = (not np.isnan(rsi)) and rsi < config.RSI_OVERSOLD
    ma50_sig = (
        (not np.isnan(ma50)) and ma50 > 0
        and (ma50 - price) / ma50 >= config.DIP_FROM_MA50_PCT
    )
    ma200_sig = (not np.isnan(ma200)) and price < ma200
    rng = hi52 - lo52
    week52_sig = (rng > 0) and ((price - lo52) / rng <= config.WEEK52_LOWER_BAND)

    score = sum([rsi_sig, ma50_sig, ma200_sig, week52_sig])
    return score >= min_score, score


# ---------------------------------------------------------------------------
# Single-ticker back-test engine
# ---------------------------------------------------------------------------

def backtest_ticker(
    symbol: str,
    history: pd.DataFrame,
    min_score: int = config.MIN_DIP_SCORE,
    take_profit: float = config.TAKE_PROFIT_PCT,
    stop_loss: float = config.STOP_LOSS_PCT,
) -> list[dict]:
    """
    Walk forward through *history* and simulate all trades for one ticker.

    Returns a list of trade dicts (one per closed or open-at-end position).
    """
    if history is None or history.empty:
        return []

    # Ensure we have OHLCV columns (at minimum Close + Open for next-day entry)
    if "Close" not in history.columns:
        return []
    if "Open" not in history.columns:
        history = history.copy()
        history["Open"] = history["Close"]   # graceful fallback

    closes = history["Close"].dropna()
    opens  = history["Open"].reindex(closes.index)
    dates  = closes.index

    # We need at least MA_SLOW bars of warm-up before we can start scoring
    warmup = config.MA_SLOW + 5
    if len(closes) <= warmup + 1:
        return []

    trades: list[dict] = []
    position: dict | None = None   # currently open trade

    for i in range(warmup, len(dates) - 1):
        today      = dates[i]
        tomorrow   = dates[i + 1]
        entry_open = float(opens.iloc[i + 1])   # we enter at *next* day's open

        # ── manage open position ─────────────────────────────────────────────
        if position is not None:
            current_price = float(closes.iloc[i])
            entry_price   = position["entry_price"]
            gain          = (current_price - entry_price) / entry_price

            hit_tp = gain >= take_profit
            hit_sl = gain <= -stop_loss

            if hit_tp or hit_sl:
                exit_price  = float(opens.iloc[i + 1])  # execute at next open
                exit_gain   = (exit_price - entry_price) / entry_price
                position.update({
                    "exit_date":   str(tomorrow.date()),
                    "exit_price":  round(exit_price, 4),
                    "pct_gain":    round(exit_gain * 100, 2),
                    "outcome":     "TP" if hit_tp else "SL",
                    "hold_days":   (tomorrow - pd.Timestamp(position["entry_date"])).days,
                })
                trades.append(position)
                position = None
            continue   # don't enter a new position on the same bar we exit

        # ── check for dip signal ─────────────────────────────────────────────
        window = closes.iloc[: i + 1]
        is_dip, score = _detect_dip(window, min_score)

        if is_dip:
            position = {
                "symbol":       symbol,
                "entry_date":   str(today.date()),
                "entry_price":  round(entry_open, 4),
                "score":        score,
                "exit_date":    None,
                "exit_price":   None,
                "pct_gain":     None,
                "outcome":      None,
                "hold_days":    None,
            }

    # ── close any position still open at end of window ───────────────────────
    if position is not None:
        final_price = float(closes.iloc[-1])
        entry_price = position["entry_price"]
        gain        = (final_price - entry_price) / entry_price
        last_date   = dates[-1]
        position.update({
            "exit_date":  str(last_date.date()),
            "exit_price": round(final_price, 4),
            "pct_gain":   round(gain * 100, 2),
            "outcome":    "OPEN",
            "hold_days":  (last_date - pd.Timestamp(position["entry_date"])).days,
        })
        trades.append(position)

    return trades


# ---------------------------------------------------------------------------
# Portfolio-level aggregation
# ---------------------------------------------------------------------------

def _compute_stats(trades: list[dict]) -> dict:
    closed = [t for t in trades if t["outcome"] != "OPEN"]
    wins   = [t for t in closed if (t["pct_gain"] or 0) > 0]
    losses = [t for t in closed if (t["pct_gain"] or 0) <= 0]
    gains  = [t["pct_gain"] for t in closed if t["pct_gain"] is not None]

    n_total   = len(trades)
    n_closed  = len(closed)
    n_open    = n_total - n_closed
    n_wins    = len(wins)
    win_rate  = n_wins / n_closed if n_closed else 0.0
    avg_gain  = sum(gains) / len(gains) if gains else 0.0
    avg_win   = sum(t["pct_gain"] for t in wins)   / len(wins)   if wins   else 0.0
    avg_loss  = sum(t["pct_gain"] for t in losses) / len(losses) if losses else 0.0

    # Expectancy = win_rate × avg_win + (1 - win_rate) × avg_loss
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    # Simple compounded equity curve (equal weight, 100% in each trade)
    # Not a portfolio sim — used for rough CAGR estimate
    cum_mult = 1.0
    for g in sorted(
        [t["pct_gain"] for t in closed if t["pct_gain"] is not None]
    ):
        cum_mult *= (1 + g / 100)

    hold_days = [t["hold_days"] for t in closed if t["hold_days"] is not None]
    avg_hold  = sum(hold_days) / len(hold_days) if hold_days else 0.0

    return {
        "n_total":    n_total,
        "n_closed":   n_closed,
        "n_open":     n_open,
        "n_wins":     n_wins,
        "n_losses":   len(losses),
        "win_rate":   win_rate,
        "avg_gain":   avg_gain,
        "avg_win":    avg_win,
        "avg_loss":   avg_loss,
        "expectancy": expectancy,
        "cum_return": (cum_mult - 1) * 100,
        "avg_hold":   avg_hold,
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _pct_colour(val: float | None, positive_good: bool = True) -> str:
    if val is None:
        return "N/A"
    s = f"{val:+.2f}%"
    if positive_good:
        colour = _GREEN if val > 0 else (_RED if val < 0 else "")
    else:
        colour = _RED if val > 0 else (_GREEN if val < 0 else "")
    return f"{colour}{s}{_RESET}"


def _outcome_colour(outcome: str | None) -> str:
    if outcome == "TP":
        return f"{_GREEN}TP{_RESET}"
    if outcome == "SL":
        return f"{_RED}SL{_RESET}"
    if outcome == "OPEN":
        return f"{_YELLOW}OPEN{_RESET}"
    return outcome or "—"


# ---------------------------------------------------------------------------
# Main back-test runner
# ---------------------------------------------------------------------------

def _run_one_pass(
    watchlist: list[str],
    price_cache: dict[str, tuple],
    fund_cache: dict[str, bool],
    use_fundamentals: bool,
    years: int,
    min_score: int,
    take_profit: float,
    stop_loss: float,
) -> tuple[list[dict], dict[str, list[dict]], list[str]]:
    """
    Run one simulation pass (dip-only OR dip+fundamentals).

    Returns (all_trades, per_ticker_trades, excluded_symbols).
    price_cache and fund_cache are shared across passes to avoid re-fetching.
    """
    import edgar as _edgar
    from screener import screen_fundamental

    all_trades: list[dict] = []
    per_ticker: dict[str, list[dict]] = {}
    excluded: list[str] = []

    for symbol in watchlist:
        # ── price history (cached) ────────────────────────────────────────
        if symbol not in price_cache:
            hist, src = _edgar.get_price_history(symbol, period_years=years)
            price_cache[symbol] = (hist, src)
        hist, _src = price_cache[symbol]

        if hist is None or hist.empty:
            excluded.append(symbol)
            continue

        # ── fundamental filter (cached) ───────────────────────────────────
        if use_fundamentals:
            if symbol not in fund_cache:
                info    = _edgar.get_fundamentals(symbol)
                profile = screen_fundamental(symbol, info=info)
                fund_cache[symbol] = profile.passes
            if not fund_cache[symbol]:
                excluded.append(symbol)
                continue

        trades = backtest_ticker(symbol, hist, min_score, take_profit, stop_loss)
        per_ticker[symbol] = trades
        all_trades.extend(trades)

    return all_trades, per_ticker, excluded


def _build_trade_table(all_trades: list[dict], label: str) -> list[str]:
    """Return lines for the per-trade log table."""
    trade_hdr = (
        f"  {'#':<4} {'SYMBOL':<7} {'ENTRY DATE':<12} {'ENTRY $':>9} "
        f"{'EXIT DATE':<12} {'EXIT $':>9} {'GAIN':>8} {'OUTCOME':<8} {'DAYS':>5} {'SCORE'}"
    )
    rows = []
    for idx, t in enumerate(sorted(all_trades, key=lambda x: x["entry_date"]), 1):
        exit_px_str = f"{t['exit_price']:.2f}" if t["exit_price"] else "—"
        rows.append(
            f"  {idx:<4} {t['symbol']:<7} {t['entry_date']:<12} {t['entry_price']:>9.2f} "
            f"{(t['exit_date'] or '—'):<12} {exit_px_str:>9} "
            f"{_pct_colour(t['pct_gain']):>8} {_outcome_colour(t['outcome']):<18} "
            f"{(t['hold_days'] or 0):>5} {t['score']}/4"
        )
    return [SEP2, f"  TRADE LOG — {label}", SEP2, trade_hdr, SEP] + rows + [SEP]


def _build_ticker_table(per_ticker: dict[str, list[dict]], label: str) -> list[str]:
    """Return lines for the per-ticker summary table."""
    hdr = (
        f"  {'SYMBOL':<8} {'TRADES':>7} {'WINS':>5} {'WIN%':>7} "
        f"{'AVG GAIN':>10} {'AVG WIN':>9} {'AVG LOSS':>10} {'EXPECT':>9} {'CUM RET':>10}"
    )
    rows = []
    for sym, trades in sorted(per_ticker.items()):
        if not trades:
            continue
        s = _compute_stats(trades)
        rows.append(
            f"  {sym:<8} {s['n_total']:>7} {s['n_wins']:>5} {s['win_rate']:>6.0%}  "
            f"{_pct_colour(s['avg_gain']):>10} {_pct_colour(s['avg_win']):>9} "
            f"{_pct_colour(s['avg_loss'], positive_good=False):>10} "
            f"{_pct_colour(s['expectancy']):>9} {_pct_colour(s['cum_return']):>10}"
        )
    return [SEP2, f"  PER-TICKER — {label}", SEP2, hdr, SEP] + rows + [SEP]


def _stats_block(stats: dict, label: str, n_universe: int, n_with_data: int,
                 n_excluded: int, years: int) -> list[str]:
    """Return lines for one statistics block."""
    tp = stats.get("n_tp", 0)
    sl = stats.get("n_sl", 0)
    return [
        f"  ┌─ {label} {'─'*(54 - len(label))}┐",
        f"  │  {'Universe':<26}: {n_universe} tickers  ({n_with_data} with price data, {n_excluded} excluded)",
        f"  │  {'Total trades':<26}: {stats['n_total']}  ({stats['n_closed']} closed, {stats['n_open']} open)",
        f"  │  {'TP hits / SL hits':<26}: {tp} TP  /  {sl} SL",
        f"  │  {'Win rate':<26}: {stats['win_rate']:.1%}",
        f"  │  {'Avg gain per trade':<26}: {_pct_colour(stats['avg_gain'])}",
        f"  │  {'Avg win / Avg loss':<26}: {_pct_colour(stats['avg_win'])}  /  {_pct_colour(stats['avg_loss'], positive_good=False)}",
        f"  │  {'Expectancy per trade':<26}: {_pct_colour(stats['expectancy'])}",
        f"  │  {'Avg holding period':<26}: {stats['avg_hold']:.1f} days",
        f"  │  {'Cumulative return':<26}: {_pct_colour(stats['cum_return'])}",
        f"  └{'─'*57}┘",
    ]


def _comparison_table(stats_dip: dict, stats_fund: dict) -> list[str]:
    """Side-by-side comparison of dip-only vs dip+fundamentals."""
    def fmt(val, pct=True, good=True):
        if val is None:
            return "N/A"
        s = f"{val:+.2f}%" if pct else f"{val:.1f}"
        colour = (_GREEN if val > 0 else _RED) if good else (_RED if val > 0 else _GREEN)
        return f"{colour}{s}{_RESET}"

    rows = [
        SEP2,
        "  COMPARISON — Dip Signals Only  vs.  Dip + Fundamental Filter",
        SEP2,
        f"  {'Metric':<28} {'DIP ONLY':>16} {'DIP + FUND':>16}  {'DELTA':>12}",
        SEP,
    ]

    metrics = [
        ("Total trades",       "n_total",    False, True),
        ("Win rate",           "win_rate",   True,  True),
        ("Avg gain/trade",     "avg_gain",   True,  True),
        ("Avg win",            "avg_win",    True,  True),
        ("Avg loss",           "avg_loss",   True,  False),
        ("Expectancy",         "expectancy", True,  True),
        ("Avg hold (days)",    "avg_hold",   False, True),
        ("Cumulative return",  "cum_return", True,  True),
    ]

    for label, key, is_pct, positive_good in metrics:
        d = stats_dip.get(key, 0) or 0
        f = stats_fund.get(key, 0) or 0
        delta = f - d

        if is_pct:
            d_str     = fmt(d * 100 if key == "win_rate" else d, pct=True,  good=positive_good)
            f_str     = fmt(f * 100 if key == "win_rate" else f, pct=True,  good=positive_good)
            delta_str = fmt(delta * 100 if key == "win_rate" else delta, pct=True, good=positive_good)
            if key == "win_rate":
                d_str     = f"{d:.1%}"
                f_str     = f"{f:.1%}"
                delta_str = fmt(delta * 100, pct=True, good=positive_good)
        else:
            d_str     = f"{d:.0f}"
            f_str     = f"{f:.0f}"
            delta_str = fmt(delta, pct=False, good=positive_good) if delta != 0 else "—"

        rows.append(f"  {label:<28} {d_str:>16} {f_str:>16}  {delta_str:>12}")

    rows += [
        SEP,
        "  NOTE: Cumulative return = sequential equal-dollar trades (rough signal only).",
        "        A positive DELTA means the fundamental filter improved that metric.",
        SEP,
    ]
    return rows


def run_backtest(
    watchlist: list[str],
    years: int = 3,
    min_score: int = config.MIN_DIP_SCORE,
    take_profit: float = config.TAKE_PROFIT_PCT,
    stop_loss: float = config.STOP_LOSS_PCT,
    use_fundamentals: bool = False,
    verbose: bool = False,
) -> None:
    """Fetch history, run simulation(s), print results, save files."""
    import re

    date_str = datetime.date.today().isoformat()
    txt_path = os.path.join(RESULTS_DIR, f"backtest_{date_str}.txt")
    csv_path = os.path.join(RESULTS_DIR, "backtest_trades.csv")

    fund_label = "Dip + Fundamental Filter" if use_fundamentals else "Dip Signals Only"
    print(f"\n{'═'*60}")
    print(f"  BUY-THE-SAUCE BACK-TEST  ({date_str})")
    print(f"  Tickers        : {len(watchlist)}")
    print(f"  Window         : {years} year(s)")
    print(f"  Min dip score  : {min_score}/4   TP: +{take_profit:.0%}   SL: −{stop_loss:.0%}")
    print(f"  Fundamental filter: {'ON — running BOTH passes for comparison' if use_fundamentals else 'OFF (use --fundamentals to enable)'}")
    print(f"{'═'*60}\n")

    # ── shared caches so we only fetch price/fundamentals once ───────────────
    price_cache: dict = {}
    fund_cache:  dict = {}

    # ── Pass 1: dip signals only ─────────────────────────────────────────────
    print("  [Pass 1/2] Dip signals only …" if use_fundamentals else "  Fetching data …")
    import edgar as _edgar
    # Pre-fetch prices with progress output
    for symbol in watchlist:
        print(f"    {symbol} …", end="\r", flush=True)
        if symbol not in price_cache:
            hist, src = _edgar.get_price_history(symbol, period_years=years)
            price_cache[symbol] = (hist, src)
    print(" " * 40, end="\r")

    trades_dip, per_ticker_dip, excluded_dip = _run_one_pass(
        watchlist, price_cache, fund_cache,
        use_fundamentals=False,
        years=years, min_score=min_score,
        take_profit=take_profit, stop_loss=stop_loss,
    )

    # ── Pass 2: dip + fundamentals (only if requested) ───────────────────────
    trades_fund: list[dict] = []
    per_ticker_fund: dict   = {}
    excluded_fund: list[str] = []

    if use_fundamentals:
        print("  [Pass 2/2] Fetching fundamentals from SEC EDGAR …")
        for symbol in watchlist:
            print(f"    {symbol} …", end="\r", flush=True)
            if symbol not in fund_cache:
                info    = _edgar.get_fundamentals(symbol)
                from screener import screen_fundamental
                profile = screen_fundamental(symbol, info=info)
                fund_cache[symbol] = profile.passes
        print(" " * 40, end="\r")

        trades_fund, per_ticker_fund, excluded_fund = _run_one_pass(
            watchlist, price_cache, fund_cache,
            use_fundamentals=True,
            years=years, min_score=min_score,
            take_profit=take_profit, stop_loss=stop_loss,
        )
        print()

    if not trades_dip:
        print("  No trades generated. Try lowering --min-score or extending --years.")
        return

    # ── compute stats ─────────────────────────────────────────────────────────
    def _with_tp_sl(trades):
        s = _compute_stats(trades)
        s["n_tp"] = sum(1 for t in trades if t["outcome"] == "TP")
        s["n_sl"] = sum(1 for t in trades if t["outcome"] == "SL")
        return s

    stats_dip  = _with_tp_sl(trades_dip)
    stats_fund = _with_tp_sl(trades_fund) if use_fundamentals else {}

    n_with_data_dip  = len(per_ticker_dip)
    n_with_data_fund = len(per_ticker_fund)

    # ── build output sections ─────────────────────────────────────────────────
    header_lines = [
        f"BUY-THE-SAUCE BACK-TEST — {date_str}",
        f"  Min dip score : {min_score}/4",
        f"  Take-profit   : +{take_profit:.0%}   Stop-loss: −{stop_loss:.0%}",
        f"  Look-back     : {years} year(s)",
        f"  Tickers       : {', '.join(watchlist)}",
        "",
    ]

    # Trade logs
    table_dip  = _build_trade_table(trades_dip,  "DIP SIGNALS ONLY")
    table_fund = _build_trade_table(trades_fund, "DIP + FUNDAMENTAL FILTER") if use_fundamentals else []

    # Per-ticker summaries
    ticker_dip  = _build_ticker_table(per_ticker_dip,  "DIP SIGNALS ONLY")
    ticker_fund = _build_ticker_table(per_ticker_fund, "DIP + FUNDAMENTAL FILTER") if use_fundamentals else []

    # Stats blocks
    stats_sec = [SEP2, "  OVERALL STATISTICS", SEP2]
    stats_sec += _stats_block(stats_dip,  "DIP SIGNALS ONLY",
                               len(watchlist), n_with_data_dip,  len(excluded_dip),  years)
    if use_fundamentals:
        fund_excl_names = ", ".join(excluded_fund) if excluded_fund else "none"
        stats_sec += [""]
        stats_sec += _stats_block(stats_fund, "DIP + FUNDAMENTAL FILTER",
                                   len(watchlist), n_with_data_fund, len(excluded_fund), years)
        stats_sec += [f"  Fundamentally excluded: {fund_excl_names}"]

    stats_sec += [SEP]

    # Comparison table (only when fundamentals pass is run)
    comparison = _comparison_table(stats_dip, stats_fund) if use_fundamentals else []

    all_sections = (
        header_lines
        + [""] + table_dip
        + ([""] + table_fund if table_fund else [])
        + [""] + ticker_dip
        + ([""] + ticker_fund if ticker_fund else [])
        + [""] + stats_sec
        + ([""] + comparison if comparison else [])
    )
    output = "\n".join(all_sections) + "\n"
    print(output)

    # ── save .txt (ANSI-stripped) ─────────────────────────────────────────────
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    clean_output = ansi_escape.sub("", output)
    with open(txt_path, "w") as f:
        f.write(clean_output)
    print(f"  Saved : {txt_path}")

    # ── append CSV ────────────────────────────────────────────────────────────
    CSV_FIELDS = [
        "backtest_date", "symbol", "entry_date", "entry_price",
        "exit_date", "exit_price", "pct_gain", "outcome", "hold_days",
        "score", "years_window", "min_score", "take_profit_pct", "stop_loss_pct",
        "fundamental_filter",
    ]
    csv_exists = os.path.exists(csv_path)

    def _write_trades(trades, fund_filter_flag):
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if not csv_exists:
                writer.writeheader()
            for t in trades:
                writer.writerow({
                    "backtest_date":     date_str,
                    "symbol":            t["symbol"],
                    "entry_date":        t["entry_date"],
                    "entry_price":       t["entry_price"],
                    "exit_date":         t["exit_date"] or "",
                    "exit_price":        t["exit_price"] or "",
                    "pct_gain":          t["pct_gain"] or "",
                    "outcome":           t["outcome"] or "",
                    "hold_days":         t["hold_days"] or "",
                    "score":             t["score"],
                    "years_window":      years,
                    "min_score":         min_score,
                    "take_profit_pct":   f"{take_profit:.4f}",
                    "stop_loss_pct":     f"{stop_loss:.4f}",
                    "fundamental_filter": fund_filter_flag,
                })

    _write_trades(trades_dip, False)
    if use_fundamentals:
        _write_trades(trades_fund, True)

    print(f"  Logged: {csv_path}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    from run_screen import WATCHLIST as _DEFAULT_WATCHLIST

    parser = argparse.ArgumentParser(
        description="Back-test the buy-the-sauce dip-buying strategy on historical price data."
    )
    parser.add_argument(
        "--tickers", nargs="+", metavar="TICKER", default=None,
        help="Override watchlist (space-separated, e.g. --tickers AAPL MSFT NVDA)",
    )
    parser.add_argument(
        "--years", type=int, default=3,
        help="Years of price history to use (default: 3)",
    )
    parser.add_argument(
        "--min-score", type=int, default=config.MIN_DIP_SCORE, dest="min_score",
        help=f"Minimum dip score to trigger a simulated entry (default: {config.MIN_DIP_SCORE})",
    )
    parser.add_argument(
        "--take-profit", type=float, default=config.TAKE_PROFIT_PCT, dest="take_profit",
        help=f"Take-profit as a decimal (default: {config.TAKE_PROFIT_PCT})",
    )
    parser.add_argument(
        "--stop-loss", type=float, default=config.STOP_LOSS_PCT, dest="stop_loss",
        help=f"Stop-loss as a decimal (default: {config.STOP_LOSS_PCT})",
    )
    parser.add_argument(
        "--fundamentals", action="store_true",
        help=(
            "Also apply the fundamental screen (FCF, P/E, D/E, ROE …) and show "
            "a side-by-side comparison of dip-only vs dip+fundamentals performance."
        ),
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print extra debug information",
    )
    args = parser.parse_args()
    args.watchlist = args.tickers if args.tickers else _DEFAULT_WATCHLIST
    return args


if __name__ == "__main__":
    args = _parse_args()
    run_backtest(
        watchlist=args.watchlist,
        years=args.years,
        min_score=args.min_score,
        take_profit=args.take_profit,
        stop_loss=args.stop_loss,
        use_fundamentals=args.fundamentals,
        verbose=args.verbose,
    )
