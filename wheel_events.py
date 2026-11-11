"""
wheel_events.py — events that move implied volatility: scheduled macro
releases (FOMC, CPI, jobs report) from macro_calendar.csv, plus per-symbol
earnings dates from Yahoo Finance.

macro_calendar.csv holds only dates taken from the official schedules
(federalreserve.gov, bls.gov); add rows there for new years or other
releases (PPI, PCE, GDP...). Lines starting with '#' are comments.
"""

from __future__ import annotations

import logging
import time
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

MACRO_CSV = Path(__file__).with_name("macro_calendar.csv")
_EARNINGS_TTL = 6 * 3600
# symbol -> (fetched_at, [(date, timing)]) ; timing "BMO" (before the open),
# "AMC" (after the close) or "?" — decides which close is the last one
# before the announcement.
_earnings_cache: dict[str, tuple[float, list[tuple[date, str]]]] = {}


def load_macro() -> pd.DataFrame:
    """Columns: date (datetime.date), time_et, event, category, source."""
    try:
        df = pd.read_csv(MACRO_CSV, comment="#", dtype=str).dropna(subset=["date"])
    except FileNotFoundError:
        logger.warning("%s not found — no macro events.", MACRO_CSV)
        return pd.DataFrame(columns=["date", "time_et", "event", "category", "source"])
    df["date"] = pd.to_datetime(df["date"].str.strip()).dt.date
    return df.sort_values("date").reset_index(drop=True)


def _timing(ts: pd.Timestamp) -> str:
    """Yahoo stamps the announcement time in US/Eastern: morning = before the
    open, afternoon/evening = after the close. Midnight means 'not given'."""
    if ts.hour == 0 and ts.minute == 0:
        return "?"
    return "BMO" if ts.hour < 12 else "AMC"


def earnings_events(symbol: str) -> list[tuple[date, str]]:
    """All known (date, timing) earnings events for *symbol*, oldest first,
    cached 6 hours. Empty when Yahoo has nothing (ETFs such as SMH, or a
    failed lookup)."""
    hit = _earnings_cache.get(symbol)
    if hit and time.time() - hit[0] < _EARNINGS_TTL:
        return hit[1]
    events: dict[date, str] = {}
    try:
        import yfinance as yf
        df = yf.Ticker(symbol).get_earnings_dates(limit=28)   # ~6-7 years of quarters
        if df is not None and not df.empty:
            for ts in df.index:
                ts = pd.Timestamp(ts)
                events[ts.date()] = _timing(ts)
    except Exception as exc:
        logger.info("%s: earnings lookup failed: %s", symbol, exc)
    out = sorted(events.items())
    _earnings_cache[symbol] = (time.time(), out)
    return out


def earnings_dates(symbol: str) -> tuple[list[date], list[date]]:
    """(past, upcoming) earnings dates for *symbol*."""
    today = date.today()
    ds = [d for d, _t in earnings_events(symbol)]
    return [d for d in ds if d < today], [d for d in ds if d >= today]


def next_earnings(symbol: str) -> Optional[date]:
    _past, upcoming = earnings_dates(symbol)
    return upcoming[0] if upcoming else None


def earnings_before(symbol: str, expiry: date) -> Optional[date]:
    """First earnings date between today and *expiry* (inclusive), else None."""
    d = next_earnings(symbol)
    return d if d is not None and d <= expiry else None


def events_between(symbols: list[str], start: date, end: date) -> pd.DataFrame:
    """Macro events plus earnings for *symbols* in [start, end].
    Columns: date, time_et, event, category, symbol, scope."""
    macro = load_macro()
    macro = macro[(macro["date"] >= start) & (macro["date"] <= end)].copy()
    macro["symbol"] = ""
    macro["scope"] = "Market-wide"
    rows = []
    for sym in symbols:
        past, upcoming = earnings_dates(sym)
        for d in past + upcoming:
            if start <= d <= end:
                rows.append(dict(date=d, time_et="", event=f"{sym} earnings", category="Earnings",
                                 symbol=sym, scope="Single stock", source="Yahoo Finance"))
    out = pd.concat([macro, pd.DataFrame(rows)], ignore_index=True) if rows else macro
    return out.sort_values(["date", "category"]).reset_index(drop=True)


def days_until(d: date) -> int:
    return (d - date.today()).days
