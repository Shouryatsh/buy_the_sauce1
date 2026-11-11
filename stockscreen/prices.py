"""
stockscreen/prices.py — daily price history for the whole universe in a few
batched yfinance calls, cached on disk.

Prices are *split-adjusted but not dividend-adjusted* closes (yfinance
auto_adjust=False 'Close'): they are the prices you actually see and trade at,
so ranges, percentiles and moving averages line up with a normal chart. All
high/low/percentile/indicator maths in the screener uses this one series.

The cache holds one pickle per symbol. A cached series younger than the TTL is
reused without any network call; otherwise the full history is re-downloaded
(rather than appended to) so a split or other corporate action can never leave
a stale unadjusted tail in the cache.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import CACHE_DIR

logger = logging.getLogger(__name__)

PRICE_DIR = CACHE_DIR / "prices"
COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _path(symbol: str) -> Path:
    return PRICE_DIR / f"{symbol.upper().replace('/', '_')}.pkl"


def load_cached(symbol: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_pickle(_path(symbol))
        return df if isinstance(df, pd.DataFrame) and not df.empty else None
    except (FileNotFoundError, ValueError, OSError, EOFError):
        return None


def cache_age_hours(symbol: str) -> Optional[float]:
    try:
        return (time.time() - _path(symbol).stat().st_mtime) / 3600
    except FileNotFoundError:
        return None


def _save(symbol: str, df: pd.DataFrame) -> None:
    PRICE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_pickle(_path(symbol))


def _clean(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Standard columns, tz-naive date index, ascending, no all-NaN closes."""
    if df is None or df.empty:
        return None
    df = df[[c for c in COLUMNS if c in df.columns]].copy()
    if "Close" not in df.columns:
        return None
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["Close"])
    df = df[df["Close"] > 0]
    return df if not df.empty else None


def _download(symbols: list[str], years: int) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    out: dict[str, pd.DataFrame] = {}
    raw = yf.download(symbols, period=f"{years}y", interval="1d", auto_adjust=False,
                      group_by="ticker", threads=True, progress=False)
    if raw is None or raw.empty:
        return out
    for sym in symbols:
        try:
            sub = raw[sym] if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            continue
        df = _clean(sub)
        if df is not None:
            out[sym] = df
    return out


def fetch_prices(symbols: list[str], years: int = 4, ttl_hours: float = 6.0, force: bool = False,
                 batch_size: int = 80, progress=None) -> dict[str, pd.DataFrame]:
    """Price history for *symbols*, from the cache when fresh, else downloaded
    in batches. Symbols that fail a batch are retried individually. Returns
    only symbols that have data."""
    result: dict[str, pd.DataFrame] = {}
    todo: list[str] = []
    for sym in symbols:
        age = cache_age_hours(sym)
        cached = load_cached(sym) if (age is not None and age < ttl_hours and not force) else None
        if cached is not None:
            result[sym] = cached
        else:
            todo.append(sym)

    done = 0
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        try:
            got = _download(batch, years)
        except Exception as exc:
            logger.warning("price batch %d failed: %s", i, exc)
            got = {}
        for sym in batch:
            if sym not in got:                 # individual retry, then fall back to a stale cache
                try:
                    got.update(_download([sym], years))
                except Exception as exc:
                    logger.info("%s: price download failed: %s", sym, exc)
            if sym in got:
                _save(sym, got[sym])
                result[sym] = got[sym]
            else:
                stale = load_cached(sym)
                if stale is not None:
                    result[sym] = stale        # better than nothing; its date is shown, never hidden
        done += len(batch)
        if progress:
            progress(done, len(todo))
    return result


def last_bar(df: pd.DataFrame) -> tuple[float, str]:
    return float(df["Close"].iloc[-1]), df.index[-1].strftime("%Y-%m-%d")
