"""
wheel_universe.py — One-time broad universe scan to build a shortlist,
instead of hand-maintaining wheel_watchlist.py's ~44 names.

Pipeline
--------
1. get_universe_tickers(): pull the first N tickers from SEC EDGAR's own
   company_tickers.json, which is roughly market-cap descending (no paid
   index-constituent feed needed -- iShares/Wikipedia-style sources are
   blocked by anti-bot measures from this environment).
2. bulk_fetch_prices(): get a recent close for all of them via yfinance's
   batched download (a handful of requests for hundreds of tickers), since
   IBKR's historical-data pacing limits (~60 req/10min) and Stooq's blocked
   CSV export make a per-symbol price lookup impractical at this scale.
   Fundamentals (FCF, margin, ROE) still come entirely from SEC EDGAR.
3. scan_universe(): screen every ticker concurrently (ThreadPoolExecutor)
   against SEC EDGAR, injecting the bulk-fetched price so
   edgar.get_fundamentals() doesn't also try a per-symbol price fetch.
   edgar.py's shared rate limiter caps the *aggregate* request rate across
   every worker thread, honoring SEC's ~10 req/s guidance regardless of how
   many threads are calling in concurrently.
4. Symbols with net margin >= UNIVERSE_MIN_NET_MARGIN and FCF > 0 become
   the shortlist, written to wheel_config.SHORTLIST_PATH -- wheel_watchlist.py
   loads from there automatically once it exists, so the rest of the
   dashboard (Fundamentals Screener's own FCF-yield/price-cap gate, Options
   Radar, Ranked Recommendations) needs no further changes to use it.

This is meant to be run occasionally (see scan_universe.py), not on every
dashboard page load -- screening ~1000+ names against SEC EDGAR takes
several minutes no matter how it's parallelized, since SEC's own rate
guidance is the bottleneck, not our code.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Optional

import edgar
import wheel_config as wc
from wheel_fundamentals import WheelFundamentalProfile, screen_wheel_candidate

logger = logging.getLogger(__name__)


def get_universe_tickers(n: int = wc.UNIVERSE_SIZE) -> list[str]:
    """First *n* tickers from SEC's own ticker map (roughly market-cap
    descending), filtered to plausible plain-vanilla common-stock symbols."""
    raw = edgar.get_ticker_universe(n)
    return [t for t in raw if _looks_like_common_stock(t)]


def _looks_like_common_stock(ticker: str) -> bool:
    """Drop obvious non-common-stock listings (warrants, units, rights,
    multiple share classes with odd suffixes) that would just fail the
    fundamentals screen anyway but cost a wasted EDGAR request first."""
    if not ticker or len(ticker) > 6:
        return False
    return all(c.isalpha() or c in ("-", ".") for c in ticker)


def bulk_fetch_prices(tickers: list[str], batch_size: int = wc.UNIVERSE_PRICE_BATCH_SIZE) -> dict:
    """Latest close price for every ticker in *tickers*, via yfinance's
    batched download (a handful of requests instead of one per symbol).

    yfinance is used here for price only -- fundamentals (FCF, margin, ROE)
    stay entirely on SEC EDGAR, unchanged from the rest of this project.
    A ticker missing from the result simply had no usable price data.
    """
    import yfinance as yf

    prices: dict = {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        try:
            df = yf.download(batch, period="5d", progress=False, group_by="ticker", threads=True)
        except Exception as exc:
            logger.warning("yfinance bulk download failed for batch starting at %d: %s", i, exc)
            continue

        for ticker in batch:
            try:
                if len(batch) == 1:
                    closes = df["Close"]
                else:
                    closes = df[ticker]["Close"]
                closes = closes.dropna()
                if not closes.empty:
                    prices[ticker] = float(closes.iloc[-1])
            except (KeyError, IndexError):
                continue

        logger.info("Bulk price batch %d-%d: got %d/%d prices", i, i + len(batch), len(prices), i + len(batch))

    return prices


def _meets_scan_criteria(profile: WheelFundamentalProfile) -> bool:
    return (
        profile.net_margin is not None
        and profile.net_margin >= wc.UNIVERSE_MIN_NET_MARGIN
        and profile.free_cash_flow is not None
        and profile.free_cash_flow > wc.UNIVERSE_MIN_FCF
    )


def scan_universe(
    tickers: Optional[list[str]] = None,
    max_workers: int = wc.UNIVERSE_SCAN_WORKERS,
    progress_callback=None,
) -> list[WheelFundamentalProfile]:
    """Screen every symbol in *tickers* (default: the full universe)
    concurrently against SEC EDGAR, with prices pre-fetched in bulk.

    *progress_callback*, if given, is called as `progress_callback(done, total)`
    after each symbol finishes -- used by scan_universe.py to show a live
    counter during the multi-minute run.
    """
    if tickers is None:
        tickers = get_universe_tickers()

    logger.info("Bulk-fetching prices for %d tickers via yfinance...", len(tickers))
    prices = bulk_fetch_prices(tickers)
    logger.info("Got prices for %d/%d tickers", len(prices), len(tickers))

    profiles: list[WheelFundamentalProfile] = []
    total = len(tickers)
    done = 0

    def _screen_one(symbol: str) -> WheelFundamentalProfile:
        info = edgar.get_fundamentals(symbol, latest_price=prices.get(symbol))
        return screen_wheel_candidate(symbol, info=info)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_screen_one, symbol): symbol for symbol in tickers}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                profiles.append(future.result())
            except Exception as exc:
                logger.warning("%s: universe scan failed — %s", symbol, exc)
            done += 1
            if progress_callback:
                progress_callback(done, total)

    return profiles


def build_shortlist(profiles: list[WheelFundamentalProfile]) -> list[str]:
    """Symbols meeting the net-margin/FCF bar, sorted by quality score."""
    shortlisted = [p for p in profiles if _meets_scan_criteria(p)]
    shortlisted.sort(key=lambda p: p.quality_score or 0, reverse=True)
    return [p.symbol for p in shortlisted]


def save_shortlist(symbols: list[str], path: str = wc.SHORTLIST_PATH) -> None:
    with open(path, "w") as f:
        json.dump(symbols, f, indent=2)


def load_shortlist(path: str = wc.SHORTLIST_PATH) -> Optional[list[str]]:
    """Returns None (not an empty list) if no shortlist has been generated
    yet, so callers can distinguish "not run" from "ran, nothing qualified"."""
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None
