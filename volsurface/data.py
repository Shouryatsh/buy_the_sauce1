"""
volsurface/data.py â€” where the option quotes come from.

  Yahoo Finance   free, ~15-minute delayed, works without TWS (and for any optionable ticker, SPX included).
                  Bid/ask are only meaningful while the market is open.
  IBKR            live, via TWS (read-only: contract details and market-data snapshots only â€” never an order).
                  SPX options need your options market-data entitlement; index options settle in cash.

Both return a QuoteSet with the same columns (surface.QUOTE_COLUMNS) so everything downstream is source
agnostic. Quotes are never repaired or invented: a missing bid/ask stays missing.
"""

from __future__ import annotations

import asyncio
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd

from .surface import NY, QUOTE_COLUMNS

logger = logging.getLogger(__name__)

INDEX_SYMBOLS = {"SPX", "NDX", "RUT", "VIX", "XSP", "DJX"}
YAHOO_INDEX = {s: f"^{s}" for s in INDEX_SYMBOLS}
DEFAULT_TARGET_DAYS = (7, 14, 30, 45, 60, 90, 120, 180, 270, 365, 540, 730)


@dataclass
class QuoteSet:
    symbol: str
    source: str
    asof: pd.Timestamp
    spot: Optional[float]
    quotes: pd.DataFrame
    all_expiries: list = field(default_factory=list)       # every listed expiry (YYYY-MM-DD)
    notes: list = field(default_factory=list)


def now_et() -> pd.Timestamp:
    return pd.Timestamp.now(tz=NY)


def root_of(contract_symbol: str, fallback: str) -> str:
    m = re.match(r"^([A-Z]+)\d{6}[CP]", contract_symbol or "")
    return m.group(1) if m else fallback


def default_expiries(listed: list[str], asof: pd.Timestamp, targets=DEFAULT_TARGET_DAYS, min_days: float = 3.0) -> list[str]:
    """One listed expiry nearest each target maturity (days), so the term structure is covered evenly."""
    asof_d = asof.tz_localize(None) if asof.tzinfo is None else asof.tz_convert(NY).tz_localize(None)
    days = {e: (pd.Timestamp(e) - asof_d.normalize()).days for e in listed}
    pool = {e: d for e, d in days.items() if d >= min_days}
    if not pool:
        return []
    chosen: list[str] = []
    for t in targets:
        e = min(pool, key=lambda x: abs(pool[x] - t))
        if e not in chosen and abs(pool[e] - t) <= max(5, 0.35 * t):
            chosen.append(e)
    return sorted(chosen)


# ---------------------------------------------------------------------------
# Yahoo
# ---------------------------------------------------------------------------

def _yahoo_symbol(symbol: str) -> str:
    return YAHOO_INDEX.get(symbol.upper(), symbol.upper().replace(".", "-"))


def yahoo_listing(symbol: str) -> tuple[list[str], Optional[float]]:
    import yfinance as yf
    t = yf.Ticker(_yahoo_symbol(symbol))
    expiries = list(t.options or [])
    spot = None
    try:
        h = t.history(period="5d")
        if h is not None and not h.empty:
            spot = float(h["Close"].iloc[-1])
    except Exception as exc:
        logger.info("%s: yahoo spot failed: %s", symbol, exc)
    return expiries, spot


def fetch_yahoo(symbol: str, expiries: Optional[list[str]] = None, workers: int = 6) -> QuoteSet:
    import yfinance as yf
    sym = symbol.upper()
    t = yf.Ticker(_yahoo_symbol(sym))
    listed = list(t.options or [])
    if not listed:
        raise ValueError(f"Yahoo lists no options for {sym}")
    asof = now_et()
    spot = None
    try:
        h = t.history(period="5d")
        spot = float(h["Close"].iloc[-1]) if h is not None and not h.empty else None
    except Exception:
        pass
    chosen = expiries or default_expiries(listed, asof)
    notes: list[str] = []

    def one(e: str) -> pd.DataFrame:
        ch = t.option_chain(e)
        frames = []
        for df, right in ((ch.calls, "C"), (ch.puts, "P")):
            if df is None or df.empty:
                continue
            d = pd.DataFrame({
                "expiry": e, "root": [root_of(cs, sym) for cs in df["contractSymbol"]], "strike": df["strike"].astype(float),
                "right": right, "bid": df["bid"].astype(float), "ask": df["ask"].astype(float),
                "volume": df.get("volume"), "open_interest": df.get("openInterest")})
            frames.append(d)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=QUOTE_COLUMNS)

    frames = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for e, fut in [(e, pool.submit(one, e)) for e in chosen if e in listed]:
            try:
                frames.append(fut.result())
            except Exception as exc:
                notes.append(f"{e}: {type(exc).__name__}: {exc}")
    quotes = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=QUOTE_COLUMNS)
    if not quotes.empty and not ((quotes["bid"] > 0) & (quotes["ask"] > 0)).any():
        notes.append("Yahoo returned no two-sided quotes â€” the market is probably closed; bid/ask only exist while it is open.")
    return QuoteSet(sym, "Yahoo (delayed)", asof, spot, quotes[QUOTE_COLUMNS], listed, notes)


# ---------------------------------------------------------------------------
# IBKR (read-only)
# ---------------------------------------------------------------------------

def pick_trading_class(params: list[dict], ymd: str, pm_preferred: bool = True) -> Optional[dict]:
    """The option class to quote for one expiry (YYYYMMDD): the PM-settled one (SPXW over SPX) when listed, SMART-routed first."""
    cands = [p for p in params if ymd in set(p["expirations"])]
    if not cands:
        return None
    cands.sort(key=lambda p: (0 if (pm_preferred and p["tradingClass"].upper().endswith("W")) else 1,
                              0 if p["exchange"].upper() == "SMART" else 1))
    return cands[0]


def thin_strikes(strikes: list[float], spot: float, band: tuple[float, float], target: int) -> list[float]:
    """At most about *target* strikes inside *band* x spot: the money region (88%-106% of spot) keeps most of the budget and is
    thinned evenly, the wings share the rest. Order and exact strikes are preserved (nothing is interpolated)."""
    ks = sorted({float(k) for k in strikes if band[0] * spot <= k <= band[1] * spot})
    if len(ks) <= target:
        return ks
    core = [k for k in ks if 0.88 * spot <= k <= 1.06 * spot]
    wings = [k for k in ks if not (0.88 * spot <= k <= 1.06 * spot)]
    n_core = min(len(core), max(int(target * 0.7), 1))
    n_wing = min(len(wings), max(target - n_core, 0))

    def even(xs, n):
        if n >= len(xs):
            return list(xs)
        if n <= 0:
            return []
        return [xs[i] for i in sorted(set(np.linspace(0, len(xs) - 1, n).round().astype(int)))]
    return sorted(even(core, n_core) + even(wings, n_wing))


def _clean(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if np.isfinite(v) and v > 0 else float("nan")


async def stream_quotes(ib, contracts: list, mdt: int, two_sided, batch: int = 80, max_wait: float = 5.0, settle: float = 1.0) -> list:
    """Quotes for many contracts by STREAMING market data in batches (subscribe, wait until the quotes stop arriving, read, cancel).
    Snapshot requests each carry a multi-second server-side hold, which made a 1,300-contract chain take minutes; a streamed batch
    is ready in a few seconds. Batches stay under the market-data line limit so nothing is rejected for exceeding it."""
    ib.reqMarketDataType(mdt)
    loop = asyncio.get_running_loop()
    out: list = []
    for i in range(0, len(contracts), batch):
        chunk = contracts[i:i + batch]
        tickers = [ib.reqMktData(c, "", False, False) for c in chunk]
        t0, last_change, seen = loop.time(), loop.time(), 0
        while loop.time() - t0 < max_wait:
            await asyncio.sleep(0.4)
            n = sum(1 for t in tickers if two_sided(t))
            if n != seen:
                seen, last_change = n, loop.time()
            if n >= 0.97 * len(tickers) or (n > 0 and loop.time() - last_change >= settle):
                break
        for c in chunk:
            ib.cancelMktData(c)
        out.extend(tickers)
    return out


async def _fetch_ibkr_async(client, symbol: str, expiries: list[str], spot: Optional[float], band, max_contracts: int,
                            progress=None) -> QuoteSet:
    from ib_insync import Index, Option, Stock
    from wheel_options import has_two_sided_quote

    ib = client.ib
    sym = symbol.upper()
    und = Index(sym, "CBOE", "USD") if sym in INDEX_SYMBOLS else Stock(sym, "SMART", "USD")
    if not await ib.qualifyContractsAsync(und):
        raise ValueError(f"IBKR does not know {sym}")
    raw = await ib.reqSecDefOptParamsAsync(und.symbol, "", und.secType, und.conId)
    params = [{"tradingClass": p.tradingClass, "exchange": p.exchange, "expirations": list(p.expirations),
               "strikes": list(p.strikes), "multiplier": p.multiplier} for p in raw]
    listed = sorted({f"{e[:4]}-{e[4:6]}-{e[6:]}" for p in params for e in p["expirations"]})
    notes: list[str] = []
    asof = now_et()
    if spot is None:
        ib.reqMarketDataType(2)
        tk = (await ib.reqTickersAsync(und))[0]
        px = tk.marketPrice()
        spot = float(px) if px and np.isfinite(px) and px > 0 else None
        if spot is None:
            bars = await ib.reqHistoricalDataAsync(und, "", "5 D", "1 day", "TRADES" if sym not in INDEX_SYMBOLS else "TRADES",
                                                   True, formatDate=1)
            spot = float(bars[-1].close) if bars else None
    if spot is None:
        raise ValueError(f"could not determine a price for {sym} to choose strikes")
    chosen = expiries or default_expiries(listed, asof)
    per_expiry = max(int(max_contracts / max(len(chosen), 1) / 1.15), 16)        # strikes per side; puts below the money + calls above
    sem = asyncio.Semaphore(4)

    async def contracts_of(e: str):
        """Every REAL contract of one expiry, from a single contract-details request (a guessed strike may not exist)."""
        ymd = e.replace("-", "")
        cls = pick_trading_class(params, ymd)
        if cls is None:
            return e, None, "not listed"
        proto = Option(sym, ymd, 0.0, "", cls["exchange"], cls.get("multiplier") or "100", "USD", tradingClass=cls["tradingClass"])
        async with sem:
            try:
                det = await asyncio.wait_for(ib.reqContractDetailsAsync(proto), 90)
            except asyncio.TimeoutError:
                return e, None, "contract list timed out"
        return e, [d.contract for d in det if d.contract.conId], ""

    qualified = []
    not_listed: list[str] = []
    for e, cs, why in await asyncio.gather(*(contracts_of(e) for e in chosen)):
        if not cs:
            if why == "not listed":
                not_listed.append(e)
            else:
                notes.append(f"{e}: {why or 'no contracts'}")
            continue
        puts = {c.strike: c for c in cs if c.right == "P"}
        calls = {c.strike: c for c in cs if c.right == "C"}
        for table, side in ((puts, "P"), (calls, "C")):
            lo, hi = (band[0], 1.04) if side == "P" else (0.96, band[1])          # OTM half plus a few at-the-money pairs for the forward
            for k in thin_strikes(list(table), spot, (lo, hi), per_expiry):
                qualified.append(table[k])
    if not_listed:
        shown = ", ".join(not_listed[:4]) + (f" … (+{len(not_listed) - 4} more)" if len(not_listed) > 4 else "")
        notes.append(f"{len(not_listed)} expiries are not in IBKR's option chain for {sym} and were skipped: {shown}")
    if progress:
        progress("contracts", len(qualified), len(qualified))
    if not qualified:
        raise ValueError(f"IBKR lists no matching option contracts for {sym}")
    async def wave(mdt: int) -> list:
        return await stream_quotes(ib, qualified, mdt, has_two_sided_quote)

    tickers = await wave(2)                                              # live while open, last live quote after the close
    if not any(has_two_sided_quote(t) for t in tickers):
        notes.append("no live two-sided quotes (options entitlement?) â€” retried with 15-minute delayed data")
        tickers = await wave(4)
    rows = []
    for t in tickers:
        c = t.contract
        e = c.lastTradeDateOrContractMonth
        rows.append({"expiry": f"{e[:4]}-{e[4:6]}-{e[6:8]}", "root": c.tradingClass or sym, "strike": float(c.strike),
                     "right": c.right, "bid": _clean(t.bid), "ask": _clean(t.ask), "volume": _clean(t.volume), "open_interest": np.nan})
    quotes = pd.DataFrame(rows, columns=QUOTE_COLUMNS)
    kind = {1: "live", 2: "frozen", 3: "delayed", 4: "delayed-frozen"}.get(getattr(tickers[0], "marketDataType", 0) if tickers else 0, "?")
    return QuoteSet(sym, f"IBKR ({kind})", asof, spot, quotes, listed, notes)


def fetch_ibkr(client, symbol: str, expiries: Optional[list[str]] = None, spot: Optional[float] = None,
               band: tuple[float, float] = (0.70, 1.30), max_contracts: int = 900) -> QuoteSet:
    if client is None:
        raise ValueError("not connected to IBKR")
    return client.run(lambda: _fetch_ibkr_async(client, symbol, expiries or [], spot, band, max_contracts), timeout=900)


