"""
straddle_options.py — IBKR quote pull for at-the-money straddles (call + put, same strike and expiry).

Same rules as the put pull in wheel_options.py: snapshot quotes on market data type 2 (live while open, frozen after
the close) with a type-4 (delayed) wave only if nothing came back, premiums only from real two-sided quotes (never the
last trade), and one open-interest pass on type 1. Read-only: nothing here builds or sends an order.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import wheel_config as wc
from wheel_options import QUOTE_TYPES, has_two_sided_quote, is_monthly_expiry, option_tickers

logger = logging.getLogger(__name__)


@dataclass
class StraddleQuote:
    symbol: str
    expiry: str              # YYYYMMDD
    dte: int
    strike: float
    underlying_price: float
    call_bid: float
    call_ask: float
    put_bid: float
    put_ask: float
    call_iv: Optional[float] = None    # broker model IV, when IBKR sends greeks
    put_iv: Optional[float] = None
    quote_type: str = "unknown"
    call_open_interest: Optional[int] = None
    put_open_interest: Optional[int] = None

    @property
    def bid(self) -> float:
        return self.call_bid + self.put_bid

    @property
    def ask(self) -> float:
        return self.call_ask + self.put_ask

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    @property
    def spread_pct(self) -> float:
        return (self.ask - self.bid) / self.mid if self.mid > 0 else float("inf")

    @property
    def min_open_interest(self) -> Optional[int]:
        if self.call_open_interest is None or self.put_open_interest is None:
            return None
        return min(self.call_open_interest, self.put_open_interest)


def _finite(x) -> Optional[float]:
    return float(x) if x is not None and math.isfinite(x) else None


def _dte(expiry: str) -> int:
    return (datetime.strptime(expiry, "%Y%m%d").date() - datetime.now().date()).days


async def _straddle_survivors_async(client, symbol: str, underlying_price: float, dte_min: int, dte_max: int,
                                    max_expiries: int, strikes_per_side: int, quote_types: tuple = (2, 4),
                                    ) -> tuple[list[tuple[StraddleQuote, object, object]], bool]:
    """Per monthly expiry in the window, the strike nearest the stock price (McMillan's at-the-money straddle) whose call
    AND put both have two-sided quotes, paired with the two qualified contracts (for the open-interest pass), plus
    whether any quote came back."""
    from ib_insync import Option, Stock

    ib = client.ib
    underlying = Stock(symbol, "SMART", "USD")
    await ib.qualifyContractsAsync(underlying)
    try:
        chains = await ib.reqSecDefOptParamsAsync(underlying.symbol, "", underlying.secType, underlying.conId)
    except Exception as exc:
        logger.warning("%s: option chain lookup failed — %s", symbol, exc)
        return [], False
    if not chains:
        return [], False
    chain = next((c for c in chains if c.exchange == "SMART"), chains[0])

    today = datetime.now().date()
    expiries = sorted(e for e in chain.expirations
                      if dte_min <= (datetime.strptime(e, "%Y%m%d").date() - today).days <= dte_max
                      and is_monthly_expiry(e, chain.expirations))[:max_expiries]
    out: list[tuple[StraddleQuote, object, object]] = []
    saw_quote = False
    for expiry in expiries:
        near = sorted(chain.strikes, key=lambda s: abs(s - underlying_price))[: 2 * strikes_per_side]
        contracts = [Option(symbol, expiry, k, right, "SMART", tradingClass=chain.tradingClass)
                     for k in near for right in ("C", "P")]
        qualified = await ib.qualifyContractsAsync(*contracts) if contracts else []
        qualified = [c for c in qualified if c is not None and getattr(c, "conId", 0)]
        if not qualified:
            continue
        tickers = await option_tickers(ib, qualified, quote_types)
        legs: dict[float, dict[str, object]] = {}
        for t in tickers:
            if not has_two_sided_quote(t):
                continue
            saw_quote = True
            legs.setdefault(t.contract.strike, {})[t.contract.right] = t
        pairs = [k for k, v in legs.items() if "C" in v and "P" in v]
        if not pairs:
            continue
        k = min(pairs, key=lambda s: abs(s - underlying_price))
        c, p = legs[k]["C"], legs[k]["P"]
        q = StraddleQuote(symbol=symbol, expiry=expiry, dte=_dte(expiry), strike=float(k),
                          underlying_price=underlying_price, call_bid=float(c.bid), call_ask=float(c.ask),
                          put_bid=float(p.bid), put_ask=float(p.ask),
                          call_iv=_finite(c.modelGreeks.impliedVol) if c.modelGreeks else None,
                          put_iv=_finite(p.modelGreeks.impliedVol) if p.modelGreeks else None,
                          quote_type=QUOTE_TYPES.get(c.marketDataType, "unknown"))
        out.append((q, c.contract, p.contract))
    return out, saw_quote


async def option_open_interest(ib, contracts: list) -> list[Optional[int]]:
    """Open interest per option contract (call or put), or None where IBKR sent none. Generic tick 101 arrives only
    on market data type 1; the type is set and every request sent with no await in between."""
    if not contracts:
        return []
    ib.reqMarketDataType(1)
    tickers = [ib.reqMktData(c, "101", False, False) for c in contracts]
    await asyncio.sleep(wc.OI_WAIT_SECONDS)
    out: list[Optional[int]] = []
    for c, t in zip(contracts, tickers):
        ib.cancelMktData(c)
        oi = getattr(t, "callOpenInterest" if c.right == "C" else "putOpenInterest", None)
        out.append(int(oi) if oi is not None and math.isfinite(oi) and oi >= 0 else None)
    return out


def iv_history_batch(client, requests: list[tuple[str, str]], per_request_timeout: float = 45.0) -> dict:
    """{symbol: daily implied-vol Series} for [(symbol, IBKR duration string), ...], MAX_CONCURRENT_CHAINS at a time.
    Historical-data requests only. Each request has its own timeout, so one slow symbol cannot stall the batch; a
    symbol whose request failed, timed out or returned nothing is omitted."""
    from wheel_vol import _iv_history
    sem: dict = {}

    async def _one(symbol, duration):
        async with sem.setdefault("s", asyncio.Semaphore(wc.MAX_CONCURRENT_CHAINS)):
            try:
                return await asyncio.wait_for(_iv_history(client.ib, symbol, duration), per_request_timeout)
            except asyncio.TimeoutError:
                logger.info("%s: IV history timed out after %.0fs", symbol, per_request_timeout)
                return None

    factories = [(lambda s=s, d=d: _one(s, d)) for s, d in requests]
    waves = -(-len(requests) // max(wc.MAX_CONCURRENT_CHAINS, 1))
    results = client.run_many(factories, timeout=60.0 + per_request_timeout * waves)
    out = {}
    for (symbol, _d), (res, exc) in zip(requests, results):
        if exc is None and res is not None and not res.empty:
            out[symbol] = res
        elif exc is not None:
            logger.info("%s: IV history failed — %s", symbol, exc)
    return out


def fetch_straddle_quotes_batch(client, requests: list[tuple], dte_min: int = wc.STRADDLE_DTE_MIN,
                                dte_max: int = wc.STRADDLE_DTE_MAX, max_expiries: int = 2,
                                strikes_per_side: int = 2) -> dict:
    """{symbol: [StraddleQuote, ...]} for [(symbol, underlying_price), ...], pulled concurrently (at most
    MAX_CONCURRENT_CHAINS at once). A symbol whose pull raised is omitted and logged. Open interest is attached to
    both legs but not filtered here: the recommendation layer reports why a straddle is rejected."""
    sem: dict = {}
    seen_quote = {"any": False}

    async def _limited(symbol, price, quote_type):
        s = sem.setdefault("s", asyncio.Semaphore(wc.MAX_CONCURRENT_CHAINS))
        async with s:
            res, saw = await _straddle_survivors_async(client, symbol, price, dte_min, dte_max, max_expiries,
                                                       strikes_per_side, quote_types=(quote_type,))
            seen_quote["any"] |= saw
            return res

    def _wave(quote_type):
        factories = [(lambda symbol=symbol, price=price: _limited(symbol, price, quote_type)) for symbol, price in requests]
        return client.run_many(factories, timeout=max(120.0, 20.0 * len(requests)))

    results = _wave(2)
    if requests and not seen_quote["any"]:
        results = _wave(4)

    found: dict = {}
    for (symbol, _price), (result, exc) in zip(requests, results):
        if exc is not None:
            logger.warning("%s: straddle quote fetch failed — %s", symbol, exc)
            continue
        found[symbol] = result
    everyone = [(q, c, p) for trip in found.values() for (q, c, p) in trip]
    contracts = [x for _q, c, p in everyone for x in (c, p)]
    ois = client.run(lambda: option_open_interest(client.ib, contracts), timeout=60) if contracts else []
    for i, (q, _c, _p) in enumerate(everyone):
        q.call_open_interest, q.put_open_interest = ois[2 * i], ois[2 * i + 1]
    return {sym: [q for q, _c, _p in trip] for sym, trip in found.items()}
