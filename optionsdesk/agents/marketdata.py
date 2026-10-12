"""MarketDataAgent: option chains, underlying prices, IV/Greeks from IBKR (or a simulator).

Live and delayed data are kept apart: every quote carries a DataQuality tag
(LIVE / DELAYED / ESTIMATED) taken from IBKR's per-tick marketDataType.
"""

from __future__ import annotations

import asyncio
import math
import time
import zlib
from datetime import date, timedelta
from typing import Any, Optional, Protocol, Sequence

from ..backtest import synthetic_chain
from ..events import AgentStatus, ChainSnapshot, DataQuality, OptionQuote, Quote
from ..settings import Settings
from .base import BaseAgent, Context

MIN_HISTORY_DAYS = 60


class MarketDataSource(Protocol):
    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...
    def is_connected(self) -> bool: ...
    async def get_quote(self, symbol: str) -> tuple[float, DataQuality]: ...
    async def get_chain(self, symbol: str, dte_min: int, dte_max: int) -> tuple[float, list[OptionQuote], DataQuality]: ...
    async def get_daily_closes(self, symbol: str, days: int) -> list[tuple[date, float]]: ...
    async def get_iv_history(self, symbol: str, days: int) -> list[tuple[date, float]]: ...


# ---------------------------------------------------------------------------
# Pacing
# ---------------------------------------------------------------------------

class RateLimiter:
    """Spaces calls so at most `rate` start per second."""

    def __init__(self, rate: float) -> None:
        self._interval = 1.0 / rate if rate > 0 else 0.0
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._next - now
            self._next = max(now, self._next) + self._interval
        if wait > 0:
            await asyncio.sleep(wait)


# ---------------------------------------------------------------------------
# Simulated source (no broker): deterministic random walk + modelled chain
# ---------------------------------------------------------------------------

class SimulatedMarketData:
    """Never touches IBKR. Prices random-walk; chain is Black-Scholes-modelled and tagged ESTIMATED."""

    def __init__(self, base_vol: float = 0.28, seed: int = 0) -> None:
        self._connected = False
        self._vol = base_vol
        self._seed = seed
        self._spot: dict[str, float] = {}
        self._rng: dict[str, Any] = {}

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def _state(self, symbol: str):
        import random
        if symbol not in self._rng:
            h = zlib.crc32(symbol.encode()) ^ self._seed
            self._rng[symbol] = random.Random(h)
            self._spot[symbol] = 50.0 + (h % 25000) / 100.0
        return self._rng[symbol]

    async def get_quote(self, symbol: str) -> tuple[float, DataQuality]:
        rng = self._state(symbol)
        self._spot[symbol] *= math.exp(rng.gauss(0, self._vol / math.sqrt(252 * 78)))  # ~5-min step
        return round(self._spot[symbol], 2), DataQuality.ESTIMATED

    async def get_chain(self, symbol: str, dte_min: int, dte_max: int):
        spot, _ = await self.get_quote(symbol)
        today = date.today()
        exps = [today + timedelta(d) for d in range(dte_min, dte_max + 1) if (today + timedelta(d)).weekday() == 4]
        chain = synthetic_chain(symbol, spot, today, self._vol, exps[:4], "CP", (0.80, 1.20), 0.045, [], 1.0)
        return spot, chain, DataQuality.ESTIMATED

    async def get_daily_closes(self, symbol: str, days: int) -> list[tuple[date, float]]:
        rng = self._state(symbol)
        px, out = self._spot[symbol], []
        d = date.today()
        for _ in range(days):
            out.append((d, px))
            px /= math.exp(rng.gauss(0, self._vol / math.sqrt(252)))
            d -= timedelta(1)
        return [x for x in reversed(out) if x[0].weekday() < 5]

    async def get_iv_history(self, symbol: str, days: int) -> list[tuple[date, float]]:
        rng = self._state(symbol)
        iv, d, out = self._vol, date.today(), []
        for _ in range(days):
            out.append((d, max(0.08, iv)))
            iv += 0.15 * (self._vol - iv) + rng.gauss(0, 0.02)
            d -= timedelta(1)
        return [x for x in reversed(out) if x[0].weekday() < 5]


# ---------------------------------------------------------------------------
# IBKR source
# ---------------------------------------------------------------------------

def _finite(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def quality_from_market_data_type(mdt: Any) -> DataQuality:
    """IBKR marketDataType: 1 live, 2 frozen, 3 delayed, 4 delayed-frozen."""
    return DataQuality.LIVE if mdt == 1 else DataQuality.DELAYED


def parse_option_ticker(symbol: str, contract: Any, ticker: Any, expiry: date) -> OptionQuote:
    """Build an OptionQuote from an ib_insync Ticker (duck-typed so it can be mocked)."""
    bid, ask, last = _finite(getattr(ticker, "bid", None)), _finite(getattr(ticker, "ask", None)), _finite(getattr(ticker, "last", None))
    g = getattr(ticker, "modelGreeks", None) or getattr(ticker, "lastGreeks", None)
    right = contract.right
    oi_raw = getattr(ticker, "callOpenInterest" if right == "C" else "putOpenInterest", None)
    oi, vol = _finite(oi_raw), _finite(getattr(ticker, "volume", None))
    iv = _finite(getattr(g, "impliedVol", None)) if g else None
    return OptionQuote(
        symbol=symbol, expiry=expiry, strike=float(contract.strike), right=right,
        bid=bid if bid and bid > 0 else None, ask=ask if ask and ask > 0 else None,
        last=last if last and last > 0 else None,
        iv=iv if iv and iv > 0 else None,
        delta=_finite(getattr(g, "delta", None)) if g else None,
        gamma=_finite(getattr(g, "gamma", None)) if g else None,
        theta=_finite(getattr(g, "theta", None)) if g else None,
        vega=_finite(getattr(g, "vega", None)) if g else None,
        open_interest=int(oi) if oi is not None and oi >= 0 else None,
        volume=int(vol) if vol is not None and vol >= 0 else None,
        quality=quality_from_market_data_type(getattr(ticker, "marketDataType", 3)),
    )


class IBKRMarketData:
    """Read-only ib_insync connection. Streams market data in small batches to respect line limits.

    Not exercised against a live TWS in tests (see README "Unverified")."""

    MAX_EXPIRIES = 3
    MAX_STRIKES_PER_EXPIRY = 30

    def __init__(self, settings: Settings) -> None:
        from ib_insync import IB
        self._s = settings
        self._ib = IB()
        self._limiter = RateLimiter(settings.ibkr.max_requests_per_second)
        self._stocks: dict[str, Any] = {}

    async def connect(self) -> None:
        c = self._s.ibkr
        await self._ib.connectAsync(c.host, self._s.ibkr_port, clientId=c.client_id,
                                    timeout=c.connect_timeout, readonly=True)
        self._ib.reqMarketDataType(c.market_data_type)

    async def disconnect(self) -> None:
        if self._ib.isConnected():
            self._ib.disconnect()

    def is_connected(self) -> bool:
        return bool(self._ib.isConnected())

    async def _stock(self, symbol: str):
        from ib_insync import Stock
        if symbol not in self._stocks:
            st = Stock(symbol, "SMART", "USD")
            await self._limiter.acquire()
            q = await self._ib.qualifyContractsAsync(st)
            if not q:
                raise LookupError(f"cannot qualify {symbol}")
            self._stocks[symbol] = st
        return self._stocks[symbol]

    async def get_quote(self, symbol: str) -> tuple[float, DataQuality]:
        st = await self._stock(symbol)
        await self._limiter.acquire()
        t = self._ib.reqMktData(st, "", False, False)
        try:
            await asyncio.sleep(2.0)
            price = _finite(t.marketPrice()) or _finite(t.last) or _finite(t.close)
        finally:
            self._ib.cancelMktData(st)
        if not price or price <= 0:
            raise LookupError(f"no price for {symbol}")
        return price, quality_from_market_data_type(getattr(t, "marketDataType", 3))

    async def get_chain(self, symbol: str, dte_min: int, dte_max: int):
        from ib_insync import Option
        st = await self._stock(symbol)
        spot, _ = await self.get_quote(symbol)
        await self._limiter.acquire()
        chains = await self._ib.reqSecDefOptParamsAsync(st.symbol, "", st.secType, st.conId)
        if not chains:
            return spot, [], DataQuality.DELAYED
        chain = next((c for c in chains if c.exchange == "SMART"), chains[0])
        today = date.today()
        exps = sorted(e for e in chain.expirations
                      if dte_min <= (date(int(e[:4]), int(e[4:6]), int(e[6:])) - today).days <= dte_max)[: self.MAX_EXPIRIES]
        strikes = sorted(k for k in chain.strikes if 0.85 * spot <= k <= 1.15 * spot)
        if len(strikes) > self.MAX_STRIKES_PER_EXPIRY:
            strikes = sorted(sorted(strikes, key=lambda k: abs(k - spot))[: self.MAX_STRIKES_PER_EXPIRY])

        contracts = [Option(symbol, e, k, r, "SMART", tradingClass=chain.tradingClass)
                     for e in exps for k in strikes for r in ("P", "C")]
        await self._limiter.acquire()
        qualified = [c for c in await self._ib.qualifyContractsAsync(*contracts) if c is not None and c.conId]
        quotes: list[OptionQuote] = []
        batch = self._s.ibkr.batch_size
        for i in range(0, len(qualified), batch):
            part = qualified[i: i + batch]
            tickers = []
            for c in part:
                await self._limiter.acquire()
                tickers.append(self._ib.reqMktData(c, "100,101,106", False, False))
            await asyncio.sleep(self._s.ibkr.mktdata_wait_seconds)  # let model greeks arrive
            for c, t in zip(part, tickers):
                self._ib.cancelMktData(c)
                exp = date(int(c.lastTradeDateOrContractMonth[:4]), int(c.lastTradeDateOrContractMonth[4:6]),
                           int(c.lastTradeDateOrContractMonth[6:8]))
                quotes.append(parse_option_ticker(symbol, c, t, exp))
        quality = DataQuality.LIVE if quotes and all(q.quality is DataQuality.LIVE for q in quotes) else DataQuality.DELAYED
        return spot, quotes, quality

    async def _bars(self, symbol: str, days: int, what: str) -> list[tuple[date, float]]:
        st = await self._stock(symbol)
        await self._limiter.acquire()
        bars = await self._ib.reqHistoricalDataAsync(
            st, endDateTime="", durationStr=f"{min(days, 365)} D", barSizeSetting="1 day",
            whatToShow=what, useRTH=True, formatDate=1)
        out = []
        for b in bars or []:
            v = _finite(b.close)
            if v and v > 0:
                out.append((b.date if isinstance(b.date, date) else date.fromisoformat(str(b.date)[:10]), v))
        return out

    async def get_daily_closes(self, symbol: str, days: int) -> list[tuple[date, float]]:
        return await self._bars(symbol, days, "TRADES")

    async def get_iv_history(self, symbol: str, days: int) -> list[tuple[date, float]]:
        return await self._bars(symbol, days, "OPTION_IMPLIED_VOLATILITY")


def make_source(settings: Settings) -> MarketDataSource:
    if settings.data_source == "simulated":
        return SimulatedMarketData()
    return IBKRMarketData(settings)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class MarketDataAgent(BaseAgent):
    name = "market_data"

    def __init__(self, ctx: Context, source: MarketDataSource) -> None:
        super().__init__(ctx)
        self.source = source
        self._chain_cache: dict[str, tuple[float, ChainSnapshot]] = {}
        self._bootstrapped: set[str] = set()

    async def close(self) -> None:
        await self.source.disconnect()

    def _dte_window(self) -> tuple[int, int]:
        w, s = self.settings.wheel, self.settings.straddle
        return min(w.dte_min, s.dte_min), max(w.dte_max, s.dte_max)

    async def _bootstrap(self, symbol: str) -> None:
        """Seed price/IV history so IV rank and realized vol work from day one."""
        if symbol in self._bootstrapped:
            return
        self._bootstrapped.add(symbol)
        today = self.ctx.today()
        if len(self.db.closes(symbol, until=today)) < MIN_HISTORY_DAYS:
            try:
                for d, px in await self.source.get_daily_closes(symbol, 260):
                    self.db.upsert_close(symbol, d, px)
            except Exception as exc:  # noqa: BLE001
                self.log.warning("%s: price history bootstrap failed: %s", symbol, exc)
        if len(self.db.iv_history(symbol, until=today)) < MIN_HISTORY_DAYS:
            try:
                for d, iv in await self.source.get_iv_history(symbol, 260):
                    self.db.upsert_iv(symbol, d, iv)
            except Exception as exc:  # noqa: BLE001
                self.log.warning("%s: IV history bootstrap failed: %s", symbol, exc)

    async def poll_symbol(self, symbol: str) -> None:
        await self._bootstrap(symbol)
        cached = self._chain_cache.get(symbol)
        if cached and time.monotonic() - cached[0] < self.settings.ibkr.quote_cache_ttl_seconds:
            return
        spot, quality = await self.source.get_quote(symbol)
        self.db.upsert_close(symbol, self.ctx.today(), spot)
        quote = Quote(symbol=symbol, price=spot, quality=quality)
        self.bus.publish(quote)
        dmin, dmax = self._dte_window()
        spot2, quotes, cq = await self.source.get_chain(symbol, dmin, dmax)
        snap = quote.derive(ChainSnapshot, symbol=symbol, underlying_price=spot2 or spot, quotes=quotes, quality=cq)
        self._chain_cache[symbol] = (time.monotonic(), snap)  # type: ignore[assignment]
        self.bus.publish(snap)

    async def poll_all(self) -> int:
        ok = 0
        for sym in self.settings.tickers:
            try:
                await self.poll_symbol(sym)
                ok += 1
            except Exception as exc:  # noqa: BLE001 - one ticker must not stop the rest
                self.degrade(exc, f"polling {sym}")
                if not self.source.is_connected():
                    break
        return ok

    async def run(self) -> None:
        backoff = self.settings.ibkr.reconnect_backoff_seconds
        cap = self.settings.ibkr.reconnect_max_backoff_seconds
        while True:
            if not self.source.is_connected():
                try:
                    await self.source.connect()
                    backoff = self.settings.ibkr.reconnect_backoff_seconds
                    self.beat(AgentStatus.RUNNING)
                except Exception as exc:  # noqa: BLE001
                    self.degrade(exc, "connect")
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, cap)
                    continue
            ok = await self.poll_all()
            if ok == len(self.settings.tickers):
                self.beat(AgentStatus.RUNNING, ran=True)
            elif ok:
                self.beat(ran=True)
            await asyncio.sleep(self.settings.poll_interval_seconds)
