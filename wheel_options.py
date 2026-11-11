"""
wheel_options.py — Option chain pull, Black-Scholes greeks, and the Phase 1
"richness" proxy for the options wheel selection engine.

Richness score (Phase 1 proxy)
-------------------------------
Compares the market's implied volatility at/near the target delta against
the underlying's trailing realized volatility. A market IV well above
realized vol suggests the option is "rich" (overpriced) relative to how the
stock has actually been moving -- a reasonable place to sell premium.

Phase 3 replaces the realized-vol comparison with a fair-value IV backed out
of a Merton jump-diffusion model calibrated to the same underlying's ATM
options (see wheel_jump_diffusion.py, currently stubbed).

Greeks
------
IBKR only reports model greeks/IV with a live OPRA market data subscription.
Where that's unavailable (delayed/no options data), we compute delta
ourselves via Black-Scholes using IBKR's IV if present, else trailing
realized vol as a substitute -- and mark the value as estimated so it's
never confused with a broker-reported greek.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
from scipy.stats import norm

import wheel_config as wc

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Black-Scholes
# ---------------------------------------------------------------------------

def bs_delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "put") -> float:
    """Black-Scholes delta. T is in years, sigma is annualized volatility.

    Returns the *signed* delta (negative for puts). Callers that want the
    assignment-probability proxy should use abs(delta).
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    if option_type == "put":
        return norm.cdf(d1) - 1.0
    return norm.cdf(d1)


def bs_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "put") -> float:
    """Black-Scholes theoretical price, used for the stress-exposure repricing
    in wheel_risk.py (Phase 1 proxy -- see that module's docstring)."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, (K - S) if option_type == "put" else (S - K))
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "put":
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


# ---------------------------------------------------------------------------
# Realized volatility
# ---------------------------------------------------------------------------

def realized_volatility(price_history, lookback_days: int = wc.RICHNESS_LOOKBACK_DAYS) -> Optional[float]:
    """Annualized close-to-close realized volatility from a price DataFrame
    with a 'Close' column (oldest-first), as returned by edgar.get_price_history."""
    if price_history is None or price_history.empty:
        return None
    closes = price_history["Close"].tail(lookback_days + 1)
    if len(closes) < 10:
        return None
    log_returns = np.log(closes / closes.shift(1)).dropna()
    if log_returns.empty:
        return None
    return float(log_returns.std() * math.sqrt(252))


# ---------------------------------------------------------------------------
# Candidate option contract (one strike/expiry on one underlying)
# ---------------------------------------------------------------------------

@dataclass
class OptionCandidate:
    symbol: str
    expiry: str            # YYYYMMDD
    strike: float
    dte: int
    underlying_price: float
    premium: float          # mid price, per share
    delta: float             # signed; puts are negative
    delta_is_estimated: bool  # True if computed via BS proxy rather than broker greeks
    market_iv: Optional[float]
    realized_vol: Optional[float]
    richness_ratio: Optional[float]   # market_iv / realized_vol
    overpriced: bool
    annualized_yield: float           # on collateral (strike * 100)
    premium_efficiency: float         # annualized_yield / |delta|
    capital_required: float           # strike * 100 per contract
    bid: Optional[float] = None
    ask: Optional[float] = None
    quote_type: str = "unknown"       # IBKR market data type: live / frozen / delayed / delayed-frozen
    open_interest: Optional[int] = None

    @property
    def spread_pct(self) -> Optional[float]:
        if self.bid is None or self.ask is None or self.premium <= 0:
            return None
        return (self.ask - self.bid) / self.premium

    @property
    def is_delayed(self) -> bool:
        return self.quote_type != "live"


# ib_insync Ticker.marketDataType → label.
QUOTE_TYPES = {1: "live", 2: "frozen", 3: "delayed", 4: "delayed-frozen"}


def has_two_sided_quote(t) -> bool:
    return (t.bid is not None and t.ask is not None and math.isfinite(t.bid) and math.isfinite(t.ask)
            and 0 < t.bid <= t.ask)


async def option_tickers(ib, contracts: list, types: tuple = (2, 4)) -> list:
    """Snapshot quotes that work in every session and subscription state.

    Market data type 2 returns live quotes while the market is open and the
    last live quote ("frozen") after the close — the same numbers TWS shows.
    If that yields no two-sided quote at all (no live options entitlement),
    retry with type 4 (15-minute delayed, frozen after the close). Type 1
    alone returns no bid/ask after the close, only a stale last trade; types
    3/4 alone return nothing after the close for a live-entitled account.

    IBKR applies the connection's CURRENT type when it answers, so callers
    running several of these concurrently must pass a single type and do
    any fallback as a separate wave (see fetch_put_candidates_batch).
    """
    tickers: list = []
    for t in types:
        ib.reqMarketDataType(t)
        tickers = await ib.reqTickersAsync(*contracts)
        if not contracts or any(has_two_sided_quote(x) for x in tickers):
            break
    return tickers


def third_friday(year: int, month: int) -> date:
    d = date(year, month, 1)
    return d + timedelta(days=(4 - d.weekday()) % 7 + 14)


def is_monthly_expiry(expiry: str, all_expiries) -> bool:
    """Standard monthly (third-Friday) expiry. When that Friday is a market
    holiday (e.g. Good Friday) the monthly expires the Thursday before, which
    shows up as the Thursday being listed and the Friday not."""
    d = datetime.strptime(expiry, "%Y%m%d").date()
    tf = third_friday(d.year, d.month)
    if d == tf:
        return True
    return d == tf - timedelta(days=1) and tf.strftime("%Y%m%d") not in all_expiries


def _dte(expiry: str) -> int:
    exp_date = datetime.strptime(expiry, "%Y%m%d").date()
    return (exp_date - datetime.now().date()).days


def build_candidate(
    symbol: str,
    expiry: str,
    strike: float,
    underlying_price: float,
    premium: float,
    market_iv: Optional[float],
    realized_vol: Optional[float],
    delta: Optional[float] = None,
    risk_free_rate: float = wc.RISK_FREE_RATE,
    bid: Optional[float] = None,
    ask: Optional[float] = None,
    quote_type: str = "unknown",
) -> Optional[OptionCandidate]:
    """Assemble an OptionCandidate, computing delta via Black-Scholes when the
    caller doesn't already have a broker-reported greek. Returns None if the
    contract is too far out/in to price sensibly (T<=0)."""
    dte = _dte(expiry)
    if dte <= 0:
        return None
    T = dte / 365.0

    delta_is_estimated = delta is None
    if delta is None:
        sigma = market_iv or realized_vol
        if not sigma:
            return None
        delta = bs_delta(underlying_price, strike, T, risk_free_rate, sigma, option_type="put")

    delta_abs = abs(delta)
    if delta_abs <= 0:
        return None

    richness_ratio = None
    overpriced = False
    if market_iv and realized_vol:
        richness_ratio = market_iv / realized_vol
        overpriced = richness_ratio >= wc.RICHNESS_OVERPRICED_RATIO

    annualized_yield = (premium / strike) * (365.0 / dte) if strike > 0 else 0.0
    premium_efficiency = annualized_yield / delta_abs if delta_abs > 0 else 0.0

    return OptionCandidate(
        symbol=symbol,
        expiry=expiry,
        strike=strike,
        dte=dte,
        underlying_price=underlying_price,
        premium=premium,
        delta=delta,
        delta_is_estimated=delta_is_estimated,
        market_iv=market_iv,
        realized_vol=realized_vol,
        richness_ratio=richness_ratio,
        overpriced=overpriced,
        annualized_yield=annualized_yield,
        premium_efficiency=premium_efficiency,
        capital_required=strike * 100.0,
        bid=bid,
        ask=ask,
        quote_type=quote_type,
    )


# ---------------------------------------------------------------------------
# IBKR option chain pull
# ---------------------------------------------------------------------------

async def _fetch_put_candidates_async(
    client,
    symbol: str,
    underlying_price: float,
    realized_vol: Optional[float],
    dte_min: int,
    dte_max: int,
    max_delta_abs: float,
    max_expiries: int,
    strike_band_low_pct: float,
    strike_band_high_pct: float,
    min_premium_per_contract: float,
    monthly_only: Optional[bool] = None,   # None → wc.MONTHLY_EXPIRIES_ONLY at call time
    min_open_interest: Optional[int] = None,   # None → wc.MIN_OPEN_INTEREST at call time
) -> list[OptionCandidate]:
    """One symbol end to end: quotes and filters, then the open-interest
    liquidity filter. Awaits directly rather than going through client.run()
    per call so it can be scheduled as a single coroutine."""
    survivors, _saw_quote = await _put_survivors_async(
        client, symbol, underlying_price, realized_vol, dte_min, dte_max, max_delta_abs, max_expiries,
        strike_band_low_pct, strike_band_high_pct, min_premium_per_contract, monthly_only)
    candidates = await _with_open_interest(client.ib, survivors, min_open_interest)
    candidates.sort(key=lambda c: c.premium_efficiency, reverse=True)
    return candidates


async def _put_survivors_async(
    client,
    symbol: str,
    underlying_price: float,
    realized_vol: Optional[float],
    dte_min: int,
    dte_max: int,
    max_delta_abs: float,
    max_expiries: int,
    strike_band_low_pct: float,
    strike_band_high_pct: float,
    min_premium_per_contract: float,
    monthly_only: Optional[bool] = None,   # None → wc.MONTHLY_EXPIRIES_ONLY at call time
    quote_types: tuple = (2, 4),
) -> tuple[list[tuple[OptionCandidate, object]], bool]:
    """Quote stage: every put clearing the quote, spread, delta and premium
    filters, paired with its qualified contract (for the open-interest pass),
    plus whether ANY contract had a two-sided quote (False suggests the
    requested market data type returned nothing)."""
    if monthly_only is None:
        monthly_only = wc.MONTHLY_EXPIRIES_ONLY
    from ib_insync import Stock, Option

    ib = client.ib
    underlying = Stock(symbol, "SMART", "USD")
    await ib.qualifyContractsAsync(underlying)

    try:
        chains = await ib.reqSecDefOptParamsAsync(
            underlying.symbol, "", underlying.secType, underlying.conId
        )
    except Exception as exc:
        logger.warning("%s: option chain lookup failed — %s", symbol, exc)
        return [], False
    if not chains:
        return [], False
    chain = next((c for c in chains if c.exchange == "SMART"), chains[0])

    today = datetime.now().date()
    expiries = sorted(
        e for e in chain.expirations
        if dte_min <= (datetime.strptime(e, "%Y%m%d").date() - today).days <= dte_max
        and (not monthly_only or is_monthly_expiry(e, chain.expirations))
    )[:max_expiries]   # cap requests per symbol -- each one costs a ~11s IBKR snapshot hold
    if not expiries:
        return [], False

    strikes = sorted(
        s for s in chain.strikes
        if strike_band_low_pct * underlying_price <= s <= strike_band_high_pct * underlying_price
    )

    survivors: list[tuple[OptionCandidate, object]] = []   # (candidate, qualified contract)
    saw_quote = False
    for expiry in expiries:
        contracts = [
            Option(symbol, expiry, strike, "P", "SMART", tradingClass=chain.tradingClass)
            for strike in strikes
        ]
        qualified = await ib.qualifyContractsAsync(*contracts) if contracts else []
        if not qualified:
            continue

        tickers = await option_tickers(ib, qualified, quote_types)
        for t in tickers:
            strike = t.contract.strike

            market_iv = t.modelGreeks.impliedVol if t.modelGreeks else None
            if market_iv is not None and not math.isfinite(market_iv):
                market_iv = None   # IBKR reports NaN greeks when a snapshot has no computed model, not "0 vol"
            broker_delta = t.modelGreeks.delta if t.modelGreeks else None
            if broker_delta is not None and not math.isfinite(broker_delta):
                broker_delta = None   # falls back to the Black-Scholes proxy below instead of a NaN delta

            # Premium must come from a real two-sided quote. The last trade is
            # never used: on an illiquid strike it can be days old and far from
            # what you could sell at now (e.g. last 0.75 vs bid 0.30 / ask 1.95).
            # NaN and IBKR's -1 "no quote" both fail the checks below.
            if not has_two_sided_quote(t):
                continue
            saw_quote = True
            mid = (t.bid + t.ask) / 2.0
            if (t.ask - t.bid) / mid > wc.MAX_SPREAD_PCT:
                continue   # too wide to trust the mid, and costly to trade

            candidate = build_candidate(
                symbol=symbol,
                expiry=expiry,
                strike=strike,
                underlying_price=underlying_price,
                premium=mid,
                market_iv=market_iv,
                realized_vol=realized_vol,
                delta=broker_delta,
                bid=t.bid,
                ask=t.ask,
                quote_type=QUOTE_TYPES.get(t.marketDataType, "unknown"),
            )
            if candidate is None:
                continue
            # Belt and suspenders: NaN fails every comparison (nan < x and
            # nan > x are both False), so a stray NaN premium/delta from
            # anywhere upstream would otherwise sail straight through both
            # filters below instead of being rejected by either of them.
            if not math.isfinite(candidate.premium) or not math.isfinite(candidate.delta):
                continue
            if abs(candidate.delta) > max_delta_abs:
                continue
            if candidate.premium * 100 < min_premium_per_contract:
                continue
            survivors.append((candidate, t.contract))
    return survivors, saw_quote


async def put_open_interest(ib, contracts: list) -> list[Optional[int]]:
    """Open interest for put contracts, or None where IBKR sent none.

    Open interest is a streamed tick (generic 101) and only arrives on market
    data type 1: the "frozen" type 2 used for after-hours quotes carries none.
    Setting the type and sending the requests happens without an await in
    between, so concurrent chain pulls can't interleave a different type.
    """
    if not contracts:
        return []
    ib.reqMarketDataType(1)
    tickers = [ib.reqMktData(c, "101", False, False) for c in contracts]
    await asyncio.sleep(wc.OI_WAIT_SECONDS)
    out: list[Optional[int]] = []
    for c, t in zip(contracts, tickers):
        ib.cancelMktData(c)
        oi = getattr(t, "putOpenInterest", None)
        out.append(int(oi) if oi is not None and math.isfinite(oi) and oi >= 0 else None)
    return out


async def _with_open_interest(ib, survivors: list, min_open_interest: Optional[int] = None) -> list[OptionCandidate]:
    """Attach each contract's put open interest and keep those at or above
    *min_open_interest*. A contract with no open-interest figure is dropped:
    an unknown chain is treated as a dead one."""
    if min_open_interest is None:
        min_open_interest = wc.MIN_OPEN_INTEREST
    kept = []
    ois = await put_open_interest(ib, [contract for _c, contract in survivors])
    for (cand, _contract), oi in zip(survivors, ois):
        cand.open_interest = oi
        if oi is not None and oi >= min_open_interest:
            kept.append(cand)
    return kept


def fetch_put_candidates(
    client,
    symbol: str,
    underlying_price: float,
    realized_vol: Optional[float],
    dte_min: int = wc.DEFAULT_DTE_MIN,
    dte_max: int = wc.DEFAULT_DTE_MAX,
    max_delta_abs: float = wc.MAX_DELTA_ABS,
    max_expiries: int = wc.MAX_EXPIRIES_PER_SYMBOL,
    strike_band_low_pct: float = wc.STRIKE_BAND_LOW_PCT,
    strike_band_high_pct: float = wc.STRIKE_BAND_HIGH_PCT,
    min_premium_per_contract: float = wc.MIN_PREMIUM_PER_CONTRACT,
) -> list[OptionCandidate]:
    """Pull the put chain for *symbol* from a connected wheel_ibkr.WheelIBKRClient,
    filter to `dte_min`-`dte_max` DTE, |delta| <= max_delta_abs, and premium
    >= min_premium_per_contract, and return OptionCandidate objects ranked
    by premium efficiency (best first).

    Requires live/delayed options market data (OPRA) for real IV/greeks --
    without a subscription, IBKR returns no greeks and we fall back to the
    Black-Scholes/realized-vol proxy (candidate.delta_is_estimated=True).

    Single-symbol convenience wrapper around _fetch_put_candidates_async --
    for more than a couple of symbols, use fetch_put_candidates_batch
    instead, which runs them concurrently rather than one at a time (each
    IBKR snapshot request has its own ~11s hold regardless of how few
    contracts are in it, so N symbols done serially costs N x that wait).
    """
    return client.run(lambda: _fetch_put_candidates_async(
        client, symbol, underlying_price, realized_vol,
        dte_min, dte_max, max_delta_abs, max_expiries,
        strike_band_low_pct, strike_band_high_pct, min_premium_per_contract,
    ))


def fetch_put_candidates_batch(
    client,
    requests: list[tuple],   # [(symbol, underlying_price, realized_vol), ...]
    dte_min: int = wc.DEFAULT_DTE_MIN,
    dte_max: int = wc.DEFAULT_DTE_MAX,
    max_delta_abs: float = wc.MAX_DELTA_ABS,
    max_expiries: int = wc.MAX_EXPIRIES_PER_SYMBOL,
    strike_band_low_pct: float = wc.STRIKE_BAND_LOW_PCT,
    strike_band_high_pct: float = wc.STRIKE_BAND_HIGH_PCT,
    min_premium_per_contract: float = wc.MIN_PREMIUM_PER_CONTRACT,
    min_open_interest: Optional[int] = None,   # None → wc.MIN_OPEN_INTEREST at call time
) -> dict:
    """Pull option chains for many symbols concurrently instead of one at a
    time. Returns {symbol: list[OptionCandidate]} -- a symbol is simply
    omitted (and logged) if its fetch raised, so one bad symbol doesn't
    abort the whole batch.
    """
    # IBKR answers each request with the connection's CURRENT market data
    # type, so types are never mixed while requests are in flight:
    #   wave 1: every symbol's quotes on type 2 (live / after-close frozen)
    #   wave 2: only if NO symbol got any two-sided quote (no live options
    #           entitlement): all of them again on type 4 (delayed)
    #   wave 3: one open-interest pass on type 1 for all survivors.
    # At most MAX_CONCURRENT_CHAINS symbols in flight per wave: each opens
    # ~25-50 option snapshots, and too many at once trips IBKR's
    # simultaneous market-data limit. Semaphore created lazily on the loop.
    sem: dict = {}
    seen_quote = {"any": False}

    async def _limited(symbol, price, rv, quote_type):
        s = sem.setdefault("s", asyncio.Semaphore(wc.MAX_CONCURRENT_CHAINS))
        async with s:
            survivors, saw = await _put_survivors_async(
                client, symbol, price, rv,
                dte_min, dte_max, max_delta_abs, max_expiries,
                strike_band_low_pct, strike_band_high_pct, min_premium_per_contract,
                quote_types=(quote_type,),
            )
            seen_quote["any"] |= saw
            return survivors

    def _wave(quote_type):
        factories = [
            (lambda symbol=symbol, price=price, rv=rv: _limited(symbol, price, rv, quote_type))
            for symbol, price, rv in requests
        ]
        return client.run_many(factories, timeout=max(120.0, 20.0 * len(requests)))

    results = _wave(2)
    if requests and not seen_quote["any"]:
        results = _wave(4)

    survivors_by_symbol: dict = {}
    for (symbol, _price, _rv), (result, exc) in zip(requests, results):
        if exc is not None:
            logger.warning("%s: option chain fetch failed — %s", symbol, exc)
            continue
        survivors_by_symbol[symbol] = result

    everyone = [pair for pairs in survivors_by_symbol.values() for pair in pairs]
    kept = set(map(id, client.run(lambda: _with_open_interest(client.ib, everyone, min_open_interest),
                                  timeout=60)))
    candidates_by_symbol: dict = {}
    for symbol, pairs in survivors_by_symbol.items():
        cands = [c for c, _contract in pairs if id(c) in kept]
        cands.sort(key=lambda c: c.premium_efficiency, reverse=True)
        candidates_by_symbol[symbol] = cands
    return candidates_by_symbol
