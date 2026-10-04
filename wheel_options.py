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

import logging
import math
from dataclasses import dataclass
from datetime import datetime
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
) -> list[OptionCandidate]:
    """Async core of the option chain pull -- awaits directly rather than
    going through client.run() per call, so a whole symbol's worth of work
    can be scheduled as a single coroutine (used by both the single-symbol
    and the concurrent batch entry points below)."""
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
        return []
    if not chains:
        return []
    chain = next((c for c in chains if c.exchange == "SMART"), chains[0])

    today = datetime.now().date()
    expiries = sorted(
        e for e in chain.expirations
        if dte_min <= (datetime.strptime(e, "%Y%m%d").date() - today).days <= dte_max
    )[:max_expiries]   # cap requests per symbol -- each one costs a ~11s IBKR snapshot hold
    if not expiries:
        return []

    strikes = sorted(
        s for s in chain.strikes
        if strike_band_low_pct * underlying_price <= s <= strike_band_high_pct * underlying_price
    )

    candidates: list[OptionCandidate] = []
    for expiry in expiries:
        contracts = [
            Option(symbol, expiry, strike, "P", "SMART", tradingClass=chain.tradingClass)
            for strike in strikes
        ]
        qualified = await ib.qualifyContractsAsync(*contracts) if contracts else []
        if not qualified:
            continue

        tickers = await ib.reqTickersAsync(*qualified)
        for t in tickers:
            strike = t.contract.strike

            market_iv = t.modelGreeks.impliedVol if t.modelGreeks else None
            if market_iv is not None and not math.isfinite(market_iv):
                market_iv = None   # IBKR reports NaN greeks when a snapshot has no computed model, not "0 vol"
            broker_delta = t.modelGreeks.delta if t.modelGreeks else None
            if broker_delta is not None and not math.isfinite(broker_delta):
                broker_delta = None   # falls back to the Black-Scholes proxy below instead of a NaN delta

            # NaN is a valid float, and `bool(float('nan'))` is True -- a
            # plain truthiness check on t.last treats "no trade data" (NaN,
            # IBKR's actual response for illiquid far-OTM strikes) as a real
            # price. Require an actual positive, finite number instead.
            mid = None
            if (
                t.bid is not None and t.ask is not None
                and math.isfinite(t.bid) and math.isfinite(t.ask)
                and t.bid > 0 and t.ask > 0
            ):
                mid = (t.bid + t.ask) / 2.0
            elif t.last is not None and math.isfinite(t.last) and t.last > 0:
                mid = t.last
            if mid is None:
                continue

            candidate = build_candidate(
                symbol=symbol,
                expiry=expiry,
                strike=strike,
                underlying_price=underlying_price,
                premium=mid,
                market_iv=market_iv,
                realized_vol=realized_vol,
                delta=broker_delta,
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
            candidates.append(candidate)

    candidates.sort(key=lambda c: c.premium_efficiency, reverse=True)
    return candidates


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
) -> dict:
    """Pull option chains for many symbols concurrently instead of one at a
    time. Returns {symbol: list[OptionCandidate]} -- a symbol is simply
    omitted (and logged) if its fetch raised, so one bad symbol doesn't
    abort the whole batch.
    """
    factories = [
        (lambda symbol=symbol, price=price, rv=rv: _fetch_put_candidates_async(
            client, symbol, price, rv,
            dte_min, dte_max, max_delta_abs, max_expiries,
            strike_band_low_pct, strike_band_high_pct, min_premium_per_contract,
        ))
        for symbol, price, rv in requests
    ]
    results = client.run_many(factories, timeout=max(120.0, 20.0 * len(requests)))

    candidates_by_symbol: dict = {}
    for (symbol, _price, _rv), (result, exc) in zip(requests, results):
        if exc is not None:
            logger.warning("%s: option chain fetch failed — %s", symbol, exc)
            continue
        candidates_by_symbol[symbol] = result
    return candidates_by_symbol
