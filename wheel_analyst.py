"""
wheel_analyst.py — strict cash-secured-put screen, portfolio fit and roll view
for the "Wheel Analyst" dashboard tab.

Read-only: uses WheelIBKRClient (connected readonly=True) and never builds an
order. Filter thresholds below are the user's own rules and are deliberately
separate from wheel_config.py's quality-ranked radar.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

import wheel_config as wc

WATCHLIST = ["AXP", "SAP", "NVDA", "CHWY", "SRAD", "MSFT", "HIMS", "PG", "MA", "V", "WFC"]

MAX_ABS_DELTA = 0.15
MIN_CUSHION = 0.05       # strike at least 5% below spot
MIN_OPEN_INTEREST = 100
# Spread limit, (ask-bid)/mid, is shared with Recommendations: wheel_config.MAX_SPREAD_PCT.
STRIKE_BAND = (0.72, 0.95)


def third_friday_next_month(today: Optional[date] = None) -> date:
    today = today or date.today()
    y, m = (today.year + 1, 1) if today.month == 12 else (today.year, today.month + 1)
    d = date(y, m, 1)
    d += timedelta(days=(4 - d.weekday()) % 7)   # first Friday
    return d + timedelta(days=14)


def _clean(x) -> Optional[float]:
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) or x <= 0 else x


@dataclass
class PutCandidate:
    symbol: str
    spot: float
    strike: float
    expiry: str
    dte: int
    delta: float
    bid: float
    ask: float
    open_interest: int
    delayed: bool
    earnings: str = "unverified"        # "none before expiry" | "YYYY-MM-DD" | "unverified"

    @property
    def mid(self) -> float: return (self.bid + self.ask) / 2
    @property
    def pct_below_spot(self) -> float: return (self.spot - self.strike) / self.spot
    @property
    def capital(self) -> float: return self.strike * 100
    @property
    def premium_contract_mid(self) -> float: return self.mid * 100
    @property
    def roi_mid(self) -> float: return self.mid / self.strike
    @property
    def roi_bid(self) -> float: return self.bid / self.strike
    @property
    def ann_mid(self) -> float: return self.roi_mid * 365 / self.dte
    @property
    def ann_bid(self) -> float: return self.roi_bid * 365 / self.dte
    @property
    def breakeven(self) -> float: return self.strike - self.mid
    @property
    def breakeven_pct_below(self) -> float: return (self.spot - self.breakeven) / self.spot
    @property
    def earnings_blocks(self) -> bool: return self.earnings not in ("none before expiry", "unverified")


def screen_put(symbol: str, spot: float, strike: float, expiry: str, dte: int, q: dict,
               delayed: bool, earnings: str = "unverified") -> tuple[Optional[PutCandidate], Optional[str]]:
    """Apply the user's filters to one raw quote. Returns (candidate, None) or (None, reason)."""
    bid, ask = _clean(q.get("bid")), _clean(q.get("ask"))
    delta, oi = q.get("delta"), q.get("oi")
    if bid is None or ask is None:
        return None, "missing bid/ask"
    if bid > ask:
        return None, "bad quote (bid > ask)"
    if delta is None or (isinstance(delta, float) and math.isnan(delta)):
        return None, "missing delta"
    if delta > 0:
        return None, "bad delta sign for a put"
    if oi is None or (isinstance(oi, float) and math.isnan(oi)):
        return None, "missing open interest"
    # Delta must fit strike/price: a >=5%-OTM put cannot have |delta| above ~0.5.
    if strike >= spot or abs(delta) > 0.5:
        return None, "delta does not fit strike/price"
    if (spot - strike) / spot < MIN_CUSHION:
        return None, "less than 5% below spot"
    if abs(delta) > MAX_ABS_DELTA:
        return None, "|delta| above 0.15"
    mid = (bid + ask) / 2
    if (ask - bid) / mid > wc.MAX_SPREAD_PCT:
        return None, f"spread over {wc.MAX_SPREAD_PCT:.0%} of mid"
    if oi < MIN_OPEN_INTEREST:
        return None, "open interest under 100"
    return PutCandidate(symbol, spot, strike, expiry, dte, abs(delta), bid, ask, int(oi), delayed, earnings), None


def select_portfolio(cands: list[PutCandidate], budget: float) -> list[PutCandidate]:
    """One put per underlying, greedy by annualized mid ROI under the budget.
    Names with earnings before expiry are never selected."""
    best: dict[str, PutCandidate] = {}
    for c in cands:
        if c.earnings_blocks:
            continue
        if c.symbol not in best or c.ann_mid > best[c.symbol].ann_mid:
            best[c.symbol] = c
    chosen, used = [], 0.0
    for c in sorted(best.values(), key=lambda c: c.ann_mid, reverse=True):
        if used + c.capital <= budget:
            chosen.append(c); used += c.capital
    return chosen


def blended_annualized(chosen: list[PutCandidate]) -> float:
    cap = sum(c.capital for c in chosen)
    if not cap:
        return 0.0
    # capital-weighted; each name's ROI is annualized from its own DTE
    return sum(c.ann_mid * c.capital for c in chosen) / cap


def put_status(spot: float, strike: float, abs_delta: Optional[float]) -> str:
    """safe / watch / roll. Delta may be unknown (market closed): distance alone decides then."""
    dist = (spot - strike) / spot
    if dist < 0.02 or (abs_delta is not None and abs_delta > 0.30):
        return "roll"
    if dist < MIN_CUSHION or (abs_delta is not None and abs_delta > MAX_ABS_DELTA):
        return "watch"
    return "safe"


def earnings_status(symbol: str, expiry: date) -> str:
    """Via wheel_events (Yahoo earnings dates). No upcoming date known → 'unverified'."""
    import wheel_events
    nxt = wheel_events.next_earnings(symbol)
    if nxt is None:
        return "unverified"
    return nxt.isoformat() if nxt <= expiry else "none before expiry"


# ---------------------------------------------------------------------------
# IBKR quote fetch (runs on WheelIBKRClient's loop thread; read-only)
# ---------------------------------------------------------------------------

def _first(*vals):
    for v in vals:
        v = _clean(v)
        if v is not None:
            return v
    return None


def _quote(t) -> dict:
    g = t.modelGreeks
    delta = g.delta if g is not None and g.delta is not None and not math.isnan(g.delta) else None
    oi = t.putOpenInterest
    return {
        "bid": _first(t.bid, getattr(t, "delayedBid", None)),
        "ask": _first(t.ask, getattr(t, "delayedAsk", None)),
        "delta": delta,
        "oi": None if oi is None or (isinstance(oi, float) and math.isnan(oi)) else oi,
        "mdt": t.marketDataType,
    }


async def _stream_quotes(ib, contracts: list, wait: float) -> list[dict]:
    """Streaming quotes with the same type-2-then-type-4 fallback as
    wheel_options.option_tickers: live (or the last live quote after the
    close), else 15-minute delayed. Open interest is filled in from a
    separate type-1 request where the quote stream didn't carry it (it never
    does on the after-hours "frozen" type)."""
    from wheel_options import put_open_interest
    out: list[dict] = []
    for mdt in (2, 4):
        ib.reqMarketDataType(mdt)
        tickers = [ib.reqMktData(o, "101", False, False) for o in contracts]
        await asyncio.sleep(wait)
        out = [_quote(t) for t in tickers]
        for o in contracts:
            ib.cancelMktData(o)
        if any(q["bid"] and q["ask"] for q in out):
            break
    missing = [i for i, q in enumerate(out) if q["oi"] is None]
    if missing:
        ois = await put_open_interest(ib, [contracts[i] for i in missing])
        for i, oi in zip(missing, ois):
            out[i]["oi"] = oi
    return out


async def _fetch_symbol(ib, symbol: str, expiry: str, wait: float = 7.0) -> dict:
    from ib_insync import Stock, Option
    ib.reqMarketDataType(2)           # live, or last live quote after the close
    stock = Stock(symbol, "SMART", "USD")
    await ib.qualifyContractsAsync(stock)
    st = ib.reqMktData(stock, "", False, False)
    await asyncio.sleep(3)
    spot = _first(st.last, getattr(st, "delayedLast", None), st.marketPrice(), st.close)
    ib.cancelMktData(stock)
    if spot is None:
        return {"spot": None, "quotes": []}

    chains = await ib.reqSecDefOptParamsAsync(symbol, "", "STK", stock.conId)
    strikes: set[float] = set()
    for c in chains:
        if expiry in c.expirations and c.tradingClass == symbol:
            strikes |= set(c.strikes)
    ks = sorted(k for k in strikes if STRIKE_BAND[0] * spot <= k <= STRIKE_BAND[1] * spot)
    opts = await ib.qualifyContractsAsync(*[Option(symbol, expiry, k, "P", "SMART", tradingClass=symbol) for k in ks])
    opts = [o for o in opts if o and o.conId]

    quotes = []
    for i in range(0, len(opts), 40):
        batch = opts[i:i + 40]
        for o, q in zip(batch, await _stream_quotes(ib, batch, wait)):
            quotes.append({"strike": o.strike, **q})
    return {"spot": spot, "quotes": quotes}


async def _fetch_held(ib, symbol: str, expiry: str, strike: float) -> dict:
    from ib_insync import Option
    o = Option(symbol, expiry, strike, "P", "SMART")
    await ib.qualifyContractsAsync(o)
    return (await _stream_quotes(ib, [o], 6))[0]


def fetch_symbol(client, symbol: str, expiry: str) -> dict:
    return client.run(lambda: _fetch_symbol(client.ib, symbol, expiry), timeout=120)


def fetch_held(client, symbol: str, expiry: str, strike: float) -> dict:
    return client.run(lambda: _fetch_held(client.ib, symbol, expiry, strike), timeout=60)
