"""
straddle_recommend.py — McMillan straddle buying for the dashboard ("My Favorite Strategy", *McMillan on Options*
ch. 7). The rules live in premia/mcmillan.py and are the same ones the backtest uses
(premia/research/mcmillan_bt.py, premia/MCMILLAN.md):

  1. IV at or below the 10th percentile of all its past readings; options with at least three months of life
     (the expiry nearest six months is used, as in the book's example);
  2. P(the stock EVER touches strike +/- straddle price before expiry) >= 80 %, lognormal with the minimum of the
     10/20/50/100-day historical volatilities;
  3. the stock has made the required move in at least half of the same-length windows of the past five years;
  4. no fundamental reason for cheap options (a takeover): a price-pattern proxy here, plus your own news check.

Follow-up for straddles you hold: risk 60 % of the price paid; get out with about a month of life left; when a
breakeven is crossed, sell the losing side and a third of the winning side and trail the rest with the 20-day moving
average. Nothing in this module talks to a broker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

import wheel_config as wc
from premia import mcmillan as mm
from straddle_options import StraddleQuote

SLIP = 0.5            # expected fill: mid + half of (ask - mid), as in the backtest


def wheel_universe() -> list[str]:
    """Every name the wheel evaluates (the watchlist), minus the manual exclusions."""
    from wheel_watchlist import WHEEL_WATCHLIST, load_exclusions
    ex = load_exclusions()
    return [s for s in dict.fromkeys(s.upper() for s in WHEEL_WATCHLIST) if s not in ex]


def stored_iv(symbol: str) -> pd.Series:
    """The research store's IBKR implied-vol history (cleaned), empty if the name was never downloaded."""
    from premia import data, store
    try:
        iv = store.load_iv(symbol)
    except Exception:
        return pd.Series(dtype=float)
    if iv is None or len(iv) == 0:
        return pd.Series(dtype=float)
    clean, _bad = data.clean_iv(iv)
    return clean.dropna()


def recent_duration(stored: pd.Series, today: Optional[date] = None) -> str:
    """How much recent IV to request from IBKR so that stored + recent has no gap."""
    today = today or date.today()
    if stored.empty:
        return "10 Y"
    gap = (today - stored.index[-1].date()).days
    return f"{min(max(gap + 7, 30), 365)} D" if gap < 360 else "10 Y"


def combine_iv(stored: pd.Series, recent: Optional[pd.Series]) -> pd.Series:
    if recent is None or recent.empty:
        return stored
    r = recent.copy()
    r.index = pd.to_datetime(r.index).normalize()
    s = stored[stored.index < r.index[0]] if not stored.empty else stored
    return pd.concat([s, r]).sort_index()


@dataclass
class McMillanIdea:
    symbol: str
    expiry: str
    dte: int
    trading_days: int
    strike: float
    underlying_price: float
    bid: float
    ask: float
    mid: float
    cost: float                       # expected fill per share (call + put)
    straddle_iv: Optional[float]
    criteria: mm.Criteria
    spread_pct: float
    call_open_interest: Optional[int]
    put_open_interest: Optional[int]
    quote_type: str
    contracts_suggested: int
    cost_total: float
    reasons: list = field(default_factory=list)       # every unmet criterion or limit

    @property
    def passes(self) -> bool:
        return not self.reasons

    @property
    def min_open_interest(self) -> Optional[int]:
        if self.call_open_interest is None or self.put_open_interest is None:
            return None
        return min(self.call_open_interest, self.put_open_interest)

    @property
    def stop_value(self) -> float:
        return (1 - mm.RISK_FRAC) * self.cost

    @property
    def time_exit(self) -> date:
        return datetime.strptime(self.expiry, "%Y%m%d").date() - timedelta(days=mm.TIME_EXIT_DAYS)


def pick_expiry(quotes: list[StraddleQuote]) -> Optional[StraddleQuote]:
    """The quoted expiry nearest six months with at least three months of life (criterion 1)."""
    ok = [q for q in quotes if q.dte >= mm.MIN_LIFE_DAYS]
    return min(ok, key=lambda q: abs(q.dte - mm.TARGET_LIFE_DAYS)) if ok else None


def evaluate_quote(q: StraddleQuote, ohlc: pd.DataFrame, iv_pct: Optional[float], nlv: float,
                   committed_premium: float = 0.0, already_held: bool = False,
                   today: Optional[date] = None) -> McMillanIdea:
    """All four criteria, the premia liquidity limits and the size for one quoted straddle. `ohlc` is daily history
    (Open/High/Low/Close, oldest first) up to the latest close."""
    today = today or date.today()
    hist = ohlc.rename(columns=str.lower)[["open", "high", "low", "close"]].astype(float).dropna()
    cost = q.mid + SLIP * (q.ask - q.mid)
    exp = datetime.strptime(q.expiry, "%Y%m%d").date()
    tdays = int(max(np.busday_count(today, exp), 1))
    min_hv = float(mm.min_historical_vol(hist["close"]).iloc[-1]) if len(hist) > max(mm.HV_WINDOWS) else None
    crit = mm.evaluate(q.underlying_price, q.strike, cost, q.dte, tdays, iv_pct, min_hv, hist)
    reasons = list(crit.reasons)
    if q.spread_pct > wc.STRADDLE_MAX_SPREAD_PCT:
        reasons.append(f"limit: straddle spread {q.spread_pct:.0%} of mid > {wc.STRADDLE_MAX_SPREAD_PCT:.0%}")
    oi = None if q.call_open_interest is None or q.put_open_interest is None else min(q.call_open_interest, q.put_open_interest)
    if oi is None:
        reasons.append("limit: open interest unknown on a leg")
    elif oi < wc.STRADDLE_MIN_OPEN_INTEREST:
        reasons.append(f"limit: open interest {oi:,} < {wc.STRADDLE_MIN_OPEN_INTEREST:,} on a leg")
    if already_held:
        reasons.append("a long straddle on this name is already held")
    per = cost * mm.MULT
    n, why = mm.size(cost, nlv, committed_premium, wc.STRADDLE_BUDGET_PCT_NLV, wc.STRADDLE_ONE_CONTRACT_MAX_PCT_NLV,
                     wc.STRADDLE_MAX_TOTAL_PCT_NLV)
    if why:
        reasons.append(why)
    ivs = [x for x in (q.call_iv, q.put_iv) if x]
    return McMillanIdea(q.symbol, q.expiry, q.dte, tdays, q.strike, q.underlying_price, q.bid, q.ask, q.mid, cost,
                        float(np.mean(ivs)) if ivs else None, crit, q.spread_pct, q.call_open_interest,
                        q.put_open_interest, q.quote_type, n, per * n, reasons)


def rank(ideas: list[McMillanIdea]) -> tuple[list[McMillanIdea], list[McMillanIdea]]:
    """(buys, the rest): buys best 'ever' probability first; the rest by how far they got."""
    buys = sorted((i for i in ideas if i.passes), key=lambda i: -(i.criteria.ever_prob or 0))
    rest = sorted((i for i in ideas if not i.passes), key=lambda i: (len(i.reasons), -(i.criteria.ever_prob or 0)))
    return buys, rest


# ---------------------------------------------------------------------------
# follow-up on straddles already held
# ---------------------------------------------------------------------------

@dataclass
class HeldStraddle:
    symbol: str
    strike: float
    expiry: str
    contracts: int
    cost: float                # per share, call + put, from IBKR average cost
    value: Optional[float]     # per share, call + put, from IBKR market price
    action: mm.FollowUp
    breakeven_up: float
    breakeven_down: float


def held_straddles(positions) -> list[tuple[str, float, str, object, object]]:
    """(symbol, strike, expiry, call position, put position) for every long call + long put at the same strike/expiry."""
    legs: dict = {}
    for p in positions:
        if p.sec_type == "OPT" and p.quantity > 0 and p.right in ("C", "P"):
            legs.setdefault((p.symbol, p.strike, p.expiry), {})[p.right] = p
    return [(s, k, e, d["C"], d["P"]) for (s, k, e), d in legs.items() if "C" in d and "P" in d]


def follow_up_for(symbol: str, strike: float, expiry: str, call, put, close: float, sma20: float,
                  today: Optional[date] = None) -> HeldStraddle:
    """The book's follow-up for a held straddle (IBKR avg cost is per contract, i.e. x100)."""
    today = today or date.today()
    cost = (call.avg_cost + put.avg_cost) / 100.0
    value = (call.market_price + put.market_price) if call.market_price is not None and put.market_price is not None else None
    n = int(min(call.quantity, put.quantity))
    exp = datetime.strptime(expiry, "%Y%m%d").date()
    f = mm.follow_up("both", n, n, close, sma20, value if value is not None else cost, cost, strike,
                     (exp - today).days, int(np.busday_count(today, exp)))
    return HeldStraddle(symbol, strike, expiry, n, cost, value, f, strike + cost, strike - cost)
