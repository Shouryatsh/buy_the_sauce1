"""
premia/mcmillan.py — McMillan's straddle-buying strategy, as written in *McMillan on Options* ch. 7, "Buy Low and
Sell High — Volatility, That Is" ("My Favorite Strategy", pp. 221-235). Shared by the backtest and the dashboard.

Entry: all four criteria (p. 229).
  1. Implied volatility in the 10th percentile or lower of its past readings, options with at least three months of
     life (pp. 221-223). The book's example uses six months (130 trading days, p. 225).
  2. Monte Carlo "ever" probability >= 80 % that the stock trades at one or the other breakeven (strike +/- straddle
     price) at any time before expiration, under a lognormal model whose volatility is the MINIMUM of the 10-, 20-,
     50- and 100-day historical volatilities, the conservative choice for a buyer (pp. 224-228).
  3. Past prices show the stock has frequently made a move of the required distance in the allotted time (p. 228).
  4. No fundamental reason for the options to be cheap, e.g. a cash takeover bid (p. 229).
Follow-up (pp. 232-233): leave the straddle alone; risk 60 % of the initial price; that usually lets you hold until
about one month of life remains. If a breakeven is crossed: sell all of the losing side, sell about one-third of the
winning side, and ride the rest with the 20-day moving average of the stock as a trailing stop (close below it for
calls, above it for puts). No delta-neutral adjustments ("trading against the straddle" is rejected, p. 232).

Readings that are mine, not the book's (each one a named constant below):
  * "Ever" probability is computed in closed form for continuous monitoring (the book's simulator is a Monte Carlo
    of the same lognormal model), zero drift in log price.
  * "Frequently" (criterion 3) = in at least HIST_FREQ_MIN of the past HIST_YEARS of overlapping windows of the
    same length, the stock's high or low reached the required percentage move. Percent, not points, so the test
    is comparable across price levels.
  * Percentile of "all past readings" = all readings available up to the day (point in time), at least
    MIN_IV_HISTORY of them.
  * Criterion 4 can only be checked by reading the news. The backtest uses a mechanical proxy for a pinned takeover
    target (deal_pinned).
  * Time exit at TIME_EXIT_DAYS calendar days before expiry for the untouched straddle; the trend-riding remainder
    is closed TREND_EXIT_DAYS trading days before expiry at the latest ("not wise to hold until expiration", p. 221).
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

IV_PCT_MAX = 10.0
MIN_IV_HISTORY = 504
MIN_LIFE_DAYS = 90
TARGET_LIFE_DAYS = 182          # six months, the book's example
EVER_PROB_MIN = 0.80
HV_WINDOWS = (10, 20, 50, 100)
HIST_FREQ_MIN = 0.50
HIST_YEARS = 5
RISK_FRAC = 0.60
TIME_EXIT_DAYS = 30
PARTIAL_FRAC = 1 / 3
TRAIL_MA = 20
TREND_EXIT_DAYS = 5
TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# criterion 1: implied volatility percentile against all past readings
# ---------------------------------------------------------------------------

def iv_percentile_all(iv: pd.Series, min_history: int = MIN_IV_HISTORY) -> pd.Series:
    """Point-in-time percentile (0-100) of each day's IV among ALL readings before it (strictly below)."""
    vals = iv.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    seen: list[float] = []
    for i, v in enumerate(vals):
        if not np.isfinite(v):
            continue
        if len(seen) >= min_history:
            out[i] = 100.0 * bisect.bisect_left(seen, v) / len(seen)
        bisect.insort(seen, v)
    return pd.Series(out, index=iv.index)


def latest_iv_percentile(iv: pd.Series, min_history: int = MIN_IV_HISTORY) -> Optional[float]:
    s = iv.dropna()
    if len(s) <= min_history:
        return None
    past = s.iloc[:-1].to_numpy()
    return float(100.0 * (past < s.iloc[-1]).mean())


# ---------------------------------------------------------------------------
# criterion 2: probability of EVER touching either breakeven (lognormal, minimum historical vol)
# ---------------------------------------------------------------------------

def historical_vols(close: pd.Series) -> pd.DataFrame:
    r = np.log(close.astype(float)).diff()
    return pd.DataFrame({f"hv{w}": r.rolling(w).std() * math.sqrt(TRADING_DAYS) for w in HV_WINDOWS})


def min_historical_vol(close: pd.Series) -> pd.Series:
    """The option BUYER's volatility input: the minimum of the 10/20/50/100-day historical volatilities (p. 226)."""
    return historical_vols(close).min(axis=1, skipna=False)


def ever_probability(S: float, upper: float, lower: float, sigma: float, T: float, mu: float = 0.0,
                     terms: int = 400) -> float:
    """P(the price touches `upper` or `lower` at any time within T years) for log-price Brownian motion with drift mu
    and volatility sigma, continuous monitoring (eigenfunction series for Brownian motion killed at two barriers)."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    if S >= upper or (lower > 0 and S <= lower):
        return 1.0
    a = math.log(upper / S)
    st = sigma * math.sqrt(T)
    if lower <= 0:                                           # only the upper breakeven exists
        return float(norm.cdf((-a + mu * T) / st) + math.exp(2 * mu * a / sigma ** 2) * norm.cdf((-a - mu * T) / st))
    b = math.log(lower / S)
    L, x = a - b, -b
    c = mu / sigma ** 2
    n = np.arange(1, terms + 1)
    k = n * math.pi / L
    integral = k * (1 - (-1.0) ** n * math.exp(c * L)) / (c * c + k * k)
    series = np.sin(k * x) * np.exp(-0.5 * sigma ** 2 * k * k * T) * integral
    survive = (2 / L) * math.exp(-c * x - mu * mu * T / (2 * sigma ** 2)) * float(series.sum())
    return float(min(max(1.0 - survive, 0.0), 1.0))


def ever_probability_mc(S: float, upper: float, lower: float, sigma: float, T: float, mu: float = 0.0,
                        n_paths: int = 20000, steps: int = 2000, seed: int = 0) -> float:
    """Monte Carlo of the same model (the book's method); used to test the closed form. A Brownian-bridge test between
    steps catches touches that happen between two sampled points, so the result targets continuous monitoring."""
    rng = np.random.default_rng(seed)
    dt = T / steps
    v = sigma * sigma * dt
    x = np.zeros(n_paths)
    hit = np.zeros(n_paths, bool)
    a, b = math.log(upper / S), (math.log(lower / S) if lower > 0 else -np.inf)
    for _ in range(steps):
        x1 = x + mu * dt + sigma * math.sqrt(dt) * rng.standard_normal(n_paths)
        p_up = np.exp(-2 * np.maximum(a - x, 0) * np.maximum(a - x1, 0) / v)
        p_dn = np.exp(-2 * np.maximum(x - b, 0) * np.maximum(x1 - b, 0) / v) if np.isfinite(b) else 0.0
        u = rng.random(n_paths)
        hit |= (x1 >= a) | (x1 <= b) | (u < p_up) | (rng.random(n_paths) < p_dn)
        x = x1
    return float(hit.mean())


# ---------------------------------------------------------------------------
# criterion 3: has the stock frequently made the required move in the allotted time?
# ---------------------------------------------------------------------------

def move_frequency(df: pd.DataFrame, i: int, horizon: int, up_pct: float, down_pct: float,
                   years: int = HIST_YEARS) -> Optional[float]:
    """Share of the past `years` of overlapping `horizon`-trading-day windows (all fully known at row i) in which the
    high reached +up_pct or the low reached -down_pct from the window's starting close."""
    start = max(0, i - years * TRADING_DAYS)
    last = i - horizon                                       # last window start whose window ends by row i
    if last - start < TRADING_DAYS:
        return None
    hi = df["high"].to_numpy(float)[start:i + 1]
    lo = df["low"].to_numpy(float)[start:i + 1]
    c0 = df["close"].to_numpy(float)[start:last + 1]
    fmax = pd.Series(hi[::-1]).rolling(horizon, min_periods=horizon).max().to_numpy()[::-1]    # max of hi[s..s+h-1]
    fmin = pd.Series(lo[::-1]).rolling(horizon, min_periods=horizon).min().to_numpy()[::-1]
    m = last - start + 1
    up = fmax[1:m + 1] >= c0 * (1 + up_pct)                  # window (s, s + horizon]: from the day after s
    dn = fmin[1:m + 1] <= c0 * (1 - down_pct)
    ok = np.isfinite(fmax[1:m + 1]) & np.isfinite(c0)
    return float((up | dn)[ok].mean()) if ok.any() else None


# ---------------------------------------------------------------------------
# criterion 4 proxy: a takeover target pinned to the deal price
# ---------------------------------------------------------------------------

def deal_pinned(close: pd.Series, i: int, jump: float = 0.15, lookback: int = 126, ratio: float = 0.35) -> bool:
    """True when, within `lookback` days, the stock jumped >= `jump` in a day and its 20-day volatility has since
    collapsed below `ratio` x its 100-day volatility before the jump: what a pending cash takeover looks like."""
    r = np.log(close.astype(float)).diff().to_numpy()
    lo = max(101, i - lookback)
    big = np.where(np.abs(r[lo:i + 1]) >= jump)[0]
    if not len(big):
        return False
    j = lo + int(big[-1])
    pre = np.nanstd(r[j - 100:j]) * math.sqrt(TRADING_DAYS)
    if i - j < 10:
        return False
    cur = np.nanstd(r[i - 19:i + 1]) * math.sqrt(TRADING_DAYS)
    return bool(pre > 0 and cur < ratio * pre)


# ---------------------------------------------------------------------------
# all four criteria together
# ---------------------------------------------------------------------------

@dataclass
class Criteria:
    iv_percentile: Optional[float]
    life_days: int
    breakeven_up: float
    breakeven_down: float
    min_hv: Optional[float]
    ever_prob: Optional[float]
    hist_freq: Optional[float]
    pinned: bool
    reasons: list = field(default_factory=list)

    @property
    def passes(self) -> bool:
        return not self.reasons


def evaluate(S: float, K: float, cost: float, life_days: int, trading_days: int, iv_pct: Optional[float],
             min_hv: Optional[float], hist: Optional[pd.DataFrame], i: Optional[int] = None,
             hist_freq_min: float = HIST_FREQ_MIN, iv_pct_max: float = IV_PCT_MAX) -> Criteria:
    """McMillan's four criteria for buying the K straddle at `cost` per share. `hist` is the daily OHLC history (rows
    up to and including today when `i` is None)."""
    up, down = K + cost, K - cost
    reasons = []
    if iv_pct is None:
        reasons.append("1: not enough implied-volatility history")
    elif iv_pct > iv_pct_max:
        reasons.append(f"1: IV percentile {iv_pct:.0f} > {iv_pct_max:.0f}")
    if life_days < MIN_LIFE_DAYS:
        reasons.append(f"1: {life_days} days of life < {MIN_LIFE_DAYS}")
    p = None
    if min_hv is None or not np.isfinite(min_hv):
        reasons.append("2: no historical volatility")
    else:
        p = ever_probability(S, up, down, float(min_hv), trading_days / TRADING_DAYS)
        if p < EVER_PROB_MIN:
            reasons.append(f"2: P(ever touch a breakeven) {p:.0%} < {EVER_PROB_MIN:.0%}")
    freq, pinned = None, False
    if hist is not None:
        idx = len(hist) - 1 if i is None else i
        freq = move_frequency(hist, idx, trading_days, up / S - 1, 1 - down / S)
        if freq is None:
            reasons.append("3: not enough price history")
        elif freq < hist_freq_min:
            reasons.append(f"3: required move made in {freq:.0%} of past windows < {hist_freq_min:.0%}")
        pinned = deal_pinned(hist["close"], idx)
        if pinned:
            reasons.append("4: price pinned after a jump (looks like a pending takeover)")
    return Criteria(iv_pct, life_days, up, down, min_hv, p, freq, pinned, reasons)


# ---------------------------------------------------------------------------
# size and liquidity (not in the book): one function for the backtest and the dashboard, so both open the same trade
# ---------------------------------------------------------------------------

MULT = 100


def straddle_spread_frac(bid: float, ask: float) -> float:
    """(ask - bid) / mid of the straddle (call + put summed)."""
    mid = (bid + ask) / 2
    return (ask - bid) / mid if mid > 0 else float("inf")


def size(cost: float, nav: float, committed: float, budget: float, one_contract_max: float,
         max_total: float) -> tuple[int, Optional[str]]:
    """Straddles to buy at `cost` per share: `budget` of NAV each (one allowed up to `one_contract_max` of NAV), cut
    down to fit what is left under `max_total` of NAV for all long straddles. Returns (n, reason when n == 0)."""
    per = cost * MULT
    if per <= 0 or nav <= 0:
        return 0, "size: no price"
    n = int(budget * nav // per)
    if n == 0 and per <= one_contract_max * nav:
        n = 1
    if n == 0:
        return 0, f"size: one straddle costs ${per:,.0f}, over {one_contract_max:.1%} of NLV"
    room = max_total * nav - committed
    if per * n > room:
        n = int(max(room, 0) // per)
        if n == 0:
            return 0, f"size: long straddles already use ${committed:,.0f} of the {max_total:.0%} cap"
    return n, None


# ---------------------------------------------------------------------------
# follow-up action
# ---------------------------------------------------------------------------

@dataclass
class FollowUp:
    sell_calls: int
    sell_puts: int
    mode: str                  # "both" | "call" (riding calls) | "put" (riding puts) | "closed"
    reason: str = ""


def follow_up(mode: str, n_calls: int, n_puts: int, close: float, sma: float, value: float, cost: float,
              K: float, days_left: int, trading_days_left: int) -> FollowUp:
    """The book's follow-up for one straddle, decided on the day's close (orders go in the next morning, p. 233).
    `value` is the straddle's current value per share (call + put), `cost` what was paid per share."""
    if mode == "both":
        if close >= K + cost and n_calls > 0:
            sell = int(round(n_calls * PARTIAL_FRAC))
            return FollowUp(sell, n_puts, "call" if n_calls - sell > 0 else "closed",
                            "upside breakeven crossed: sell the puts and a third of the calls")
        if close <= K - cost and n_puts > 0:
            sell = int(round(n_puts * PARTIAL_FRAC))
            return FollowUp(n_calls, sell, "put" if n_puts - sell > 0 else "closed",
                            "downside breakeven crossed: sell the calls and a third of the puts")
        if value <= (1 - RISK_FRAC) * cost:
            return FollowUp(n_calls, n_puts, "closed", "lost 60 % of the initial price")
        if days_left <= TIME_EXIT_DAYS:
            return FollowUp(n_calls, n_puts, "closed", "about one month of life left")
        return FollowUp(0, 0, "both")
    if mode == "call":
        if np.isfinite(sma) and close < sma:
            return FollowUp(n_calls, 0, "closed", f"closed below the {TRAIL_MA}-day moving average")
        if trading_days_left <= TREND_EXIT_DAYS:
            return FollowUp(n_calls, 0, "closed", "expiration approaching")
        return FollowUp(0, 0, "call")
    if mode == "put":
        if np.isfinite(sma) and close > sma:
            return FollowUp(0, n_puts, "closed", f"closed above the {TRAIL_MA}-day moving average")
        if trading_days_left <= TREND_EXIT_DAYS:
            return FollowUp(0, n_puts, "closed", "expiration approaching")
        return FollowUp(0, 0, "put")
    return FollowUp(0, 0, "closed")
