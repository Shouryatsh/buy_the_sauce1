"""
wheel_vol.py — implied and realized volatility, each ranked against ITS OWN
history (McMillan, "Profits with Options"): implied vol is not compared with
realized vol directly — only with past implied vol — and realized vol only
with past realized vol. Where each sits now is expressed as a percentile.

Percentile = % of days in the lookback window on which the series was lower
than it is today (0 = lowest in the window, 100 = highest).

Implied-vol history comes from IBKR (whatToShow=OPTION_IMPLIED_VOLATILITY:
IBKR's 30-day implied vol for the underlying). Realized (historical) vol is
computed here from daily closes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252
LOOKBACKS = {"1Y": 252, "2Y": 504, "3Y": 756}
HV_WINDOWS = (10, 20, 50, 100)   # McMillan's usual realized-vol windows


def realized_vol(close: pd.Series, window: int) -> pd.Series:
    """Annualized close-to-close realized vol over a rolling *window* of days."""
    rets = np.log(close.astype(float)).diff()
    return rets.rolling(window).std() * math.sqrt(TRADING_DAYS)


def percentile_rank(series: pd.Series, lookback: int) -> Optional[float]:
    """Percent of the last *lookback* observations (before today) below today's value."""
    s = series.dropna()
    if len(s) < 20:
        return None
    current, hist = s.iloc[-1], s.iloc[-(lookback + 1):-1]
    if hist.empty:
        return None
    return float((hist < current).mean() * 100)


def rolling_percentile_bands(series: pd.Series, lookback: int, qs=(0.10, 0.50, 0.90)) -> pd.DataFrame:
    """Trailing quantiles so the shaded band on a chart shows 'normal' at each date."""
    s = series.dropna()
    minp = max(20, lookback // 4)
    return pd.DataFrame({f"p{int(q * 100)}": s.rolling(lookback, min_periods=minp).quantile(q) for q in qs})


def label(pct: Optional[float]) -> str:
    if pct is None:
        return "n/a"
    if pct >= 90:
        return "very high"
    if pct >= 75:
        return "high"
    if pct <= 10:
        return "very low"
    if pct <= 25:
        return "low"
    return "normal"


@dataclass
class VolRead:
    value: Optional[float]
    pct: Optional[float]
    lo: Optional[float]
    median: Optional[float]
    hi: Optional[float]

    @property
    def label(self) -> str:
        return label(self.pct)


def read(series: pd.Series, lookback: int) -> VolRead:
    s = series.dropna()
    if s.empty:
        return VolRead(None, None, None, None, None)
    q = [None if pd.isna(v) else float(v) for v in s.iloc[-(lookback + 1):-1].quantile([0.10, 0.50, 0.90])]
    return VolRead(float(s.iloc[-1]), percentile_rank(s, lookback), *q)


def interpretation(iv: VolRead, has_event_soon: bool) -> str:
    """Plain-language read of where implied vol sits vs its own history."""
    if iv.pct is None:
        return "Not enough implied-vol history to rank."
    if iv.pct >= 75:
        base = (f"Options are expensive versus their own history ({iv.pct:.0f}th percentile) — "
                "historically a better time to be a premium seller.")
        if has_event_soon:
            base += " An event is coming up, which often explains elevated IV; it usually drops after the event."
        return base
    if iv.pct <= 25:
        return (f"Options are cheap versus their own history ({iv.pct:.0f}th percentile) — "
                "selling premium pays less than usual; buyers get better value.")
    return f"Implied vol is in its normal range ({iv.pct:.0f}th percentile)."


# ---------------------------------------------------------------------------
# Earnings: IV run-up into the event and crush after it, aligned in trading
# days on the last close BEFORE the announcement (k = 0). k = -5 is one
# trading week before; k = +1 is the first close after the news.
# ---------------------------------------------------------------------------

def pre_event_pos(index: pd.DatetimeIndex, event_date, timing: str) -> Optional[int]:
    """Position in *index* of the last close before the announcement.
    After the close (AMC): that day's close. Before the open (BMO) or
    unknown: the previous session's close. None if the data has a gap there."""
    e = pd.Timestamp(event_date)
    days = index.normalize()
    pos = days.searchsorted(e, side="right" if timing == "AMC" else "left") - 1
    if pos < 0 or (e - days[pos]).days > 5:
        return None
    return int(pos)


def event_paths(iv: pd.Series, events: list, k_min: int = -20, k_max: int = 5) -> pd.DataFrame:
    """IV around each event: rows k (trading days from the pre-event close),
    one column per event date. Cells outside the data are NaN."""
    s = iv.dropna()
    cols = {}
    for d, timing in events:
        pos = pre_event_pos(s.index, d, timing)
        if pos is None:
            continue
        cols[pd.Timestamp(d).strftime("%Y-%m-%d")] = [
            s.iloc[pos + k] if 0 <= pos + k < len(s) else np.nan for k in range(k_min, k_max + 1)]
    return pd.DataFrame(cols, index=pd.RangeIndex(k_min, k_max + 1, name="k"))


def earnings_table(iv: pd.Series, close: Optional[pd.Series], events: list) -> pd.DataFrame:
    """One row per past event that has a post-event close: IV a week before,
    IV at the last pre-event close, run-up, IV after, crush, and the stock's
    actual close-to-close move through the event."""
    s = iv.dropna()
    px = None
    if close is not None and not close.empty:
        px = close.copy()
        px.index = pd.DatetimeIndex(px.index).tz_localize(None).normalize() if getattr(px.index, "tz", None) \
            else pd.DatetimeIndex(px.index).normalize()
    rows = []
    for d, timing in events:
        pos = pre_event_pos(s.index, d, timing)
        if pos is None or pos + 1 >= len(s) or pos - 5 < 0:
            continue
        pre, post, wk = float(s.iloc[pos]), float(s.iloc[pos + 1]), float(s.iloc[pos - 5])
        move = None
        if px is not None:
            a, b = s.index[pos].normalize(), s.index[pos + 1].normalize()
            if a in px.index and b in px.index:
                move = float(px[b] / px[a] - 1)
        rows.append({"Earnings": pd.Timestamp(d).date(), "Timing": timing, "IV 1 week before": wk,
                     "IV last close before": pre, "Run-up": pre / wk - 1, "IV after": post,
                     "Crush": post / pre - 1, "Stock move": move})
    return pd.DataFrame(rows)


def trading_days_to_pre_close(last_date, event_date, timing: str) -> int:
    """Trading days (weekdays; exchange holidays ignored) from the latest bar
    to the pre-event close. 0 = the latest bar IS the pre-event close."""
    e = np.datetime64(pd.Timestamp(event_date).date())
    pre = e if timing == "AMC" else np.busday_offset(e, -1, roll="backward")
    return int(np.busday_count(np.datetime64(pd.Timestamp(last_date).date()), pre))


@dataclass
class EarningsCompare:
    k: int                      # today's position relative to the coming pre-event close
    today: float
    same_point: pd.Series       # IV at the same k before each past event
    pct: Optional[float]        # % of past events with lower IV at this point
    typical_peak: Optional[float]           # median IV at the last close before past events
    typical_runup_from_here: Optional[float]  # median (IV at k=0 / IV at k) - 1


def compare_to_past_earnings(iv: pd.Series, past_events: list, upcoming: tuple) -> Optional[EarningsCompare]:
    s = iv.dropna()
    if s.empty or not upcoming:
        return None
    k = -trading_days_to_pre_close(s.index[-1], *upcoming)
    if k > 0:
        return None
    paths = event_paths(s, past_events, k_min=min(k, -20), k_max=1)
    if paths.empty or k not in paths.index:
        return None
    at_k, at_0 = paths.loc[k].dropna(), paths.loc[0]
    today = float(s.iloc[-1])
    runup = (at_0[at_k.index] / at_k - 1).dropna()
    return EarningsCompare(
        k=k, today=today, same_point=at_k,
        pct=float((at_k < today).mean() * 100) if len(at_k) else None,
        typical_peak=float(at_0.median()) if at_0.notna().any() else None,
        typical_runup_from_here=float(runup.median()) if len(runup) else None,
    )


# ---------------------------------------------------------------------------
# IBKR implied-vol history (read-only historical data request)
# ---------------------------------------------------------------------------

async def _iv_history(ib, symbol: str, duration: str) -> pd.Series:
    from ib_insync import Stock
    c = Stock(symbol, "SMART", "USD")
    await ib.qualifyContractsAsync(c)
    bars = await ib.reqHistoricalDataAsync(c, "", duration, "1 day", "OPTION_IMPLIED_VOLATILITY",
                                           True, formatDate=1, timeout=60)
    if not bars:
        return pd.Series(dtype=float, name="IV")
    s = pd.Series([b.close for b in bars], index=pd.to_datetime([b.date for b in bars]), name="IV")
    return s[s > 0]


def iv_history(client, symbol: str, duration: str = "3 Y") -> pd.Series:
    return client.run(lambda: _iv_history(client.ib, symbol, duration), timeout=90)
