"""premia/markets_calendar.py — listed-expiry helpers shared by the backtester and the live tools."""

from __future__ import annotations

import pandas as pd


def third_friday(y: int, m: int) -> pd.Timestamp:
    d = pd.Timestamp(y, m, 15)
    return d + pd.Timedelta(days=(4 - d.weekday()) % 7)


def monthly_iso(asof, lo_days: int, hi_days: int) -> list[str]:
    """Standard monthly expiries (third Fridays) between lo_days and hi_days calendar days after *asof*."""
    a = pd.Timestamp(asof)
    a = (a.tz_localize(None) if a.tzinfo else a).normalize()
    out = []
    y, m = a.year, a.month
    for _ in range(14):
        f = third_friday(y, m)
        if lo_days <= (f - a).days <= hi_days:
            out.append(f.strftime("%Y-%m-%d"))
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out
