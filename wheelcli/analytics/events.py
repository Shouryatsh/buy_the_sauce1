"""
wheelcli/analytics/events.py — Event-risk discount calculation.

Philosophy
----------
Selling puts into an earnings announcement or a major macro event (FOMC,
CPI, etc.) dramatically increases the probability of a large adverse move.
We discount the attractiveness of any trade whose holding period overlaps
such an event.

Rules (applied multiplicatively so penalties stack)
----------------------------------------------------
1. Symbol NOT in earnings calendar → UNKNOWN_EARNINGS × 0.90
2. Symbol IS tracked AND earnings fall within [opened, expiry + window_days]
   → EARNINGS × 0.60
3. Any macro event date falls within [opened, expiry]
   → MACRO × 0.85

Penalties stack: e.g. EARNINGS+MACRO → 0.60 × 0.85 = 0.51

Public API
----------
compute_event_multiplier(symbol, expiry, date_opened, ...) → (multiplier, flag_string)
"""

from __future__ import annotations

from datetime import date, timedelta

from ..data.earnings import EarningsProvider


def compute_event_multiplier(
    symbol: str,
    expiry: date,
    date_opened: date,
    earnings_provider: EarningsProvider,
    macro_events: list[date],
    event_window_days: int = 3,
    earnings_penalty: float = 0.6,
    macro_penalty: float = 0.85,
    unknown_earnings_penalty: float = 0.9,
) -> tuple[float, str]:
    """
    Compute the event-risk score multiplier for a CSP trade.

    Parameters
    ----------
    symbol                  : underlying ticker
    expiry                  : option expiry date
    date_opened             : assumed trade open date (typically today)
    earnings_provider       : EarningsProvider instance for the calendar lookup
    macro_events            : sorted list of known macro event dates
    event_window_days       : grace window after earnings before expiry
    earnings_penalty        : multiplier when trade crosses earnings
    macro_penalty           : multiplier when trade crosses a macro event
    unknown_earnings_penalty: multiplier when symbol is not in earnings calendar

    Returns
    -------
    (multiplier, event_flag) where event_flag is a '+'-joined string of
    active risk labels:  "" | "EARNINGS" | "MACRO" | "UNKNOWN_EARNINGS"
    or any combination, e.g. "EARNINGS+MACRO".
    """
    multiplier = 1.0
    flags: list[str] = []

    # ── Earnings risk ──────────────────────────────────────────────────────
    if not earnings_provider.is_tracked(symbol):
        # Symbol absent from the earnings calendar → conservative discount
        multiplier *= unknown_earnings_penalty
        flags.append("UNKNOWN_EARNINGS")
    else:
        # Symbol is in the calendar: look for the next earnings after we open
        next_earnings = earnings_provider.next_earnings(symbol, date_opened)

        if next_earnings is not None:
            # Check whether earnings fall inside the holding + window period
            in_window = date_opened <= next_earnings <= expiry + timedelta(
                days=event_window_days
            )
            if in_window:
                multiplier *= earnings_penalty
                flags.append("EARNINGS")
        # If next_earnings is None the symbol is tracked but has no upcoming
        # dates → no penalty (we know the calendar is up-to-date)

    # ── Macro events ───────────────────────────────────────────────────────
    macro_in_period = [e for e in macro_events if date_opened <= e <= expiry]
    if macro_in_period:
        multiplier *= macro_penalty
        flags.append("MACRO")

    return multiplier, "+".join(flags)
