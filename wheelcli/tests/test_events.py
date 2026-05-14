"""
wheelcli/tests/test_events.py — Unit tests for analytics.events module.

Tests cover all combinations of the three risk sources:
  • Unknown earnings (symbol absent from calendar)
  • Earnings within the holding + window period
  • Macro event within the holding period
  • Stacked penalties
  • Known earnings well outside the window → no penalty
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import pytest

from wheelcli.analytics.events import compute_event_multiplier
from wheelcli.data.earnings import EarningsProvider


# =============================================================================
# Test double: configurable mock earnings provider
# =============================================================================


class MockEarningsProvider(EarningsProvider):
    """
    Simple test double.

    tracked=True  → is_tracked returns True
    next_date     → next_earnings always returns this date (or None)
    """

    def __init__(
        self,
        next_date: Optional[date] = None,
        tracked: bool = True,
    ) -> None:
        self._next_date = next_date
        self._tracked = tracked

    def is_tracked(self, symbol: str) -> bool:
        return self._tracked

    def next_earnings(self, symbol: str, as_of: date) -> Optional[date]:
        if self._next_date is not None and self._next_date >= as_of:
            return self._next_date
        return None


# =============================================================================
# Convenience wrapper
# =============================================================================

_OPEN = date(2026, 5, 14)     # trade open date (today)
_EXPIRY = date(2026, 6, 20)   # option expiry (~37 DTE)


def _mult(
    provider: EarningsProvider,
    macro_events: list[date] | None = None,
    expiry: date = _EXPIRY,
    opened: date = _OPEN,
    **kwargs,
) -> tuple[float, str]:
    return compute_event_multiplier(
        symbol="AAPL",
        expiry=expiry,
        date_opened=opened,
        earnings_provider=provider,
        macro_events=macro_events or [],
        **kwargs,
    )


# =============================================================================
# Unknown earnings (symbol not in calendar)
# =============================================================================


class TestUnknownEarnings:
    def test_unknown_symbol_applies_penalty(self):
        provider = MockEarningsProvider(tracked=False)
        mult, flag = _mult(provider, unknown_earnings_penalty=0.9)
        assert mult == pytest.approx(0.9)
        assert flag == "UNKNOWN_EARNINGS"

    def test_unknown_symbol_custom_penalty(self):
        provider = MockEarningsProvider(tracked=False)
        mult, flag = _mult(provider, unknown_earnings_penalty=0.75)
        assert mult == pytest.approx(0.75)

    def test_tracked_but_no_future_earnings_no_penalty(self):
        """Symbol is tracked but has no future dates → no penalty."""
        provider = MockEarningsProvider(next_date=None, tracked=True)
        mult, flag = _mult(provider)
        assert mult == pytest.approx(1.0)
        assert flag == ""


# =============================================================================
# Known earnings within window
# =============================================================================


class TestKnownEarningsInWindow:
    def test_earnings_on_expiry_day_penalizes(self):
        provider = MockEarningsProvider(next_date=_EXPIRY, tracked=True)
        mult, flag = _mult(provider, event_window_days=3, earnings_penalty=0.6)
        assert mult == pytest.approx(0.6)
        assert "EARNINGS" in flag

    def test_earnings_within_grace_window_penalizes(self):
        """Earnings 2 days after expiry but within 3-day grace window → penalty."""
        earnings = date(2026, 6, 22)  # 2 days after expiry
        provider = MockEarningsProvider(next_date=earnings, tracked=True)
        mult, flag = _mult(provider, event_window_days=3, earnings_penalty=0.6)
        assert mult == pytest.approx(0.6)
        assert "EARNINGS" in flag

    def test_earnings_just_outside_window_no_penalty(self):
        """Earnings 4 days after expiry, window is 3 → no penalty."""
        earnings = date(2026, 6, 24)  # 4 days after expiry
        provider = MockEarningsProvider(next_date=earnings, tracked=True)
        mult, flag = _mult(provider, event_window_days=3, earnings_penalty=0.6)
        assert mult == pytest.approx(1.0)
        assert flag == ""

    def test_earnings_well_after_expiry_no_penalty(self):
        """Earnings months away → no penalty."""
        provider = MockEarningsProvider(next_date=date(2026, 9, 15), tracked=True)
        mult, flag = _mult(provider)
        assert mult == pytest.approx(1.0)
        assert flag == ""

    def test_earnings_on_open_date_penalizes(self):
        """Earnings exactly on the day we open the trade → within window."""
        provider = MockEarningsProvider(next_date=_OPEN, tracked=True)
        mult, flag = _mult(provider, event_window_days=3, earnings_penalty=0.6)
        assert mult == pytest.approx(0.6)
        assert "EARNINGS" in flag

    def test_custom_earnings_penalty(self):
        provider = MockEarningsProvider(next_date=_EXPIRY, tracked=True)
        mult, _ = _mult(provider, earnings_penalty=0.5)
        assert mult == pytest.approx(0.5)


# =============================================================================
# Macro events
# =============================================================================


class TestMacroEvents:
    def test_macro_within_holding_period_penalizes(self):
        provider = MockEarningsProvider(tracked=True)
        macro = [date(2026, 6, 1)]  # between open and expiry
        mult, flag = _mult(provider, macro_events=macro, macro_penalty=0.85)
        assert mult == pytest.approx(0.85)
        assert "MACRO" in flag

    def test_macro_on_open_date_penalizes(self):
        provider = MockEarningsProvider(tracked=True)
        macro = [_OPEN]
        mult, flag = _mult(provider, macro_events=macro)
        assert "MACRO" in flag

    def test_macro_on_expiry_date_penalizes(self):
        provider = MockEarningsProvider(tracked=True)
        macro = [_EXPIRY]
        mult, flag = _mult(provider, macro_events=macro)
        assert "MACRO" in flag

    def test_macro_before_open_no_penalty(self):
        provider = MockEarningsProvider(tracked=True)
        macro = [date(2026, 5, 13)]  # one day before trade opens
        mult, flag = _mult(provider, macro_events=macro)
        assert mult == pytest.approx(1.0)
        assert "MACRO" not in flag

    def test_macro_after_expiry_no_penalty(self):
        provider = MockEarningsProvider(tracked=True)
        macro = [date(2026, 6, 21)]  # one day after expiry
        mult, flag = _mult(provider, macro_events=macro)
        assert mult == pytest.approx(1.0)
        assert "MACRO" not in flag

    def test_multiple_macro_events_single_penalty(self):
        """Multiple macro events in window → only one penalty applied."""
        provider = MockEarningsProvider(tracked=True)
        macro = [date(2026, 5, 20), date(2026, 6, 10)]
        mult, flag = _mult(provider, macro_events=macro, macro_penalty=0.85)
        assert mult == pytest.approx(0.85)  # not 0.85 ** 2


# =============================================================================
# Stacked penalties
# =============================================================================


class TestStackedPenalties:
    def test_earnings_and_macro_multiply(self):
        provider = MockEarningsProvider(next_date=_EXPIRY, tracked=True)
        macro = [date(2026, 6, 1)]
        mult, flag = _mult(
            provider,
            macro_events=macro,
            earnings_penalty=0.6,
            macro_penalty=0.85,
        )
        assert mult == pytest.approx(0.6 * 0.85)
        assert "EARNINGS" in flag
        assert "MACRO" in flag

    def test_unknown_earnings_and_macro_multiply(self):
        provider = MockEarningsProvider(tracked=False)
        macro = [date(2026, 6, 1)]
        mult, flag = _mult(
            provider,
            macro_events=macro,
            unknown_earnings_penalty=0.9,
            macro_penalty=0.85,
        )
        assert mult == pytest.approx(0.9 * 0.85)
        assert "UNKNOWN_EARNINGS" in flag
        assert "MACRO" in flag

    def test_no_events_multiplier_is_one(self):
        provider = MockEarningsProvider(tracked=True)
        mult, flag = _mult(provider)
        assert mult == pytest.approx(1.0)
        assert flag == ""
