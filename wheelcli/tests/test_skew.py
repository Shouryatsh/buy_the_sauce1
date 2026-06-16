"""
wheelcli/tests/test_skew.py — Unit tests for analytics.skew module.

Tests cover:
  • find_atm_iv  : ATM selection, 25-delta fallback, unavailable case
  • compute_skew  : ratio / diff computation, ITM exclusion, missing-IV cases
  • compute_skew_bonus : all threshold combinations
"""

from __future__ import annotations

from datetime import date

import pytest

from wheelcli.analytics.skew import compute_skew, compute_skew_bonus, find_atm_iv
from wheelcli.models import OptionContract


# =============================================================================
# Fixtures / helpers
# =============================================================================


def _put(strike: float, iv: float | None, delta: float | None = None) -> OptionContract:
    return OptionContract(
        symbol="AAPL",
        expiry=date(2026, 6, 20),
        strike=strike,
        right="P",
        delta=delta,
        iv=iv,
        bid=1.00,
        ask=1.10,
        mid=1.05,
        dte=30,
    )


# =============================================================================
# find_atm_iv
# =============================================================================


class TestFindATMIV:
    def test_returns_iv_of_closest_strike_to_spot(self):
        spot = 150.0
        contracts = [_put(145, 0.25), _put(150, 0.22), _put(155, 0.20)]
        iv, method = find_atm_iv(contracts, spot)
        assert iv == pytest.approx(0.22)
        assert method == "atm"

    def test_skips_none_iv_for_nearest_strikes(self):
        """If the closest strikes have no IV, continue searching."""
        spot = 150.0
        contracts = [
            _put(150, None),
            _put(148, None),
            _put(145, 0.28),
        ]
        iv, method = find_atm_iv(contracts, spot)
        assert iv == pytest.approx(0.28)
        assert method == "atm"

    def test_falls_back_to_25d_when_nearby_iv_missing(self):
        """Only the far-OTM 25-delta put has IV; all strikes closer to spot have None."""
        spot = 150.0
        contracts = [
            _put(150, None),
            _put(149, None),
            _put(148, None),
            _put(147, None),
            _put(146, None),
            _put(130, 0.32, delta=0.25),  # outside the 5-nearest window
        ]
        iv, method = find_atm_iv(contracts, spot)
        assert iv == pytest.approx(0.32)
        assert method == "25d_approx"

    def test_returns_unavailable_when_no_iv_at_all(self):
        contracts = [_put(150, None), _put(140, None)]
        iv, method = find_atm_iv(contracts, 150.0)
        assert iv is None
        assert method == "unavailable"

    def test_empty_list_returns_unavailable(self):
        iv, method = find_atm_iv([], 150.0)
        assert iv is None
        assert method == "unavailable"


# =============================================================================
# compute_skew
# =============================================================================


class TestComputeSkew:
    def test_basic_skew_ratio_and_diff(self):
        spot = 150.0
        contracts = [
            _put(150, 0.20),  # ATM
            _put(140, 0.25),  # OTM
            _put(130, 0.32),  # Far OTM
        ]
        result = compute_skew(contracts, spot)

        assert 140.0 in result
        assert 130.0 in result
        assert result[140.0]["skew_ratio"] == pytest.approx(0.25 / 0.20)
        assert result[140.0]["skew_diff"] == pytest.approx(0.25 - 0.20)
        assert result[130.0]["skew_ratio"] == pytest.approx(0.32 / 0.20)

    def test_atm_strike_not_in_result(self):
        """ATM strikes (>= spot) must be excluded from the output dict."""
        spot = 150.0
        contracts = [_put(150, 0.20), _put(140, 0.25)]
        result = compute_skew(contracts, spot)
        assert 150.0 not in result

    def test_itm_put_excluded(self):
        """A put with strike > spot is ITM and must be excluded."""
        spot = 150.0
        contracts = [_put(150, 0.20), _put(160, 0.15)]
        result = compute_skew(contracts, spot)
        assert 160.0 not in result

    def test_no_atm_iv_sets_warning(self):
        """If NO nearby contract has IV, ATM IV is unavailable → warning set."""
        spot = 150.0
        # Both the near-ATM and the OTM put have no IV
        contracts = [_put(150, None), _put(140, None)]
        result = compute_skew(contracts, spot)
        assert result[140.0]["skew_ratio"] is None
        assert result[140.0]["warning"] == "no_atm_iv"

    def test_no_otm_iv_sets_warning(self):
        spot = 150.0
        contracts = [_put(150, 0.20), _put(140, None)]
        result = compute_skew(contracts, spot)
        assert result[140.0]["skew_ratio"] is None
        assert result[140.0]["warning"] == "no_otm_iv"

    def test_empty_contracts(self):
        result = compute_skew([], 150.0)
        assert result == {}

    def test_atm_iv_stored_in_result(self):
        spot = 150.0
        contracts = [_put(150, 0.22), _put(140, 0.28)]
        result = compute_skew(contracts, spot)
        assert result[140.0]["atm_iv"] == pytest.approx(0.22)


# =============================================================================
# compute_skew_bonus
# =============================================================================


class TestComputeSkewBonus:
    def test_high_ratio_awards_bonus(self):
        assert compute_skew_bonus(1.15, 0.02) == 1.0

    def test_high_diff_awards_bonus(self):
        assert compute_skew_bonus(1.05, 0.05) == 1.0

    def test_both_conditions_met_still_one(self):
        assert compute_skew_bonus(1.20, 0.08) == 1.0

    def test_neither_condition_no_bonus(self):
        assert compute_skew_bonus(1.05, 0.01) == 0.0

    def test_exact_ratio_threshold_passes(self):
        assert compute_skew_bonus(1.10, 0.0, ratio_threshold=1.10) == 1.0

    def test_exact_diff_threshold_passes(self):
        assert compute_skew_bonus(1.05, 0.03, diff_threshold=0.03) == 1.0

    def test_none_ratio_uses_diff(self):
        assert compute_skew_bonus(None, 0.05) == 1.0

    def test_none_diff_uses_ratio(self):
        assert compute_skew_bonus(1.15, None) == 1.0

    def test_both_none_no_bonus(self):
        assert compute_skew_bonus(None, None) == 0.0

    def test_custom_thresholds(self):
        # Tighter threshold: ratio must be ≥ 1.20
        assert compute_skew_bonus(1.15, 0.01, ratio_threshold=1.20, diff_threshold=0.05) == 0.0
        assert compute_skew_bonus(1.25, 0.01, ratio_threshold=1.20, diff_threshold=0.05) == 1.0
