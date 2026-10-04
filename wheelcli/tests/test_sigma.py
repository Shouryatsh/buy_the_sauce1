"""
wheelcli/tests/test_sigma.py — Unit tests for analytics.sigma module.

Tests cover:
  • compute_sigma_distance: basic case, boundary, edge/invalid inputs
  • passes_filter: all four combinations of delta/sigma availability
"""

from __future__ import annotations

import math

import pytest

from wheelcli.analytics.sigma import compute_sigma_distance, passes_filter


# =============================================================================
# compute_sigma_distance
# =============================================================================


class TestComputeSigmaDistance:
    def test_basic_case(self):
        """Formula: (S − K) / (S × IV × √T)"""
        S, K, iv, T = 150.0, 120.0, 0.30, 30 / 365
        expected = (S - K) / (S * iv * math.sqrt(T))
        result = compute_sigma_distance(S, K, iv, T)
        assert result == pytest.approx(expected, rel=1e-9)

    def test_result_is_positive_for_otm_put(self):
        """Strike below spot → positive sigma distance."""
        result = compute_sigma_distance(100.0, 80.0, 0.25, 45 / 365)
        assert result is not None
        assert result > 0

    def test_result_is_zero_for_atm_put(self):
        """Strike equal to spot → sigma distance = 0."""
        result = compute_sigma_distance(100.0, 100.0, 0.30, 30 / 365)
        assert result is not None
        assert result == pytest.approx(0.0)

    def test_itm_put_negative_sigma(self):
        """Strike above spot → negative sigma distance."""
        result = compute_sigma_distance(100.0, 110.0, 0.30, 30 / 365)
        assert result is not None
        assert result < 0

    def test_zero_iv_returns_none(self):
        assert compute_sigma_distance(100.0, 80.0, 0.0, 30 / 365) is None

    def test_negative_iv_returns_none(self):
        assert compute_sigma_distance(100.0, 80.0, -0.10, 30 / 365) is None

    def test_zero_t_returns_none(self):
        assert compute_sigma_distance(100.0, 80.0, 0.30, 0.0) is None

    def test_negative_t_returns_none(self):
        assert compute_sigma_distance(100.0, 80.0, 0.30, -0.05) is None

    def test_zero_spot_returns_none(self):
        assert compute_sigma_distance(0.0, 80.0, 0.30, 30 / 365) is None

    def test_scales_with_strike(self):
        """Deeper OTM strike → larger sigma distance."""
        far = compute_sigma_distance(100.0, 70.0, 0.30, 30 / 365)
        near = compute_sigma_distance(100.0, 90.0, 0.30, 30 / 365)
        assert far is not None and near is not None
        assert far > near

    def test_scales_with_iv(self):
        """Higher IV → smaller sigma distance (same strike is 'less far')."""
        low_vol = compute_sigma_distance(100.0, 80.0, 0.20, 30 / 365)
        high_vol = compute_sigma_distance(100.0, 80.0, 0.60, 30 / 365)
        assert low_vol is not None and high_vol is not None
        assert low_vol > high_vol

    def test_scales_with_dte(self):
        """More time → larger one-sigma move → smaller sigma distance."""
        short = compute_sigma_distance(100.0, 80.0, 0.30, 10 / 365)
        long_ = compute_sigma_distance(100.0, 80.0, 0.30, 45 / 365)
        assert short is not None and long_ is not None
        assert short > long_


# =============================================================================
# passes_filter
# =============================================================================


class TestPassesFilter:
    def test_passes_on_delta_alone(self):
        assert passes_filter(delta=0.04, sigma_distance=1.0, max_delta=0.05) is True

    def test_exact_delta_boundary_passes(self):
        assert passes_filter(delta=0.05, sigma_distance=None, max_delta=0.05) is True

    def test_delta_too_high_fails_without_sigma(self):
        assert passes_filter(delta=0.10, sigma_distance=None, max_delta=0.05) is False

    def test_passes_on_sigma_when_delta_fails(self):
        assert (
            passes_filter(delta=0.10, sigma_distance=2.5, max_delta=0.05, sigma_threshold=2.0)
            is True
        )

    def test_exact_sigma_boundary_passes(self):
        assert (
            passes_filter(delta=0.10, sigma_distance=2.0, max_delta=0.05, sigma_threshold=2.0)
            is True
        )

    def test_fails_when_both_conditions_fail(self):
        assert (
            passes_filter(delta=0.10, sigma_distance=1.5, max_delta=0.05, sigma_threshold=2.0)
            is False
        )

    def test_none_delta_passes_via_sigma(self):
        assert (
            passes_filter(delta=None, sigma_distance=2.5, max_delta=0.05, sigma_threshold=2.0)
            is True
        )

    def test_none_sigma_passes_via_delta(self):
        assert passes_filter(delta=0.03, sigma_distance=None, max_delta=0.05) is True

    def test_both_none_fails_conservatively(self):
        assert passes_filter(delta=None, sigma_distance=None) is False
