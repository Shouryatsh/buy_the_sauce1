"""
test_valuation.py — Unit tests for the conservative valuation engine.

Tests cover:
  - WACC estimation with various inputs
  - Quality scoring and tier assignment
  - All 8 individual valuation models
  - Composite valuation (weighted median, signals, grades)
  - Edge cases (missing data, negative FCF, zero shares)
"""

import math
import pytest
from valuation import (
    _estimate_wacc,
    _assess_quality,
    _dcf_two_stage,
    _reverse_dcf,
    _earnings_power_value,
    _graham_number,
    _excess_returns,
    _dividend_discount,
    _relative_pe,
    _asset_floor,
    _estimate_net_debt,
    _get_equity,
    _weighted_median,
    valuate,
    ValuationResult,
    WACC_FLOOR,
    WACC_CAP,
    QUALITY_HIGH_THRESHOLD,
    QUALITY_MEDIUM_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Fixtures — reusable mock data
# ---------------------------------------------------------------------------

def _high_quality_info():
    """Mock fundamentals for a high-quality large-cap company."""
    return {
        "marketCap": 200_000_000_000,
        "_fcf_history": [15e9, 13e9, 11e9, 10e9],
        "_equity": 50_000_000_000,
        "totalStockholderEquity": 50_000_000_000,
        "_long_term_debt": 20_000_000_000,
        "longTermDebt": 20_000_000_000,
        "_total_debt": 25_000_000_000,
        "currentDebt": 5_000_000_000,
        "_cash": 15_000_000_000,
        "cash": 15_000_000_000,
        "_net_income": 12_000_000_000,
        "_operating_cf": 18_000_000_000,
        "freeCashflow": 15_000_000_000,
        "capitalExpenditures": -3_000_000_000,
        "profitMargins": 0.20,
        "revenue_current": 60_000_000_000,
        "revenue_prior": 55_000_000_000,
        "revenue_growth": 0.091,
        "debtToEquity": 50.0,
        "returnOnEquity": 0.24,
        "roic": 0.171,
        "roic_prior": 0.165,
        "fcf_margin": 0.25,
        "fcf_margin_prior": 0.236,
        "cash_conversion": 1.25,
        "cash_conversion_prior": 1.18,
        "accruals_ratio": -0.03,
        "trailingPE": 16.7,
        "trailingEPS": 6.0,
        "epsCurrentYear": 6.5,
        "epsTrailingTwelveMonths": 6.0,
        "sector_pe": 20.0,
        "fiftyTwoWeekHigh": 120.0,
        "fiftyTwoWeekLow": 80.0,
        "currentPrice": 100.0,
        "sharesOutstanding": 2_000_000_000,
        "latestPrice": 100.0,
        "beta": 1.1,
    }


def _speculative_info():
    """Mock fundamentals for a speculative / low-quality company."""
    return {
        "marketCap": 5_000_000_000,
        "_fcf_history": [-500e6, 200e6, -100e6],
        "_equity": 1_000_000_000,
        "totalStockholderEquity": 1_000_000_000,
        "_long_term_debt": 3_000_000_000,
        "longTermDebt": 3_000_000_000,
        "_total_debt": 3_500_000_000,
        "currentDebt": 500_000_000,
        "_cash": 500_000_000,
        "cash": 500_000_000,
        "_net_income": -200_000_000,
        "freeCashflow": -500_000_000,
        "profitMargins": -0.04,
        "revenue_current": 5_000_000_000,
        "revenue_prior": 5_500_000_000,
        "revenue_growth": -0.091,
        "debtToEquity": 350.0,
        "returnOnEquity": -0.20,
        "roic": -0.05,
        "trailingPE": None,
        "trailingEPS": None,
        "epsCurrentYear": None,
        "sector_pe": 20.0,
        "sharesOutstanding": 500_000_000,
        "latestPrice": 10.0,
        "beta": 2.0,
    }


# ---------------------------------------------------------------------------
# WACC Tests
# ---------------------------------------------------------------------------

class TestWACC:
    def test_wacc_within_bounds(self):
        info = _high_quality_info()
        wacc = _estimate_wacc(info)
        assert WACC_FLOOR <= wacc <= WACC_CAP

    def test_wacc_floor_enforced(self):
        # Very low beta, low D/E → should still be >= WACC_FLOOR
        info = {"beta": 0.3, "debtToEquity": 10.0, "marketCap": 1e12}
        wacc = _estimate_wacc(info)
        assert wacc >= WACC_FLOOR

    def test_wacc_cap_enforced(self):
        # Very high beta → should be capped
        info = {"beta": 5.0, "debtToEquity": 500.0, "marketCap": 1e9}
        wacc = _estimate_wacc(info)
        assert wacc <= WACC_CAP

    def test_wacc_missing_beta_defaults(self):
        info = {}
        wacc = _estimate_wacc(info)
        assert WACC_FLOOR <= wacc <= WACC_CAP


# ---------------------------------------------------------------------------
# Quality Tests
# ---------------------------------------------------------------------------

class TestQuality:
    def test_high_quality_company(self):
        info = _high_quality_info()
        score, tier, moat = _assess_quality(info)
        assert tier == "HIGH"
        assert score >= QUALITY_HIGH_THRESHOLD
        assert len(moat) > 0

    def test_speculative_company(self):
        info = _speculative_info()
        score, tier, moat = _assess_quality(info)
        assert tier == "SPECULATIVE"
        assert score < QUALITY_MEDIUM_THRESHOLD

    def test_moat_indicators_detected(self):
        info = _high_quality_info()
        _, _, moat = _assess_quality(info)
        # Should detect ROE, FCF margin, cash conversion, accruals
        moat_text = " ".join(moat)
        assert "ROE" in moat_text
        assert "FCF" in moat_text or "cash" in moat_text.lower()


# ---------------------------------------------------------------------------
# Individual Model Tests
# ---------------------------------------------------------------------------

class TestDCFTwoStage:
    def test_positive_fcf_produces_value(self):
        info = _high_quality_info()
        wacc = 0.10
        shares = 2_000_000_000
        result = _dcf_two_stage(info, wacc, shares)
        assert result.fair_value_per_share is not None
        assert result.fair_value_per_share > 0

    def test_negative_fcf_returns_error(self):
        info = _speculative_info()
        wacc = 0.10
        shares = 500_000_000
        result = _dcf_two_stage(info, wacc, shares)
        assert result.fair_value_per_share is None
        assert "Negative FCF" in result.error

    def test_no_fcf_data_returns_error(self):
        result = _dcf_two_stage({}, 0.10, 1e9)
        assert result.fair_value_per_share is None
        assert result.error != ""


class TestEPV:
    def test_positive_earnings_produces_value(self):
        info = _high_quality_info()
        result = _earnings_power_value(info, 0.10, 2e9)
        assert result.fair_value_per_share is not None
        assert result.fair_value_per_share > 0

    def test_negative_earnings_returns_error(self):
        info = _speculative_info()
        result = _earnings_power_value(info, 0.10, 5e8)
        assert result.fair_value_per_share is None


class TestGrahamNumber:
    def test_positive_eps_bvps(self):
        info = _high_quality_info()
        result = _graham_number(info, 2e9)
        assert result.fair_value_per_share is not None
        assert result.fair_value_per_share > 0

    def test_negative_income_returns_error(self):
        info = _speculative_info()
        result = _graham_number(info, 5e8)
        assert result.fair_value_per_share is None


class TestExcessReturns:
    def test_roic_above_wacc(self):
        info = _high_quality_info()
        result = _excess_returns(info, 0.10, 2e9)
        assert result.fair_value_per_share is not None
        # With ROIC > WACC, should be worth MORE than book value
        bvps = 50e9 / 2e9
        assert result.fair_value_per_share > bvps

    def test_roic_below_wacc(self):
        info = _high_quality_info()
        info["roic"] = 0.05  # below WACC
        result = _excess_returns(info, 0.10, 2e9)
        # Should be approximately book value
        assert result.fair_value_per_share is not None


class TestReverseDCF:
    def test_implied_growth_calculated(self):
        info = _high_quality_info()
        result = _reverse_dcf(info, 0.10, 2e9, 100.0)
        assert result.inputs.get("implied_growth") is not None
        assert result.inputs.get("reasonableness") is not None


class TestDDM:
    def test_positive_fcf_produces_value(self):
        info = _high_quality_info()
        result = _dividend_discount(info, 0.10, 2e9)
        assert result.fair_value_per_share is not None
        assert result.fair_value_per_share > 0


class TestRelativePE:
    def test_produces_value(self):
        info = _high_quality_info()
        result = _relative_pe(info, 2e9)
        assert result.fair_value_per_share is not None
        assert result.fair_value_per_share > 0

    def test_conservative_pe_cap(self):
        info = _high_quality_info()
        result = _relative_pe(info, 2e9)
        # Conservative P/E should never exceed 25x (70% of 35)
        assert result.inputs["conservative_pe"] <= 25.0


class TestAssetFloor:
    def test_positive_equity(self):
        info = _high_quality_info()
        result = _asset_floor(info, 2e9)
        assert result.fair_value_per_share is not None
        # Should be tangible book value per share
        # tangible = 50B - 0 (goodwill) - 0 (intangibles) = 50B
        # per share = 50B / 2B = 25
        assert result.fair_value_per_share > 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_net_debt_with_raw_data(self):
        info = {"_total_debt": 25e9, "_cash": 15e9}
        assert _estimate_net_debt(info) == 10e9

    def test_net_debt_no_cash(self):
        info = {"_total_debt": 25e9}
        nd = _estimate_net_debt(info)
        assert nd > 0

    def test_equity_from_raw(self):
        info = {"_equity": 50e9}
        assert _get_equity(info) == 50e9

    def test_weighted_median_simple(self):
        vals = [(10.0, 1.0), (20.0, 1.0), (30.0, 1.0)]
        assert _weighted_median(vals) == 20.0

    def test_weighted_median_weighted(self):
        # Heavy weight on 10 → median should be 10
        vals = [(10.0, 10.0), (20.0, 1.0), (30.0, 1.0)]
        assert _weighted_median(vals) == 10.0


# ---------------------------------------------------------------------------
# Composite Valuation Tests
# ---------------------------------------------------------------------------

class TestCompositeValuation:
    def test_high_quality_undervalued(self):
        info = _high_quality_info()
        # Set price well below likely fair value
        result = valuate("TEST", info, current_price=40.0, shares_outstanding=2e9)
        assert result.valuation_signal in ("DEEP_VALUE", "UNDERVALUED")
        assert result.valuation_grade in ("A", "B")
        assert result.fair_value is not None
        assert result.buy_price is not None
        assert result.upside_pct > 0

    def test_high_quality_overvalued(self):
        info = _high_quality_info()
        # Set price way above likely fair value
        result = valuate("TEST", info, current_price=500.0, shares_outstanding=2e9)
        assert result.valuation_signal in ("OVERVALUED", "EXPENSIVE")
        assert result.valuation_grade in ("D", "F")
        assert result.upside_pct < 0

    def test_speculative_insufficient_data(self):
        info = _speculative_info()
        result = valuate("SPEC", info, current_price=10.0, shares_outstanding=5e8)
        # Should still produce some output (asset floor, etc.)
        assert result.symbol == "SPEC"
        assert result.quality_tier == "SPECULATIVE"

    def test_no_shares_returns_insufficient(self):
        info = _high_quality_info()
        info.pop("marketCap", None)
        result = valuate("TEST", info, current_price=100.0, shares_outstanding=0)
        assert result.valuation_signal == "INSUFFICIENT_DATA"

    def test_margin_of_safety_tiers(self):
        # High quality → lower MoS
        info = _high_quality_info()
        result = valuate("HQ", info, current_price=100.0, shares_outstanding=2e9)
        assert result.margin_of_safety_pct <= 0.25  # high quality = 20%

        # Speculative → higher MoS
        info2 = _speculative_info()
        result2 = valuate("LQ", info2, current_price=10.0, shares_outstanding=5e8)
        assert result2.margin_of_safety_pct >= 0.35  # speculative = 40%

    def test_model_count(self):
        info = _high_quality_info()
        result = valuate("TEST", info, current_price=100.0, shares_outstanding=2e9)
        # Should have 9 models (8 base + reverse DCF since price is provided)
        assert len(result.models) >= 8

    def test_summary_populated(self):
        info = _high_quality_info()
        result = valuate("TEST", info, current_price=100.0, shares_outstanding=2e9)
        assert result.summary != ""
        assert "Fair Value" in result.summary or "Insufficient" in result.summary
