"""Tests for the stock screener fundamental filters."""

import pytest

from screener import (
    filter_strong_companies,
    free_cash_flow_yield,
    has_low_debt_to_equity,
    has_positive_revenue_growth,
    has_strong_fcf_yield,
    has_strong_roe,
    is_fundamentally_strong,
)


# ---------------------------------------------------------------------------
# free_cash_flow_yield
# ---------------------------------------------------------------------------

class TestFreeCashFlowYield:
    def test_basic_calculation(self):
        # FCF 10, market cap 100 -> 10%
        assert free_cash_flow_yield(10, 100) == pytest.approx(10.0)

    def test_five_percent(self):
        assert free_cash_flow_yield(5, 100) == pytest.approx(5.0)

    def test_below_five_percent(self):
        assert free_cash_flow_yield(4, 100) == pytest.approx(4.0)

    def test_negative_fcf(self):
        assert free_cash_flow_yield(-5, 100) == pytest.approx(-5.0)

    def test_zero_market_cap_raises(self):
        with pytest.raises(ValueError):
            free_cash_flow_yield(10, 0)

    def test_negative_market_cap_raises(self):
        with pytest.raises(ValueError):
            free_cash_flow_yield(10, -1)


# ---------------------------------------------------------------------------
# has_strong_fcf_yield
# ---------------------------------------------------------------------------

class TestHasStrongFcfYield:
    def test_above_threshold_passes(self):
        assert has_strong_fcf_yield(6, 100) is True  # 6% > 5%

    def test_exactly_at_threshold_passes(self):
        assert has_strong_fcf_yield(5, 100) is True  # exactly 5%

    def test_below_threshold_fails(self):
        assert has_strong_fcf_yield(4, 100) is False  # 4% < 5%

    def test_custom_threshold(self):
        assert has_strong_fcf_yield(3, 100, min_yield_pct=3.0) is True

    def test_negative_fcf_fails(self):
        assert has_strong_fcf_yield(-1, 100) is False


# ---------------------------------------------------------------------------
# has_low_debt_to_equity
# ---------------------------------------------------------------------------

class TestHasLowDebtToEquity:
    def test_below_max_passes(self):
        assert has_low_debt_to_equity(50, 100) is True  # 0.5 <= 1.0

    def test_exactly_at_max_passes(self):
        assert has_low_debt_to_equity(100, 100) is True  # 1.0 <= 1.0

    def test_above_max_fails(self):
        assert has_low_debt_to_equity(150, 100) is False  # 1.5 > 1.0

    def test_custom_max_ratio(self):
        assert has_low_debt_to_equity(40, 100, max_ratio=0.3) is False

    def test_zero_equity_raises(self):
        with pytest.raises(ValueError):
            has_low_debt_to_equity(100, 0)


# ---------------------------------------------------------------------------
# has_strong_roe
# ---------------------------------------------------------------------------

class TestHasStrongRoe:
    def test_above_threshold_passes(self):
        assert has_strong_roe(20, 100) is True  # 20% >= 15%

    def test_exactly_at_threshold_passes(self):
        assert has_strong_roe(15, 100) is True  # exactly 15%

    def test_below_threshold_fails(self):
        assert has_strong_roe(10, 100) is False  # 10% < 15%

    def test_custom_threshold(self):
        assert has_strong_roe(10, 100, min_roe_pct=10.0) is True

    def test_zero_equity_raises(self):
        with pytest.raises(ValueError):
            has_strong_roe(10, 0)


# ---------------------------------------------------------------------------
# has_positive_revenue_growth
# ---------------------------------------------------------------------------

class TestHasPositiveRevenueGrowth:
    def test_growth_passes(self):
        assert has_positive_revenue_growth(110, 100) is True

    def test_no_growth_fails(self):
        assert has_positive_revenue_growth(100, 100) is False

    def test_decline_fails(self):
        assert has_positive_revenue_growth(90, 100) is False

    def test_zero_previous_raises(self):
        with pytest.raises(ValueError):
            has_positive_revenue_growth(100, 0)

    def test_negative_previous_raises(self):
        with pytest.raises(ValueError):
            has_positive_revenue_growth(100, -10)


# ---------------------------------------------------------------------------
# is_fundamentally_strong
# ---------------------------------------------------------------------------

class TestIsFundamentallyStrong:
    def _strong_company_kwargs(self):
        """Return kwargs that satisfy all default filter thresholds."""
        return dict(
            free_cash_flow=6,      # 6% FCF yield on market cap of 100
            market_cap=100,
            debt=50,               # D/E = 0.5
            equity=100,
            net_income=20,         # ROE = 20%
            current_revenue=110,
            previous_revenue=100,
        )

    def test_all_filters_pass(self):
        assert is_fundamentally_strong(**self._strong_company_kwargs()) is True

    def test_low_fcf_yield_fails(self):
        kwargs = self._strong_company_kwargs()
        kwargs["free_cash_flow"] = 4  # 4% < 5%
        assert is_fundamentally_strong(**kwargs) is False

    def test_high_de_ratio_fails(self):
        kwargs = self._strong_company_kwargs()
        kwargs["debt"] = 200  # D/E = 2.0 > 1.0
        assert is_fundamentally_strong(**kwargs) is False

    def test_low_roe_fails(self):
        kwargs = self._strong_company_kwargs()
        kwargs["net_income"] = 10  # ROE = 10% < 15%
        assert is_fundamentally_strong(**kwargs) is False

    def test_revenue_decline_fails(self):
        kwargs = self._strong_company_kwargs()
        kwargs["current_revenue"] = 90  # decline from 100
        assert is_fundamentally_strong(**kwargs) is False

    def test_exactly_at_fcf_yield_threshold_passes(self):
        kwargs = self._strong_company_kwargs()
        kwargs["free_cash_flow"] = 5  # exactly 5%
        assert is_fundamentally_strong(**kwargs) is True


# ---------------------------------------------------------------------------
# filter_strong_companies
# ---------------------------------------------------------------------------

class TestFilterStrongCompanies:
    def _make_company(self, name, fcf, market_cap, debt, equity, net_income, cur_rev, prev_rev):
        return dict(
            name=name,
            free_cash_flow=fcf,
            market_cap=market_cap,
            debt=debt,
            equity=equity,
            net_income=net_income,
            current_revenue=cur_rev,
            previous_revenue=prev_rev,
        )

    def test_returns_only_strong_companies(self):
        strong = self._make_company("StrongCo", 6, 100, 50, 100, 20, 110, 100)
        weak_fcf = self._make_company("WeakFCF", 4, 100, 50, 100, 20, 110, 100)
        weak_de = self._make_company("WeakDE", 6, 100, 200, 100, 20, 110, 100)
        weak_roe = self._make_company("WeakROE", 6, 100, 50, 100, 10, 110, 100)
        weak_rev = self._make_company("WeakRev", 6, 100, 50, 100, 20, 90, 100)

        result = filter_strong_companies([strong, weak_fcf, weak_de, weak_roe, weak_rev])
        assert len(result) == 1
        assert result[0]["name"] == "StrongCo"

    def test_empty_list_returns_empty(self):
        assert filter_strong_companies([]) == []

    def test_all_strong_returns_all(self):
        c1 = self._make_company("Co1", 6, 100, 50, 100, 20, 110, 100)
        c2 = self._make_company("Co2", 10, 100, 30, 100, 25, 120, 100)
        result = filter_strong_companies([c1, c2])
        assert len(result) == 2

    def test_custom_thresholds(self):
        # Relax thresholds: min FCF yield 3%, max D/E 2.0, min ROE 10%
        borderline = self._make_company("BorderCo", 3, 100, 180, 100, 10, 110, 100)
        result = filter_strong_companies([borderline], min_fcf_yield_pct=3.0, max_de_ratio=2.0, min_roe_pct=10.0)
        assert len(result) == 1
        assert result[0]["name"] == "BorderCo"
