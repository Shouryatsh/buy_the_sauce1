"""
tests/test_screener.py — Unit tests for screener.py.

All tests are offline (no network calls) — yfinance data is injected via
pre-built ``info`` dicts and mock Ticker objects so tests are fast and
deterministic.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

import config
from screener import screen_fundamental, FundamentalProfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ticker(fcf_values: list[float] | None = None) -> MagicMock:
    """Return a mock yf.Ticker whose cash-flow statement contains *fcf_values*.

    *fcf_values* should be ordered newest → oldest (matching yfinance column order).
    Pass None to simulate a ticker with no cash-flow data.
    """
    ticker = MagicMock()
    if fcf_values is None:
        ticker.cashflow = pd.DataFrame()
    else:
        cols = pd.date_range(end="2024-01-01", periods=len(fcf_values), freq="YE")[::-1]
        ticker.cashflow = pd.DataFrame(
            {"Free Cash Flow": fcf_values},
            index=["Free Cash Flow"],
            columns=cols,
        )
    return ticker


# A minimal passing info dict — satisfies every filter comfortably.
GOOD_INFO: dict = {
    "forwardPE": 20.0,
    "profitMargins": 0.15,          # 15%
    "debtToEquity": 50.0,           # 0.5× after /100
    "freeCashflow": 5_000_000_000,  # $5 B
    "marketCap": 100_000_000_000,   # $100 B → FCF yield = 5%
    "returnOnEquity": 0.20,         # 20%
    "capitalExpenditures": -1_000_000_000,  # $1 B capex → capex/FCF = 20%
}

# Increasing FCF over 3 years (newest first)
INCREASING_FCF = [5_000_000_000, 4_000_000_000, 3_000_000_000]


# ---------------------------------------------------------------------------
# Full-pass scenario
# ---------------------------------------------------------------------------

class TestFullPass:
    def test_good_stock_passes(self):
        ticker = _make_ticker(INCREASING_FCF)
        profile = screen_fundamental("GOOD", info=GOOD_INFO.copy(), _ticker=ticker)
        assert profile.passes is True
        assert profile.fail_reasons == []

    def test_profile_fields_populated(self):
        ticker = _make_ticker(INCREASING_FCF)
        profile = screen_fundamental("GOOD", info=GOOD_INFO.copy(), _ticker=ticker)
        assert profile.pe_ratio == pytest.approx(20.0)
        assert profile.profit_margin == pytest.approx(0.15)
        assert profile.debt_to_equity == pytest.approx(0.50)
        assert profile.free_cash_flow == pytest.approx(5_000_000_000)
        assert profile.fcf_yield == pytest.approx(0.05)
        assert profile.fcf_increasing is True
        assert profile.return_on_equity == pytest.approx(0.20)
        assert profile.capex_to_fcf == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# P/E filter
# ---------------------------------------------------------------------------

class TestPeFilter:
    def test_high_pe_fails(self):
        info = {**GOOD_INFO, "forwardPE": config.MAX_PE_RATIO + 1}
        profile = screen_fundamental("HIGH_PE", info=info, _ticker=_make_ticker(INCREASING_FCF))
        assert profile.passes is False
        assert any("PE" in r for r in profile.fail_reasons)

    def test_zero_pe_fails(self):
        info = {**GOOD_INFO, "forwardPE": 0.0}
        profile = screen_fundamental("ZERO_PE", info=info, _ticker=_make_ticker(INCREASING_FCF))
        assert profile.passes is False

    def test_missing_pe_passes(self):
        info = {k: v for k, v in GOOD_INFO.items() if k not in ("forwardPE", "trailingPE")}
        profile = screen_fundamental("NO_PE", info=info, _ticker=_make_ticker(INCREASING_FCF))
        assert profile.passes is True


# ---------------------------------------------------------------------------
# Profit margin filter
# ---------------------------------------------------------------------------

class TestProfitMarginFilter:
    def test_low_margin_fails(self):
        info = {**GOOD_INFO, "profitMargins": config.MIN_PROFIT_MARGIN - 0.01}
        profile = screen_fundamental("LOW_MARGIN", info=info, _ticker=_make_ticker(INCREASING_FCF))
        assert profile.passes is False
        assert any("profitMargin" in r for r in profile.fail_reasons)

    def test_exactly_at_threshold_passes(self):
        info = {**GOOD_INFO, "profitMargins": config.MIN_PROFIT_MARGIN}
        profile = screen_fundamental("EXACT_MARGIN", info=info, _ticker=_make_ticker(INCREASING_FCF))
        assert profile.passes is True


# ---------------------------------------------------------------------------
# Debt / equity filter
# ---------------------------------------------------------------------------

class TestDebtToEquityFilter:
    def test_high_de_fails(self):
        # yfinance D/E is ×100, so set to (MAX + 0.1) * 100
        info = {**GOOD_INFO, "debtToEquity": (config.MAX_DEBT_TO_EQUITY + 0.1) * 100}
        profile = screen_fundamental("HIGH_DE", info=info, _ticker=_make_ticker(INCREASING_FCF))
        assert profile.passes is False
        assert any("D/E" in r for r in profile.fail_reasons)

    def test_acceptable_de_passes(self):
        info = {**GOOD_INFO, "debtToEquity": config.MAX_DEBT_TO_EQUITY * 100}
        profile = screen_fundamental("OK_DE", info=info, _ticker=_make_ticker(INCREASING_FCF))
        assert profile.passes is True


# ---------------------------------------------------------------------------
# Free cash flow filters
# ---------------------------------------------------------------------------

class TestFreeCashFlowFilter:
    def test_negative_fcf_fails(self):
        info = {**GOOD_INFO, "freeCashflow": -1}
        profile = screen_fundamental("NEG_FCF", info=info, _ticker=_make_ticker(INCREASING_FCF))
        assert profile.passes is False
        assert any("FCF" in r and "negative" in r for r in profile.fail_reasons)

    def test_zero_fcf_fails(self):
        info = {**GOOD_INFO, "freeCashflow": 0}
        profile = screen_fundamental("ZERO_FCF", info=info, _ticker=_make_ticker(INCREASING_FCF))
        assert profile.passes is False

    def test_missing_fcf_passes_with_benefit_of_doubt(self):
        info = {k: v for k, v in GOOD_INFO.items() if k != "freeCashflow"}
        profile = screen_fundamental("NO_FCF", info=info, _ticker=_make_ticker(INCREASING_FCF))
        # FCF field missing → skip check → should still pass other filters
        assert profile.free_cash_flow is None


# ---------------------------------------------------------------------------
# FCF yield filter
# ---------------------------------------------------------------------------

class TestFcfYieldFilter:
    def test_low_fcf_yield_fails(self):
        # FCF yield = freeCashflow / marketCap; set yield just below MIN_FCF_YIELD
        market_cap = 100_000_000_000
        fcf = int(market_cap * (config.MIN_FCF_YIELD - 0.005))   # ~1.5% yield
        info = {**GOOD_INFO, "freeCashflow": fcf, "marketCap": market_cap}
        profile = screen_fundamental("LOW_YIELD", info=info, _ticker=_make_ticker(INCREASING_FCF))
        assert profile.passes is False
        assert any("FCF yield" in r for r in profile.fail_reasons)

    def test_sufficient_fcf_yield_passes(self):
        market_cap = 100_000_000_000
        fcf = int(market_cap * 0.05)   # 5% yield — well above 2% threshold
        info = {**GOOD_INFO, "freeCashflow": fcf, "marketCap": market_cap}
        profile = screen_fundamental("GOOD_YIELD", info=info, _ticker=_make_ticker(INCREASING_FCF))
        assert profile.passes is True


# ---------------------------------------------------------------------------
# FCF trend filter
# ---------------------------------------------------------------------------

class TestFcfTrendFilter:
    def test_declining_fcf_fails(self):
        declining = [3_000_000_000, 4_000_000_000, 5_000_000_000]   # oldest → newest reversed
        ticker = _make_ticker(declining)
        profile = screen_fundamental("DEC_FCF", info=GOOD_INFO.copy(), _ticker=ticker)
        assert profile.fcf_increasing is False
        assert profile.passes is False
        assert any("FCF not increasing" in r for r in profile.fail_reasons)

    def test_increasing_fcf_passes(self):
        ticker = _make_ticker(INCREASING_FCF)
        profile = screen_fundamental("INC_FCF", info=GOOD_INFO.copy(), _ticker=ticker)
        assert profile.fcf_increasing is True

    def test_missing_cashflow_data_passes_with_benefit_of_doubt(self):
        ticker = _make_ticker(None)   # empty cashflow DataFrame
        profile = screen_fundamental("NO_CF", info=GOOD_INFO.copy(), _ticker=ticker)
        assert profile.fcf_increasing is None
        # None means data unavailable → benefit of the doubt, not a hard fail
        assert not any("FCF not increasing" in r for r in profile.fail_reasons)


# ---------------------------------------------------------------------------
# ROE filter
# ---------------------------------------------------------------------------

class TestRoeFilter:
    def test_low_roe_fails(self):
        info = {**GOOD_INFO, "returnOnEquity": config.MIN_RETURN_ON_EQUITY - 0.01}
        profile = screen_fundamental("LOW_ROE", info=info, _ticker=_make_ticker(INCREASING_FCF))
        assert profile.passes is False
        assert any("ROE" in r for r in profile.fail_reasons)

    def test_exactly_at_roe_threshold_passes(self):
        info = {**GOOD_INFO, "returnOnEquity": config.MIN_RETURN_ON_EQUITY}
        profile = screen_fundamental("OK_ROE", info=info, _ticker=_make_ticker(INCREASING_FCF))
        assert profile.passes is True

    def test_missing_roe_passes_with_benefit_of_doubt(self):
        info = {k: v for k, v in GOOD_INFO.items() if k != "returnOnEquity"}
        profile = screen_fundamental("NO_ROE", info=info, _ticker=_make_ticker(INCREASING_FCF))
        assert profile.return_on_equity is None


# ---------------------------------------------------------------------------
# Capex-to-FCF filter
# ---------------------------------------------------------------------------

class TestCapexToFcfFilter:
    def test_high_capex_ratio_fails(self):
        fcf = 5_000_000_000
        # Capex > 50% of FCF (yfinance reports as negative)
        capex = -int(fcf * (config.MAX_CAPEX_TO_FCF + 0.10))
        info = {**GOOD_INFO, "freeCashflow": fcf, "capitalExpenditures": capex}
        profile = screen_fundamental("HIGH_CAPEX", info=info, _ticker=_make_ticker(INCREASING_FCF))
        assert profile.passes is False
        assert any("Capex/FCF" in r for r in profile.fail_reasons)

    def test_acceptable_capex_ratio_passes(self):
        fcf = 5_000_000_000
        capex = -int(fcf * 0.20)   # 20% — well within 50% limit
        info = {**GOOD_INFO, "freeCashflow": fcf, "capitalExpenditures": capex}
        profile = screen_fundamental("OK_CAPEX", info=info, _ticker=_make_ticker(INCREASING_FCF))
        assert profile.passes is True

    def test_missing_capex_skips_check(self):
        info = {k: v for k, v in GOOD_INFO.items() if k != "capitalExpenditures"}
        profile = screen_fundamental("NO_CAPEX", info=info, _ticker=_make_ticker(INCREASING_FCF))
        assert profile.capex_to_fcf is None


# ---------------------------------------------------------------------------
# Multiple failures
# ---------------------------------------------------------------------------

class TestMultipleFailures:
    def test_all_failures_reported(self):
        """A truly bad stock should accumulate multiple fail reasons."""
        bad_info = {
            "forwardPE": 200.0,             # too high
            "profitMargins": -0.05,         # negative margin
            "debtToEquity": 500.0,          # 5× D/E
            "freeCashflow": -100_000,       # negative FCF
            "marketCap": 1_000_000_000,
            "returnOnEquity": -0.01,        # negative ROE
            "capitalExpenditures": -900_000,
        }
        ticker = _make_ticker([1_000_000, 2_000_000, 3_000_000])   # declining when reversed
        profile = screen_fundamental("BAD", info=bad_info, _ticker=ticker)
        assert profile.passes is False
        assert len(profile.fail_reasons) >= 3
