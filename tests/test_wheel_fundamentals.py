"""
tests/test_wheel_fundamentals.py — Unit tests for wheel_fundamentals.py.

All offline — EDGAR fundamentals are injected via pre-built `info` dicts;
price is monkeypatched via edgar.get_price_history so no network calls happen.
"""

from __future__ import annotations

import pandas as pd
import pytest

import wheel_config as wc
import wheel_fundamentals as wf


def _price_df(price: float) -> pd.DataFrame:
    return pd.DataFrame({"Close": [price - 1, price]}, index=pd.date_range("2026-01-01", periods=2))


GOOD_INFO = {
    "freeCashflow": 5_000_000_000,
    "marketCap": 50_000_000_000,      # FCF yield = 10%
    "returnOnEquity": 0.15,
    "_fcf_history": [5_000_000_000, 4_000_000_000, 3_000_000_000],
    "_net_income_history": [1_100_000_000, 1_000_000_000],
    "_revenue_history": [10_000_000_000, 9_000_000_000],
}


class TestPriceGate:
    def test_price_above_max_assignment_fails(self, monkeypatch):
        monkeypatch.setattr(wf.edgar, "get_price_history", lambda *a, **k: (_price_df(wc.MAX_ASSIGNMENT_PRICE + 50), "Stooq"))
        profile = wf.screen_wheel_candidate("EXPENSIVE", info=GOOD_INFO.copy())
        assert profile.passes is False
        assert any("MAX_ASSIGNMENT_PRICE" in r for r in profile.fail_reasons)

    def test_price_under_max_assignment_passes_gate(self, monkeypatch):
        monkeypatch.setattr(wf.edgar, "get_price_history", lambda *a, **k: (_price_df(100), "Stooq"))
        profile = wf.screen_wheel_candidate("CHEAP", info=GOOD_INFO.copy())
        assert not any("MAX_ASSIGNMENT_PRICE" in r for r in profile.fail_reasons)


class TestFcfHardFilter:
    def test_negative_fcf_fails(self, monkeypatch):
        monkeypatch.setattr(wf.edgar, "get_price_history", lambda *a, **k: (_price_df(100), "Stooq"))
        info = {**GOOD_INFO, "freeCashflow": -1}
        profile = wf.screen_wheel_candidate("NEGFCF", info=info)
        assert profile.passes is False
        assert any("FCF" in r for r in profile.fail_reasons)

    def test_missing_fcf_fails_and_flags_unreliable(self, monkeypatch):
        monkeypatch.setattr(wf.edgar, "get_price_history", lambda *a, **k: (_price_df(100), "Stooq"))
        info = {k: v for k, v in GOOD_INFO.items() if k != "freeCashflow"}
        profile = wf.screen_wheel_candidate("NOFCF", info=info)
        assert profile.passes is False
        assert profile.data_reliable is False


class TestFcfYieldEliteOverride:
    def test_standard_bar_used_when_not_elite(self, monkeypatch):
        monkeypatch.setattr(wf.edgar, "get_price_history", lambda *a, **k: (_price_df(100), "Stooq"))
        # 3% yield fails standard bar (5%) and this stock isn't elite
        info = {**GOOD_INFO, "marketCap": GOOD_INFO["freeCashflow"] / 0.03, "returnOnEquity": 0.05}
        profile = wf.screen_wheel_candidate("MEDIOCRE", info=info)
        assert profile.is_elite is False
        assert profile.passes is False
        assert any("FCF_YIELD_MIN" in r and "ELITE" not in r for r in profile.fail_reasons)

    def test_elite_bar_lets_lower_yield_pass(self, monkeypatch):
        monkeypatch.setattr(wf.edgar, "get_price_history", lambda *a, **k: (_price_df(100), "Stooq"))
        # 3% yield: fails standard (5%) but clears elite bar (2%)
        info = {
            **GOOD_INFO,
            "marketCap": GOOD_INFO["freeCashflow"] / 0.03,
            "returnOnEquity": 0.30,                                   # > ELITE_MIN_ROE
            "_net_income_history": [2_500_000_000, 2_000_000_000],    # 25% margin, growing
            "_revenue_history": [10_000_000_000, 8_000_000_000],      # 25% revenue growth
        }
        profile = wf.screen_wheel_candidate("ELITE", info=info)
        assert profile.is_elite is True
        assert not any("FCF" in r and "yield" in r for r in profile.fail_reasons)


class TestQualityScore:
    def test_score_is_bounded_0_100(self, monkeypatch):
        monkeypatch.setattr(wf.edgar, "get_price_history", lambda *a, **k: (_price_df(100), "Stooq"))
        profile = wf.screen_wheel_candidate("GOOD", info=GOOD_INFO.copy())
        assert 0.0 <= profile.quality_score <= 100.0

    def test_stronger_trends_score_higher(self, monkeypatch):
        monkeypatch.setattr(wf.edgar, "get_price_history", lambda *a, **k: (_price_df(100), "Stooq"))
        weak = wf.screen_wheel_candidate("WEAK", info={
            **GOOD_INFO,
            "_fcf_history": [3_000_000_000, 4_000_000_000, 5_000_000_000],  # declining
            "_net_income_history": [900_000_000, 1_000_000_000],            # declining margin
        })
        strong = wf.screen_wheel_candidate("STRONG", info=GOOD_INFO.copy())
        assert strong.quality_score > weak.quality_score

    def test_missing_data_does_not_crash_and_flags_unreliable(self, monkeypatch):
        monkeypatch.setattr(wf.edgar, "get_price_history", lambda *a, **k: (_price_df(100), "Stooq"))
        profile = wf.screen_wheel_candidate("SPARSE", info={"freeCashflow": 1_000_000, "marketCap": 20_000_000})
        assert profile.data_reliable is False
        assert profile.quality_score is not None


class TestFcfTrendScore:
    def test_all_up_years_scores_one(self):
        assert wf._fcf_trend_score([5, 4, 3, 2]) == 1.0

    def test_all_down_years_scores_zero(self):
        assert wf._fcf_trend_score([2, 3, 4, 5]) == 0.0

    def test_insufficient_history_returns_none(self):
        assert wf._fcf_trend_score([5]) is None
        assert wf._fcf_trend_score([]) is None
