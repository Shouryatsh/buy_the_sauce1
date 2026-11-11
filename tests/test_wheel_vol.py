"""
tests/test_wheel_vol.py — volatility percentiles and the earnings IV
run-up / crush analysis in wheel_vol.py, on synthetic series with known answers.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

import wheel_vol as wv


def _bdays(start: str, n: int) -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


class TestPercentiles:
    def test_top_and_bottom(self):
        assert wv.percentile_rank(pd.Series(np.arange(1.0, 102.0)), 100) == 100.0
        assert wv.percentile_rank(pd.Series(list(np.arange(2.0, 102.0)) + [1.0]), 100) == 0.0

    def test_labels(self):
        assert wv.label(95) == "very high" and wv.label(80) == "high"
        assert wv.label(50) == "normal" and wv.label(20) == "low" and wv.label(5) == "very low"

    def test_realized_vol_of_constant_returns_is_zero(self):
        close = pd.Series(100 * 1.01 ** np.arange(60))
        assert wv.realized_vol(close, 20).dropna().abs().max() < 1e-9


class TestPreEventPosition:
    idx = _bdays("2026-01-05", 30)   # Mon Jan 5 ...

    def test_before_open_uses_previous_close(self):
        # Thu Jan 15 before the open → last pre-event close is Wed Jan 14
        pos = wv.pre_event_pos(self.idx, date(2026, 1, 15), "BMO")
        assert self.idx[pos].date() == date(2026, 1, 14)

    def test_after_close_uses_same_day(self):
        pos = wv.pre_event_pos(self.idx, date(2026, 1, 15), "AMC")
        assert self.idx[pos].date() == date(2026, 1, 15)

    def test_unknown_timing_is_treated_as_before_open(self):
        pos = wv.pre_event_pos(self.idx, date(2026, 1, 15), "?")
        assert self.idx[pos].date() == date(2026, 1, 14)

    def test_event_outside_data_returns_none(self):
        assert wv.pre_event_pos(self.idx, date(2025, 6, 1), "BMO") is None


def _iv_with_event(idx, pre_pos, base=0.20, peak=0.40, after=0.22):
    """Flat base IV, linear run-up over the 10 sessions before pre_pos, crush after."""
    v = np.full(len(idx), base)
    v[pre_pos - 10:pre_pos + 1] = np.linspace(base, peak, 11)
    v[pre_pos + 1:] = after
    return pd.Series(v, index=idx)


class TestEarningsTable:
    def test_run_up_and_crush(self):
        idx = _bdays("2026-01-05", 40)
        pre = 20                                     # pre-event close at idx[20]
        iv = _iv_with_event(idx, pre)
        event_day = idx[pre + 1].date()              # reported before the open next session
        close = pd.Series(100.0, index=idx)
        close.iloc[pre + 1:] = 95.0                  # -5% gap on the news
        t = wv.earnings_table(iv, close, [(event_day, "BMO")])
        row = t.iloc[0]
        wk = np.linspace(0.20, 0.40, 11)[5]          # IV 5 sessions before the pre-event close
        assert row["IV 1 week before"] == pytest.approx(wk)
        assert row["IV last close before"] == pytest.approx(0.40)
        assert row["Run-up"] == pytest.approx(0.40 / wk - 1)
        assert row["IV after"] == pytest.approx(0.22)
        assert row["Crush"] == pytest.approx(0.22 / 0.40 - 1)
        assert row["Stock move"] == pytest.approx(-0.05)

    def test_upcoming_event_has_no_row(self):
        idx = _bdays("2026-01-05", 25)
        iv = pd.Series(0.2, index=idx)
        t = wv.earnings_table(iv, None, [(idx[-1].date() + pd.Timedelta(days=10), "BMO")])
        assert t.empty


class TestEventPaths:
    def test_aligned_on_pre_event_close(self):
        idx = _bdays("2026-01-05", 60)
        iv = _iv_with_event(idx, 30)
        paths = wv.event_paths(iv, [(idx[31].date(), "BMO")], k_min=-20, k_max=5)
        col = paths.iloc[:, 0]
        assert col.loc[0] == pytest.approx(0.40)
        assert col.loc[1] == pytest.approx(0.22)
        assert col.loc[-20] == pytest.approx(0.20)


class TestTradingDaysToPreClose:
    def test_axp_oct_2026(self):
        # Latest bar Tue Oct 13 2026; AXP reports Fri Oct 23 before the open →
        # pre-event close Thu Oct 22; sessions Oct 13–21 in between = 7.
        assert wv.trading_days_to_pre_close("2026-10-13", date(2026, 10, 23), "BMO") == 7

    def test_after_close_same_day_is_zero(self):
        assert wv.trading_days_to_pre_close("2026-10-13", date(2026, 10, 13), "AMC") == 0


class TestCompareToPastEarnings:
    def test_today_versus_same_point_in_past_cycles(self):
        # Three past events with identical paths; "today" sits 5 sessions
        # before the next pre-event close with IV above all past same-point values.
        idx = _bdays("2025-01-06", 200)
        v = np.full(len(idx), 0.20)
        pres = [40, 100, 160]
        for p in pres:
            v[p - 10:p + 1] = np.linspace(0.20, 0.40, 11)
            v[p + 1:p + 3] = 0.22
        v[-1] = 0.50                                  # today: unusually high
        iv = pd.Series(v, index=idx)
        past = [(idx[p + 1].date(), "BMO") for p in pres]
        last = idx[-1]
        nxt = (np.busday_offset(np.datetime64(last.date()), 6, roll="forward")).astype(object)  # pre-close in 5 sessions
        cmp_ = wv.compare_to_past_earnings(iv, past, (nxt, "BMO"))
        assert cmp_.k == -5
        same_point = np.linspace(0.20, 0.40, 11)[5]
        assert cmp_.same_point.round(6).tolist() == [round(same_point, 6)] * 3
        assert cmp_.pct == 100.0
        assert cmp_.typical_peak == pytest.approx(0.40)
        assert cmp_.typical_runup_from_here == pytest.approx(0.40 / same_point - 1)
