"""
tests/test_stockscreen_fundamentals.py — the fundamental screen on synthetic
quarterly/annual observations: FCF consistency, margin, growth, yield, size,
trend classification, status precedence, and every boundary case.
"""

from __future__ import annotations

from datetime import date

import pytest

from stockscreen import fundamentals as fd
from stockscreen.config import ScreenConfig

TODAY = date(2026, 10, 20)
ENDS = ["2023-09-30", "2023-12-31", "2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31",
        "2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31", "2026-03-31", "2026-06-30"]   # 12 quarters


M = 1_000_000          # tests speak in $ millions; observations are stored in dollars


def q(values, ends=ENDS):
    return [{"end": e, "value": v * M, "basis": "reported 3M", "filed": "2026-08-01", "first_filed": "2026-08-01"}
            for e, v in zip(ends, values)]


def yr(values, ends=("2024-12-31", "2025-12-31")):
    return [{"start": "", "end": e, "value": v * M, "basis": "reported FY", "filed": "2026-02-01", "first_filed": "2026-02-01"}
            for e, v in zip(ends, values)]


def make_fund(ocf=None, capex=None, rev=None, ni=None, ocf_y=None, capex_y=None, rev_y=None, shares=100.0 * M,
              share_end="2026-07-15", ends=ENDS):
    flat = lambda v: [v] * len(ends)                                         # noqa: E731
    return {
        "fetch": "ok", "latest_filed": "2026-08-01",
        "quarters": {"ocf": q(ocf or flat(30), ends), "capex": q(capex or flat(10), ends),
                     "revenue": q(rev or flat(100), ends), "net_income": q(ni or flat(15), ends), "disc_ops": []},
        "years": {"ocf": yr(ocf_y or [120, 120]), "capex": yr(capex_y or [40, 40]), "revenue": yr(rev_y or [400, 400]),
                  "net_income": yr([60, 60])},
        "shares": [{"value": shares, "end": share_end, "source": "test", "approx": False, "classes": 1}],
    }


def good_fund(**kw):
    """A company that clears every criterion: FCF $80M TTM, margin 18%, growth 20%, yield 5.3%, cap $1.5B."""
    base = dict(rev=[25] * 8 + [30] * 4, ni=[4] * 8 + [5.4] * 4, shares=150 * M)
    base.update(kw)
    return make_fund(**base)


def price(last=10.0):
    return {"last": last, "asof": "2026-10-20", "qe_close": {e: last for e in ENDS}}


def crit(ev, key):
    return next(c for c in ev["criteria"] if c["key"] == key)


# --- A. FCF consistency -----------------------------------------------------------------

def test_fcf_is_operating_cash_flow_minus_capex():
    ev = fd.evaluate_fundamentals(make_fund(ocf=[50] * 12, capex=[20] * 12), price(), ScreenConfig(), TODAY)
    assert ev["metrics"]["fcf_quarters"][-1]["fcf"] == 30 * M
    assert ev["metrics"]["ttm_fcf"] == 120 * M


def test_all_four_quarters_and_two_years_positive_passes():
    ev = fd.evaluate_fundamentals(make_fund(), price(), ScreenConfig(), TODAY)
    assert crit(ev, "fcf_consistency")["status"] == "pass"


def test_zero_fcf_in_one_quarter_fails():
    ocf = [30] * 11 + [10]                  # latest quarter: 10 - 10 capex = 0
    ev = fd.evaluate_fundamentals(make_fund(ocf=ocf), price(), ScreenConfig(), TODAY)
    c = crit(ev, "fcf_consistency")
    assert c["status"] == "fail" and "2026-06-30" in c["detail"]


def test_negative_fiscal_year_fails_even_when_quarters_are_positive():
    ev = fd.evaluate_fundamentals(make_fund(ocf_y=[120, 30], capex_y=[40, 40]), price(), ScreenConfig(), TODAY)
    assert crit(ev, "fcf_consistency")["status"] == "fail"


def test_missing_capex_is_not_treated_as_zero():
    fund = good_fund()
    fund["quarters"]["capex"] = fund["quarters"]["capex"][:-1]               # latest quarter's capex absent
    ev = fd.evaluate_fundamentals(fund, price(), ScreenConfig(), TODAY)
    assert crit(ev, "fcf_consistency")["status"] == "na"
    assert ev["status"] == "INSUFFICIENT_DATA"
    cfg = ScreenConfig(allow_missing_capex_as_zero=True)                      # only when explicitly enabled
    ev2 = fd.evaluate_fundamentals(fund, price(), cfg, TODAY)
    assert crit(ev2, "fcf_consistency")["status"] == "pass"
    assert any("assumed 0" in f for f in ev2["flags"])


def test_missing_quarter_or_year_is_insufficient_never_pass():
    fund = make_fund()
    del fund["quarters"]["ocf"][-2]                                           # a hole in the last four quarters
    assert crit(fd.evaluate_fundamentals(fund, price(), ScreenConfig(), TODAY), "fcf_consistency")["status"] == "na"
    fund = make_fund()
    fund["years"]["ocf"] = fund["years"]["ocf"][-1:]                          # only one fiscal year reported
    assert crit(fd.evaluate_fundamentals(fund, price(), ScreenConfig(), TODAY), "fcf_consistency")["status"] == "na"


# --- B. net margin (strictly greater than 10%) ------------------------------------------------

def test_margin_exactly_ten_percent_fails():
    ev = fd.evaluate_fundamentals(make_fund(rev=[100] * 12, ni=[10] * 12), price(), ScreenConfig(), TODAY)
    c = crit(ev, "net_margin")
    assert c["value"] == pytest.approx(0.10) and c["status"] == "fail"


def test_margin_just_above_ten_percent_passes():
    ev = fd.evaluate_fundamentals(make_fund(rev=[100] * 12, ni=[10.1] * 12), price(), ScreenConfig(), TODAY)
    assert crit(ev, "net_margin")["status"] == "pass"


def test_ttm_margin_uses_four_quarters():
    ni = [5] * 8 + [20, 20, 20, 20]
    ev = fd.evaluate_fundamentals(make_fund(ni=ni), price(), ScreenConfig(), TODAY)
    assert ev["metrics"]["net_margin"] == pytest.approx(80 / 400)


# --- C. revenue growth (strictly greater than 10%) --------------------------------------------------

def test_growth_exactly_ten_percent_fails_despite_float_noise():
    rev = [100] * 8 + [110] * 4                                   # TTM 440 vs prior TTM 400 → +10%... but 8..11 are the latest 4
    fund = make_fund(rev=[25] * 8 + [27.5] * 4)                   # TTM 110 vs 100
    ev = fd.evaluate_fundamentals(fund, price(), ScreenConfig(), TODAY)
    c = crit(ev, "revenue_growth")
    assert c["status"] == "fail"                                  # 110/100 - 1 is 0.10000000000000009 in floating point


def test_growth_above_ten_percent_passes_and_is_labelled_ttm():
    fund = make_fund(rev=[25] * 8 + [28] * 4)                     # +12%
    ev = fd.evaluate_fundamentals(fund, price(), ScreenConfig(), TODAY)
    c = crit(ev, "revenue_growth")
    assert c["status"] == "pass" and "TTM" in c["detail"]


def test_growth_falls_back_to_fiscal_year_and_says_so():
    fund = make_fund()
    fund["quarters"]["revenue"] = fund["quarters"]["revenue"][-5:]            # too few quarters for TTM growth
    fund["years"]["revenue"] = yr([400, 460], ends=("2024-12-31", "2025-12-31"))
    ev = fd.evaluate_fundamentals(fund, price(), ScreenConfig(), TODAY)
    c = crit(ev, "revenue_growth")
    assert c["status"] == "pass" and "fiscal-year" in c["detail"]
    assert any("fiscal-year basis" in f for f in ev["flags"])


def test_growth_unavailable_is_na():
    fund = make_fund()
    fund["quarters"]["revenue"] = fund["quarters"]["revenue"][-5:]
    fund["years"]["revenue"] = []
    assert crit(fd.evaluate_fundamentals(fund, price(), ScreenConfig(), TODAY), "revenue_growth")["status"] == "na"


# --- D. FCF yield (at least 5%) and E. size (at least $1B) --------------------------------------------

def test_fcf_yield_exactly_five_percent_passes():
    # TTM FCF = 4 x (30-10) = $80M; shares x price = 80 / 0.05 = $1,600M
    ev = fd.evaluate_fundamentals(make_fund(shares=160 * M), price(10.0), ScreenConfig(), TODAY)
    c = crit(ev, "fcf_yield")
    assert c["value"] == pytest.approx(0.05) and c["status"] == "pass"


def test_fcf_yield_just_below_five_percent_fails():
    ev = fd.evaluate_fundamentals(make_fund(shares=161 * M), price(10.0), ScreenConfig(), TODAY)
    assert crit(ev, "fcf_yield")["status"] == "fail"


def test_market_cap_exactly_one_billion_passes_and_below_fails():
    cfg = ScreenConfig()
    ev = fd.evaluate_fundamentals(make_fund(shares=100_000_000.0), price(10.0), cfg, TODAY)
    assert ev["metrics"]["market_cap"] == 1_000_000_000 and crit(ev, "market_cap")["status"] == "pass"
    ev = fd.evaluate_fundamentals(make_fund(shares=99_999_999.0), price(10.0), cfg, TODAY)
    assert crit(ev, "market_cap")["status"] == "fail"


def test_stale_share_count_is_never_used():
    fund = make_fund(share_end="2018-11-30")                                  # eight years old, like WDAY's SEC cover page
    ev = fd.evaluate_fundamentals(fund, price(), ScreenConfig(), TODAY)
    assert ev["metrics"]["market_cap"] is None
    assert crit(ev, "market_cap")["status"] == "na" and crit(ev, "fcf_yield")["status"] == "na"


def test_external_share_fallback_is_used_and_labelled():
    fund = make_fund(share_end="2018-11-30")
    fund["shares_external"] = {"value": 500.0, "end": "2026-10-20", "source": "Yahoo Finance share count (fallback)", "approx": False}
    ev = fd.evaluate_fundamentals(fund, price(10.0), ScreenConfig(), TODAY)
    assert ev["metrics"]["market_cap"] == 5000.0
    assert "Yahoo" in ev["metrics"]["shares"]["source"]


# --- F. trend classification ---------------------------------------------------------------------------------

CFG = ScreenConfig()


def test_trend_improving_declining_stable():
    assert fd.classify_trend([100, 110, 121, 133, 146], "level", 0.05, CFG)["label"] == "Improving"
    assert fd.classify_trend([100, 92, 85, 78, 71], "level", 0.05, CFG)["label"] == "Declining"
    assert fd.classify_trend([100, 101, 100, 102, 101], "level", 0.05, CFG)["label"] == "Stable"


def test_one_noisy_quarter_is_not_improving_or_declining():
    assert fd.classify_trend([100, 100, 100, 100, 112], "level", 0.05, CFG)["label"] == "Stable"   # one jump
    assert fd.classify_trend([100, 100, 100, 100, 94], "level", 0.05, CFG)["label"] == "Stable"    # one dip, within severe limit


def test_severe_drop_is_declining_even_if_sudden():
    assert fd.classify_trend([100, 100, 100, 100, 60], "level", 0.05, CFG)["label"] == "Declining"   # -40% > 3 x 5%


def test_change_exactly_at_tolerance_is_stable():
    assert fd.classify_trend([100, 101.25, 102.5, 103.75, 105], "level", 0.05, CFG)["label"] == "Stable"   # exactly +5%


def test_margin_trend_uses_percentage_points():
    assert fd.classify_trend([0.20, 0.19, 0.18, 0.17, 0.15], "pp", 0.01, CFG)["label"] == "Declining"
    assert fd.classify_trend([0.20, 0.2005, 0.199, 0.2002, 0.2], "pp", 0.01, CFG)["label"] == "Stable"


def test_too_few_points_is_insufficient_data():
    assert fd.classify_trend([100, 110, 120], "level", 0.05, CFG)["label"] == "Insufficient data"


def test_critical_declining_metric_blocks_eligibility():
    rev = [100] * 8 + [100, 90, 80, 70]                           # revenue falling hard
    ev = fd.evaluate_fundamentals(make_fund(rev=rev), price(), ScreenConfig(), TODAY)
    assert ev["trends"]["revenue"]["label"] == "Declining"
    assert crit(ev, "trend")["status"] == "fail" and ev["status"] == "FAILED"


def test_non_critical_declining_metric_does_not_block():
    cfg = ScreenConfig(critical_trend_metrics=("fcf",))
    rev = [100] * 8 + [100, 90, 80, 70]
    ev = fd.evaluate_fundamentals(make_fund(rev=rev, ni=[8] * 8 + [8, 7.2, 6.4, 5.6]), price(), cfg, TODAY)
    assert crit(ev, "trend")["status"] == "pass"


def test_yield_decline_from_a_rising_price_is_not_called_deterioration():
    fund = make_fund(shares=160 * M)
    px = {"last": 20.0, "asof": "2026-10-20", "qe_close": dict(zip(ENDS, [5, 5, 5, 5, 5, 6, 8, 10, 12, 14, 17, 20]))}
    ev = fd.evaluate_fundamentals(fund, px, ScreenConfig(), TODAY)
    ytr = ev["trends"]["fcf_yield"]
    assert ev["trends"]["fcf"]["label"] == "Stable"
    assert ytr["label"] == "Stable" and "price rose" in ytr["note"]


# --- status precedence and staleness ------------------------------------------------------------------------------

def test_a_company_that_meets_everything_passes():
    ev = fd.evaluate_fundamentals(good_fund(), price(10.0), ScreenConfig(), TODAY)
    assert ev["status"] == "PASSED", ev["reasons"]


def test_stale_data_is_not_a_pass():
    ev = fd.evaluate_fundamentals(good_fund(), price(10.0), ScreenConfig(), date(2027, 3, 1))   # latest quarter 244 days old
    assert ev["status"] == "STALE_DATA" and ev["stale"]


def test_failure_takes_precedence_over_missing_data():
    fund = good_fund()
    fund["shares"] = []                                                       # yield/market cap unknown
    fund["quarters"]["net_income"] = q([1] * 12)                              # margin 4%: a definite failure
    ev = fd.evaluate_fundamentals(fund, price(), ScreenConfig(), TODAY)
    assert ev["status"] == "FAILED"


def test_unavailable_fundamentals_are_reported_not_passed():
    ev = fd.evaluate_fundamentals({"fetch": "no_statements", "error": "fund or ETF"}, None, ScreenConfig(), TODAY)
    assert ev["status"] == "INSUFFICIENT_DATA" and "fund or ETF" in ev["reasons"][0]
    ev = fd.evaluate_fundamentals({"fetch": "error", "error": "SEC fetch failed"}, None, ScreenConfig(), TODAY)
    assert ev["status"] == "ERROR"


# --- independence of the income-statement criteria from the FCF inputs ------------------------------------------

def test_bank_without_capex_still_gets_margin_and_growth():
    fund = good_fund()
    fund["quarters"]["capex"] = []                    # banks and insurers report no capex line
    fund["years"]["capex"] = []
    ev = fd.evaluate_fundamentals(fund, price(10.0), ScreenConfig(), TODAY)
    assert crit(ev, "net_margin")["status"] == "pass" and crit(ev, "revenue_growth")["status"] == "pass"
    assert crit(ev, "market_cap")["status"] == "pass"
    assert crit(ev, "fcf_consistency")["status"] == "na" and crit(ev, "fcf_yield")["status"] == "na"
    assert ev["status"] == "INSUFFICIENT_DATA"        # FCF cannot be computed, so it is never a pass
    assert ev["trends"]["revenue"]["label"] != "Insufficient data" and ev["trends"]["net_margin"]["label"] != "Insufficient data"


# --- financial companies whose investment flows sit inside operating cash flow (Blackstone) ----------------------------

def bx_like(sic="6282", investing_m=-4.0, ocf_year=4663, capex_year=4.0):
    """FY: OCF and capex as given; reported investing total as given. Quarterly data is a clean passing company."""
    fund = good_fund()
    fund["years"]["ocf"] = yr([ocf_year / 2, ocf_year])
    fund["years"]["capex"] = yr([capex_year, capex_year])
    fund["years"]["investing"] = yr([investing_m, investing_m])
    fund["sic"], fund["sic_desc"] = sic, "Investment Advice"
    return fund


def test_investing_equal_to_capex_at_an_asset_manager_makes_fcf_not_meaningful():
    ev = fd.evaluate_fundamentals(bx_like(), price(10.0), ScreenConfig(), TODAY)
    assert ev["metrics"]["fcf_not_meaningful"] is True
    assert crit(ev, "fcf_consistency")["status"] == "na" and crit(ev, "fcf_yield")["status"] == "na"
    assert "investing activities" in crit(ev, "fcf_yield")["detail"] and "inside operating cash flow" in ev["metrics"]["fcf_note"]
    assert ev["status"] == "INSUFFICIENT_DATA"                                      # never a pass
    assert ev["metrics"]["fcf_yield"] is None and ev["metrics"]["fcf_yield_as_filed"] is not None   # shown, not hidden
    assert ev["trends"]["fcf"]["label"] == "Insufficient data"
    assert crit(ev, "net_margin")["status"] == "pass" and crit(ev, "revenue_growth")["status"] == "pass"   # other criteria still work


def test_same_pattern_outside_finance_is_untouched():
    ev = fd.evaluate_fundamentals(bx_like(sic="7372"), price(10.0), ScreenConfig(), TODAY)
    assert not ev["metrics"].get("fcf_not_meaningful") and ev["status"] == "PASSED"


def test_financial_company_with_real_investing_flows_keeps_its_fcf_but_is_flagged():
    ev = fd.evaluate_fundamentals(bx_like(investing_m=-399.0), price(10.0), ScreenConfig(), TODAY)   # SEIC-like: investing >> capex
    assert not ev["metrics"].get("fcf_not_meaningful") and ev["status"] == "PASSED"
    assert any("financial company" in f for f in ev["flags"])


def test_reits_are_not_treated_as_financial():
    assert fd.financial_group("6798") is None and fd.financial_group("6510") is None
    assert [fd.financial_group(s) for s in ("6021", "6141", "6211", "6282", "6331", "6411", "6770", "7372", "")] == \
        ["bank", "credit", "broker/asset manager", "broker/asset manager", "insurance", "insurance", "investment holding", None, None]


def test_policy_flag_and_off():
    flag = fd.evaluate_fundamentals(bx_like(), price(10.0), ScreenConfig(financial_fcf_policy="flag"), TODAY)
    assert not flag["metrics"].get("fcf_not_meaningful") and flag["status"] == "PASSED"
    assert any("not meaningful" in f for f in flag["flags"])                      # warned, not blanked
    off = fd.evaluate_fundamentals(bx_like(), price(10.0), ScreenConfig(financial_fcf_policy="off"), TODAY)
    assert off["status"] == "PASSED" and not any("not meaningful" in f for f in off["flags"])


def test_missing_investing_total_or_capex_never_triggers_the_guard():
    fund = bx_like()
    fund["years"]["investing"] = []
    assert not fd.evaluate_fundamentals(fund, price(10.0), ScreenConfig(), TODAY)["metrics"].get("fcf_not_meaningful")
    fund = bx_like()
    fund["years"]["capex"] = []                                                  # banks: no capex line at all
    assert not fd.evaluate_fundamentals(fund, price(10.0), ScreenConfig(), TODAY)["metrics"].get("fcf_not_meaningful")


def test_fcf_components_are_left_out_of_the_score_when_not_meaningful():
    from stockscreen import scoring as sc
    ev = fd.evaluate_fundamentals(bx_like(), price(10.0), ScreenConfig(), TODAY)
    comp = sc.fundamental_score(ev, ScreenConfig())["components"]
    assert comp["fcf_consistency"] is None and comp["fcf_trend"] is None and comp["fcf_yield"] is None
    assert comp["net_margin"] is not None
