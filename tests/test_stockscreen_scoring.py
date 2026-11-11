"""
tests/test_stockscreen_scoring.py — opportunity classification, the three
heuristic scores, missing-data handling, and the config round trip.
"""

from __future__ import annotations

import json

import pytest

from stockscreen import scoring as sc
from stockscreen.config import ScreenConfig, load_config, save_config

CFG = ScreenConfig()
PASS = [{"key": "fcf_consistency", "status": "pass"}, {"key": "trend", "status": "pass"}]


def opp(status="PASSED", criteria=PASS, zone="Low-price candidate", label="Healthy pullback", signals=()):
    return sc.classify_opportunity(status, criteria, zone, {"label": label, "signals": list(signals)}, CFG)["code"]


# --- opportunity classes ---------------------------------------------------------------------------

def test_quality_stock_not_near_lows_is_monitor():
    assert opp(zone="Mid-range") == "A"
    assert opp(zone="Near highs") == "A"
    assert opp(zone="Insufficient history") == "A"


def test_low_price_with_healthy_pullback_is_c():
    assert opp(zone="Deep value-zone candidate", label="Healthy pullback") == "C"
    assert opp(label="Strong uptrend") == "C"


def test_low_price_with_confirmed_reversal_is_d_and_unconfirmed_is_b():
    assert opp(label="Trend reversal candidate", signals=["price reclaimed SMA20", "MACD bullish"]) == "D"
    assert opp(label="Trend reversal candidate", signals=["price reclaimed SMA20"]) == "B"


def test_low_price_without_technical_confirmation_is_b():
    assert opp(label="Sideways/consolidating") == "B"
    assert opp(label="Insufficient data") == "B"


def test_low_price_in_a_downtrend_is_a_warning_not_a_buy():
    assert opp(label="Downtrend") == "E"


def test_low_price_with_declining_fundamentals_is_a_warning_and_kept_visible():
    crit = [{"key": "fcf_consistency", "status": "pass"}, {"key": "net_margin", "status": "pass"},
            {"key": "trend", "status": "fail"}]
    assert opp(status="FAILED", criteria=crit, zone="Low-price candidate") == "E"
    assert opp(status="FAILED", criteria=crit, zone="Mid-range") == "X"           # not low: just ineligible


def test_failing_other_criteria_is_not_eligible_even_when_cheap():
    crit = [{"key": "fcf_yield", "status": "fail"}, {"key": "trend", "status": "fail"}]
    assert opp(status="FAILED", criteria=crit, zone="Deep value-zone candidate") == "X"


def test_not_passed_never_reaches_buy_zones():
    for st in ("INSUFFICIENT_DATA", "STALE_DATA", "ERROR"):
        assert opp(status=st, criteria=[], zone="Deep value-zone candidate", label="Healthy pullback") == "X"


# --- scores ------------------------------------------------------------------------------------------

def test_combine_renormalises_over_available_components():
    r = sc._combine({"a": 100.0, "b": 0.0, "c": None}, {"a": 0.5, "b": 0.25, "c": 0.25}, 0.6)
    assert r["score"] == pytest.approx(100 * 0.5 / 0.75, abs=0.05) and r["coverage"] == pytest.approx(0.75)
    assert r["missing"] == ["c"]


def test_score_is_none_when_too_little_is_computable():
    r = sc._combine({"a": 100.0, "b": None, "c": None}, {"a": 0.2, "b": 0.4, "c": 0.4}, 0.6)
    assert r["score"] is None and r["coverage"] == pytest.approx(0.2)


def test_missing_data_never_scores_best():
    full = sc._combine({"a": 40.0, "b": 40.0}, {"a": 0.5, "b": 0.5}, 0.6)["score"]
    missing = sc._combine({"a": 40.0, "b": None}, {"a": 0.5, "b": 0.5}, 0.6)
    assert missing["score"] is None or missing["score"] <= full


def test_scale_function_hits_threshold_and_cap():
    assert sc._scale(0.10, 0.10, 0.30) == 50.0 and sc._scale(0.30, 0.10, 0.30) == 100.0
    assert sc._scale(0.5, 0.10, 0.30) == 100.0 and sc._scale(0.0, 0.10, 0.30) == 0.0 and sc._scale(None, 0.1, 0.3) is None


def test_price_score_rewards_the_bottom_of_the_range_and_skips_unavailable_lookbacks():
    low = {"52w": {"available": True, "range_pos": 5, "percentile": 5, "pct_below_high": 50},
           "2y": {"available": True, "range_pos": 5, "percentile": 5}, "3y": {"available": True, "range_pos": 5, "percentile": 5}}
    high = {k: dict(v, range_pos=95, percentile=95) for k, v in low.items()}
    high["52w"]["pct_below_high"] = 2
    assert sc.price_score(low, CFG)["score"] > 85 > 15 > sc.price_score(high, CFG)["score"]
    short = {"52w": low["52w"], "2y": {"available": False}, "3y": {"available": False}}
    r = sc.price_score(short, CFG)
    assert r["score"] is not None and r["coverage"] < 1 and "range_2y" in r["missing"]


def test_technical_score_orders_trends():
    strong = {"price": 120, "sma20": 115, "sma50": 110, "sma200": 100, "sma50_slope": .02, "sma200_slope": .01, "rsi": 58,
              "macd_hist": 1.0, "macd_hist_prev": 0.5}
    weak = {"price": 80, "sma20": 85, "sma50": 90, "sma200": 100, "sma50_slope": -.02, "sma200_slope": -.01, "rsi": 25,
            "macd_hist": -1.0, "macd_hist_prev": -0.5}
    s = sc.technical_score(strong, {"label": "Strong uptrend"}, CFG)["score"]
    w = sc.technical_score(weak, {"label": "Downtrend"}, CFG)["score"]
    assert s > 90 and w < 25


def test_fundamental_score_counts_missing_trends_out():
    fe = {"metrics": {"fcf_quarters": [{"fcf": 1}] * 8, "net_margin": 0.2, "revenue_growth": 0.2, "fcf_yield": 0.08},
          "trends": {"fcf": {"label": "Improving"}, "revenue": {"label": "Insufficient data"}}}
    r = sc.fundamental_score(fe, CFG)
    assert r["score"] is not None and r["coverage"] == 1.0                      # stability uses the one classified trend
    assert r["components"]["fcf_trend"] == 100.0


def test_composite_uses_weights_and_requires_the_fundamental_score():
    assert sc.composite(80, 60, 40, CFG) == pytest.approx(0.45 * 80 + 0.35 * 60 + 0.20 * 40)
    assert sc.composite(None, 60, 40, CFG) is None
    assert sc.composite(80, None, None, CFG) == 80.0                            # renormalised over what exists


# --- config persistence --------------------------------------------------------------------------------

def test_config_round_trip_and_unknown_keys(tmp_path):
    path = tmp_path / "cfg.json"
    cfg = ScreenConfig(min_fcf_yield=0.07, critical_trend_metrics=("fcf",), weights_rank={"fundamental": 0.6, "price": 0.2, "technical": 0.2})
    save_config(cfg, path)
    loaded = load_config(path)
    assert loaded.min_fcf_yield == 0.07 and loaded.critical_trend_metrics == ("fcf",)
    assert loaded.weights_rank["fundamental"] == 0.6
    data = json.loads(path.read_text())
    data["not_a_setting"] = 1
    data["min_market_cap"] = "2000000000"
    path.write_text(json.dumps(data))
    again = load_config(path)
    assert again.min_market_cap == 2e9 and not hasattr(again, "not_a_setting")
    assert load_config(tmp_path / "missing.json").min_fcf_yield == 0.05          # defaults when no file
