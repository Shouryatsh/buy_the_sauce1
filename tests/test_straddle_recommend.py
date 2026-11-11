"""Dashboard straddle page logic: McMillan's criteria on a quoted straddle, limits, sizing, expiry pick, follow-up."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import wheel_config as wc
from premia import mcmillan as mm
from premia import pricing
from straddle_options import StraddleQuote
from straddle_recommend import (combine_iv, evaluate_quote, follow_up_for, held_straddles, pick_expiry, rank,
                                recent_duration)
from tests.premia.fixtures import panel
from wheel_ibkr import PositionRecord

TODAY = date(2026, 11, 10)


def _ohlc(seed=1, n=1600, vol=0.30):
    df = panel(n=n, seed=seed, vol=vol, start="2020-11-10")
    return df[["open", "high", "low", "close"]].rename(columns=str.capitalize)


def _quote(S, iv=0.20, dte=180, spread=0.02, oi=2000, symbol="TST", K=None):
    K = K or round(S)
    T = dte / 365
    c = float(pricing.price(S, [K], T, 0.04, 0.0, [iv], "C")[0])
    p = float(pricing.price(S, [K], T, 0.04, 0.0, [iv], "P")[0])
    h = spread / 2
    exp = (TODAY + timedelta(days=dte)).strftime("%Y%m%d")
    return StraddleQuote(symbol, exp, dte, K, S, c * (1 - h), c * (1 + h), p * (1 - h), p * (1 + h), iv, iv, "live", oi, oi)


def test_cheap_straddle_on_a_volatile_stock_meets_all_criteria():
    h = _ohlc(vol=0.40)
    S = float(h["Close"].iloc[-1])
    r = evaluate_quote(_quote(S, iv=0.15), h, iv_pct=5.0, nlv=1e6, today=TODAY)
    assert r.criteria.ever_prob >= mm.EVER_PROB_MIN and r.criteria.hist_freq >= mm.HIST_FREQ_MIN
    assert r.passes and r.contracts_suggested >= 1
    assert r.criteria.breakeven_up == pytest.approx(r.strike + r.cost)
    assert r.quote_type == "live" and r.stop_value == pytest.approx(0.4 * r.cost)


def test_each_criterion_and_limit_reports_why():
    h = _ohlc(vol=0.40)
    S = float(h["Close"].iloc[-1])
    text = lambda r: " ".join(r.reasons)
    assert "1: IV percentile" in text(evaluate_quote(_quote(S, iv=0.15), h, iv_pct=40.0, nlv=1e6, today=TODAY))
    assert "2: P(ever" in text(evaluate_quote(_quote(S, iv=0.90), h, iv_pct=5.0, nlv=1e6, today=TODAY))   # far breakevens
    assert "limit: straddle spread" in text(evaluate_quote(_quote(S, iv=0.15, spread=0.3), h, 5.0, 1e6, today=TODAY))
    assert "limit: open interest" in text(evaluate_quote(_quote(S, iv=0.15, oi=50), h, 5.0, 1e6, today=TODAY))
    assert "already held" in text(evaluate_quote(_quote(S, iv=0.15), h, 5.0, 1e6, already_held=True, today=TODAY))
    calm = _ohlc(vol=0.08)
    Sc = float(calm["Close"].iloc[-1])
    r = evaluate_quote(_quote(Sc, iv=0.30), calm, 5.0, 1e6, today=TODAY)
    assert any(x.startswith("2:") or x.startswith("3:") for x in r.reasons)


def test_size_budget_one_contract_rule_and_total_cap():
    h = _ohlc(vol=0.40)
    S = float(h["Close"].iloc[-1])
    q = _quote(S, iv=0.15)
    r = evaluate_quote(q, h, 5.0, nlv=1e6, today=TODAY)
    per = r.cost * 100
    assert r.contracts_suggested == max(int(wc.STRADDLE_BUDGET_PCT_NLV * 1e6 // per), 1)
    tiny = evaluate_quote(q, h, 5.0, nlv=per / wc.STRADDLE_ONE_CONTRACT_MAX_PCT_NLV * 0.9, today=TODAY)
    assert tiny.contracts_suggested == 0 and any(x.startswith("size") for x in tiny.reasons)
    full = evaluate_quote(q, h, 5.0, nlv=1e6, committed_premium=wc.STRADDLE_MAX_TOTAL_PCT_NLV * 1e6, today=TODAY)
    assert full.contracts_suggested == 0 and not full.passes


def test_pick_expiry_nearest_six_months_with_three_months_minimum():
    qs = [_quote(100, dte=d) for d in (45, 95, 150, 215)]
    assert pick_expiry(qs).dte == 150
    assert pick_expiry([_quote(100, dte=60)]) is None


def test_rank_puts_full_passes_first_by_ever_probability():
    h = _ohlc(vol=0.40)
    S = float(h["Close"].iloc[-1])
    a = evaluate_quote(_quote(S, iv=0.15, symbol="A"), h, 5.0, 1e6, today=TODAY)
    b = evaluate_quote(_quote(S, iv=0.12, symbol="B"), h, 5.0, 1e6, today=TODAY)
    c = evaluate_quote(_quote(S, iv=0.15, symbol="C"), h, 50.0, 1e6, today=TODAY)
    buys, rest = rank([a, b, c])
    assert [x.symbol for x in buys] == sorted(["A", "B"], key=lambda s: -{"A": a, "B": b}[s].criteria.ever_prob)
    assert rest == [c]


def test_iv_history_top_up():
    idx = pd.bdate_range("2020-01-01", periods=600)
    stored = pd.Series(0.3, index=idx)
    recent = pd.Series(0.2, index=pd.bdate_range(idx[-10], periods=30))
    both = combine_iv(stored, recent)
    assert both.index.is_monotonic_increasing and not both.index.duplicated().any() and both.iloc[-1] == 0.2
    assert recent_duration(pd.Series(dtype=float)) == "10 Y"
    assert recent_duration(stored, today=idx[-1].date() + timedelta(days=3)) == "30 D"
    assert recent_duration(stored, today=idx[-1].date() + timedelta(days=100)) == "107 D"


def test_follow_up_on_held_straddle():
    exp = (TODAY + timedelta(days=120)).strftime("%Y%m%d")
    c = PositionRecord("XYZ", "OPT", "C", 100.0, exp, 2, avg_cost=600.0, market_price=2.0)
    p = PositionRecord("XYZ", "OPT", "P", 100.0, exp, 2, avg_cost=500.0, market_price=2.2)
    pairs = held_straddles([c, p, PositionRecord("XYZ", "OPT", "P", 90.0, exp, -1)])
    assert len(pairs) == 1
    h = follow_up_for("XYZ", 100.0, exp, c, p, close=100.5, sma20=100.0, today=TODAY)
    assert h.cost == pytest.approx(11.0) and h.action.mode == "closed" and "60" in h.action.reason   # 4.2 <= 4.4
    up = follow_up_for("XYZ", 100.0, exp, c, PositionRecord("XYZ", "OPT", "P", 100.0, exp, 2, avg_cost=500.0,
                                                            market_price=0.5), close=113.0, sma20=105.0, today=TODAY)
    assert up.action.mode == "call" and up.action.sell_puts == 2 and up.action.sell_calls == 1
