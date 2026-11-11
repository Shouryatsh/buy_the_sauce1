"""
tests/test_wheel_recommend.py — composite ranking uses implied vol as a
percentile of its OWN history (McMillan), not IV / realized vol.
"""

from __future__ import annotations

import pytest

import wheel_recommend as wr
import wheel_options as wo
from wheel_fundamentals import WheelFundamentalProfile


def _profile(sym, quality=80.0):
    p = WheelFundamentalProfile(symbol=sym)
    p.quality_score = quality
    return p


def _cand(sym, market_iv=0.30, realized_vol=0.20):
    from datetime import datetime, timedelta
    expiry = (datetime.now().date() + timedelta(days=38)).strftime("%Y%m%d")
    c = wo.build_candidate(sym, expiry, 90.0, 100.0, 1.50, market_iv, realized_vol, delta=-0.10,
                           bid=1.45, ask=1.55, quote_type="live")
    c.open_interest = 800
    return c


def _rank(iv_pct: dict, **cand_kwargs):
    syms = list(iv_pct)
    return wr.rank_recommendations(
        {s: _profile(s) for s in syms}, {s: [_cand(s, **cand_kwargs)] for s in syms},
        spare_capital_remaining=50_000, existing_exposure_by_symbol={}, total_book=100_000,
        iv_percentile_by_symbol=iv_pct)


def test_higher_iv_percentile_ranks_higher():
    recs = _rank({"LOW": 10.0, "HIGH": 90.0, "MID": 50.0})
    assert [r.symbol for r in recs] == ["HIGH", "MID", "LOW"]
    assert recs[0].iv_percentile == 90.0


def test_missing_iv_history_is_scored_neutral():
    recs = _rank({"LOW": 10.0, "NONE": None, "HIGH": 90.0})
    assert [r.symbol for r in recs] == ["HIGH", "NONE", "LOW"]
    neutral = next(r for r in recs if r.symbol == "NONE")
    mid = _rank({"MID": 50.0})[0]
    assert neutral.composite_rank == pytest.approx(mid.composite_rank)


def test_iv_over_realized_vol_no_longer_affects_rank():
    # Same IV percentile, very different IV/realized ratios → identical rank.
    a = _rank({"X": 60.0}, market_iv=0.60, realized_vol=0.20)[0]
    b = _rank({"X": 60.0}, market_iv=0.21, realized_vol=0.20)[0]
    assert a.composite_rank == b.composite_rank


def test_quote_and_liquidity_carried_through():
    r = _rank({"X": 60.0})[0]
    assert (r.bid, r.ask, r.quote_type, r.open_interest) == (1.45, 1.55, "live", 800)
