"""
tests/test_stockscreen_alerts.py — alert rules and the state machine:
fire once, deduplicate, cool down, re-fire after exit and re-entry, never fire
on stale / unverifiable data, and deterioration only for stocks that qualified.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from stockscreen import alerts, store
from stockscreen.config import ScreenConfig

NOW = datetime(2026, 10, 20, 18, 0)


@pytest.fixture()
def db(tmp_path):
    return str(tmp_path / "alerts.db")


def ev(symbol="ABC", status="PASSED", zone="Low-price candidate", opp="C", age=60, stale=False, price_stale=False,
       tech="Healthy pullback"):
    row = {"symbol": symbol, "name": "ABC Corp", "status": status, "price_zone": zone, "opp_code": opp,
           "data_age_days": age, "stale": stale, "price_stale": price_stale, "tech_trend": tech, "fund_trend": "Stable",
           "price": 50.0, "fcf_yield": 0.08, "pos_52w": 12.0, "pct_52w": 9.0, "pct_2y": None, "pct_3y": None,
           "zone_lookbacks": "52w", "opp_why": "low-price zone with a constructive long-term trend",
           "latest_quarter_end": "2026-06-30", "price_asof": "2026-10-20", "reasons": ""}
    return {symbol: {"row": row}}


def run(db, evals, cfg=None, now=NOW):
    return alerts.process(evals, cfg or ScreenConfig(), now=now, path=db)


def kinds(db):
    return [a["kind"] for a in store.list_alerts(path=db)]


def test_low_price_alert_fires_once_with_full_content(db):
    stats = run(db, ev())
    assert stats["fired"] == 2                                              # low_price + healthy_pullback
    got = {a["kind"]: a for a in store.list_alerts(path=db)}
    a = got["low_price"]
    assert (a["symbol"], a["name"], a["price"], a["tech_status"]) == ("ABC", "ABC Corp", 50.0, "Healthy pullback")
    assert a["fcf_yield"] == 0.08 and a["percentile"] == 9.0 and "PASSED" in a["fund_status"]
    assert a["data_asof"] == "2026-06-30" and a["price_asof"] == "2026-10-20" and a["trigger_text"]


def test_unchanged_condition_is_deduplicated_across_scans(db):
    run(db, ev())
    for day in range(1, 6):
        stats = run(db, ev(), now=NOW + timedelta(days=day))
        assert stats["fired"] == 0 and stats["deduplicated"] == 2
    assert kinds(db).count("low_price") == 1


def test_leaving_the_zone_clears_state_and_reentry_after_cooldown_fires_again(db):
    run(db, ev())
    run(db, ev(zone="Mid-range", opp="A"), now=NOW + timedelta(days=1))        # exits the zone
    assert store.get_alert_state("ABC", "low_price", db)["active"] == 0
    stats = run(db, ev(), now=NOW + timedelta(days=10))                        # re-enters after the 7-day cooldown
    assert stats["fired"] == 2 and kinds(db).count("low_price") == 2


def test_reentry_within_cooldown_is_suppressed(db):
    run(db, ev())
    run(db, ev(zone="Mid-range", opp="A"), now=NOW + timedelta(days=1))
    stats = run(db, ev(), now=NOW + timedelta(days=3))
    assert stats["fired"] == 0 and stats["suppressed_cooldown"] == 2 and kinds(db).count("low_price") == 1


def test_stale_data_never_alerts_and_does_not_reset_state(db):
    run(db, ev(stale=True, age=300))
    assert kinds(db) == []
    run(db, ev())                                                              # fresh data: alert fires
    n = len(kinds(db))
    run(db, ev(stale=True, age=300), now=NOW + timedelta(days=1))             # then data goes stale: nothing changes
    assert len(kinds(db)) == n and store.get_alert_state("ABC", "low_price", db)["active"] == 1


def test_stale_price_bar_never_alerts(db):
    assert run(db, ev(price_stale=True))["fired"] == 0


def test_insufficient_or_error_status_is_unverifiable(db):
    for status in ("INSUFFICIENT_DATA", "STALE_DATA", "ERROR"):
        stats = run(db, ev(status=status))
        assert stats["fired"] == 0 and stats["unverifiable"] >= 3
    assert kinds(db) == []


def test_zone_with_insufficient_history_is_unverifiable(db):
    assert run(db, ev(zone="Insufficient history", opp="A"))["fired"] == 0


def test_reversal_alert_only_for_class_d(db):
    run(db, ev(opp="D", tech="Trend reversal candidate"))
    assert "reversal_confirmation" in kinds(db) and "healthy_pullback" not in kinds(db)


def test_deterioration_only_after_the_stock_qualified(db):
    run(db, ev(status="FAILED", zone="Mid-range", opp="X"))                    # never passed: nothing to deteriorate from
    assert kinds(db) == []
    run(db, ev(status="PASSED", zone="Mid-range", opp="A"), now=NOW + timedelta(days=1))
    stats = run(db, ev(status="FAILED", zone="Mid-range", opp="X"), now=NOW + timedelta(days=2))
    assert "fundamental_deterioration" in kinds(db) and stats["fired"] == 1
    run(db, ev(status="FAILED", zone="Mid-range", opp="X"), now=NOW + timedelta(days=3))
    assert kinds(db).count("fundamental_deterioration") == 1                   # not repeated while still failing


def test_disabled_kinds_do_not_fire(db):
    cfg = ScreenConfig(alert_kinds=("reversal_confirmation",))
    assert run(db, ev(), cfg)["fired"] == 0


def test_alerts_can_be_marked_read(db):
    run(db, ev())
    assert len(store.list_alerts(unread_only=True, path=db)) == 2
    store.mark_alerts_read(path=db)
    assert store.list_alerts(unread_only=True, path=db) == []
