"""
tests/test_stockscreen_exclusions.py — manual exclusions are absolute: an excluded ticker is never scanned,
never shown, never scored and never alerted on, whichever path asks for it, including results and
alerts an older scan left behind before the exclusion existed.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import wheel_config as wc
import wheel_ui
import wheel_watchlist as ww
from stockscreen import alerts, scan, store, universe
from stockscreen import config as sconfig
from stockscreen.config import ScreenConfig
from tests.stockscreen_fixtures import good_fund, price_frame, record

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(wc, "WHEEL_DB_PATH", str(tmp_path / "ex.db"))
    monkeypatch.setattr(ww, "EXCLUSIONS_PATH", tmp_path / "exclusions.json")
    monkeypatch.setattr(sconfig, "CONFIG_PATH", tmp_path / "cfg.json")
    st.cache_data.clear()
    return tmp_path


def seed(*symbols):
    rid = store.start_run("Wheel universe (shortlist)", len(symbols), "")
    for s in symbols:
        store.upsert_result(s, rid, record(s))
        store.put_fund_cache(s, good_fund(s))
    store.finish_run(rid, "complete", len(symbols), 0, 0)


def test_apply_exclusions_purges_every_table(env):
    seed("KEEP", "GONE")
    store.add_alert({"symbol": "GONE", "kind": "low_price", "name": "x"})
    store.add_alert({"symbol": "KEEP", "kind": "low_price", "name": "y"})
    store.set_alert_state("GONE", "low_price", True)
    ww.save_exclusions(["gone"])                                   # lower case on purpose
    counts = universe.apply_exclusions()
    assert counts == {"results": 1, "cache": 1, "alerts": 1, "states": 1}
    assert [r["symbol"] for r in store.load_results()] == ["KEEP"]
    assert [a["symbol"] for a in store.list_alerts()] == ["KEEP"]
    assert store.get_fund_cache("GONE", 30) is None
    assert universe.apply_exclusions() == {"results": 0, "cache": 0, "alerts": 0, "states": 0}   # idempotent


def test_no_exclusions_changes_nothing(env):
    seed("A", "B")
    assert universe.apply_exclusions()["results"] == 0 and len(store.load_results()) == 2


def test_scan_never_processes_an_excluded_ticker_even_when_asked_explicitly(env, monkeypatch):
    seen = []
    monkeypatch.setattr(scan, "load_fundamentals", lambda sym, cfg, force=False: (seen.append(sym), good_fund(sym))[1])
    monkeypatch.setattr(scan.pr, "fetch_prices", lambda syms, *a, **k: {s: price_frame(np.linspace(50, 100, 900)) for s in syms})
    monkeypatch.setattr(scan, "_fallback_shares", lambda *a, **k: None)
    ww.save_exclusions(["BAD"])
    mgr = scan.ScanManager()
    assert mgr.start(["GOOD", "BAD", "bad", "ALSO"], "explicit", "", ScreenConfig(scan_workers=2))
    mgr.join(30)
    assert sorted(seen) == ["ALSO", "GOOD"]
    assert {r["symbol"] for r in store.load_results()} == {"ALSO", "GOOD"}
    run = store.latest_run()
    assert run["universe_size"] == 2 and run["status"] == "complete"          # sized by what was actually scanned


def test_scan_removes_stale_results_for_tickers_excluded_since_the_last_scan(env, monkeypatch):
    seed("OLD", "STAY")
    ww.save_exclusions(["OLD"])
    monkeypatch.setattr(scan, "load_fundamentals", lambda sym, cfg, force=False: good_fund(sym))
    monkeypatch.setattr(scan.pr, "fetch_prices", lambda syms, *a, **k: {s: price_frame(np.linspace(50, 100, 900)) for s in syms})
    monkeypatch.setattr(scan, "_fallback_shares", lambda *a, **k: None)
    mgr = scan.ScanManager()
    mgr.start(["STAY"], "x", "", ScreenConfig(scan_workers=1))
    mgr.join(30)
    assert [r["symbol"] for r in store.load_results()] == ["STAY"]


def test_custom_list_cannot_smuggle_in_an_excluded_ticker(env, monkeypatch):
    monkeypatch.setattr(ww, "WHEEL_WATCHLIST", ["AAA", "BBB"])
    ww.save_exclusions(["NVDA", "BBB"])
    opts = scan.universe_options(holdings=["BBB", "ZZZ"], custom="nvda, sap, bbb")
    assert opts["Custom list"]["symbols"] == ["SAP"]
    assert "NVDA" in opts["Custom list"]["note"] and "BBB" in opts["Custom list"]["note"]
    assert opts["Wheel universe + holdings"]["symbols"] == ["AAA", "ZZZ"]
    assert all("BBB" not in o["symbols"] for o in opts.values())


def test_alert_engine_ignores_excluded_tickers(env):
    from datetime import datetime
    from tests.test_stockscreen_alerts import ev
    ww.save_exclusions(["ABC"])
    stats = alerts.process(ev("ABC"), ScreenConfig(), now=datetime(2026, 10, 20, 18, 0))
    assert stats["fired"] == 0 and store.list_alerts() == []


def test_overview_alert_feed_hides_excluded_tickers(env):
    class Stub:
        def get_account_summary(self):
            from wheel_ibkr import AccountSummary
            return AccountSummary(net_liquidation=100_000, total_cash=50_000, buying_power=1, margin_used=1, is_paper=False)

        def get_positions(self):
            return []
    for s in ("SHOWN", "HIDDEN"):
        store.add_alert({"symbol": s, "name": s, "kind": "low_price", "price": 1.0, "trigger_text": "t",
                         "fund_status": "f", "tech_status": "t", "reason": "r"})
    ww.save_exclusions(["HIDDEN"])
    titles = [a.title for a in wheel_ui.compute_alerts(Stub())]
    assert any("SHOWN" in t for t in titles) and not any("HIDDEN" in t for t in titles)


# --- the page ------------------------------------------------------------------------------------------------------

def open_page():
    at = AppTest.from_file(os.path.join(ROOT, "wheel_dashboard.py"), default_timeout=60).run()
    at.switch_page("app_pages/screener.py").run()
    assert not at.exception, [e.message for e in at.exception]
    return at


def shown_symbols(at):
    for df in at.dataframe:
        if "symbol" in df.value.columns and "status" in df.value.columns:
            return sorted(df.value["symbol"])
    return []


def test_page_hides_excluded_tickers_that_an_older_scan_stored(env):
    seed("PASSHI", "PASSLOW", "OLDSCAN")
    store.add_alert({"symbol": "OLDSCAN", "kind": "low_price", "name": "Old", "price": 1.0, "trigger_text": "t",
                     "fund_status": "f", "tech_status": "t", "reason": "r"})
    at = open_page()
    assert shown_symbols(at) == ["OLDSCAN", "PASSHI", "PASSLOW"]
    assert next(m.value for m in at.metric if m.label == "Stocks scanned") == "3"

    ww.save_exclusions(["OLDSCAN"])                                  # exclude after the scan, as the user did
    at.run()
    assert shown_symbols(at) == ["PASSHI", "PASSLOW"]
    assert next(m.value for m in at.metric if m.label == "Stocks scanned") == "2"
    assert not any("OLDSCAN" in s.value or "OLDSCAN" in str(s) for s in at.subheader)
    page_text = " ".join(e.value for group in (at.markdown, at.caption, at.success, at.warning, at.info) for e in group)
    assert "OLDSCAN" not in page_text                                # not in alerts, banners or captions either
    assert "OLDSCAN" not in [r["symbol"] for r in store.load_results()]    # and it is gone from storage, not merely hidden


def test_excluded_ticker_is_not_even_offered_in_the_detail_picker(env):
    seed("PASSHI", "GONE")
    ww.save_exclusions(["GONE"])
    at = open_page()
    assert "GONE" not in at.selectbox(key="detail_pick").options


def test_restoring_a_ticker_does_not_resurrect_stale_data(env):
    seed("PASSHI", "BACK")
    ww.save_exclusions(["BACK"])
    open_page()                                                       # purge happens
    ww.save_exclusions([])                                            # restored
    at = open_page()
    assert shown_symbols(at) == ["PASSHI"]                            # BACK reappears only after it is scanned again
