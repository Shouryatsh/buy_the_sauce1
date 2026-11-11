"""
tests/test_stockscreen_scan.py — scan orchestration with stubbed providers:
completeness labelling, partial scans, provider failures, caching, cancellation,
universe selection, and the auto-scan schedule. No network, throwaway database.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime

import numpy as np
import pytest

import wheel_config as wc
import wheel_watchlist as ww
from stockscreen import prices as pr
from stockscreen import scan, store
from stockscreen import sec_data as sd
from stockscreen.config import ScreenConfig
from tests.stockscreen_fixtures import good_fund, price_frame, record


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(wc, "WHEEL_DB_PATH", str(tmp_path / "scan.db"))
    monkeypatch.setattr(ww, "EXCLUSIONS_PATH", tmp_path / "exclusions.json")      # never read the real exclusion list
    monkeypatch.setattr(pr, "PRICE_DIR", tmp_path / "prices")
    monkeypatch.setattr(sd, "fetch_sic", lambda cik, retries=3: ("7372", "Services-Prepackaged Software"))   # no network
    return tmp_path


def run_scan(symbols, fund_for, prices_for=None, cfg=None, monkeypatch=None, wait=True):
    mp = monkeypatch
    mp.setattr(scan, "load_fundamentals", lambda sym, cfg, force=False: fund_for(sym))
    mp.setattr(scan.pr, "fetch_prices", lambda syms, *a, **k: (prices_for or (lambda s: {x: price_frame(np.linspace(50, 100, 900)) for x in s}))(syms))
    mp.setattr(scan, "_fallback_shares", lambda *a, **k: None)
    mgr = scan.ScanManager()
    assert mgr.start(symbols, "test universe", "note", cfg or ScreenConfig(scan_workers=2))
    if wait:
        mgr.join(30)
    return mgr


def test_complete_scan_is_labelled_complete_and_counts_exclusions(env, monkeypatch):
    def fund(sym):
        return {"fetch": "no_statements", "error": "ETF"} if sym == "ETF" else good_fund(sym)
    mgr = run_scan(["GOOD", "ETF"], fund, monkeypatch=monkeypatch)
    run = store.latest_run()
    assert run["status"] == "complete" and run["universe_size"] == 2
    assert (run["evaluated"], run["no_statements"], run["errors"]) == (1, 1, 0)
    assert mgr.progress.state == "done" and mgr.progress.done == 2
    assert {r["symbol"] for r in store.load_results()} == {"GOOD", "ETF"}


def test_provider_failure_gives_a_partial_scan_without_aborting_the_rest(env, monkeypatch):
    def fund(sym):
        if sym == "BAD":
            raise RuntimeError("SEC is down")
        return good_fund(sym)
    mgr = run_scan(["AAA", "BAD", "CCC"], fund, monkeypatch=monkeypatch)
    run = store.latest_run()
    assert run["status"] == "partial" and run["errors"] == 1 and run["evaluated"] == 2       # NOT labelled complete
    bad = next(r for r in store.load_results() if r["symbol"] == "BAD")
    assert bad["fund"]["fetch"] == "error" and "SEC is down" in bad["fund"]["error"]
    assert {r["symbol"] for r in store.load_results()} == {"AAA", "BAD", "CCC"}


def test_missing_prices_still_record_the_stock_and_flag_it(env, monkeypatch):
    mgr = run_scan(["AAA"], lambda s: good_fund(s), prices_for=lambda syms: {}, monkeypatch=monkeypatch)
    rec = store.load_results()[0]
    assert rec["price"] == {} and rec["tech"] == {}
    from stockscreen import evaluate as ev
    row = ev.evaluate_record(rec, ScreenConfig())["row"]
    assert row["price_zone"] == "Insufficient history" and row["tech_trend"] == "Insufficient data"
    assert row["status"] != "PASSED"                                          # no price → no market cap → cannot pass


def test_cancelled_scan_is_partial(env, monkeypatch):
    gate = threading.Event()

    def fund(sym):
        gate.wait(5)
        return good_fund(sym)
    syms = [f"S{i}" for i in range(12)]
    mgr = run_scan(syms, fund, cfg=ScreenConfig(scan_workers=1), monkeypatch=monkeypatch, wait=False)
    time.sleep(0.3)
    mgr.cancel()
    gate.set()
    mgr.join(30)
    run = store.latest_run()
    assert mgr.progress.state == "cancelled" and run["status"] == "partial"


def test_only_one_scan_runs_at_a_time(env, monkeypatch):
    gate = threading.Event()
    mgr = run_scan(["A", "B"], lambda s: (gate.wait(5), good_fund(s))[1], monkeypatch=monkeypatch, wait=False)
    assert mgr.start(["C"], "x", "", ScreenConfig()) is False
    gate.set()
    mgr.join(30)


def test_scan_fires_alerts_only_for_stocks_that_qualify(env, monkeypatch):
    mgr = run_scan(["GOOD"], lambda s: good_fund(s), monkeypatch=monkeypatch)
    assert isinstance(mgr.progress.alert_stats, dict) and "fired" in mgr.progress.alert_stats


# --- fundamentals cache and SEC handling ------------------------------------------------------------------------

def facts_for(symbol):
    return {"cik": 1, "entityName": symbol, "facts": {"us-gaap": {"NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [
        {"start": "2025-01-01", "end": "2025-03-31", "val": 10, "filed": "2025-05-01", "form": "10-Q", "accn": "a"}]}}}}}


def test_fundamentals_cache_prevents_refetch_and_parser_version_invalidates_it(env, monkeypatch):
    calls = []
    monkeypatch.setattr(sd, "get_cik", lambda s: 123)
    monkeypatch.setattr(sd, "fetch_company_facts", lambda cik, retries=3: calls.append(cik) or facts_for("X"))
    cfg = ScreenConfig()
    first = scan.load_fundamentals("X", cfg)
    assert first["fetch"] == "ok" and calls == [123]
    scan.load_fundamentals("X", cfg)
    assert calls == [123]                                                     # served from the cache
    scan.load_fundamentals("X", cfg, force=True)
    assert len(calls) == 2                                                    # force bypasses it
    cached = store.get_fund_cache("X", 1)
    cached["parser_version"] = 0                                              # stale parsing rules
    store.put_fund_cache("X", cached)
    scan.load_fundamentals("X", cfg)
    assert len(calls) == 3


def test_sec_errors_are_reported_per_symbol(env, monkeypatch):
    import requests
    monkeypatch.setattr(sd, "get_cik", lambda s: 123 if s != "NOCIK" else None)

    def boom(cik, retries=3):
        resp = requests.Response()
        resp.status_code = 404
        raise requests.HTTPError("404", response=resp)
    monkeypatch.setattr(sd, "fetch_company_facts", boom)
    assert scan.load_fundamentals("NOCIK", ScreenConfig())["fetch"] == "no_cik"
    assert scan.load_fundamentals("FUND", ScreenConfig())["fetch"] == "no_statements"
    monkeypatch.setattr(sd, "fetch_company_facts", lambda cik, retries=3: (_ for _ in ()).throw(ConnectionError("timeout")))
    r = scan.load_fundamentals("X", ScreenConfig())
    assert r["fetch"] == "error" and "timeout" in r["error"]


# --- prices ----------------------------------------------------------------------------------------------------------

def test_fresh_price_cache_avoids_any_download(env, monkeypatch):
    df = price_frame(np.linspace(10, 20, 400))
    pr._save("AAA", df)
    monkeypatch.setattr(pr, "_download", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not download")))
    got = pr.fetch_prices(["AAA"], ttl_hours=6)
    assert len(got["AAA"]) == 400


def test_failed_download_falls_back_to_a_stale_cache_but_never_invents_data(env, monkeypatch):
    pr._save("OLD", price_frame(np.linspace(10, 20, 400)))
    monkeypatch.setattr(pr, "_download", lambda *a, **k: {})
    got = pr.fetch_prices(["OLD", "NEVER"], ttl_hours=0)
    assert "OLD" in got and "NEVER" not in got


def test_price_cleaning_drops_bad_rows_and_orders_dates():
    import pandas as pd
    idx = pd.to_datetime(["2026-01-03", "2026-01-02", "2026-01-02", "2026-01-04"], utc=True)
    df = pd.DataFrame({"Close": [3.0, 2.0, 2.5, float("nan")]}, index=idx)
    out = pr._clean(df)
    assert list(out["Close"]) == [2.5, 3.0] and out.index.is_monotonic_increasing and out.index.tz is None


# --- universe + schedule ---------------------------------------------------------------------------------------------

def test_universe_respects_manual_exclusions_and_labels_it_a_shortlist(env, monkeypatch, tmp_path):
    import wheel_watchlist as ww
    monkeypatch.setattr(ww, "EXCLUSIONS_PATH", tmp_path / "ex.json")
    monkeypatch.setattr(ww, "WHEEL_WATCHLIST", ["AAA", "BBB", "CCC"])
    ww.save_exclusions(["BBB"])
    opts = scan.universe_options(holdings=["ZZZ", "BBB"], custom="nvda, sap")
    short = opts["Wheel universe (shortlist)"]
    assert short["symbols"] == ["AAA", "CCC"] and "not the whole market" in short["note"]
    assert opts["Wheel universe + holdings"]["symbols"] == ["AAA", "CCC", "ZZZ"]      # an excluded holding stays out
    assert opts["Custom list"]["symbols"] == ["NVDA", "SAP"]


def test_auto_scan_schedule(env, monkeypatch):
    cfg = ScreenConfig(auto_scan_enabled=True, auto_scan_time="17:30")
    tue = lambda h, m: datetime(2026, 10, 20, h, m)                           # noqa: E731  (a Tuesday)
    assert scan.should_auto_scan(cfg, tue(17, 31)) is True
    assert scan.should_auto_scan(cfg, tue(17, 29)) is False                   # too early
    assert scan.should_auto_scan(cfg, datetime(2026, 10, 24, 18, 0)) is False  # Saturday
    assert scan.should_auto_scan(ScreenConfig(auto_scan_enabled=False), tue(18, 0)) is False
    monkeypatch.setattr(store, "last_complete_run", lambda: {"finished_at": "2026-10-20T17:35:00"})
    assert scan.should_auto_scan(cfg, tue(18, 0)) is False                    # already completed a scan today
    monkeypatch.setattr(store, "last_complete_run", lambda: {"finished_at": "2026-10-19T17:35:00"})
    assert scan.should_auto_scan(cfg, tue(18, 0)) is True                     # last complete scan was yesterday
    monkeypatch.setattr(store, "last_complete_run", lambda: None)
    assert scan.should_auto_scan(cfg, tue(18, 0)) is True                     # never scanned


# --- SIC cache and interrupted runs ----------------------------------------------------------------------------------------

def test_sic_is_fetched_once_ever_and_failures_are_retried_next_time(env, monkeypatch):
    calls = []
    monkeypatch.setattr(sd, "get_cik", lambda s: 123)
    monkeypatch.setattr(sd, "fetch_company_facts", lambda cik, retries=3: facts_for("X"))
    monkeypatch.setattr(sd, "fetch_sic", lambda cik, retries=3: (calls.append(cik), ("6282", "Investment Advice"))[1])
    cfg = ScreenConfig()
    assert scan.load_fundamentals("X", cfg)["sic"] == "6282"
    scan.load_fundamentals("X", cfg, force=True)                              # a forced SEC refresh does not re-ask for the SIC
    assert calls == [123] and store.get_sic("X") == ("6282", "Investment Advice")

    monkeypatch.setattr(sd, "fetch_sic", lambda cik, retries=3: ("", ""))     # SEC unreachable for another company
    assert scan.load_fundamentals("Y", cfg)["sic"] == ""
    monkeypatch.setattr(sd, "fetch_sic", lambda cik, retries=3: ("6211", "Security Brokers"))
    assert scan.load_fundamentals("Y", cfg)["sic"] == "6211"                  # the cached payload is upgraded on the next scan


def test_a_run_left_running_by_a_killed_process_is_recorded_as_interrupted(env):
    rid = store.start_run("Wheel universe (shortlist)", 100, "")
    assert store.latest_run()["status"] == "running"
    assert store.interrupt_orphans() == 1
    run = store.latest_run()
    assert run["status"] == "partial" and "interrupted" in run["note"] and run["finished_at"]
    assert store.interrupt_orphans() == 0
