"""
tests/test_ivscreen_scan.py — universe building and the background scan, with IBKR replaced by fakes:
what is requested (and what never is), the data-availability statuses recorded for each failure mode,
incremental refreshes, exclusions, duplicates, cancellation.
"""

from __future__ import annotations

import asyncio
from datetime import date

import numpy as np
import pandas as pd
import pytest

import wheel_config as wc
import wheel_watchlist as ww
from ivscreen import scan, store
from ivscreen.config import IVConfig

TODAY = date.today()


def iv_series(n=600, lo=0.2, hi=0.5):
    idx = pd.bdate_range(end=pd.Timestamp(TODAY), periods=n)
    return pd.Series(np.linspace(lo, hi, n), index=idx, name="IV")


class FakeIB:
    """Stands in for the IBKR session: a table of known contracts and the answer each history request gets."""

    def __init__(self):
        self.contracts = {}          # symbol -> info dict (or None = unknown ticker)
        self.answers = {}            # symbol -> (series, errors) or callable(duration) -> (series, errors)
        self.resolved, self.fetched, self.earn = [], [], []
        self.on_fetch = None

    def info(self, sym, con_id, stock_type="COMMON", exchange="NYSE", sector="Technology"):
        self.contracts[sym] = {"con_id": con_id, "sec_type": "STK", "stock_type": stock_type, "exchange": exchange,
                               "currency": "USD", "name": f"{sym} Inc", "sector": sector, "industry": "Software",
                               "subcategory": "Apps", "listings": 1}


class FakeClient:
    def __init__(self, fib):
        self.fib = fib
        self.ib = fib

    def run(self, factory, timeout=30.0):
        return asyncio.run(factory())


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(wc, "WHEEL_DB_PATH", str(tmp_path / "scan.db"))
    monkeypatch.setattr(ww, "EXCLUSIONS_PATH", tmp_path / "exclusions.json")
    monkeypatch.setattr(ww, "WHEEL_WATCHLIST", ["AAA", "BBB", "BRK-B"])
    fib = FakeIB()

    async def resolve(ib, symbol, timeout=25.0):
        fib.resolved.append(symbol)
        return fib.contracts.get(symbol)

    async def fetch(ib, symbol, info, duration, timeout):
        fib.fetched.append((symbol, duration))
        if fib.on_fetch:
            fib.on_fetch(symbol)
        ans = fib.answers.get(symbol, (iv_series(), []))
        return ans(duration) if callable(ans) else ans

    def earnings(symbol):
        fib.earn.append(symbol)
        return ("2026-12-01", "AMC")

    monkeypatch.setattr(scan, "_resolve", resolve)
    monkeypatch.setattr(scan, "_fetch_iv_bars", fetch)
    monkeypatch.setattr(scan, "_earnings", earnings)
    monkeypatch.setattr(scan.pr, "fetch_prices", lambda *a, **k: {})
    return fib, tmp_path


def run_scan(fib, symbols, cfg=None, force=False, sources=None):
    mgr = scan.IVScanManager()
    uni = {"symbols": symbols, "sources": sources or {s: ["Wheel universe"] for s in symbols}, "note": "t"}
    assert mgr.start(FakeClient(fib), uni, "test", cfg or IVConfig(concurrency=2, skip_fresh_hours=0), force=force)
    mgr.join(30)
    return mgr


# ---------------------------------------------------------------------------
# universe
# ---------------------------------------------------------------------------

def test_symbols_are_normalised_and_duplicates_merged(env):
    cfg = IVConfig(etf_symbols=["SPY", "brk.b"])
    u = scan.build_universe([scan.SOURCE_WHEEL, scan.SOURCE_ETF, scan.SOURCE_CUSTOM, scan.SOURCE_HOLDINGS], holdings=["aaa"],
                            custom="brk.b, ccc;", cfg=cfg)
    assert u["symbols"].count("BRK-B") == 1 and u["duplicates"] >= 3
    assert u["sources"]["BRK-B"] == [scan.SOURCE_WHEEL, scan.SOURCE_ETF, scan.SOURCE_CUSTOM]
    assert u["sources"]["AAA"] == [scan.SOURCE_WHEEL, scan.SOURCE_HOLDINGS]
    assert "CCC" in u["symbols"] and "CCC;" not in u["symbols"]
    assert u["symbols"][:2] == ["AAA", "BBB"]


def test_etfs_are_only_included_when_asked(env):
    cfg = IVConfig(etf_symbols=["SPY", "QQQ"])
    assert "SPY" not in scan.build_universe([scan.SOURCE_WHEEL], cfg=cfg)["symbols"]
    assert "SPY" in scan.build_universe([scan.SOURCE_WHEEL, scan.SOURCE_ETF], cfg=cfg)["symbols"]


def test_excluded_tickers_never_enter_the_universe_in_any_spelling(env):
    fib, tmp = env
    ww.save_exclusions({"BBB", "BRK.B", "CCC", "SPY"})
    u = scan.build_universe([scan.SOURCE_WHEEL, scan.SOURCE_HOLDINGS, scan.SOURCE_ETF, scan.SOURCE_CUSTOM],
                            holdings=["CCC"], custom="spy", cfg=IVConfig(etf_symbols=["SPY"]))
    assert u["symbols"] == ["AAA"] and set(u["dropped"]) >= {"BRK-B", "CCC", "SPY"} and "BBB" not in u["symbols"]
    assert "manually excluded" in u["note"]


# ---------------------------------------------------------------------------
# scan outcomes
# ---------------------------------------------------------------------------

def test_a_normal_symbol_is_stored_with_its_contract_facts(env):
    fib, _ = env
    fib.info("AAA", 1)
    fib.info("SPY", 2, stock_type="ETF", exchange="ARCA", sector="")
    mgr = run_scan(fib, ["AAA", "SPY"])
    rows = store.load_symbols()
    a, s = rows["AAA"], rows["SPY"]
    assert a["fetch_status"] == "ok" and a["category"] == "Common stock" and a["exchange"] == "NYSE" and a["sector"] == "Technology"
    assert a["iv_n"] == 600 and a["iv_last"] == pd.bdate_range(end=pd.Timestamp(TODAY), periods=1)[0].strftime("%Y-%m-%d")
    assert s["category"] == "ETF" and s["next_earnings"] is None            # ETFs have no earnings lookup
    assert a["next_earnings"] == "2026-12-01" and fib.earn == ["AAA"]
    assert len(store.load_iv("AAA")) == 600
    assert mgr.progress.state == "done" and mgr.progress.ok == 2 and store.latest_run()["status"] == "complete"


def test_no_options_data_is_recorded_as_unknown_not_as_data(env):
    fib, _ = env
    fib.info("AAA", 1)
    fib.answers["AAA"] = (pd.Series(dtype=float), [(162, "Historical Market Data Service error message: HMDS query returned no data")])
    mgr = run_scan(fib, ["AAA"])
    row = store.load_symbols()["AAA"]
    assert row["fetch_status"] == "no_data" and "no data" in row["fetch_detail"] and row["iv_n"] is None
    assert len(store.load_iv("AAA")) == 0 and mgr.progress.no_options == 1 and fib.earn == []


def test_unknown_ticker_unsupported_instruments_and_duplicates(env):
    fib, _ = env
    fib.contracts["NOPE"] = None
    fib.contracts["PREF"] = {**{"con_id": 5, "sec_type": "STK", "exchange": "NYSE", "currency": "USD", "name": "P", "sector": "",
                                "industry": "", "subcategory": "", "listings": 1}, "stock_type": "PREFERRED"}
    fib.contracts["OPT"] = {"unsupported_only": True}
    fib.info("AAA", 7)
    fib.info("AAA2", 7)                                                        # same IBKR contract under another spelling
    mgr = run_scan(fib, ["NOPE", "PREF", "OPT", "AAA", "AAA2"])
    st = {s: r["fetch_status"] for s, r in store.load_symbols().items()}
    assert st["NOPE"] == "unresolved" and st["PREF"] == "unsupported" and st["OPT"] == "unsupported"
    assert sorted([st["AAA"], st["AAA2"]]) == ["duplicate", "ok"]
    dup = next(r for r in store.load_symbols().values() if r["fetch_status"] == "duplicate")
    assert "same IBKR contract" in dup["fetch_detail"]
    assert mgr.progress.duplicates == 1 and mgr.progress.ok == 1 and mgr.progress.unsupported == 3
    assert sum(1 for s, _d in fib.fetched) == 1                                  # nothing was requested for the three non-stocks


def test_classify_failure_separates_no_data_from_a_failed_request():
    assert scan._classify_failure([(162, "HMDS query returned no data")], 0.4, 60)[0] == "no_data"
    assert scan._classify_failure([(200, "No security definition")], 0.2, 60)[0] == "no_data"
    assert scan._classify_failure([], 0.3, 60)[0] == "no_data"
    assert scan._classify_failure([], 60.0, 60)[0] == "error"                    # silence for the whole timeout = failed, not "no data"
    assert scan._classify_failure([(162, "Historical data request pacing violation")], 0.2, 60)[0] == "error"
    assert scan._classify_failure([(1100, "Connectivity between IB and TWS has been lost")], 0.2, 60)[0] == "error"


def test_a_crash_on_one_symbol_is_a_failed_fetch_and_the_rest_continue(env):
    fib, _ = env
    fib.info("AAA", 1)
    fib.info("BBB", 2)
    fib.answers["AAA"] = lambda d: (_ for _ in ()).throw(RuntimeError("socket closed"))
    mgr = run_scan(fib, ["AAA", "BBB"])
    rows = store.load_symbols()
    assert rows["AAA"]["fetch_status"] == "error" and "socket closed" in rows["AAA"]["fetch_detail"]
    assert rows["BBB"]["fetch_status"] == "ok"
    assert mgr.progress.errors == 1 and store.latest_run()["status"] == "partial"


# ---------------------------------------------------------------------------
# refreshes
# ---------------------------------------------------------------------------

def test_second_scan_only_tops_up_and_does_not_resolve_again(env):
    fib, _ = env
    fib.info("AAA", 1)
    cfg = IVConfig(concurrency=1, skip_fresh_hours=0)
    run_scan(fib, ["AAA"], cfg)
    run_scan(fib, ["AAA"], cfg)
    assert fib.resolved == ["AAA"]
    assert fib.fetched == [("AAA", cfg.iv_duration), ("AAA", cfg.refresh_duration)]
    run_scan(fib, ["AAA"], cfg, force=True)
    assert fib.fetched[-1] == ("AAA", cfg.iv_duration)


def test_full_history_is_redownloaded_after_the_refresh_interval_or_a_long_gap(env):
    fib, _ = env
    fib.info("AAA", 1)
    cfg = IVConfig(concurrency=1, full_refresh_days=30, skip_fresh_hours=0)
    run_scan(fib, ["AAA"], cfg)
    store.upsert_symbol("AAA", iv_full_at="2020-01-01T00:00:00")
    run_scan(fib, ["AAA"], cfg)
    assert fib.fetched[-1][1] == cfg.iv_duration
    assert scan._durations(cfg, (TODAY.replace(year=TODAY.year - 1)).isoformat(), store.now_iso(), False, TODAY)[1] is True


def test_an_empty_top_up_falls_back_to_the_full_download(env):
    fib, _ = env
    fib.info("AAA", 1)
    cfg = IVConfig(concurrency=1, skip_fresh_hours=0)
    run_scan(fib, ["AAA"], cfg)
    fib.answers["AAA"] = lambda d: (pd.Series(dtype=float), []) if d == cfg.refresh_duration else (iv_series(650), [])
    run_scan(fib, ["AAA"], cfg)
    assert [d for _s, d in fib.fetched][-2:] == [cfg.refresh_duration, cfg.iv_duration]
    assert store.load_symbols()["AAA"]["fetch_status"] == "ok" and len(store.load_iv("AAA")) >= 650


def test_a_failed_refresh_never_downgrades_stored_data_to_no_data(env):
    fib, _ = env
    fib.info("AAA", 1)
    cfg = IVConfig(concurrency=1, skip_fresh_hours=0)
    run_scan(fib, ["AAA"], cfg)
    fib.answers["AAA"] = (pd.Series(dtype=float), [])
    run_scan(fib, ["AAA"], cfg)
    row = store.load_symbols()["AAA"]
    assert row["fetch_status"] == "error" and "keeping data through" in row["fetch_detail"]
    assert len(store.load_iv("AAA")) == 600                                      # history untouched


# ---------------------------------------------------------------------------
# exclusions and cancellation
# ---------------------------------------------------------------------------

def test_excluded_symbols_are_never_requested_and_stored_traces_are_purged(env):
    fib, _ = env
    fib.info("AAA", 1)
    fib.info("BBB", 2)
    run_scan(fib, ["AAA", "BBB"])
    assert {"AAA", "BBB"} <= set(store.load_symbols())
    ww.save_exclusions({"BBB"})
    fib.resolved.clear()
    fib.fetched.clear()
    run_scan(fib, ["AAA", "BBB"])
    assert [s for s, _d in fib.fetched] == ["AAA"] and "BBB" not in fib.resolved
    assert "BBB" not in store.load_symbols() and "BBB" not in store.load_all_iv()


def test_an_exclusion_added_mid_scan_is_purged_before_the_scan_ends(env):
    fib, _ = env
    for i, s in enumerate(["AAA", "BBB", "CCC"], 1):
        fib.info(s, i)
    fib.on_fetch = lambda sym: ww.save_exclusions({"BBB"}) if sym == "AAA" else None
    run_scan(fib, ["AAA", "BBB", "CCC"], IVConfig(concurrency=1, skip_fresh_hours=0))
    assert "BBB" not in store.load_symbols() and "BBB" not in store.load_all_iv()


def test_cancel_stops_new_requests_and_marks_the_run_partial(env):
    fib, _ = env
    syms = [f"S{i}" for i in range(12)]
    for i, s in enumerate(syms, 1):
        fib.info(s, i)
    mgr = scan.IVScanManager()
    fib.on_fetch = lambda sym: mgr.cancel()
    uni = {"symbols": syms, "sources": {}, "note": ""}
    mgr.start(FakeClient(fib), uni, "cancel", IVConfig(concurrency=1, skip_fresh_hours=0))
    mgr.join(30)
    assert mgr.progress.state == "cancelled" and len(fib.fetched) < len(syms)
    assert store.latest_run()["status"] == "partial"


def test_only_one_scan_runs_at_a_time(env):
    fib, _ = env
    fib.info("AAA", 1)
    mgr = scan.IVScanManager()
    gate = asyncio.Event()
    started = []

    async def slow(ib, symbol, info, duration, timeout):
        started.append(1)
        await asyncio.sleep(0.4)
        return iv_series(), []
    import ivscreen.scan as sm
    orig = sm._fetch_iv_bars
    sm._fetch_iv_bars = slow
    try:
        uni = {"symbols": ["AAA"], "sources": {}, "note": ""}
        assert mgr.start(FakeClient(fib), uni, "a", IVConfig())
        assert not mgr.start(FakeClient(fib), uni, "b", IVConfig())
        mgr.join(30)
    finally:
        sm._fetch_iv_bars = orig
    assert len(started) == 1


# ---------------------------------------------------------------------------
# resilience: earnings never breaks IV, failures are retried, fresh symbols are skipped
# ---------------------------------------------------------------------------

def test_a_missing_or_crashing_earnings_lookup_never_changes_the_iv_status(env, monkeypatch):
    fib, _ = env
    fib.info("AAA", 1)
    fib.info("BBB", 2)
    monkeypatch.setattr(scan, "_earnings", lambda sym: (None, "") if sym == "AAA" else (_ for _ in ()).throw(RuntimeError("yahoo down")))
    mgr = run_scan(fib, ["AAA", "BBB"])
    rows = store.load_symbols()
    assert rows["AAA"]["fetch_status"] == "ok" and rows["BBB"]["fetch_status"] == "ok"
    assert rows["AAA"]["next_earnings"] is None and rows["BBB"]["next_earnings"] is None
    assert mgr.progress.ok == 2 and mgr.progress.errors == 0 and store.latest_run()["status"] == "complete"


def test_a_known_future_earnings_date_survives_a_lookup_that_finds_nothing(env, monkeypatch):
    fib, _ = env
    fib.info("AAA", 1)
    run_scan(fib, ["AAA"])
    assert store.load_symbols()["AAA"]["next_earnings"] == "2026-12-01"
    store.upsert_symbol("AAA", next_earnings="2099-01-01", earnings_checked_at="2020-01-01T00:00:00")
    monkeypatch.setattr(scan, "_earnings", lambda sym: (None, ""))
    run_scan(fib, ["AAA"])
    assert store.load_symbols()["AAA"]["next_earnings"] == "2099-01-01"


def test_a_transient_failure_is_retried_once_and_recovers(env):
    fib, _ = env
    fib.info("AAA", 1)
    calls = []

    def flaky(duration):
        calls.append(duration)
        if len(calls) == 1:
            raise TimeoutError("request timed out")
        return iv_series(), []
    fib.answers["AAA"] = flaky
    mgr = run_scan(fib, ["AAA"])
    assert len(calls) == 2 and store.load_symbols()["AAA"]["fetch_status"] == "ok"
    assert mgr.progress.ok == 1 and mgr.progress.errors == 0 and store.latest_run()["status"] == "complete"


def test_a_persistent_failure_is_retried_once_then_reported(env):
    fib, _ = env
    fib.info("AAA", 1)
    fib.info("BBB", 2)
    calls = []

    def dead(duration):
        calls.append(duration)
        raise TimeoutError("request timed out")
    fib.answers["AAA"] = dead
    mgr = run_scan(fib, ["AAA", "BBB"])
    assert len(calls) == 2
    assert store.load_symbols()["AAA"]["fetch_status"] == "error" and store.load_symbols()["BBB"]["fetch_status"] == "ok"
    assert mgr.progress.errors == 1 and mgr.progress.ok == 1 and mgr.progress.done == 2
    assert store.latest_run()["status"] == "partial"


def test_cancel_during_the_retry_pass_stops_it(env):
    fib, _ = env
    fib.info("AAA", 1)
    mgr = scan.IVScanManager()
    calls = []

    def dead(duration):
        calls.append(1)
        mgr.cancel()
        raise TimeoutError("x")
    fib.answers["AAA"] = dead
    mgr.start(FakeClient(fib), {"symbols": ["AAA"], "sources": {}, "note": ""}, "c", IVConfig(skip_fresh_hours=0))
    mgr.join(30)
    assert len(calls) == 1 and mgr.progress.state == "cancelled"


def test_symbols_fetched_recently_are_skipped_so_a_cancelled_scan_can_resume(env):
    fib, _ = env
    fib.info("AAA", 1)
    fib.info("BBB", 2)
    cfg = IVConfig(concurrency=1, skip_fresh_hours=6)
    run_scan(fib, ["AAA"], cfg)
    fib.fetched.clear()
    mgr = run_scan(fib, ["AAA", "BBB"], cfg)
    assert [s for s, _d in fib.fetched] == ["BBB"]                       # AAA was fetched minutes ago
    assert mgr.progress.ok == 2 and store.latest_run()["status"] == "complete"
    fib.fetched.clear()
    run_scan(fib, ["AAA", "BBB"], cfg, force=True)                      # force always refetches
    assert sorted(s for s, _d in fib.fetched) == ["AAA", "BBB"]
    store.upsert_symbol("AAA", iv_checked_at="2020-01-01T00:00:00")     # old enough -> refreshed again
    fib.fetched.clear()
    run_scan(fib, ["AAA", "BBB"], cfg)
    assert [s for s, _d in fib.fetched] == ["AAA"]


def test_a_failed_symbol_is_never_treated_as_fresh(env):
    fib, _ = env
    fib.info("AAA", 1)
    fib.answers["AAA"] = (pd.Series(dtype=float), [])
    run_scan(fib, ["AAA"], IVConfig(skip_fresh_hours=6))
    assert store.load_symbols()["AAA"]["fetch_status"] == "no_data"
    fib.answers["AAA"] = (iv_series(), [])
    run_scan(fib, ["AAA"], IVConfig(skip_fresh_hours=6))
    assert store.load_symbols()["AAA"]["fetch_status"] == "ok"
