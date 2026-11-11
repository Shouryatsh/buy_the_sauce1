"""
stockscreen/scan.py — universe selection, per-symbol record building, and a
background ScanManager so a full-universe scan never blocks the dashboard.

A scan record is raw and threshold-free:
    {"symbol","name","scanned_at","fund":{quarters,years,shares,...},"price":{last,asof,positions,qe_close},"tech":{...}}
Thresholds are applied later by stockscreen.evaluate, so changing them needs no rescan.

Data sources
  Fundamentals  SEC EDGAR XBRL company facts (via the project's shared 8 req/s limiter)
  Prices        Yahoo Finance daily history (yfinance), batched, split-adjusted closes
  Shares        SEC cover page / balance sheet / weighted average; yfinance share count only
                as a labelled fallback when SEC has no current count (e.g. several dual-class names)

Run headless (e.g. from Windows Task Scheduler):
    python -m stockscreen.scan --universe wheel [--force]
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from . import alerts as alerts_mod
from . import evaluate as ev
from . import prices as pr
from . import sec_data as sd
from . import store
from . import universe
from . import technicals as tc
from .config import ROOT, ScreenConfig, load_config

logger = logging.getLogger(__name__)

KEEP_QUARTERS, KEEP_YEARS = 20, 6


# ---------------------------------------------------------------------------
# universe
# ---------------------------------------------------------------------------

SHORTLIST_NOTE = (
    "The wheel universe is a shortlist, not the whole market: scan_universe.py took the largest ~1,200 SEC-registered "
    "tickers and kept those with net margin >= 10% and positive FCF, then you removed some by hand."
)


def universe_options(holdings: Optional[list[str]] = None, custom: str = "", sec_top_n: int = 1000) -> dict[str, dict]:
    """label -> {"symbols": [...], "note": str}. Manual exclusions apply to every list."""
    from wheel_watchlist import active_watchlist, load_exclusions
    excluded = load_exclusions()
    shortlist = active_watchlist()
    out = {"Wheel universe (shortlist)": {"symbols": shortlist, "note": SHORTLIST_NOTE}}
    extra = [s for s in (holdings or []) if s.upper() not in excluded]
    out["Wheel universe + holdings"] = {"symbols": list(dict.fromkeys(shortlist + extra)),
                                        "note": SHORTLIST_NOTE + " Your current holdings are added."}
    try:
        import edgar
        from wheel_universe import _looks_like_common_stock
        top = [t for t in edgar.get_ticker_universe(sec_top_n) if _looks_like_common_stock(t) and t not in excluded]
    except Exception as exc:                                  # offline / SEC down
        logger.warning("could not build SEC top-N universe: %s", exc)
        top = []
    out[f"SEC top {sec_top_n} by size (expanded)"] = {
        "symbols": top,
        "note": (f"The first {sec_top_n} tickers in SEC's company list (roughly market-cap order), common-stock-looking "
                 "symbols only. ETFs and funds without XBRL statements are excluded automatically and counted. Still not the "
                 "whole market.")}
    custom_syms = [s.strip().upper() for s in custom.replace(",", " ").split() if s.strip()]
    skipped = [s for s in custom_syms if s in excluded]
    custom_syms = [s for s in custom_syms if s not in excluded]
    if custom_syms or skipped:
        out["Custom list"] = {"symbols": custom_syms, "note": "Tickers you typed in."
                              + (f" Skipped (manually excluded): {', '.join(skipped)}." if skipped else "")}
    return out


# ---------------------------------------------------------------------------
# record building
# ---------------------------------------------------------------------------

def fund_payload(parsed: sd.ParsedFacts) -> dict:
    return {
        "fetch": "ok", "parser_version": sd.PARSER_VERSION, "cik": parsed.cik, "name": parsed.name, "latest_filed": parsed.latest_filed,
        "sic": "", "sic_desc": "",
        "quarters": {m: [o.to_dict() for o in obs[-KEEP_QUARTERS:]] for m, obs in parsed.quarters.items()},
        "years": {m: [o.to_dict() for o in obs[-KEEP_YEARS:]] for m, obs in parsed.years.items()},
        "shares": parsed.shares,
    }


def attach_sic(symbol: str, cik: int, payload: dict) -> dict:
    """Industry code for the financial-company guard. Cached forever; a failed lookup is retried on the next scan."""
    got = store.get_sic(symbol)
    if got is None:
        sic, desc = sd.fetch_sic(cik)
        if sic:
            store.put_sic(symbol, cik, sic, desc)
            got = (sic, desc)
    payload["sic"], payload["sic_desc"] = got if got else ("", "")
    return payload


def load_fundamentals(symbol: str, cfg: ScreenConfig, force: bool = False) -> dict:
    """Cached parsed SEC data for *symbol*, refreshed after fundamentals_ttl_days."""
    if not force:
        cached = store.get_fund_cache(symbol, cfg.fundamentals_ttl_days)
        if cached is not None and cached.get("fetch") == "ok" and cached.get("parser_version") == sd.PARSER_VERSION:
            if not cached.get("sic"):                                   # SIC lookup failed earlier: try again
                attach_sic(symbol, cached.get("cik") or sd.get_cik(symbol), cached)
                store.put_fund_cache(symbol, cached)
            return cached
    cik = sd.get_cik(symbol)
    if cik is None:
        return {"fetch": "no_cik", "error": "not in SEC's company ticker list (not an SEC-registered operating company)"}
    try:
        facts = sd.fetch_company_facts(cik)
    except Exception as exc:
        code = getattr(getattr(exc, "response", None), "status_code", None)
        if code == 404:
            return {"fetch": "no_statements", "error": "SEC has no XBRL financial statements for this filer"}
        return {"fetch": "error", "error": f"SEC fetch failed: {exc}"}
    parsed = sd.parse_company_facts(facts)
    if not parsed.has_statements:
        return {"fetch": "no_statements", "error": "no operating-cash-flow statements in SEC filings (fund, trust or foreign filer)"}
    payload = fund_payload(parsed)
    attach_sic(symbol, cik, payload)
    store.put_fund_cache(symbol, payload)
    return payload


def _fallback_shares(symbol: str, fund: dict, cfg: ScreenConfig) -> None:
    """If SEC has no fresh, trustworthy share count, ask yfinance and label it."""
    from .fundamentals import pick_shares
    if pick_shares(fund, date.today(), cfg.max_data_age_days) is not None:
        return
    try:
        import yfinance as yf
        n = yf.Ticker(symbol).fast_info["shares"]
        if n and n > 0:
            fund["shares_external"] = {"value": float(n), "end": date.today().isoformat(), "filed": date.today().isoformat(),
                                       "source": "Yahoo Finance share count (fallback; SEC has none current)",
                                       "approx": False, "classes": 1}
    except Exception as exc:
        logger.info("%s: yfinance share fallback failed: %s", symbol, exc)


def price_block(df: Optional[pd.DataFrame], fund: dict, cfg: ScreenConfig) -> tuple[dict, dict]:
    if df is None or len(df) < 2:
        return {}, {}
    close = df["Close"].astype(float)
    last, asof = pr.last_bar(df)
    qe_close: dict[str, float] = {}
    for o in (fund.get("quarters", {}).get("ocf") or [])[-8:]:
        d = pd.Timestamp(o["end"])
        i = close.index.searchsorted(d, side="right") - 1
        if i >= 0 and (d - close.index[i]).days <= 7 and close.index[0] <= d:
            qe_close[o["end"]] = float(close.iloc[i])
    price = {"last": last, "asof": asof, "n_bars": int(len(close)), "first_bar": close.index[0].strftime("%Y-%m-%d"),
             "positions": tc.price_positions(close, cfg.history_grace_days), "qe_close": qe_close}
    tech = tc.indicator_snapshot(df, cfg.slope_window) if len(df) >= 30 else {}
    return price, tech


def build_record(symbol: str, cfg: ScreenConfig, price_df: Optional[pd.DataFrame], force: bool = False) -> dict:
    rec = {"symbol": symbol, "name": "", "scanned_at": store.now_iso(), "fund": {}, "price": {}, "tech": {}}
    try:
        fund = load_fundamentals(symbol, cfg, force)
        rec["name"] = fund.get("name", "")
        if fund.get("fetch") == "ok":
            _fallback_shares(symbol, fund, cfg)
        rec["fund"] = fund
        rec["price"], rec["tech"] = price_block(price_df, fund, cfg)
    except Exception as exc:
        logger.exception("%s: scan failed", symbol)
        rec["fund"] = {"fetch": "error", "error": f"{type(exc).__name__}: {exc}"}
    return rec


# ---------------------------------------------------------------------------
# background manager
# ---------------------------------------------------------------------------

@dataclass
class Progress:
    state: str = "idle"                 # idle | running | done | cancelled | error
    phase: str = ""
    total: int = 0
    done: int = 0
    current: str = ""
    evaluated: int = 0
    no_statements: int = 0
    errors: int = 0
    started: float = 0.0
    finished: float = 0.0
    run_id: Optional[int] = None
    message: str = ""
    alert_stats: dict = field(default_factory=dict)

    @property
    def fraction(self) -> float:
        return self.done / self.total if self.total else 0.0


class ScanManager:
    """One background scan at a time; thread-safe progress for the UI to poll."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._cancel = threading.Event()
        self.progress = Progress()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def cancel(self) -> None:
        self._cancel.set()

    def start(self, symbols: list[str], label: str, note: str, cfg: ScreenConfig, force: bool = False) -> bool:
        symbols, _dropped = universe.drop_excluded(symbols)          # exclusions are absolute
        universe.apply_exclusions()
        with self._lock:
            if self.running:
                return False
            store.interrupt_orphans()                                # a run left 'running' by a killed process
            self._cancel.clear()
            self.progress = Progress(state="running", phase="starting", total=len(symbols), started=time.time())
            self._thread = threading.Thread(target=self._run, args=(list(symbols), label, note, cfg, force),
                                            name="stockscreen-scan", daemon=True)
            self._thread.start()
            return True

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    # -- worker -----------------------------------------------------------------
    def _run(self, symbols: list[str], label: str, note: str, cfg: ScreenConfig, force: bool) -> None:
        p = self.progress
        run_id = store.start_run(label, len(symbols), note)
        p.run_id = run_id
        try:
            p.phase = "prices"
            prices = pr.fetch_prices(symbols, cfg.price_history_years, cfg.price_ttl_hours, force)
            p.phase = "fundamentals"
            written: list[dict] = []

            def work(sym: str) -> dict:
                if self._cancel.is_set() or sym.upper() in universe.excluded():
                    return {"symbol": sym, "cancelled": True}
                rec = build_record(sym, cfg, prices.get(sym), force)
                store.upsert_result(sym, run_id, rec)
                return rec

            with ThreadPoolExecutor(max_workers=cfg.scan_workers) as pool:
                futs = {pool.submit(work, s): s for s in symbols}
                for fut in as_completed(futs):
                    sym = futs[fut]
                    p.current = sym
                    try:
                        rec = fut.result()
                    except Exception as exc:
                        logger.exception("%s failed", sym)
                        p.errors += 1
                        rec = None
                    p.done += 1
                    if not rec or rec.get("cancelled"):
                        continue
                    written.append(rec)
                    fetch = rec["fund"].get("fetch")
                    if fetch == "ok":
                        p.evaluated += 1
                    elif fetch == "error":
                        p.errors += 1
                    else:
                        p.no_statements += 1

            universe.apply_exclusions()                                   # also covers an exclusion added mid-scan
            written = [r for r in written if r["symbol"].upper() not in universe.excluded()]
            p.phase = "alerts"
            evals = {r["symbol"]: ev.evaluate_record(r, cfg) for r in written if r["fund"].get("fetch") in ("ok", "no_statements", "no_cik")}
            p.alert_stats = alerts_mod.process(evals, cfg)
            cancelled = self._cancel.is_set()
            complete = (not cancelled) and p.done == p.total and p.errors == 0
            status = "complete" if complete else "partial"
            p.state = "cancelled" if cancelled else "done"
            p.message = (f"{p.evaluated} evaluated, {p.no_statements} without SEC statements, {p.errors} errors"
                         + (" (cancelled)" if cancelled else ""))
            store.finish_run(run_id, status, p.evaluated, p.no_statements, p.errors, p.message)
        except Exception as exc:
            logger.exception("scan failed")
            p.state, p.message = "error", f"{type(exc).__name__}: {exc}"
            store.finish_run(run_id, "failed", p.evaluated, p.no_statements, p.errors, p.message)
        finally:
            p.finished = time.time()
            p.phase = ""


def should_auto_scan(cfg: ScreenConfig, now: Optional[datetime] = None) -> bool:
    """Weekday, past the configured time, and no completed scan yet today."""
    if not cfg.auto_scan_enabled:
        return False
    now = now or datetime.now()
    if now.weekday() >= 5:
        return False
    hh, mm = (int(x) for x in cfg.auto_scan_time.split(":"))
    if (now.hour, now.minute) < (hh, mm):
        return False
    last = store.last_complete_run()
    return not (last and last.get("finished_at", "")[:10] == now.date().isoformat())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run the quality-stock scan headless.")
    ap.add_argument("--universe", default="wheel", choices=["wheel", "sec-top"], help="which universe to scan")
    ap.add_argument("--sec-top-n", type=int, default=1000)
    ap.add_argument("--force", action="store_true", help="bypass the SEC and price caches")
    ap.add_argument("--tickers", default="", help="space/comma separated custom tickers (overrides --universe)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config()
    opts = universe_options(custom=args.tickers, sec_top_n=args.sec_top_n)
    if args.tickers:
        label = "Custom list"
    elif args.universe == "sec-top":
        label = f"SEC top {args.sec_top_n} by size (expanded)"
    else:
        label = "Wheel universe (shortlist)"
    u = opts[label]
    mgr = ScanManager()
    mgr.start(u["symbols"], label, u["note"], cfg, force=args.force)
    while mgr.running:
        p = mgr.progress
        print(f"\r{p.phase:12s} {p.done}/{p.total} {p.current:8s}", end="", flush=True)
        time.sleep(1)
    print(f"\n{mgr.progress.message}; alerts: {mgr.progress.alert_stats}")
    return 0 if mgr.progress.state == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
