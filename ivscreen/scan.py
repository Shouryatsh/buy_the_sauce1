"""
ivscreen/scan.py — universe building and the background IBKR scan.

What a scan does, per symbol (all READ-ONLY: contract details + historical-data requests only —
no orders, no market-data subscriptions, no account calls):
  1. resolve the contract once (type, exchange, industry; cached in ivscreen_symbol)
  2. fetch IBKR's 30-day implied-vol history (whatToShow=OPTION_IMPLIED_VOLATILITY, daily). The first
     download is the full history; later scans fetch only the last ~2 months and merge them in
  3. record the outcome as a data-availability status — "IBKR has no options data", "request failed",
     "unsupported instrument" are all distinct from a genuinely low IV and never become a zero
  4. next earnings date (Yahoo Finance) for the event note, non-ETFs only
Daily prices for realized vol come from the screener's shared price cache (stockscreen/prices.py).

Manual ticker exclusions (wheel_exclusions.json) are absolute: excluded tickers are never requested,
and anything stored for them is purged.

Run headless (e.g. from Task Scheduler; needs TWS up):
    python -m ivscreen.scan --universe wheel [--etf] [--force]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Callable, Optional

import pandas as pd

from stockscreen import prices as pr
from stockscreen import universe as su

from . import store
from .config import IVConfig, load_config

logger = logging.getLogger(__name__)

UNSUPPORTED_STOCK_TYPES = {"PREFERRED", "RIGHT", "WARRANT", "UNIT", "CLOSED-END FUND", "TRACKING STOCK"}
INSTRUMENT_BY_STOCK_TYPE = {"COMMON": "Common stock", "ETF": "ETF", "ADR": "ADR", "REIT": "REIT / trust"}

SOURCE_WHEEL, SOURCE_HOLDINGS, SOURCE_ETF, SOURCE_CUSTOM = "Wheel universe", "Holdings", "ETFs", "Custom"
ALL_SOURCES = (SOURCE_WHEEL, SOURCE_HOLDINGS, SOURCE_ETF, SOURCE_CUSTOM)


# ---------------------------------------------------------------------------
# universe
# ---------------------------------------------------------------------------

def normalize_symbol(s: str) -> str:
    """BRK.B / brk b / BRK-B all become BRK-B (the form used everywhere else in the dashboard)."""
    return s.strip().upper().replace(".", "-").replace("/", "-").replace(" ", "-")


def ib_symbol(s: str) -> str:
    return s.replace("-", " ")


def build_universe(sources: list[str], holdings: Optional[list[str]] = None, custom: str = "",
                   cfg: Optional[IVConfig] = None) -> dict:
    """{"symbols": [...], "sources": {symbol: [source, ...]}, "dropped": [...], "duplicates": n, "note": str}.
    Manual exclusions apply to every source; duplicate spellings collapse to one symbol."""
    from wheel_watchlist import active_watchlist
    cfg = cfg or load_config()
    pools = {
        SOURCE_WHEEL: list(active_watchlist()),
        SOURCE_HOLDINGS: list(holdings or []),
        SOURCE_ETF: list(cfg.etf_symbols),
        SOURCE_CUSTOM: [t for t in custom.replace(",", " ").replace(";", " ").split() if t.strip()],
    }
    ex = su.excluded()
    order: list[str] = []
    srcs: dict[str, list[str]] = {}
    dropped: list[str] = []
    dups = 0
    for name in ALL_SOURCES:
        if name not in sources:
            continue
        for raw in pools[name]:
            sym = normalize_symbol(raw)
            if not sym:
                continue
            if sym in ex or sym.replace("-", ".") in ex or sym.replace("-", "") in ex:
                if sym not in dropped:
                    dropped.append(sym)
                continue
            if sym in srcs:
                dups += 1
                if name not in srcs[sym]:
                    srcs[sym].append(name)
                continue
            order.append(sym)
            srcs[sym] = [name]
    note = (f"{len(order)} symbols from {', '.join(s for s in ALL_SOURCES if s in sources) or 'nothing selected'}"
            + (f"; {len(dropped)} manually excluded tickers skipped" if dropped else "")
            + (f"; {dups} duplicate entries merged" if dups else ""))
    return {"symbols": order, "sources": srcs, "dropped": dropped, "duplicates": dups, "note": note}


# ---------------------------------------------------------------------------
# IBKR requests (module-level so tests can replace them)
# ---------------------------------------------------------------------------

async def _resolve(ib, symbol: str, timeout: float = 25.0) -> Optional[dict]:
    """Contract facts for *symbol*, or None when IBKR does not know it."""
    from ib_insync import Stock
    details = await asyncio.wait_for(ib.reqContractDetailsAsync(Stock(ib_symbol(symbol), "SMART", "USD")), timeout)
    stocks = [d for d in (details or []) if getattr(d.contract, "secType", "STK") == "STK"]
    if not stocks:
        return {"unsupported_only": True} if details else None
    d = stocks[0]
    return {"con_id": d.contract.conId, "sec_type": d.contract.secType, "stock_type": (d.stockType or "").upper(),
            "exchange": d.contract.primaryExchange or d.contract.exchange or "", "currency": d.contract.currency,
            "name": d.longName or "", "sector": d.industry or "", "industry": d.category or "",
            "subcategory": d.subcategory or "", "listings": len(stocks)}


async def _fetch_iv_bars(ib, symbol: str, info: dict, duration: str, timeout: float) -> tuple[pd.Series, list]:
    """(implied-vol series, [(ibkr error code, message)] raised for this contract)."""
    from ib_insync import Stock
    c = Stock(ib_symbol(symbol), "SMART", "USD", conId=info["con_id"], primaryExchange=info.get("exchange") or "")
    errors: list = []

    def on_error(req_id, code, msg, contract):
        if contract is not None and getattr(contract, "conId", None) == info["con_id"] and code not in (2104, 2106, 2158):
            errors.append((code, msg))

    ib.errorEvent += on_error
    try:
        bars = await ib.reqHistoricalDataAsync(c, "", duration, "1 day", "OPTION_IMPLIED_VOLATILITY", True,
                                               formatDate=1, timeout=timeout)
    finally:
        ib.errorEvent -= on_error
    if not bars:
        return pd.Series(dtype=float, name="IV"), errors
    s = pd.Series([b.close for b in bars], index=pd.to_datetime([b.date for b in bars]), name="IV")
    return s[s > 0], errors


def _earnings(symbol: str) -> tuple[Optional[str], str]:
    import wheel_events
    today = date.today()
    nxt = next(((d, t) for d, t in wheel_events.earnings_events(symbol) if d >= today), None)
    return (nxt[0].isoformat(), nxt[1]) if nxt else (None, "")


# ---------------------------------------------------------------------------
# per-symbol work
# ---------------------------------------------------------------------------

def _classify_failure(series_errors: list, elapsed: float, timeout: float) -> tuple[str, str]:
    """An empty answer is 'no data' only when IBKR answered; a timeout or pacing message is a failed fetch."""
    text = "; ".join(f"{c}: {m}" for c, m in series_errors)[:200]
    if elapsed >= timeout - 1:
        return "error", f"request timed out after {timeout:.0f}s"
    if any("pacing" in m.lower() for _c, m in series_errors):
        return "error", f"IBKR pacing limit — retry later ({text})"
    if any(c in (1100, 1101, 1102, 504, 502, 509) for c, _m in series_errors):
        return "error", f"connection problem ({text})"
    return "no_data", text or "IBKR returned no implied-vol history for this symbol"


def _durations(cfg: IVConfig, last_date: Optional[str], full_at: Optional[str], force: bool, today: date) -> tuple[str, bool]:
    """(IBKR duration, is_full_download)."""
    if force or not last_date or not full_at:
        return cfg.iv_duration, True
    try:
        gap = (today - datetime.strptime(last_date, "%Y-%m-%d").date()).days
        full_age = (datetime.now() - datetime.fromisoformat(full_at)).days
    except ValueError:
        return cfg.iv_duration, True
    if gap > 35 or full_age >= cfg.full_refresh_days:
        return cfg.iv_duration, True
    return cfg.refresh_duration, False


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
    ok: int = 0
    no_options: int = 0
    unsupported: int = 0
    errors: int = 0
    duplicates: int = 0
    started: float = 0.0
    finished: float = 0.0
    run_id: Optional[int] = None
    message: str = ""

    @property
    def fraction(self) -> float:
        return self.done / self.total if self.total else 0.0

    @property
    def eta_seconds(self) -> Optional[float]:
        if not self.done or self.state != "running":
            return None
        return (time.time() - self.started) / self.done * (self.total - self.done)

    def counts(self) -> dict:
        return {"ok": self.ok, "no_options": self.no_options, "unsupported": self.unsupported,
                "errors": self.errors, "duplicates": self.duplicates}


class IVScanManager:
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

    def start(self, client, universe: dict, label: str, cfg: IVConfig, force: bool = False) -> bool:
        symbols, _ = su.drop_excluded(universe["symbols"])          # exclusions are absolute
        sources = {s: universe.get("sources", {}).get(s, []) for s in symbols}
        self._purge_excluded()
        with self._lock:
            if self.running:
                return False
            store.interrupt_orphans()
            self._cancel.clear()
            self.progress = Progress(state="running", phase="starting", total=len(symbols), started=time.time())
            self._thread = threading.Thread(target=self._run, args=(client, symbols, sources, label, universe.get("note", ""),
                                                                    cfg, force), name="ivscreen-scan", daemon=True)
            self._thread.start()
            return True

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    @staticmethod
    def _purge_excluded() -> None:
        ex = su.excluded()
        if ex:
            store.purge_symbols(ex)

    # -- worker ----------------------------------------------------------------------------
    def _run(self, client, symbols: list[str], sources: dict, label: str, note: str, cfg: IVConfig, force: bool) -> None:
        p = self.progress
        run_id = store.start_run(label, len(symbols), note)
        p.run_id = run_id
        try:
            price_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ivscreen-prices")
            price_job = price_pool.submit(pr.fetch_prices, symbols, cfg.price_years, cfg.price_ttl_hours, force)
            p.phase = "implied vol"
            client.run(lambda: self._scan_async(client.ib, symbols, sources, cfg, force, run_id),
                       timeout=6 * 3600)
            p.phase = "prices"
            try:
                price_job.result(timeout=900)
            except Exception as exc:
                logger.warning("price download failed: %s", exc)
            price_pool.shutdown(wait=False)
            self._purge_excluded()                                   # also covers an exclusion added mid-scan
            cancelled = self._cancel.is_set()
            complete = (not cancelled) and p.done == p.total and p.errors == 0
            p.state = "cancelled" if cancelled else "done"
            p.message = (f"{p.ok} ranked-ready, {p.no_options} without options data, {p.unsupported} unsupported, "
                         f"{p.duplicates} duplicates, {p.errors} failed" + (" (cancelled)" if cancelled else ""))
            store.finish_run(run_id, "complete" if complete else "partial", p.counts(), p.message)
        except Exception as exc:
            logger.exception("IV scan failed")
            p.state, p.message = "error", f"{type(exc).__name__}: {exc}"
            store.finish_run(run_id, "failed", p.counts(), p.message)
        finally:
            p.finished = time.time()
            p.phase = ""

    async def _scan_async(self, ib, symbols: list[str], sources: dict, cfg: IVConfig, force: bool, run_id: int) -> None:
        p = self.progress
        sem = asyncio.Semaphore(max(1, cfg.concurrency))
        earn_sem = asyncio.Semaphore(4)
        loop = asyncio.get_running_loop()
        seen_con: dict[int, str] = {}
        today = date.today()

        retry: list[str] = []

        async def attempt(sym: str, limiter: asyncio.Semaphore) -> Optional[str]:
            """One pass over a symbol; returns the Progress counter to bump, or None if skipped (cancelled / excluded)."""
            if self._cancel.is_set() or sym in su.excluded():
                return None
            try:
                async with limiter:
                    if self._cancel.is_set():
                        return None
                    p.current = sym
                    outcome = await self._one(ib, sym, sources.get(sym, []), cfg, force, run_id, seen_con, today)
            except Exception as exc:
                logger.exception("%s: scan failed", sym)
                store.upsert_symbol(sym, fetch_status="error", fetch_detail=f"{type(exc).__name__}: {exc}"[:200],
                                    iv_checked_at=store.now_iso(), sources=",".join(sources.get(sym, [])), run_id=run_id)
                return "errors"
            if outcome == "ok":
                await self._earnings_for(sym, loop, earn_sem, today)       # never changes the implied-vol status
            return outcome

        async def first_pass(sym: str) -> None:
            outcome = await attempt(sym, sem)
            if outcome is None:
                return
            setattr(p, outcome, getattr(p, outcome) + 1)
            p.done += 1
            if outcome == "errors":
                retry.append(sym)

        await asyncio.gather(*(first_pass(s) for s in symbols))

        # Failures are usually IBKR hiccups (a request or contract lookup timing out under load): give each one more
        # try, one at a time, before reporting it as failed.
        if retry and not self._cancel.is_set():
            p.phase = f"retrying {len(retry)} failed"
            await asyncio.sleep(3)
            one_at_a_time = asyncio.Semaphore(1)
            for sym in retry:
                if self._cancel.is_set():
                    break
                p.current = sym
                outcome = await attempt(sym, one_at_a_time)
                if outcome and outcome != "errors":
                    p.errors -= 1
                    setattr(p, outcome, getattr(p, outcome) + 1)

    async def _earnings_for(self, sym: str, loop, sem: asyncio.Semaphore, today: date) -> None:
        """Next earnings date for the event note. Best effort: any problem here is logged and ignored."""
        try:
            meta = store.get_symbol(sym) or {}
            if meta.get("category") == "ETF":
                return
            known = meta.get("next_earnings") or ""
            checked = meta.get("earnings_checked_at")
            if checked and known >= today.isoformat() and datetime.fromisoformat(checked) > datetime.now() - timedelta(days=1):
                return
            async with sem:
                nxt, timing = await loop.run_in_executor(None, _earnings, sym)
            if nxt is None and known >= today.isoformat():
                nxt, timing = known, meta.get("earnings_timing") or ""      # keep a known future date
            store.upsert_symbol(sym, next_earnings=nxt, earnings_timing=timing, earnings_checked_at=store.now_iso())
        except Exception as exc:
            logger.info("%s: earnings step failed: %s", sym, exc)

    async def _one(self, ib, sym: str, srcs: list[str], cfg: IVConfig, force: bool, run_id: int,
                   seen_con: dict, today: date) -> str:
        """Scan one symbol; returns the Progress counter to bump: ok / no_options / unsupported / duplicates / errors."""
        meta = store.get_symbol(sym) or {}
        base = {"sources": ",".join(srcs), "run_id": run_id}
        info = None
        if meta.get("con_id") and meta.get("details_at") and meta.get("fetch_status") != "unresolved":
            info = {k: meta.get(k) for k in ("con_id", "exchange", "currency", "stock_type", "sec_type")}
        else:
            try:
                got = await _resolve(ib, sym)
            except asyncio.TimeoutError:
                store.upsert_symbol(sym, fetch_status="error", fetch_detail="contract lookup timed out",
                                    iv_checked_at=store.now_iso(), **base)
                return "errors"
            if got is None:
                store.upsert_symbol(sym, fetch_status="unresolved", fetch_detail="IBKR has no stock with this ticker",
                                    iv_checked_at=store.now_iso(), details_at=store.now_iso(), **base)
                return "unsupported"
            if got.get("unsupported_only"):
                store.upsert_symbol(sym, fetch_status="unsupported", fetch_detail="not a stock/ETF contract",
                                    iv_checked_at=store.now_iso(), details_at=store.now_iso(), **base)
                return "unsupported"
            category = INSTRUMENT_BY_STOCK_TYPE.get(got["stock_type"], "Other")
            store.upsert_symbol(sym, name=got["name"], con_id=got["con_id"], sec_type=got["sec_type"],
                                stock_type=got["stock_type"], category=category, exchange=got["exchange"],
                                currency=got["currency"], sector=got["sector"], industry=got["industry"],
                                subcategory=got["subcategory"], details_at=store.now_iso(), **base)
            info = got
            if got["stock_type"] in UNSUPPORTED_STOCK_TYPES:
                store.upsert_symbol(sym, fetch_status="unsupported", fetch_detail=f"{got['stock_type'].title()} is not supported",
                                    iv_checked_at=store.now_iso())
                return "unsupported"
        if meta.get("fetch_status") == "unsupported" and meta.get("stock_type") in UNSUPPORTED_STOCK_TYPES:
            store.upsert_symbol(sym, iv_checked_at=store.now_iso(), **base)
            return "unsupported"

        owner = seen_con.setdefault(int(info["con_id"]), sym)
        if owner != sym:
            store.upsert_symbol(sym, fetch_status="duplicate", fetch_detail=f"same IBKR contract as {owner}",
                                iv_checked_at=store.now_iso(), **base)
            return "duplicates"

        last = store.last_iv_date(sym)
        if (not force and last and meta.get("fetch_status") == "ok" and meta.get("iv_checked_at")
                and datetime.fromisoformat(meta["iv_checked_at"]) > datetime.now() - timedelta(hours=cfg.skip_fresh_hours)):
            store.upsert_symbol(sym, **base)                               # checked recently (e.g. resuming a cancelled scan)
            return "ok"
        duration, full = _durations(cfg, last, meta.get("iv_full_at"), force, today)
        t0 = time.time()
        series, errs = await _fetch_iv_bars(ib, sym, info, duration, cfg.request_timeout)
        if series.empty and not full:                                  # a short top-up can legitimately be empty; ask for it all
            series, errs = await _fetch_iv_bars(ib, sym, info, cfg.iv_duration, cfg.request_timeout)
            full = True
        if series.empty:
            status, detail = _classify_failure(errs, time.time() - t0, cfg.request_timeout)
            if last:                                                    # never downgrade data we already hold to "no data"
                status, detail = "error", f"refresh returned nothing ({detail}); keeping data through {last}"
            store.upsert_symbol(sym, fetch_status=status, fetch_detail=detail, iv_checked_at=store.now_iso(), **base)
            return "no_options" if status == "no_data" else "errors"

        store.put_iv(sym, series)
        full_rows = store.load_iv(sym)
        fields = dict(fetch_status="ok", fetch_detail="", iv_checked_at=store.now_iso(), iv_n=len(full_rows),
                      iv_first=full_rows.index[0].strftime("%Y-%m-%d"), iv_last=full_rows.index[-1].strftime("%Y-%m-%d"), **base)
        if full:
            fields["iv_full_at"] = store.now_iso()
        store.upsert_symbol(sym, **fields)
        return "ok"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Refresh implied-vol history for the IV percentile screener (read-only).")
    ap.add_argument("--universe", default="wheel", choices=["wheel", "none"], help="wheel = the wheel shortlist")
    ap.add_argument("--etf", action="store_true", help="also scan the configured ETF list")
    ap.add_argument("--tickers", default="", help="extra space/comma separated tickers")
    ap.add_argument("--force", action="store_true", help="re-download full histories")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from wheel_ibkr import WheelIBKRClient
    cfg = load_config()
    sources = ([SOURCE_WHEEL] if args.universe == "wheel" else []) + ([SOURCE_ETF] if args.etf else []) \
        + ([SOURCE_CUSTOM] if args.tickers else [])
    uni = build_universe(sources, custom=args.tickers, cfg=cfg)
    client = WheelIBKRClient(client_id=int(os.environ.get("WHEEL_IBKR_CLIENT_ID", "78")))
    client.connect()
    try:
        mgr = IVScanManager()
        mgr.start(client, uni, "CLI scan", cfg, force=args.force)
        while mgr.running:
            p = mgr.progress
            print(f"\r{p.phase:12s} {p.done}/{p.total} {p.current:8s}", end="", flush=True)
            time.sleep(1)
        print(f"\n{mgr.progress.message}")
        return 0 if mgr.progress.state == "done" else 1
    finally:
        client.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
