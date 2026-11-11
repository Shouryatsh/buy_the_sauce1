"""
stockscreen/store.py — SQLite persistence for the screener, in the dashboard's
existing wheel.db (path from wheel_config.WHEEL_DB_PATH).

Tables
  screener_scan_run     one row per scan: universe, counts, status (complete / partial / ...)
  screener_fund_cache   parsed SEC observations per symbol, refreshed per fundamentals_ttl_days
  screener_result       latest threshold-free scan record per symbol (JSON payload)
  screener_alert        in-app alerts (unread/read)
  screener_alert_state  per (symbol, kind) active flag + timestamps: the dedupe / cooldown state machine

Each call opens its own short-lived connection (WAL mode), so scan worker threads
and the Streamlit UI can use the store concurrently.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import wheel_config as wc

from .config import ROOT

_SCHEMA = """
CREATE TABLE IF NOT EXISTS screener_scan_run (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL, finished_at TEXT,
    universe_label TEXT, universe_size INTEGER, universe_note TEXT,
    evaluated INTEGER DEFAULT 0, no_statements INTEGER DEFAULT 0, errors INTEGER DEFAULT 0,
    status TEXT DEFAULT 'running', note TEXT
);
CREATE TABLE IF NOT EXISTS screener_fund_cache (
    symbol TEXT PRIMARY KEY, fetched_at TEXT NOT NULL, payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS screener_sic (
    symbol TEXT PRIMARY KEY, cik INTEGER, sic TEXT, sic_desc TEXT, fetched_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS screener_result (
    symbol TEXT PRIMARY KEY, scan_id INTEGER, scanned_at TEXT NOT NULL, payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_screener_result_scan ON screener_result(scan_id);
CREATE TABLE IF NOT EXISTS screener_alert (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL, symbol TEXT NOT NULL, name TEXT, kind TEXT NOT NULL,
    price REAL, trigger_text TEXT, percentile REAL, fcf_yield REAL,
    fund_status TEXT, tech_status TEXT, reason TEXT, data_asof TEXT, price_asof TEXT,
    is_read INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_screener_alert_created ON screener_alert(created_at);
CREATE INDEX IF NOT EXISTS ix_screener_alert_unread ON screener_alert(is_read);
CREATE TABLE IF NOT EXISTS screener_alert_state (
    symbol TEXT NOT NULL, kind TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 0,
    last_triggered TEXT, last_cleared TEXT,
    PRIMARY KEY (symbol, kind)
);
"""


def db_path() -> str:
    p = Path(wc.WHEEL_DB_PATH)
    return str(p if p.is_absolute() else ROOT / p)


@contextmanager
def connect(path: Optional[str] = None):
    conn = sqlite3.connect(path or db_path(), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


# --- scan runs -----------------------------------------------------------------

def start_run(label: str, size: int, note: str = "", path: Optional[str] = None) -> int:
    with connect(path) as c:
        cur = c.execute("INSERT INTO screener_scan_run (started_at, universe_label, universe_size, universe_note, status) "
                        "VALUES (?,?,?,?, 'running')", (now_iso(), label, size, note))
        return int(cur.lastrowid)


def finish_run(run_id: int, status: str, evaluated: int, no_statements: int, errors: int,
               note: str = "", path: Optional[str] = None) -> None:
    with connect(path) as c:
        c.execute("UPDATE screener_scan_run SET finished_at=?, status=?, evaluated=?, no_statements=?, errors=?, note=? WHERE id=?",
                  (now_iso(), status, evaluated, no_statements, errors, note, run_id))


def interrupt_orphans(path: Optional[str] = None) -> int:
    """A run still marked 'running' when no scan is actually running (process killed, machine slept, crash) can never
    finish: record it as an interrupted, partial scan instead of showing it as in progress forever."""
    with connect(path) as c:
        return c.execute("UPDATE screener_scan_run SET status='partial', finished_at=?, note=COALESCE(note,'') || "
                         "'interrupted before finishing' WHERE status='running'", (now_iso(),)).rowcount


def latest_run(path: Optional[str] = None) -> Optional[dict]:
    with connect(path) as c:
        row = c.execute("SELECT * FROM screener_scan_run ORDER BY id DESC LIMIT 1").fetchone()
        return dict(row) if row else None


def last_complete_run(path: Optional[str] = None) -> Optional[dict]:
    with connect(path) as c:
        row = c.execute("SELECT * FROM screener_scan_run WHERE status='complete' ORDER BY id DESC LIMIT 1").fetchone()
        return dict(row) if row else None


# --- fundamentals cache -----------------------------------------------------------

def get_fund_cache(symbol: str, max_age_days: float, path: Optional[str] = None) -> Optional[dict]:
    with connect(path) as c:
        row = c.execute("SELECT fetched_at, payload FROM screener_fund_cache WHERE symbol=?", (symbol,)).fetchone()
    if not row:
        return None
    if datetime.now() - datetime.fromisoformat(row["fetched_at"]) > timedelta(days=max_age_days):
        return None
    return json.loads(row["payload"])


def put_fund_cache(symbol: str, payload: dict, path: Optional[str] = None) -> None:
    with connect(path) as c:
        c.execute("INSERT OR REPLACE INTO screener_fund_cache (symbol, fetched_at, payload) VALUES (?,?,?)",
                  (symbol, now_iso(), json.dumps(payload)))


# --- SIC industry codes (cached permanently) ------------------------------------------------------------------

def get_sic(symbol: str, path: Optional[str] = None) -> Optional[tuple[str, str]]:
    with connect(path) as c:
        row = c.execute("SELECT sic, sic_desc FROM screener_sic WHERE symbol=?", (symbol.upper(),)).fetchone()
    return (row["sic"], row["sic_desc"]) if row and row["sic"] else None


def put_sic(symbol: str, cik: int, sic: str, desc: str, path: Optional[str] = None) -> None:
    with connect(path) as c:
        c.execute("INSERT OR REPLACE INTO screener_sic (symbol, cik, sic, sic_desc, fetched_at) VALUES (?,?,?,?,?)",
                  (symbol.upper(), cik, sic, desc, now_iso()))


# --- results ----------------------------------------------------------------------------

def upsert_result(symbol: str, scan_id: int, record: dict, path: Optional[str] = None) -> None:
    with connect(path) as c:
        c.execute("INSERT OR REPLACE INTO screener_result (symbol, scan_id, scanned_at, payload) VALUES (?,?,?,?)",
                  (symbol, scan_id, record.get("scanned_at") or now_iso(), json.dumps(record)))


def load_results(symbols: Optional[Iterable[str]] = None, path: Optional[str] = None) -> list[dict]:
    with connect(path) as c:
        rows = c.execute("SELECT payload FROM screener_result").fetchall()
    recs = [json.loads(r["payload"]) for r in rows]
    if symbols is not None:
        keep = {s.upper() for s in symbols}
        recs = [r for r in recs if r["symbol"].upper() in keep]
    return recs


def results_stamp(path: Optional[str] = None) -> str:
    """Changes whenever any stored result is added or rewritten (used as a cache key)."""
    with connect(path) as c:
        n, last = c.execute("SELECT COUNT(*), MAX(scanned_at) FROM screener_result").fetchone()
    return f"{n}|{last}"


def purge_symbols(symbols: Iterable[str], path: Optional[str] = None) -> dict:
    """Delete every stored trace of *symbols* (results, cached SEC data, alerts, alert state). Returns row counts."""
    syms = sorted({s.upper() for s in symbols})
    out = {"results": 0, "cache": 0, "alerts": 0, "states": 0}
    if not syms:
        return out
    marks = ",".join("?" * len(syms))
    with connect(path) as c:
        for key, table in (("results", "screener_result"), ("cache", "screener_fund_cache"),
                           ("alerts", "screener_alert"), ("states", "screener_alert_state")):
            out[key] = c.execute(f"DELETE FROM {table} WHERE UPPER(symbol) IN ({marks})", syms).rowcount
    return out


def count_results(path: Optional[str] = None) -> int:
    with connect(path) as c:
        return int(c.execute("SELECT COUNT(*) FROM screener_result").fetchone()[0])


# --- alerts --------------------------------------------------------------------------------

def add_alert(alert: dict, path: Optional[str] = None) -> int:
    with connect(path) as c:
        cur = c.execute(
            "INSERT INTO screener_alert (created_at, symbol, name, kind, price, trigger_text, percentile, fcf_yield, "
            "fund_status, tech_status, reason, data_asof, price_asof) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (alert.get("created_at") or now_iso(), alert["symbol"], alert.get("name"), alert["kind"], alert.get("price"),
             alert.get("trigger_text"), alert.get("percentile"), alert.get("fcf_yield"), alert.get("fund_status"),
             alert.get("tech_status"), alert.get("reason"), alert.get("data_asof"), alert.get("price_asof")))
        return int(cur.lastrowid)


def list_alerts(unread_only: bool = False, limit: int = 200, path: Optional[str] = None) -> list[dict]:
    with connect(path) as c:
        q = "SELECT * FROM screener_alert " + ("WHERE is_read=0 " if unread_only else "") + "ORDER BY id DESC LIMIT ?"
        return [dict(r) for r in c.execute(q, (limit,)).fetchall()]


def mark_alerts_read(ids: Optional[list[int]] = None, path: Optional[str] = None) -> None:
    with connect(path) as c:
        if ids is None:
            c.execute("UPDATE screener_alert SET is_read=1")
        else:
            c.executemany("UPDATE screener_alert SET is_read=1 WHERE id=?", [(i,) for i in ids])


def get_alert_state(symbol: str, kind: str, path: Optional[str] = None) -> dict:
    with connect(path) as c:
        row = c.execute("SELECT * FROM screener_alert_state WHERE symbol=? AND kind=?", (symbol, kind)).fetchone()
    return dict(row) if row else {"symbol": symbol, "kind": kind, "active": 0, "last_triggered": None, "last_cleared": None}


def set_alert_state(symbol: str, kind: str, active: bool, triggered: Optional[str] = None, cleared: Optional[str] = None,
                    path: Optional[str] = None) -> None:
    with connect(path) as c:
        old = c.execute("SELECT last_triggered, last_cleared FROM screener_alert_state WHERE symbol=? AND kind=?",
                        (symbol, kind)).fetchone()
        lt = triggered or (old["last_triggered"] if old else None)
        lc = cleared or (old["last_cleared"] if old else None)
        c.execute("INSERT OR REPLACE INTO screener_alert_state (symbol, kind, active, last_triggered, last_cleared) "
                  "VALUES (?,?,?,?,?)", (symbol, kind, int(active), lt, lc))
