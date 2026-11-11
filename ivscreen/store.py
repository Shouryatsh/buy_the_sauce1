"""
ivscreen/store.py — SQLite persistence for the IV screener, in the dashboard's wheel.db
(same file and short-lived-connection pattern as stockscreen/store.py).

Tables
  ivscreen_run     one row per scan: universe, counts, status (complete / partial / ...)
  ivscreen_symbol  one row per symbol: IBKR contract facts (type, exchange, industry), universe sources,
                   the outcome of the last IV fetch (the data-availability status) and next earnings
  ivscreen_iv      daily 30-day implied volatility, one row per (symbol, date); top-ups overwrite overlap

Only raw, threshold-free data is stored. Percentiles and bands are computed at view time.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Iterable, Optional

import pandas as pd

from stockscreen.store import db_path as _db_path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ivscreen_run (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL, finished_at TEXT,
    universe_label TEXT, universe_size INTEGER, universe_note TEXT,
    ok INTEGER DEFAULT 0, no_options INTEGER DEFAULT 0, unsupported INTEGER DEFAULT 0,
    errors INTEGER DEFAULT 0, duplicates INTEGER DEFAULT 0,
    status TEXT DEFAULT 'running', note TEXT
);
CREATE TABLE IF NOT EXISTS ivscreen_symbol (
    symbol TEXT PRIMARY KEY,
    name TEXT, con_id INTEGER, sec_type TEXT, stock_type TEXT, category TEXT,
    exchange TEXT, currency TEXT, industry TEXT, sector TEXT, subcategory TEXT,
    sources TEXT, details_at TEXT,
    fetch_status TEXT, fetch_detail TEXT,
    iv_full_at TEXT, iv_checked_at TEXT, iv_n INTEGER, iv_first TEXT, iv_last TEXT,
    next_earnings TEXT, earnings_timing TEXT, earnings_checked_at TEXT,
    run_id INTEGER
);
CREATE TABLE IF NOT EXISTS ivscreen_iv (
    symbol TEXT NOT NULL, date TEXT NOT NULL, iv REAL NOT NULL,
    PRIMARY KEY (symbol, date)
) WITHOUT ROWID;
"""

SYMBOL_COLUMNS = (
    "name", "con_id", "sec_type", "stock_type", "category", "exchange", "currency", "industry", "sector",
    "subcategory", "sources", "details_at", "fetch_status", "fetch_detail", "iv_full_at", "iv_checked_at",
    "iv_n", "iv_first", "iv_last", "next_earnings", "earnings_timing", "earnings_checked_at", "run_id",
)


@contextmanager
def connect(path: Optional[str] = None):
    conn = sqlite3.connect(path or _db_path(), timeout=30)
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


# --- runs ---------------------------------------------------------------------------------

def start_run(label: str, size: int, note: str = "", path: Optional[str] = None) -> int:
    with connect(path) as c:
        cur = c.execute("INSERT INTO ivscreen_run (started_at, universe_label, universe_size, universe_note, status) "
                        "VALUES (?,?,?,?, 'running')", (now_iso(), label, size, note))
        return int(cur.lastrowid)


def finish_run(run_id: int, status: str, counts: dict, note: str = "", path: Optional[str] = None) -> None:
    with connect(path) as c:
        c.execute("UPDATE ivscreen_run SET finished_at=?, status=?, ok=?, no_options=?, unsupported=?, errors=?, "
                  "duplicates=?, note=? WHERE id=?",
                  (now_iso(), status, counts.get("ok", 0), counts.get("no_options", 0), counts.get("unsupported", 0),
                   counts.get("errors", 0), counts.get("duplicates", 0), note, run_id))


def interrupt_orphans(path: Optional[str] = None) -> int:
    """A run still 'running' while no scan is alive (killed process, sleep, crash) is recorded as partial."""
    with connect(path) as c:
        return c.execute("UPDATE ivscreen_run SET status='partial', finished_at=?, note=COALESCE(note,'') || "
                         "' interrupted before finishing' WHERE status='running'", (now_iso(),)).rowcount


def latest_run(path: Optional[str] = None) -> Optional[dict]:
    with connect(path) as c:
        row = c.execute("SELECT * FROM ivscreen_run ORDER BY id DESC LIMIT 1").fetchone()
        return dict(row) if row else None


# --- symbols ------------------------------------------------------------------------------

def upsert_symbol(symbol: str, path: Optional[str] = None, **fields) -> None:
    """Create the row if needed and set exactly the given columns."""
    bad = set(fields) - set(SYMBOL_COLUMNS)
    if bad:
        raise ValueError(f"unknown ivscreen_symbol columns: {sorted(bad)}")
    sym = symbol.upper()
    with connect(path) as c:
        c.execute("INSERT OR IGNORE INTO ivscreen_symbol (symbol) VALUES (?)", (sym,))
        if fields:
            sets = ", ".join(f"{k}=?" for k in fields)
            c.execute(f"UPDATE ivscreen_symbol SET {sets} WHERE symbol=?", (*fields.values(), sym))


def get_symbol(symbol: str, path: Optional[str] = None) -> Optional[dict]:
    with connect(path) as c:
        row = c.execute("SELECT * FROM ivscreen_symbol WHERE symbol=?", (symbol.upper(),)).fetchone()
    return dict(row) if row else None


def load_symbols(path: Optional[str] = None) -> dict[str, dict]:
    with connect(path) as c:
        return {r["symbol"]: dict(r) for r in c.execute("SELECT * FROM ivscreen_symbol")}


# --- implied-vol history ---------------------------------------------------------------------

def put_iv(symbol: str, series: pd.Series, path: Optional[str] = None) -> int:
    s = series.dropna()
    s = s[s > 0]
    rows = [(symbol.upper(), pd.Timestamp(d).strftime("%Y-%m-%d"), float(v)) for d, v in s.items()]
    with connect(path) as c:
        c.executemany("INSERT OR REPLACE INTO ivscreen_iv (symbol, date, iv) VALUES (?,?,?)", rows)
    return len(rows)


def load_iv(symbol: str, path: Optional[str] = None) -> pd.Series:
    with connect(path) as c:
        rows = c.execute("SELECT date, iv FROM ivscreen_iv WHERE symbol=? ORDER BY date", (symbol.upper(),)).fetchall()
    return pd.Series([r["iv"] for r in rows], index=pd.to_datetime([r["date"] for r in rows]), name="IV", dtype=float)


def load_all_iv(path: Optional[str] = None) -> dict[str, pd.Series]:
    with connect(path) as c:
        rows = c.execute("SELECT symbol, date, iv FROM ivscreen_iv ORDER BY symbol, date").fetchall()
    if not rows:
        return {}
    df = pd.DataFrame(rows, columns=["symbol", "date", "iv"])
    df["date"] = pd.to_datetime(df["date"])
    return {sym: g.set_index("date")["iv"].rename("IV") for sym, g in df.groupby("symbol")}


def last_iv_date(symbol: str, path: Optional[str] = None) -> Optional[str]:
    with connect(path) as c:
        row = c.execute("SELECT MAX(date) AS d FROM ivscreen_iv WHERE symbol=?", (symbol.upper(),)).fetchone()
    return row["d"] if row and row["d"] else None


# --- housekeeping -----------------------------------------------------------------------------

def purge_symbols(symbols: Iterable[str], path: Optional[str] = None) -> dict:
    """Remove every trace of the given tickers (used for manual exclusions)."""
    syms = [s.upper() for s in symbols]
    if not syms:
        return {"symbols": 0, "iv_rows": 0}
    out = {"symbols": 0, "iv_rows": 0}
    with connect(path) as c:
        for i in range(0, len(syms), 500):
            chunk = syms[i:i + 500]
            marks = ",".join("?" * len(chunk))
            out["iv_rows"] += c.execute(f"DELETE FROM ivscreen_iv WHERE symbol IN ({marks})", chunk).rowcount
            out["symbols"] += c.execute(f"DELETE FROM ivscreen_symbol WHERE symbol IN ({marks})", chunk).rowcount
    return out


def stamp(path: Optional[str] = None) -> str:
    """Changes whenever any stored data changes (cache key for evaluated results)."""
    with connect(path) as c:
        a = c.execute("SELECT COUNT(*) AS n, COALESCE(MAX(iv_checked_at),'') AS m, "
                      "COALESCE(MAX(details_at),'') AS d, "
                      "COALESCE(MAX(earnings_checked_at),'') AS e FROM ivscreen_symbol").fetchone()
        b = c.execute("SELECT COUNT(*) AS n FROM ivscreen_iv").fetchone()
    return f"{a['n']}|{a['m']}|{a['d']}|{a['e']}|{b['n']}"
