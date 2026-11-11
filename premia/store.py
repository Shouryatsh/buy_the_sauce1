"""
premia/store.py — the research database (SQLite, `premia_data.db`, separate from the dashboard's wheel.db).

Raw data only; every derived number is recomputed from it.

  price       daily OHLCV per symbol. close = split-adjusted (not dividend-adjusted); adj_close = split+dividend adjusted
  action      corporate actions: ('div', cash per split-adjusted share) and ('split', ratio new/old)
  iv30        IBKR 30-day ATM implied volatility (OPTION_IMPLIED_VOLATILITY daily close), decimal
  series      index-level series (^VIX family, ^SKEW, ^IRX ...), keyed like price but close only
  membership  S&P 500 membership intervals (survivorship control)
  facts       SEC company facts reduced to the tags we use, gzip JSON, with every filing date kept (point-in-time)
  earnings    earnings-release dates (SEC 8-K item 2.02 filing dates)
  fetch_log   one row per (symbol, kind): status of the last download, so fetches resume and coverage is reportable
"""

from __future__ import annotations

import gzip
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterable, Optional

import pandas as pd

from .config import DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS price (
    symbol TEXT NOT NULL, date TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL NOT NULL, adj_close REAL, volume REAL,
    PRIMARY KEY (symbol, date)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS action (
    symbol TEXT NOT NULL, date TEXT NOT NULL, kind TEXT NOT NULL, value REAL NOT NULL,
    PRIMARY KEY (symbol, date, kind)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS iv30 (
    symbol TEXT NOT NULL, date TEXT NOT NULL, iv REAL NOT NULL,
    PRIMARY KEY (symbol, date)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS series (
    name TEXT NOT NULL, date TEXT NOT NULL, value REAL NOT NULL,
    PRIMARY KEY (name, date)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS membership (
    ticker TEXT NOT NULL, start TEXT NOT NULL, end TEXT,
    PRIMARY KEY (ticker, start));
CREATE TABLE IF NOT EXISTS facts (
    symbol TEXT PRIMARY KEY, cik INTEGER, name TEXT, sic TEXT, payload BLOB, fetched_at TEXT);
CREATE TABLE IF NOT EXISTS earnings (
    symbol TEXT NOT NULL, date TEXT NOT NULL, source TEXT,
    PRIMARY KEY (symbol, date)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS fetch_log (
    symbol TEXT NOT NULL, kind TEXT NOT NULL, status TEXT, detail TEXT, n INTEGER,
    first TEXT, last TEXT, fetched_at TEXT,
    PRIMARY KEY (symbol, kind));
"""


@contextmanager
def connect(path: Optional[str] = None):
    conn = sqlite3.connect(str(path or DB_PATH), timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# writers
# ---------------------------------------------------------------------------

def put_prices(symbol: str, df: pd.DataFrame, path=None) -> int:
    """df indexed by date with columns open, high, low, close, adj_close, volume."""
    if df.empty:
        return 0
    rows = [(symbol, d.strftime("%Y-%m-%d"), *(None if pd.isna(v) else float(v) for v in r))
            for d, r in zip(df.index, df[["open", "high", "low", "close", "adj_close", "volume"]].itertuples(index=False))]
    rows = [r for r in rows if r[6] is not None]
    with connect(path) as c:
        c.executemany("INSERT OR REPLACE INTO price VALUES (?,?,?,?,?,?,?,?)", rows)
    return len(rows)


def put_actions(symbol: str, rows: Iterable[tuple[str, str, float]], path=None) -> int:
    rows = [(symbol, d, k, float(v)) for d, k, v in rows]
    with connect(path) as c:
        c.execute("DELETE FROM action WHERE symbol=?", (symbol,))
        c.executemany("INSERT OR REPLACE INTO action VALUES (?,?,?,?)", rows)
    return len(rows)


def put_iv(symbol: str, s: pd.Series, path=None) -> int:
    rows = [(symbol, d.strftime("%Y-%m-%d"), float(v)) for d, v in s.items() if pd.notna(v) and v > 0]
    with connect(path) as c:
        c.executemany("INSERT OR REPLACE INTO iv30 VALUES (?,?,?)", rows)
    return len(rows)


def put_series(name: str, s: pd.Series, path=None) -> int:
    rows = [(name, d.strftime("%Y-%m-%d"), float(v)) for d, v in s.items() if pd.notna(v)]
    with connect(path) as c:
        c.executemany("INSERT OR REPLACE INTO series VALUES (?,?,?)", rows)
    return len(rows)


def put_membership(df: pd.DataFrame, path=None) -> int:
    rows = [(r.ticker, r.start_date, None if pd.isna(r.end_date) else r.end_date) for r in df.itertuples()]
    with connect(path) as c:
        c.execute("DELETE FROM membership")
        c.executemany("INSERT OR REPLACE INTO membership VALUES (?,?,?)", rows)
    return len(rows)


def put_facts(symbol: str, cik: int, name: str, sic: str, payload: dict, path=None) -> None:
    blob = gzip.compress(json.dumps(payload).encode())
    with connect(path) as c:
        c.execute("INSERT OR REPLACE INTO facts VALUES (?,?,?,?,?,?)", (symbol, cik, name, sic, blob, now_iso()))


def put_earnings(symbol: str, dates: Iterable[str], source: str, path=None) -> int:
    rows = [(symbol, d, source) for d in sorted(set(dates))]
    with connect(path) as c:
        c.execute("DELETE FROM earnings WHERE symbol=?", (symbol,))
        c.executemany("INSERT OR REPLACE INTO earnings VALUES (?,?,?)", rows)
    return len(rows)


def log_fetch(symbol: str, kind: str, status: str, detail: str = "", n: int = 0,
              first: Optional[str] = None, last: Optional[str] = None, path=None) -> None:
    with connect(path) as c:
        c.execute("INSERT OR REPLACE INTO fetch_log VALUES (?,?,?,?,?,?,?,?)",
                  (symbol, kind, status, detail[:300], n, first, last, now_iso()))


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------

def load_prices(symbol: str, path=None) -> pd.DataFrame:
    with connect(path) as c:
        df = pd.read_sql_query("SELECT date, open, high, low, close, adj_close, volume FROM price WHERE symbol=? ORDER BY date",
                               c, params=(symbol,), parse_dates=["date"])
    return df.set_index("date")


def load_actions(symbol: str, path=None) -> pd.DataFrame:
    with connect(path) as c:
        df = pd.read_sql_query("SELECT date, kind, value FROM action WHERE symbol=? ORDER BY date", c, params=(symbol,),
                               parse_dates=["date"])
    return df


def load_iv(symbol: str, path=None) -> pd.Series:
    with connect(path) as c:
        df = pd.read_sql_query("SELECT date, iv FROM iv30 WHERE symbol=? ORDER BY date", c, params=(symbol,),
                               parse_dates=["date"])
    return df.set_index("date")["iv"].rename(symbol)


def load_series(name: str, path=None) -> pd.Series:
    with connect(path) as c:
        df = pd.read_sql_query("SELECT date, value FROM series WHERE name=? ORDER BY date", c, params=(name,),
                               parse_dates=["date"])
    return df.set_index("date")["value"].rename(name)


def load_membership(path=None) -> pd.DataFrame:
    with connect(path) as c:
        return pd.read_sql_query("SELECT ticker, start, end FROM membership ORDER BY ticker, start", c)


def load_facts(symbol: str, path=None) -> Optional[dict]:
    with connect(path) as c:
        row = c.execute("SELECT cik, name, sic, payload FROM facts WHERE symbol=?", (symbol,)).fetchone()
    if not row or row[3] is None:
        return None
    return {"cik": row[0], "name": row[1], "sic": row[2], "facts": json.loads(gzip.decompress(row[3]))}


def load_earnings(symbol: str, path=None) -> list[pd.Timestamp]:
    with connect(path) as c:
        rows = c.execute("SELECT date FROM earnings WHERE symbol=? ORDER BY date", (symbol,)).fetchall()
    return [pd.Timestamp(r[0]) for r in rows]


def fetch_status(kind: Optional[str] = None, path=None) -> pd.DataFrame:
    with connect(path) as c:
        q = "SELECT * FROM fetch_log" + (" WHERE kind=?" if kind else "")
        return pd.read_sql_query(q, c, params=(kind,) if kind else ())


def symbols_with(kind: str, path=None) -> list[str]:
    table = {"iv": ("iv30", "symbol"), "price": ("price", "symbol"), "facts": ("facts", "symbol")}[kind]
    with connect(path) as c:
        return [r[0] for r in c.execute(f"SELECT DISTINCT {table[1]} FROM {table[0]}")]
