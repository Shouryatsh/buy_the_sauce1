"""
wheel_db.py — Local SQLite persistence for the wheel dashboard.

Two tables:
  screen_history — one row per (run, symbol), so the Fundamentals Screener
                    panel has a history to chart, not just today's snapshot.
  trade_log       — one row per trade leg or manually-logged closed trade.

Why trade_log matters more than it looks: IBKR's live reqExecutions API has
turned out to retain essentially nothing beyond the current session in
practice (confirmed empty even over a 90-day request) -- nowhere near
enough for a wheel strategy where positions are opened weeks before they
close. Two consequences:

1. record_fill() should be called every time live fills ARE seen, so a
   history accumulates locally going forward, even though IBKR's own API
   can't supply one after the fact.
2. For anything that closed *before* this archiving was wired up, or that
   IBKR's API simply never surfaced (a worthless expiration generates no
   execution at all -- the position just disappears), record_manual_trade()
   lets you log it directly. wheel_dashboard.py's Monthly P&L tab sums
   both sources.

Kept intentionally tiny -- no real migrations framework, just an
idempotent ALTER TABLE ADD COLUMN step for columns added after the first
release, guarded by checking PRAGMA table_info first.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime

import wheel_config as wc

_SCHEMA = """
CREATE TABLE IF NOT EXISTS screen_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at TEXT NOT NULL,
    symbol TEXT NOT NULL,
    price REAL,
    fcf_yield REAL,
    quality_score REAL,
    passes INTEGER,
    data_reliable INTEGER
);

CREATE TABLE IF NOT EXISTS trade_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at TEXT NOT NULL,
    symbol TEXT NOT NULL,
    sec_type TEXT,
    action TEXT,
    quantity REAL,
    price REAL,
    realized_pnl REAL,
    conid INTEGER,
    UNIQUE(conid, logged_at, action, quantity, price)
);
"""

# Columns added after the original release -- ALTER TABLE ADD COLUMN is
# additive and safe to run repeatedly as long as we check first.
_TRADE_LOG_MIGRATIONS = [
    ("closed_date", "TEXT"),           # the date the trade actually closed/expired (not when we logged it)
    ("notes", "TEXT"),
    ("source", "TEXT DEFAULT 'ibkr'"),  # 'ibkr' (from a live fill) or 'manual'
]


def _migrate(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(trade_log)")}
    for col, coltype in _TRADE_LOG_MIGRATIONS:
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE trade_log ADD COLUMN {col} {coltype}")


@contextmanager
def _connect(db_path: str = wc.WHEEL_DB_PATH):
    conn = sqlite3.connect(db_path)
    try:
        _migrate(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


def save_screen_snapshot(profiles: list, db_path: str = wc.WHEEL_DB_PATH) -> None:
    """Persist a batch of wheel_fundamentals.WheelFundamentalProfile rows."""
    run_at = datetime.now().isoformat(timespec="seconds")
    with _connect(db_path) as conn:
        conn.executemany(
            "INSERT INTO screen_history (run_at, symbol, price, fcf_yield, quality_score, passes, data_reliable) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (run_at, p.symbol, p.price, p.fcf_yield, p.quality_score, int(p.passes), int(p.data_reliable))
                for p in profiles
            ],
        )


def record_fill(
    symbol: str,
    sec_type: str,
    action: str,
    quantity: float,
    price: float,
    conid: int,
    realized_pnl: float = 0.0,
    closed_date: str = None,
    db_path: str = wc.WHEEL_DB_PATH,
) -> None:
    """Archive one raw fill leg as seen from IBKR's live execution feed.
    Safe to call repeatedly for the same fill -- the UNIQUE constraint on
    (conid, logged_at, action, quantity, price) makes this an upsert-once."""
    logged_at = datetime.now().isoformat(timespec="seconds")
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO trade_log "
            "(logged_at, symbol, sec_type, action, quantity, price, realized_pnl, conid, closed_date, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'ibkr')",
            (logged_at, symbol, sec_type, action, quantity, price, realized_pnl, conid, closed_date),
        )


def record_manual_trade(
    symbol: str,
    sec_type: str,
    action: str,               # e.g. "EXPIRED_WORTHLESS", "ASSIGNED", "CLOSED"
    quantity: float,           # contracts (positive)
    premium_per_contract: float,
    realized_pnl: float,
    closed_date: str,          # ISO date (YYYY-MM-DD) the trade actually closed
    notes: str = "",
    db_path: str = wc.WHEEL_DB_PATH,
) -> None:
    """Log a trade IBKR's own API never surfaced -- a worthless expiration
    generates no execution at all (the position just disappears), and
    reqExecutions' retention window is far shorter than a typical wheel
    position's holding period. See module docstring."""
    logged_at = datetime.now().isoformat(timespec="seconds")
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO trade_log "
            "(logged_at, symbol, sec_type, action, quantity, price, realized_pnl, conid, closed_date, notes, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, 'manual')",
            (logged_at, symbol, sec_type, action, quantity, premium_per_contract, realized_pnl, closed_date, notes),
        )


def get_trade_log(limit: int = 200, db_path: str = wc.WHEEL_DB_PATH) -> list[dict]:
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM trade_log ORDER BY COALESCE(closed_date, logged_at) DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_manual_realized_pnl_since(start_date: str, db_path: str = wc.WHEEL_DB_PATH) -> dict:
    """Sum realized P&L from manually-logged trades closed on/after
    *start_date* (ISO date string). Kept separate from IBKR-sourced fills
    (get_trade_stats in wheel_ibkr.py) so the two are simply additive --
    they're disjoint by construction (you log manually precisely because
    IBKR's API isn't showing it)."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT realized_pnl FROM trade_log "
            "WHERE source = 'manual' AND closed_date >= ?",
            (start_date,),
        ).fetchall()
    values = [r[0] for r in rows if r[0] is not None]
    return {"realized_pnl": sum(values), "trade_count": len(values)}


def get_screen_history(symbol: str = None, limit: int = 500, db_path: str = wc.WHEEL_DB_PATH) -> list[dict]:
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        if symbol:
            rows = conn.execute(
                "SELECT * FROM screen_history WHERE symbol = ? ORDER BY run_at DESC LIMIT ?",
                (symbol, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM screen_history ORDER BY run_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]
