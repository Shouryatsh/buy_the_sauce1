"""
volsurface/store.py — saved quote snapshots, so surface dynamics can be studied over time.

A snapshot is the raw quote table plus spot/asof/source (zlib-compressed JSON, ~50-150 KB for a full SPX
set). The fitted surface is always recomputed from it, so a change in fit settings re-analyses every saved
snapshot. A small metrics blob (constant-maturity ATM vol, skew, SVI parameters) is cached beside it for the
dynamics charts.
"""

from __future__ import annotations

import json
import sqlite3
import zlib
from contextlib import contextmanager
from io import StringIO
from typing import Optional

import pandas as pd

from stockscreen.store import db_path

from .data import QuoteSet
from .surface import QUOTE_COLUMNS

_SCHEMA = """
CREATE TABLE IF NOT EXISTS volsurface_snapshot (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL, source TEXT NOT NULL, asof TEXT NOT NULL, spot REAL, n_quotes INTEGER,
    payload BLOB NOT NULL, metrics TEXT
);
CREATE INDEX IF NOT EXISTS ix_volsurface_snapshot ON volsurface_snapshot(symbol, asof);
"""


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


def save_snapshot(qs: QuoteSet, metrics: Optional[dict] = None, min_gap_seconds: float = 120.0, path: Optional[str] = None) -> int:
    """Store a QuoteSet; returns its id. A snapshot of the same symbol and source less than *min_gap_seconds* after the
    previous one is not stored again (the previous id is returned), so a page refresh never floods the table."""
    with connect(path) as c:
        row = c.execute("SELECT id, asof FROM volsurface_snapshot WHERE symbol=? AND source=? ORDER BY asof DESC LIMIT 1",
                        (qs.symbol, qs.source)).fetchone()
        if row and abs((pd.Timestamp(qs.asof) - pd.Timestamp(row["asof"])).total_seconds()) < min_gap_seconds:
            return int(row["id"])
        body = json.dumps({"quotes": json.loads(qs.quotes.to_json(orient="split")), "all_expiries": qs.all_expiries, "notes": qs.notes})
        cur = c.execute("INSERT INTO volsurface_snapshot (symbol, source, asof, spot, n_quotes, payload, metrics) VALUES (?,?,?,?,?,?,?)",
                        (qs.symbol, qs.source, str(qs.asof), qs.spot, int(len(qs.quotes)), zlib.compress(body.encode()),
                         json.dumps(metrics) if metrics else None))
        return int(cur.lastrowid)


def update_metrics(snapshot_id: int, metrics: dict, path: Optional[str] = None) -> None:
    with connect(path) as c:
        c.execute("UPDATE volsurface_snapshot SET metrics=? WHERE id=?", (json.dumps(metrics), snapshot_id))


def list_snapshots(symbol: Optional[str] = None, limit: int = 500, path: Optional[str] = None) -> pd.DataFrame:
    q = "SELECT id, symbol, source, asof, spot, n_quotes, metrics FROM volsurface_snapshot"
    args: tuple = ()
    if symbol:
        q += " WHERE symbol=?"
        args = (symbol.upper(),)
    q += " ORDER BY asof DESC LIMIT ?"
    with connect(path) as c:
        rows = c.execute(q, (*args, limit)).fetchall()
    return pd.DataFrame([dict(r) for r in rows], columns=["id", "symbol", "source", "asof", "spot", "n_quotes", "metrics"])


def load_snapshot(snapshot_id: int, path: Optional[str] = None) -> QuoteSet:
    with connect(path) as c:
        r = c.execute("SELECT * FROM volsurface_snapshot WHERE id=?", (snapshot_id,)).fetchone()
    if r is None:
        raise KeyError(f"no snapshot {snapshot_id}")
    body = json.loads(zlib.decompress(r["payload"]).decode())
    quotes = pd.read_json(StringIO(json.dumps(body["quotes"])), orient="split")
    quotes = quotes[QUOTE_COLUMNS] if not quotes.empty else pd.DataFrame(columns=QUOTE_COLUMNS)
    quotes["expiry"] = quotes["expiry"].astype(str).str[:10]
    return QuoteSet(r["symbol"], r["source"], pd.Timestamp(r["asof"]), r["spot"], quotes, body.get("all_expiries", []), body.get("notes", []))


def snapshot_metrics_df(symbol: str, path: Optional[str] = None) -> pd.DataFrame:
    """One row per snapshot that has cached metrics, oldest first."""
    df = list_snapshots(symbol, limit=5000, path=path)
    rows = []
    for _, r in df.iterrows():
        if isinstance(r["metrics"], str) and r["metrics"]:
            m = json.loads(r["metrics"])
            rows.append({"id": r["id"], "asof": pd.Timestamp(r["asof"]), "source": r["source"], **m})
    out = pd.DataFrame(rows)
    return out.sort_values("asof").reset_index(drop=True) if len(out) else out


def delete_snapshot(snapshot_id: int, path: Optional[str] = None) -> None:
    with connect(path) as c:
        c.execute("DELETE FROM volsurface_snapshot WHERE id=?", (snapshot_id,))
