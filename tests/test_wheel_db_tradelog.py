"""
tests/test_wheel_db_tradelog.py — the trade log stores each IBKR fill once
(keyed by execId), upgrades rows archived before execIds were kept, and
cleans up the duplicates the old logged_at-based key produced.
"""

from __future__ import annotations

import sqlite3

import wheel_db


def _fill(db, **kw):
    base = dict(symbol="LULU", sec_type="OPT", action="SLD", quantity=1.0, price=1.56, conid=863422070,
                closed_date="2026-10-14", exec_id="0002920a.6ad00e59.01.01", exec_time="2026-10-14T18:02:50+00:00",
                opt_right="P", strike=85.0, expiry="20261120", multiplier=100.0)
    base.update(kw)
    wheel_db.record_fill(db_path=db, **base)


def _rows(db):
    return wheel_db.get_trade_log(db_path=db)


def test_same_execution_is_stored_once(tmp_path):
    db = str(tmp_path / "w.db")
    _fill(db)
    _fill(db)
    _fill(db)
    rows = _rows(db)
    assert len(rows) == 1
    assert (rows[0]["strike"], rows[0]["opt_right"], rows[0]["expiry"]) == (85.0, "P", "20261120")


def test_commission_arrives_later(tmp_path):
    db = str(tmp_path / "w.db")
    _fill(db)
    wheel_db.record_commission("0002920a.6ad00e59.01.01", 1.04, db_path=db)
    assert _rows(db)[0]["commission"] == 1.04


def test_legacy_row_is_upgraded_not_duplicated(tmp_path):
    db = str(tmp_path / "w.db")
    with wheel_db._connect(db) as conn:   # a row archived the old way: no exec_id
        conn.execute("INSERT INTO trade_log (logged_at, symbol, sec_type, action, quantity, price, conid, "
                     "closed_date, source) VALUES ('2026-10-14T14:47:59','LULU','OPT','SLD',1,1.56,863422070,"
                     "'2026-10-14','ibkr')")
    _fill(db)
    rows = _rows(db)
    assert len(rows) == 1 and rows[0]["exec_id"] == "0002920a.6ad00e59.01.01"


def test_migration_removes_old_duplicates(tmp_path):
    db = str(tmp_path / "old.db")
    conn = sqlite3.connect(db)   # the pre-fix schema, with the same fill logged three times
    conn.executescript("""
        CREATE TABLE trade_log (id INTEGER PRIMARY KEY AUTOINCREMENT, logged_at TEXT NOT NULL, symbol TEXT NOT NULL,
            sec_type TEXT, action TEXT, quantity REAL, price REAL, realized_pnl REAL, conid INTEGER,
            closed_date TEXT, notes TEXT, source TEXT DEFAULT 'ibkr',
            UNIQUE(conid, logged_at, action, quantity, price));
        INSERT INTO trade_log (logged_at, symbol, sec_type, action, quantity, price, conid, closed_date, source) VALUES
            ('2026-10-14T14:29:25','NVDA','OPT','SLD',1,0.81,845091401,'2026-10-14','ibkr'),
            ('2026-10-14T14:47:59','NVDA','OPT','SLD',1,0.81,845091401,'2026-10-14','ibkr'),
            ('2026-10-14T14:48:03','NVDA','OPT','SLD',1,0.81,845091401,'2026-10-14','ibkr'),
            ('2026-10-14T15:00:00','AXP','OPT','EXPIRED_WORTHLESS',1,2.86,NULL,'2026-10-16','manual');
    """)
    conn.commit(); conn.close()
    rows = _rows(db)
    assert sorted(r["symbol"] for r in rows) == ["AXP", "NVDA"]   # 3 NVDA copies → 1; manual row untouched
