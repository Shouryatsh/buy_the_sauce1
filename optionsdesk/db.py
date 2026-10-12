"""SQLite persistence (WAL). One shared connection guarded by a lock; agents call it
from asyncio via short synchronous methods (all statements are tiny)."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import date, datetime, timezone
from typing import Any, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS iv_history (
    symbol TEXT NOT NULL, day TEXT NOT NULL, iv REAL NOT NULL,
    PRIMARY KEY (symbol, day)
);
CREATE TABLE IF NOT EXISTS price_history (
    symbol TEXT NOT NULL, day TEXT NOT NULL, close REAL NOT NULL,
    PRIMARY KEY (symbol, day)
);
CREATE TABLE IF NOT EXISTS signals (
    id TEXT PRIMARY KEY, correlation_id TEXT NOT NULL, ts TEXT NOT NULL,
    strategy TEXT NOT NULL, symbol TEXT NOT NULL, action TEXT NOT NULL,
    reason TEXT, score REAL, payload TEXT
);
CREATE TABLE IF NOT EXISTS risk_decisions (
    id TEXT PRIMARY KEY, correlation_id TEXT NOT NULL, ts TEXT NOT NULL,
    signal_id TEXT NOT NULL, verdict TEXT NOT NULL, reasons TEXT, utilization TEXT
);
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY, correlation_id TEXT NOT NULL, signal_id TEXT NOT NULL,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, symbol TEXT, strategy TEXT,
    status TEXT NOT NULL, paper INTEGER NOT NULL, reason TEXT, detail TEXT,
    fill_price REAL, payload TEXT
);
CREATE TABLE IF NOT EXISTS positions (
    id TEXT PRIMARY KEY, correlation_id TEXT, symbol TEXT NOT NULL, strategy TEXT NOT NULL,
    status TEXT NOT NULL, legs TEXT NOT NULL, opened_at TEXT NOT NULL, closed_at TEXT,
    open_premium REAL DEFAULT 0, realized_pnl REAL DEFAULT 0, meta TEXT
);
CREATE TABLE IF NOT EXISTS wheel_state (
    symbol TEXT PRIMARY KEY, state TEXT NOT NULL, shares INTEGER DEFAULT 0,
    cost_basis REAL, premium_collected REAL DEFAULT 0, cycles INTEGER DEFAULT 0,
    payload TEXT, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS backtest_runs (
    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, strategy TEXT NOT NULL,
    params TEXT, status TEXT NOT NULL, metrics TEXT, equity TEXT, error TEXT
);
CREATE TABLE IF NOT EXISTS agent_health (
    agent TEXT PRIMARY KEY, status TEXT NOT NULL, last_heartbeat TEXT,
    last_run TEXT, last_error TEXT, restarts INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dumps(obj: Any) -> str:
    return json.dumps(obj, default=str)


class Database:
    def __init__(self, path: str = "optionsdesk.db") -> None:
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        if path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _exec(self, sql: str, params: tuple = ()) -> None:
        with self._lock:
            self._conn.execute(sql, params)
            self._conn.commit()

    def _all(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def _one(self, sql: str, params: tuple = ()) -> Optional[dict]:
        rows = self._all(sql, params)
        return rows[0] if rows else None

    # ---- kv / kill switch ------------------------------------------------
    def kv_set(self, key: str, value: str) -> None:
        self._exec("INSERT INTO kv(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))

    def kv_get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        row = self._one("SELECT value FROM kv WHERE key=?", (key,))
        return row["value"] if row else default

    def kill_switch_engaged(self) -> bool:
        return self.kv_get("kill_switch", "0") == "1"

    def set_kill_switch(self, engaged: bool, reason: str = "") -> None:
        self.kv_set("kill_switch", "1" if engaged else "0")
        self.kv_set("kill_switch_reason", reason)

    # ---- IV history ------------------------------------------------------
    def upsert_iv(self, symbol: str, day: date, iv: float) -> None:
        self._exec(
            "INSERT INTO iv_history(symbol,day,iv) VALUES(?,?,?) "
            "ON CONFLICT(symbol,day) DO UPDATE SET iv=excluded.iv",
            (symbol, day.isoformat(), iv),
        )

    def iv_history(self, symbol: str, until: Optional[date] = None, limit: int = 260) -> list[tuple[str, float]]:
        """Oldest-first (day, iv); `until` excludes later days (no lookahead)."""
        sql = "SELECT day, iv FROM iv_history WHERE symbol=?"
        params: list[Any] = [symbol]
        if until is not None:
            sql += " AND day<=?"
            params.append(until.isoformat())
        sql += " ORDER BY day DESC LIMIT ?"
        params.append(limit)
        rows = self._all(sql, tuple(params))
        return [(r["day"], r["iv"]) for r in reversed(rows)]

    # ---- price history ---------------------------------------------------
    def upsert_close(self, symbol: str, day: date, close: float) -> None:
        self._exec(
            "INSERT INTO price_history(symbol,day,close) VALUES(?,?,?) "
            "ON CONFLICT(symbol,day) DO UPDATE SET close=excluded.close",
            (symbol, day.isoformat(), close),
        )

    def closes(self, symbol: str, until: Optional[date] = None, limit: int = 260) -> list[tuple[str, float]]:
        sql = "SELECT day, close FROM price_history WHERE symbol=?"
        params: list[Any] = [symbol]
        if until is not None:
            sql += " AND day<=?"
            params.append(until.isoformat())
        sql += " ORDER BY day DESC LIMIT ?"
        params.append(limit)
        rows = self._all(sql, tuple(params))
        return [(r["day"], r["close"]) for r in reversed(rows)]

    # ---- signals / risk / orders ----------------------------------------
    def save_signal(self, e: Any) -> None:
        self._exec(
            "INSERT OR REPLACE INTO signals VALUES(?,?,?,?,?,?,?,?,?)",
            (e.id, e.correlation_id, e.ts.isoformat(), e.strategy, e.symbol, e.action.value,
             e.reason, e.score, e.model_dump_json()),
        )

    def save_risk_decision(self, e: Any) -> None:
        self._exec(
            "INSERT OR REPLACE INTO risk_decisions VALUES(?,?,?,?,?,?,?)",
            (e.id, e.correlation_id, e.ts.isoformat(), e.signal_id, e.verdict.value,
             _dumps(e.reasons), _dumps(e.utilization)),
        )

    def save_order_intent(self, order_id: str, e: Any, status: str, paper: bool) -> None:
        now = _now()
        self._exec(
            "INSERT OR REPLACE INTO orders(order_id,correlation_id,signal_id,created_at,updated_at,symbol,"
            "strategy,status,paper,reason,detail,fill_price,payload) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (order_id, e.correlation_id, e.signal_id, now, now, e.symbol, e.strategy, status,
             int(paper), e.reason, "", None, e.model_dump_json()),
        )

    def update_order(self, order_id: str, status: str, detail: str = "", fill_price: Optional[float] = None) -> None:
        self._exec(
            "UPDATE orders SET status=?, detail=?, fill_price=COALESCE(?,fill_price), updated_at=? WHERE order_id=?",
            (status, detail, fill_price, _now(), order_id),
        )

    def get_order(self, order_id: str) -> Optional[dict]:
        return self._one("SELECT * FROM orders WHERE order_id=?", (order_id,))

    def latest_utilization(self) -> Optional[dict]:
        row = self._one("SELECT utilization FROM risk_decisions ORDER BY ts DESC LIMIT 1")
        return json.loads(row["utilization"]) if row and row["utilization"] else None

    def cancel_stale_confirmations(self) -> None:
        self._exec("UPDATE orders SET status='cancelled', detail='stale confirmation (restart)', updated_at=? "
                   "WHERE status='pending_confirmation'", (_now(),))

    def list_orders(self, limit: int = 200) -> list[dict]:
        return self._all("SELECT order_id,correlation_id,signal_id,created_at,updated_at,symbol,strategy,status,paper,"
                         "reason,detail,fill_price FROM orders ORDER BY created_at DESC LIMIT ?", (limit,))

    def journal(self, limit: int = 200) -> list[dict]:
        """Signals joined with their risk decision and order, newest first."""
        return self._all(
            "SELECT s.id AS signal_id, s.correlation_id, s.ts, s.strategy, s.symbol, s.action, s.reason,"
            " s.score, r.verdict, r.reasons AS risk_reasons, o.order_id, o.status AS order_status,"
            " o.paper, o.fill_price FROM signals s"
            " LEFT JOIN risk_decisions r ON r.signal_id=s.id"
            " LEFT JOIN orders o ON o.signal_id=s.id ORDER BY s.ts DESC LIMIT ?",
            (limit,),
        )

    # ---- positions / wheel state ----------------------------------------
    def save_position(self, pos: dict) -> None:
        self._exec(
            "INSERT OR REPLACE INTO positions(id,correlation_id,symbol,strategy,status,legs,opened_at,"
            "closed_at,open_premium,realized_pnl,meta) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (pos["id"], pos.get("correlation_id"), pos["symbol"], pos["strategy"], pos["status"],
             _dumps(pos["legs"]), pos["opened_at"], pos.get("closed_at"), pos.get("open_premium", 0.0),
             pos.get("realized_pnl", 0.0), _dumps(pos.get("meta", {}))),
        )

    def open_positions(self) -> list[dict]:
        rows = self._all("SELECT * FROM positions WHERE status='open'")
        for r in rows:
            r["legs"] = json.loads(r["legs"])
            r["meta"] = json.loads(r["meta"] or "{}")
        return rows

    def closed_positions(self, limit: int = 200) -> list[dict]:
        rows = self._all("SELECT * FROM positions WHERE status='closed' ORDER BY closed_at DESC LIMIT ?", (limit,))
        for r in rows:
            r["legs"] = json.loads(r["legs"])
            r["meta"] = json.loads(r["meta"] or "{}")
        return rows

    def save_wheel_state(self, symbol: str, state: str, shares: int, cost_basis: Optional[float],
                         premium_collected: float, cycles: int, payload: dict) -> None:
        self._exec(
            "INSERT OR REPLACE INTO wheel_state VALUES(?,?,?,?,?,?,?,?)",
            (symbol, state, shares, cost_basis, premium_collected, cycles, _dumps(payload), _now()),
        )

    def wheel_states(self) -> list[dict]:
        rows = self._all("SELECT * FROM wheel_state ORDER BY symbol")
        for r in rows:
            r["payload"] = json.loads(r["payload"] or "{}")
        return rows

    # ---- backtests -------------------------------------------------------
    def create_backtest(self, run_id: str, strategy: str, params: dict) -> None:
        self._exec(
            "INSERT INTO backtest_runs(id,created_at,strategy,params,status) VALUES(?,?,?,?,'running')",
            (run_id, _now(), strategy, _dumps(params)),
        )

    def finish_backtest(self, run_id: str, metrics: dict, equity: list, error: Optional[str] = None) -> None:
        self._exec(
            "UPDATE backtest_runs SET status=?, metrics=?, equity=?, error=? WHERE id=?",
            ("failed" if error else "done", _dumps(metrics), _dumps(equity), error, run_id),
        )

    def get_backtest(self, run_id: str) -> Optional[dict]:
        row = self._one("SELECT * FROM backtest_runs WHERE id=?", (run_id,))
        if row:
            for k in ("params", "metrics", "equity"):
                row[k] = json.loads(row[k]) if row[k] else None
        return row

    def list_backtests(self, limit: int = 50) -> list[dict]:
        rows = self._all("SELECT id,created_at,strategy,params,status,metrics,error FROM backtest_runs "
                         "ORDER BY created_at DESC LIMIT ?", (limit,))
        for r in rows:
            r["params"] = json.loads(r["params"]) if r["params"] else None
            r["metrics"] = json.loads(r["metrics"]) if r["metrics"] else None
        return rows

    # ---- agent health ----------------------------------------------------
    def record_health(self, agent: str, status: str, last_error: Optional[str] = None,
                      ran: bool = False, restarted: bool = False) -> None:
        now = _now()
        self._exec(
            "INSERT INTO agent_health(agent,status,last_heartbeat,last_run,last_error,restarts) "
            "VALUES(?,?,?,?,?,?) ON CONFLICT(agent) DO UPDATE SET status=excluded.status,"
            " last_heartbeat=excluded.last_heartbeat,"
            " last_run=COALESCE(?, agent_health.last_run),"
            " last_error=COALESCE(excluded.last_error, agent_health.last_error),"
            " restarts=agent_health.restarts + ?",
            (agent, status, now, now if ran else None, last_error, 0, now if ran else None, 1 if restarted else 0),
        )

    def agent_health(self) -> list[dict]:
        return self._all("SELECT * FROM agent_health ORDER BY agent")
