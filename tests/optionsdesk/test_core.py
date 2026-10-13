import io
import json
import logging
from datetime import date

import pytest
from pydantic import ValidationError

from optionsdesk.bus import EventBus
from optionsdesk.db import Database
from optionsdesk.events import Heartbeat, AgentStatus, Quote, Signal, SignalAction, Leg, Event, RiskDecision, Verdict
from optionsdesk.logs import JsonFormatter, correlation
from optionsdesk.settings import Mode, Settings


def test_defaults_are_paper():
    s = Settings()
    assert s.mode is Mode.PAPER
    assert s.ibkr_port == 7497


def test_live_requires_explicit_flag():
    with pytest.raises(ValidationError):
        Settings(mode="live", data_source="ibkr", execution_backend="ibkr")
    with pytest.raises(ValidationError):
        Settings(mode="live", allow_live=True)  # simulated broker/data must not pose as live
    with pytest.raises(ValidationError):
        Settings(mode="live", allow_live=True, data_source="simulated", execution_backend="ibkr")
    s = Settings(mode="live", allow_live=True, data_source="ibkr", execution_backend="ibkr")
    assert s.ibkr_port == 7496


def test_env_nested_override(monkeypatch):
    monkeypatch.setenv("OPTIONSDESK_RISK__MAX_NET_DELTA", "123")
    assert Settings().risk.max_net_delta == 123


def test_yaml_loaded(tmp_path, monkeypatch):
    p = tmp_path / "c.yaml"
    p.write_text("tickers: [ZZZ]\nwheel:\n  roll_dte: 17\n")
    monkeypatch.setenv("OPTIONSDESK_CONFIG", str(p))
    s = Settings()
    assert s.tickers == ["ZZZ"] and s.wheel.roll_dte == 17


async def test_bus_routes_by_type_and_base_class():
    bus = EventBus()
    only_quotes = bus.subscribe(Quote)
    everything = bus.subscribe()
    bus.publish(Quote(symbol="A", price=1.0))
    bus.publish(Heartbeat(agent="x", status=AgentStatus.RUNNING))
    assert (await only_quotes.get(0.1)).symbol == "A"
    assert only_quotes.queue.empty()
    assert everything.queue.qsize() == 2


async def test_bus_drops_oldest_when_full():
    bus = EventBus()
    sub = bus.subscribe(Quote, maxsize=2)
    for i in range(4):
        bus.publish(Quote(symbol="A", price=float(i)))
    assert sub.dropped == 2
    assert [(await sub.get(0.1)).price for _ in range(2)] == [2.0, 3.0]


def test_derive_keeps_correlation_id():
    q = Quote(symbol="A", price=1.0)
    d = q.derive(Heartbeat, agent="a", status=AgentStatus.RUNNING)
    assert d.correlation_id == q.correlation_id and d.id != q.id


def test_db_iv_history_no_lookahead():
    db = Database(":memory:")
    for i, iv in enumerate([0.2, 0.3, 0.4]):
        db.upsert_iv("A", date(2024, 1, 1 + i), iv)
    hist = db.iv_history("A", until=date(2024, 1, 2))
    assert hist == [("2024-01-01", 0.2), ("2024-01-02", 0.3)]


def test_db_kill_switch_and_journal():
    db = Database(":memory:")
    assert not db.kill_switch_engaged()
    db.set_kill_switch(True, "test")
    assert db.kill_switch_engaged()
    sig = Signal(strategy="wheel", symbol="A", action=SignalAction.OPEN,
                 legs=[Leg(right="P", strike=100, expiry=date(2030, 1, 1), quantity=-1)], reason="r")
    db.save_signal(sig)
    db.save_risk_decision(sig.derive(RiskDecision, signal_id=sig.id, verdict=Verdict.APPROVE))
    row = db.journal()[0]
    assert row["signal_id"] == sig.id and row["verdict"] == "approve"


def test_db_agent_health_keeps_last_error():
    db = Database(":memory:")
    db.record_health("a", "error", last_error="boom")
    db.record_health("a", "running", ran=True, restarted=True)
    h = db.agent_health()[0]
    assert h["status"] == "running" and h["last_error"] == "boom" and h["restarts"] == 1 and h["last_run"]


def test_json_log_has_correlation_id():
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setFormatter(JsonFormatter())
    log = logging.getLogger("optionsdesk.test")
    log.handlers, log.propagate = [h], False
    log.setLevel(logging.INFO)
    with correlation("abc123"):
        log.info("hello", extra={"symbol": "AAPL"})
    rec = json.loads(buf.getvalue())
    assert rec["correlation_id"] == "abc123" and rec["symbol"] == "AAPL" and rec["msg"] == "hello"
