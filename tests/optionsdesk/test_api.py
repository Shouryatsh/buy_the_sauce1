import asyncio
import json
from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from optionsdesk.agents.backtest import BacktestAgent, merge_overrides, simulated_loader
from optionsdesk.agents.narrative import build_narrative
from optionsdesk.agents.wheel import WheelAgent
from optionsdesk.api.app import create_app, event_stream
from optionsdesk.backtest import synthetic_chain
from optionsdesk.events import (
    ChainSnapshot, DataQuality, IVMetrics, Leg, OrderStatus, OrderUpdate, Quote, Signal, SignalAction,
)
from optionsdesk.runtime import build_runtime
from optionsdesk.settings import BacktestSettings, Settings, StraddleSettings

TODAY = date(2025, 6, 2)


def settings(**kw):
    base = dict(tickers=["AAA", "BBB"], data_source="simulated", execution_backend="simulated",
                heartbeat_interval_seconds=0.05, db_path=":memory:")
    base.update(kw)
    return Settings(**base)


def make_rt(**kw):
    return build_runtime(settings(**kw), today=lambda: TODAY)


def feed(rt, symbol="AAA", spot=100.0, sigma=0.30, exps=(35,), rank=50.0, rv=0.30):
    qs = synthetic_chain(symbol, spot, TODAY, sigma, [TODAY + timedelta(d) for d in exps], "CP", (0.7, 1.3), 0.045, [], 1.0)
    snap = ChainSnapshot(symbol=symbol, underlying_price=spot, quotes=qs, quality=DataQuality.DELAYED)
    rt.tracker.update(snap)
    rt.tracker.update(IVMetrics(symbol=symbol, iv_current=sigma, iv_rank=rank, iv_percentile=rank, history_days=100,
                                realized_vol=rv, expected_move_abs=3.0, expected_move_pct=0.03))
    rt.ctx.db.upsert_close(symbol, TODAY, spot)
    return snap


@pytest.fixture
def client():
    rt = make_rt()
    feed(rt, "AAA")
    feed(rt, "BBB", spot=50.0)
    with TestClient(create_app(runtime=rt, start_agents=False)) as c:
        c.rt = rt
        yield c


def test_health_defaults_to_paper_and_simulated(client):
    h = client.get("/health").json()
    assert h["mode"] == "paper" and h["execution_backend"] == "simulated" and h["kill_switch"] is False


def test_overview_empty_book(client):
    o = client.get("/overview").json()
    assert o["portfolio_value"] == 100_000 and o["realized_pnl"] == 0 and o["capital_deployed"] == 0
    assert o["today"] == {"signals": 0, "orders": 0, "realized_pnl": 0} and o["data_quality"] == ["delayed"]


def test_overview_tracks_open_csp_unrealized_and_premium(client):
    rt = client.rt
    w: WheelAgent = rt.agents["wheel"]
    snap = rt.tracker.chains["AAA"]
    q = next(x for x in snap.quotes if x.right == "P" and x.strike == 95 and x.expiry == TODAY + timedelta(35))
    w.positions["AAA"].open_csp(95.0, TODAY + timedelta(35), q.mid + 0.50, 1, TODAY)  # sold 50c above today's mid
    w._persist(w.positions["AAA"])
    o = client.get("/overview").json()
    assert o["open_premium_collected"] == pytest.approx((q.mid + 0.5) * 100)
    assert o["unrealized_pnl"] == pytest.approx(50.0, abs=1e-6)  # premium kept less cost to close
    assert o["capital_deployed"] == pytest.approx(95 * 100 - (q.mid + 0.5) * 100)
    assert o["portfolio_value"] == pytest.approx(100_050)


def test_wheel_view_shows_state_cost_basis_and_suggested_action(client):
    rt = client.rt
    w: WheelAgent = rt.agents["wheel"]
    w.iv["AAA"] = rt.tracker.ivs["AAA"]
    rows = {r["symbol"]: r for r in client.get("/wheel").json()["tickers"]}
    assert rows["AAA"]["state"] == "SEARCHING" and rows["AAA"]["suggested_action"]["kind"] == "open_csp"
    w.positions["AAA"].open_csp(100.0, TODAY - timedelta(1), 2.0, 1, TODAY - timedelta(36))
    w.positions["AAA"].assign_put(TODAY)
    w._persist(w.positions["AAA"])
    r = {x["symbol"]: x for x in client.get("/wheel").json()["tickers"]}["AAA"]
    assert r["state"] == "ASSIGNED" and r["shares"] == 100 and r["cost_basis"] == pytest.approx(98.0)
    assert r["premium_collected"] == pytest.approx(200)


def test_straddle_view_has_gauge_inputs_curve_and_breakevens(client):
    s = client.get("/straddle", params={"mode": "long"}).json()
    row = {r["symbol"]: r for r in s["tickers"]}["AAA"]
    assert row["iv_rank"] == 50.0 and row["candidate"]["entry_ok"] is False  # rank 50 > long max 30, no event
    c = row["candidate"]
    assert c["breakevens"][0] < 100 < c["breakevens"][1] and len(c["payoff"]["spot"]) == len(c["payoff"]["pnl"]) > 10
    assert c["greeks"]["vega"] > 0 and c["max_loss"] > 0 and c["implied_move"] > 0 and c["realized_move"] > 0
    assert client.get("/straddle", params={"mode": "bogus"}).status_code == 422


def test_scanner_filters(client):
    allrows = client.get("/scanner").json()
    assert {r["strategy"] for r in allrows} == {"wheel_put", "wheel_call", "straddle"}
    puts = client.get("/scanner", params={"strategy": "wheel_put", "min_delta": 0.25, "max_delta": 0.30,
                                          "min_dte": 30, "max_dte": 45, "max_spread_pct": 0.10, "min_open_interest": 100}).json()
    assert puts and all(r["strategy"] == "wheel_put" and 0.25 <= r["delta"] <= 0.30 for r in puts)
    assert [r["score"] for r in puts] == sorted((r["score"] for r in puts), reverse=True)
    assert client.get("/scanner", params={"min_iv_rank": 80}).json() == []  # rank 50 filtered out
    assert client.get("/scanner", params={"min_open_interest": 10**9}).json() == []
    assert client.get("/scanner", params={"strategy": "nope"}).status_code == 422


def test_risk_view_utilization_scenarios_heatmap(client):
    rt = client.rt
    w: WheelAgent = rt.agents["wheel"]
    w.positions["AAA"].open_csp(95.0, TODAY + timedelta(35), 2.0, 1, TODAY)
    w._persist(w.positions["AAA"])
    r = client.get("/risk").json()
    assert 0 < r["utilization"]["capital"] < 1 and r["scenario"]["AAA"]["-0.2"] < 0 < r["scenario"]["AAA"]["0.2"]
    assert r["scenario"]["TOTAL"]["0.0"] == pytest.approx(0)
    assert (TODAY + timedelta(35)).isoformat() in r["heatmap"]["AAA"] and r["limits"]["max_net_delta"] == 500
    json.dumps(r)  # fully serializable (no NaN / numpy)


def test_signal_to_order_flow_is_journaled_with_reasoning():
    async def go():
        rt = make_rt(agents=["iv", "wheel", "risk", "execution"])
        snap = feed(rt, "AAA")
        await rt.start()
        await asyncio.sleep(0.1)  # let agents subscribe before publishing
        try:
            sub = rt.ctx.bus.subscribe(OrderUpdate)
            rt.ctx.bus.publish(snap)
            last = None
            for _ in range(2):
                last = await sub.get(timeout=3)
            await asyncio.sleep(0.05)
            return last, rt.ctx.db.journal(), rt.ctx.db.list_orders(), rt.agents["wheel"].positions["AAA"].state.value
        finally:
            await rt.stop()

    last, journal, orders, state = asyncio.run(go())
    assert last.status is OrderStatus.FILLED and state == "CSP_OPEN"
    j = journal[0]
    assert j["verdict"] == "approve" and j["order_status"] == "filled" and "Sell 1x AAA" in j["reason"] and j["paper"] == 1
    assert orders[0]["correlation_id"] == j["correlation_id"]

def test_kill_switch_endpoint_and_token_protection():
    rt = make_rt(api_token="s3cret")
    with TestClient(create_app(runtime=rt, start_agents=False)) as c:
        assert c.post("/kill-switch", json={"engaged": True}).status_code == 401
        assert c.post("/kill-switch", json={"engaged": True}, headers={"X-API-Token": "wrong"}).status_code == 401
        assert c.get("/kill-switch").json()["engaged"] is False  # reads are open on localhost
        r = c.post("/kill-switch", json={"engaged": True, "reason": "test"}, headers={"X-API-Token": "s3cret"})
        assert r.status_code == 200 and c.get("/kill-switch").json() == {"engaged": True, "reason": "test"}
        assert c.post("/orders/x/confirm", json={}).status_code == 401


def test_confirm_endpoint_errors(client):
    assert client.post("/orders/nope/confirm", json={}).status_code == 404
    assert client.post("/orders/nope/reject").status_code == 404


def test_backtest_endpoints_validate_and_queue(client):
    r = client.post("/backtests", json={"strategy": "wheel", "tickers": ["AAA"], "start": "2024-01-01", "end": "2024-06-01"})
    assert r.status_code == 202
    run = client.get(f"/backtests/{r.json()['run_id']}").json()
    assert run["status"] == "running" and run["params"]["tickers"] == ["AAA"]
    assert client.post("/backtests", json={"strategy": "bogus", "tickers": ["AAA"], "start": "2024-01-01", "end": "2024-06-01"}).status_code == 422
    assert client.post("/backtests", json={"strategy": "wheel", "tickers": ["AAA"], "start": "2024-06-01", "end": "2024-01-01"}).status_code == 422
    assert client.post("/backtests", json={"strategy": "wheel", "tickers": ["AAA"], "start": "2024-01-01", "end": "2024-06-01",
                                           "overrides": {"risk": {"max_net_delta": 1}}}).status_code == 422
    assert client.get("/backtests/zzz").status_code == 404 and len(client.get("/backtests").json()) == 1


async def test_backtest_agent_runs_in_background_and_stores_results():
    rt = make_rt()
    bt: BacktestAgent = rt.agents["backtest"]
    run_id = bt.submit("wheel", ["AAA"], date(2024, 3, 1), date(2024, 12, 31))
    task = asyncio.create_task(bt.run())
    for _ in range(200):
        row = rt.ctx.db.get_backtest(run_id)
        if row["status"] != "running":
            break
        await asyncio.sleep(0.05)
    task.cancel()
    assert row["status"] == "done" and row["metrics"]["n_trades"] >= 1 and "sharpe" in row["metrics"]
    assert len(row["equity"]["equity"]) > 100 and len(row["equity"]["drawdown"]) == len(row["equity"]["equity"])


async def test_backtest_failure_is_recorded_not_raised():
    rt = make_rt()
    def boom(*_a):
        raise LookupError("no data")
    bt = BacktestAgent(rt.ctx, loader=boom)
    run_id = bt.submit("wheel", ["AAA"], date(2024, 3, 1), date(2024, 12, 31))
    await bt.run_job(bt.queue.get_nowait())
    row = rt.ctx.db.get_backtest(run_id)
    assert row["status"] == "failed" and "no data" in row["error"]


def test_backtest_override_merging_and_compare_two_strategies():
    s = settings()
    m = merge_overrides(s, {"wheel": {"put_delta_max": 0.2}, "backtest": {"commission_per_contract": 1.0}})
    assert m.wheel.put_delta_max == 0.2 and m.backtest.commission_per_contract == 1.0 and m.wheel.dte_min == s.wheel.dte_min
    with pytest.raises(ValueError):
        merge_overrides(s, {"mode": {"x": 1}})


async def test_sse_stream_delivers_decision_events_as_json():
    rt = make_rt()
    gen = event_stream(rt)
    task = asyncio.create_task(gen.__anext__())
    await asyncio.sleep(0.02)
    rt.ctx.bus.publish(Quote(symbol="AAA", price=1.0))  # chatty: must not be streamed
    sig = Signal(strategy="wheel", symbol="AAA", action=SignalAction.OPEN, reason="r",
                 legs=[Leg(right="P", strike=95.0, expiry=TODAY, quantity=-1, price=1.0)])
    rt.ctx.bus.publish(sig)
    msg = await asyncio.wait_for(task, 1)
    await gen.aclose()
    assert msg["event"] == "Signal" and json.loads(msg["data"])["correlation_id"] == sig.correlation_id


def test_narrative_summarizes_positions_signals_and_risk(client):
    rt = client.rt
    w: WheelAgent = rt.agents["wheel"]
    w.positions["AAA"].open_csp(95.0, TODAY + timedelta(35), 2.0, 1, TODAY)
    w._persist(w.positions["AAA"])
    text = client.get("/narrative").json()["text"]
    assert "AAA" in text and "short 95P" in text and "CSP_OPEN" in text and "No signals" in text
    rt.ctx.db.set_kill_switch(True)
    assert "KILL SWITCH" in build_narrative(rt.ctx.db, TODAY, True)


def test_iv_history_endpoint(client):
    client.rt.ctx.db.upsert_iv("AAA", TODAY, 0.31)
    assert client.get("/iv/aaa").json()[-1] == {"day": TODAY.isoformat(), "iv": 0.31}


def test_first_poll_is_not_lost_at_startup_and_producer_starts_last():
    rt = make_rt(agents=["market_data", "iv", "risk"])
    assert list(rt.agents)[-1] == "market_data"

    async def go():
        await rt.start()
        try:
            for _ in range(100):
                if "AAA" in rt.tracker.ivs:
                    return True
                await asyncio.sleep(0.05)
            return False
        finally:
            await rt.stop()

    assert asyncio.run(go())
