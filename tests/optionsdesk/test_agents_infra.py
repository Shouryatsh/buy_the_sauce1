import asyncio
import math
from datetime import date, timedelta
from types import SimpleNamespace

import pytest

from optionsdesk.agents.base import BaseAgent, Context
from optionsdesk.agents.iv import IVAgent, atm_snapshot, compute_iv_metrics, realized_vol
from optionsdesk.agents.marketdata import (
    MarketDataAgent, RateLimiter, SimulatedMarketData, parse_option_ticker,
)
from optionsdesk.backtest import synthetic_chain
from optionsdesk.bus import EventBus
from optionsdesk.db import Database
from optionsdesk.events import AgentStatus, ChainSnapshot, DataQuality, IVMetrics, Quote
from optionsdesk.orchestrator import Orchestrator
from optionsdesk.settings import IBKRSettings, Settings

TODAY = date(2025, 6, 2)


def make_ctx(**kw) -> Context:
    s = Settings(heartbeat_interval_seconds=0.05, poll_interval_seconds=0.01,
                 ibkr=IBKRSettings(reconnect_backoff_seconds=0.01, quote_cache_ttl_seconds=0), **kw)
    return Context(settings=s, bus=EventBus(), db=Database(":memory:"), today=lambda: TODAY)


def chain(spot=100.0, sigma=0.30, symbol="AAA"):
    exps = [TODAY + timedelta(d) for d in (10, 30, 60)]
    return synthetic_chain(symbol, spot, TODAY, sigma, exps, "CP", (0.8, 1.2), 0.045, [], 1.0)


# ---- orchestrator ----------------------------------------------------------

class Flaky(BaseAgent):
    name = "flaky"

    def __init__(self, ctx, fail_times):
        super().__init__(ctx)
        self.fail_times, self.calls, self.ok = fail_times, 0, asyncio.Event()

    async def run(self):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError(f"boom {self.calls}")
        self.beat(AgentStatus.RUNNING, ran=True)
        self.ok.set()
        await asyncio.sleep(3600)


class Steady(BaseAgent):
    name = "steady"

    async def run(self):
        while True:
            self.beat(AgentStatus.RUNNING, ran=True)
            await asyncio.sleep(0.01)


async def test_orchestrator_restarts_crashing_agent_and_isolates_others():
    ctx = make_ctx()
    flaky, steady = Flaky(ctx, fail_times=2), Steady(ctx)
    orch = Orchestrator(ctx, [flaky, steady], backoff=(0.01, 0.02))
    await orch.start()
    await asyncio.wait_for(flaky.ok.wait(), 2)
    h = {x["agent"]: x for x in orch.health()}
    assert flaky.calls == 3 and h["flaky"]["restarts"] == 2 and "boom 2" in h["flaky"]["last_error"]
    assert h["flaky"]["status"] == "running" and h["steady"]["status"] == "running"
    assert h["steady"]["restarts"] == 0 and not h["steady"]["stale"]
    db = {r["agent"]: r for r in ctx.db.agent_health()}
    assert db["flaky"]["restarts"] == 2 and "boom" in db["flaky"]["last_error"]
    await orch.stop()
    assert all(a.status is AgentStatus.STOPPED for a in (flaky, steady))


async def test_handle_safely_isolates_bad_events():
    ctx = make_ctx()
    agent = Steady(ctx)
    q = Quote(symbol="A", price=1.0)

    def bad(_):
        raise ValueError("nope")

    await agent.handle_safely(bad, q)
    assert agent.status is AgentStatus.DEGRADED and "nope" in agent.last_error
    await agent.handle_safely(lambda e: None, q)
    assert agent.status is AgentStatus.RUNNING and "nope" in agent.last_error  # last error is retained


# ---- market data -----------------------------------------------------------

class FakeSource:
    def __init__(self, bad=(), fail_connects=0):
        self.bad, self.fail_connects, self.connected, self.connects = set(bad), fail_connects, False, 0
        self.history_calls = 0

    async def connect(self):
        self.connects += 1
        if self.connects <= self.fail_connects:
            raise ConnectionRefusedError("TWS down")
        self.connected = True

    async def disconnect(self):
        self.connected = False

    def is_connected(self):
        return self.connected

    async def get_quote(self, symbol):
        if symbol in self.bad:
            raise LookupError(f"no price for {symbol}")
        return 100.0, DataQuality.DELAYED

    async def get_chain(self, symbol, dte_min, dte_max):
        return 100.0, chain(symbol=symbol), DataQuality.DELAYED

    async def get_daily_closes(self, symbol, days):
        self.history_calls += 1
        return [(TODAY - timedelta(i), 100.0 + (i % 5)) for i in range(70, 0, -1)]

    async def get_iv_history(self, symbol, days):
        return [(TODAY - timedelta(i), 0.2 + 0.002 * (i % 40)) for i in range(70, 0, -1)]


async def test_marketdata_isolates_failing_ticker_and_shares_correlation_id():
    ctx = make_ctx(tickers=["AAA", "BAD", "BBB"])
    src = FakeSource(bad={"BAD"})
    src.connected = True
    agent = MarketDataAgent(ctx, src)
    sub = ctx.bus.subscribe(Quote, ChainSnapshot)
    assert await agent.poll_all() == 2
    got = [sub.queue.get_nowait() for _ in range(sub.queue.qsize())]
    assert {e.symbol for e in got} == {"AAA", "BBB"}
    quote, snap = got[0], got[1]
    assert isinstance(quote, Quote) and isinstance(snap, ChainSnapshot)
    assert quote.correlation_id == snap.correlation_id and snap.quality is DataQuality.DELAYED
    assert agent.status is AgentStatus.DEGRADED and "BAD" in agent.last_error
    # histories were seeded for IV rank / realized vol
    assert len(ctx.db.closes("AAA")) >= 60 and len(ctx.db.iv_history("AAA")) >= 60


async def test_marketdata_reconnects_with_backoff():
    ctx = make_ctx(tickers=["AAA"])
    src = FakeSource(fail_connects=2)
    agent = MarketDataAgent(ctx, src)
    sub = ctx.bus.subscribe(ChainSnapshot)
    task = asyncio.create_task(agent.run())
    snap = await sub.get(timeout=2)
    task.cancel()
    assert src.connects == 3 and snap.symbol == "AAA"
    assert "TWS down" in agent.last_error


async def test_rate_limiter_spaces_calls():
    rl = RateLimiter(100.0)
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    for _ in range(4):
        await rl.acquire()
    assert loop.time() - t0 >= 0.025


def _ticker(**kw):
    base = dict(bid=1.0, ask=1.2, last=float("nan"), volume=float("nan"), putOpenInterest=250.0,
                callOpenInterest=float("nan"), marketDataType=3,
                modelGreeks=SimpleNamespace(impliedVol=0.31, delta=-0.27, gamma=0.02, theta=-0.05, vega=0.1))
    base.update(kw)
    return SimpleNamespace(**base)


def test_parse_option_ticker_handles_nan_and_tags_quality():
    c = SimpleNamespace(right="P", strike=95.0)
    q = parse_option_ticker("AAA", c, _ticker(), TODAY)
    assert q.bid == 1.0 and q.mid == pytest.approx(1.1) and q.last is None and q.volume is None
    assert q.open_interest == 250 and q.delta == -0.27 and q.iv == 0.31
    assert q.quality is DataQuality.DELAYED
    assert parse_option_ticker("AAA", c, _ticker(marketDataType=1), TODAY).quality is DataQuality.LIVE
    bad = parse_option_ticker("AAA", c, _ticker(bid=-1.0, ask=float("nan"), modelGreeks=None, putOpenInterest=float("nan")), TODAY)
    assert bad.mid is None and bad.delta is None and bad.open_interest is None and bad.iv is None


def test_call_uses_call_open_interest():
    c = SimpleNamespace(right="C", strike=105.0)
    q = parse_option_ticker("AAA", c, _ticker(callOpenInterest=900.0), TODAY)
    assert q.open_interest == 900


async def test_simulated_source_is_offline_and_tagged_estimated():
    src = SimulatedMarketData(seed=1)
    await src.connect()
    spot, quotes, quality = await src.get_chain("AAA", 7, 45)
    assert quality is DataQuality.ESTIMATED and quotes and spot > 0
    assert all(q.quality is DataQuality.ESTIMATED for q in quotes)
    assert len(await src.get_daily_closes("AAA", 100)) > 50 and len(await src.get_iv_history("AAA", 100)) > 50


# ---- IV ---------------------------------------------------------------------

def snap(spot=100.0, sigma=0.30):
    return ChainSnapshot(symbol="AAA", underlying_price=spot, quotes=chain(spot, sigma), quality=DataQuality.DELAYED)


def test_atm_snapshot_uses_expiry_nearest_30dte():
    a = atm_snapshot(snap(), TODAY)
    assert a["expiry"] == TODAY + timedelta(30) and a["strike"] == 100.0
    assert a["iv"] == pytest.approx(0.30, abs=1e-6)
    assert a["straddle"] == pytest.approx(2 * 0.3 * 100 * math.sqrt(30 / 365) * 0.3989, rel=0.08)


def test_compute_iv_metrics_rank_percentile_and_expected_move():
    hist = [0.2 + 0.002 * i for i in range(100)]  # 0.200 .. 0.398
    closes = [100 * (1.01 if i % 2 else 0.99) ** 1 for i in range(80)]
    m = compute_iv_metrics(snap(), TODAY, hist, closes)
    assert m["iv_rank"] == pytest.approx((0.30 - 0.2) / 0.198 * 100, abs=0.5)
    assert m["iv_percentile"] == pytest.approx(50.0)
    assert m["expected_move_pct"] == pytest.approx(m["expected_move_abs"] / 100.0)
    assert m["realized_vol"] > 0.2 and m["history_days"] == 100


def test_iv_rank_none_without_history_and_realized_vol_needs_data():
    m = compute_iv_metrics(snap(), TODAY, [0.3] * 5, [100.0] * 10)
    assert m["iv_rank"] is None and m["realized_vol"] is None
    assert realized_vol([100 * math.exp(0.01 * ((-1) ** i)) for i in range(61)]) == pytest.approx(0.02 * math.sqrt(252), rel=0.05)


async def test_iv_agent_stores_history_and_publishes_with_same_correlation_id():
    ctx = make_ctx(tickers=["AAA"])
    for i in range(40):
        ctx.db.upsert_iv("AAA", TODAY - timedelta(i + 1), 0.2 + 0.005 * i)
        ctx.db.upsert_close("AAA", TODAY - timedelta(i + 1), 100 + (i % 3))
    agent = IVAgent(ctx)
    sub = ctx.bus.subscribe(IVMetrics)
    s = snap()
    await agent.handle_safely(agent.on_chain, s)
    m = sub.queue.get_nowait()
    assert m.correlation_id == s.correlation_id and m.symbol == "AAA"
    assert m.iv_rank is not None and 0 <= m.iv_rank <= 100 and m.history_days == 41
    assert ctx.db.iv_history("AAA", until=TODAY)[-1][1] == pytest.approx(0.30, abs=1e-6)


# ---- IV gate ----------------------------------------------------------------

async def test_iv_gate_waits_for_matching_iv_then_fires_once():
    from optionsdesk.agents.base import IVGate
    ctx = make_ctx()
    seen = []
    gate = IVGate(Steady(ctx), seen.append, timeout=5.0)
    s = snap()
    gate.on_chain(s)
    assert seen == []
    gate.on_iv(IVMetrics(symbol="AAA", iv_current=0.3, correlation_id="other"))  # unrelated IV: keep waiting
    assert seen == []
    gate.on_iv(s.derive(IVMetrics, symbol="AAA", iv_current=0.3))
    assert seen == [s]
    gate.on_iv(s.derive(IVMetrics, symbol="AAA", iv_current=0.3))
    assert seen == [s]  # released exactly once


async def test_iv_gate_falls_back_after_timeout_and_never_drops_chains():
    from optionsdesk.agents.base import IVGate
    ctx = make_ctx()
    seen = []
    gate = IVGate(Steady(ctx), seen.append, timeout=0.05)
    a, b = snap(), snap()
    gate.on_chain(a)
    await asyncio.sleep(0.12)
    assert seen == [a]  # no IV ever arrived: decided without it
    gate.on_chain(a)
    gate.on_chain(b)  # replaces a pending chain: the older one is processed, not lost
    assert seen == [a, a]
    gate.cancel()


async def test_iv_gate_handler_errors_degrade_agent_not_crash():
    from optionsdesk.agents.base import IVGate
    ctx = make_ctx()
    agent = Steady(ctx)

    def boom(_):
        raise RuntimeError("bad chain")

    gate = IVGate(agent, boom, timeout=0.01)
    gate.on_chain(snap())
    await asyncio.sleep(0.05)
    assert agent.status is AgentStatus.DEGRADED and "bad chain" in agent.last_error
