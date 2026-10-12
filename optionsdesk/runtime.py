"""Wires settings, bus, DB and agents into a runnable system."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Optional

from .agents.backtest import BacktestAgent, PriceLoader
from .agents.base import BaseAgent, Context
from .agents.execution import Broker, ExecutionAgent, make_broker
from .agents.iv import IVAgent
from .agents.marketdata import MarketDataAgent, MarketDataSource, make_source
from .agents.narrative import NarrativeAgent
from .agents.risk import RiskAgent
from .agents.straddle import StraddleAgent
from .agents.wheel import WheelAgent
from .bus import EventBus
from .db import Database
from .events import ChainSnapshot, IVMetrics, Quote
from .orchestrator import Orchestrator
from .settings import Settings


class StateTracker:
    """Latest chain / IV / quote per symbol, for the API (not kept in the bus history)."""

    def __init__(self, bus: EventBus) -> None:
        self.bus = bus
        self.chains: dict[str, ChainSnapshot] = {}
        self.ivs: dict[str, IVMetrics] = {}
        self.quotes: dict[str, Quote] = {}

    def update(self, ev) -> None:
        if isinstance(ev, ChainSnapshot):
            self.chains[ev.symbol] = ev
        elif isinstance(ev, IVMetrics):
            self.ivs[ev.symbol] = ev
        elif isinstance(ev, Quote):
            self.quotes[ev.symbol] = ev

    def spots(self) -> dict[str, float]:
        out = {s: q.price for s, q in self.quotes.items()}
        out.update({s: c.underlying_price for s, c in self.chains.items()})
        return out

    async def run(self) -> None:
        sub = self.bus.subscribe(ChainSnapshot, IVMetrics, Quote)
        try:
            while True:
                self.update(await sub.get())
        finally:
            sub.close()


@dataclass
class Runtime:
    ctx: Context
    agents: dict[str, BaseAgent]
    orchestrator: Orchestrator
    tracker: StateTracker
    _tracker_task: Optional[asyncio.Task] = field(default=None, repr=False)

    async def start(self) -> None:
        self._tracker_task = asyncio.create_task(self.tracker.run(), name="state-tracker")
        await self.orchestrator.start()

    async def stop(self) -> None:
        await self.orchestrator.stop()
        if self._tracker_task:
            self._tracker_task.cancel()
            await asyncio.gather(self._tracker_task, return_exceptions=True)
        self.ctx.db.close()


def build_runtime(
    settings: Settings,
    source: Optional[MarketDataSource] = None,
    broker: Optional[Broker] = None,
    loader: Optional[PriceLoader] = None,
    today: Callable[[], date] = date.today,
    db: Optional[Database] = None,
) -> Runtime:
    ctx = Context(settings=settings, bus=EventBus(history_exclude=(ChainSnapshot,)),
                  db=db or Database(settings.db_path), today=today)
    wanted = set(settings.agents)
    agents: dict[str, BaseAgent] = {}
    if "iv" in wanted:
        agents["iv"] = IVAgent(ctx)
    if "wheel" in wanted and "wheel" in settings.strategies:
        agents["wheel"] = WheelAgent(ctx)
    if "straddle" in wanted and "straddle" in settings.strategies:
        agents["straddle"] = StraddleAgent(ctx)
    if "risk" in wanted:
        agents["risk"] = RiskAgent(ctx)
    if "execution" in wanted:
        agents["execution"] = ExecutionAgent(ctx, broker or make_broker(settings))
    if "backtest" in wanted:
        agents["backtest"] = BacktestAgent(ctx, loader)
    if "narrative" in wanted:
        agents["narrative"] = NarrativeAgent(ctx)
    # Producer last: orchestrator starts agents in this order, so every consumer has subscribed
    # before the first market-data poll publishes.
    if "market_data" in wanted:
        agents["market_data"] = MarketDataAgent(ctx, source or make_source(settings))
    orch = Orchestrator(ctx, list(agents.values()))
    return Runtime(ctx=ctx, agents=agents, orchestrator=orch, tracker=StateTracker(ctx.bus))
