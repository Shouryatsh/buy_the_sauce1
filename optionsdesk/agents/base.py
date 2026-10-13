"""BaseAgent: one job per agent, communication only through the bus/DB."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Callable, Optional

from ..bus import EventBus, Subscription
from ..db import Database
from ..events import AgentStatus, ChainSnapshot, Event, Heartbeat, IVMetrics
from ..logs import correlation, get_logger
from ..settings import Settings


@dataclass
class Context:
    settings: Settings
    bus: EventBus
    db: Database
    today: Callable[[], date] = date.today


class SignalCooldown:
    """Suppresses re-emitting an identical signal for a while after it was blocked or rejected."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self._until: dict[tuple, float] = {}

    @staticmethod
    def key(sig: Any) -> tuple:
        return (sig.strategy, sig.symbol, sig.action.value,
                tuple((l.right, l.strike, l.expiry, l.quantity) for l in sig.legs))

    def suppress(self, key: tuple) -> None:
        self._until[key] = time.monotonic() + self.seconds

    def active(self, key: tuple) -> bool:
        until = self._until.get(key)
        if until is None:
            return False
        if time.monotonic() >= until:
            del self._until[key]
            return False
        return True

    def reset(self) -> None:
        self._until.clear()


class IVGate:
    """Hands a chain to `handler` once the IV metrics derived from it arrive (same correlation id),
    or after `timeout` seconds if they never do, so decisions use fresh IV rank when available."""

    def __init__(self, agent: "BaseAgent", handler: Callable[[ChainSnapshot], Any], timeout: float) -> None:
        self.agent, self.handler, self.timeout = agent, handler, timeout
        self._pending: dict[str, ChainSnapshot] = {}
        self._timers: dict[str, asyncio.TimerHandle] = {}

    def on_chain(self, snap: ChainSnapshot) -> None:
        old = self._pending.pop(snap.symbol, None)
        timer = self._timers.pop(snap.symbol, None)
        if timer:
            timer.cancel()
        if old is not None:  # previous chain never got its IV: don't drop it silently
            self._run(old)
        self._pending[snap.symbol] = snap
        self._timers[snap.symbol] = asyncio.get_running_loop().call_later(self.timeout, self._expire, snap.symbol, snap.id)

    def on_iv(self, iv: IVMetrics) -> None:
        snap = self._pending.get(iv.symbol)
        if snap is not None and snap.correlation_id == iv.correlation_id:
            self._release(iv.symbol)

    def _expire(self, symbol: str, snap_id: str) -> None:
        snap = self._pending.get(symbol)
        if snap is not None and snap.id == snap_id:
            self._release(symbol)

    def _release(self, symbol: str) -> None:
        snap = self._pending.pop(symbol)
        timer = self._timers.pop(symbol, None)
        if timer:
            timer.cancel()
        self._run(snap)

    def _run(self, snap: ChainSnapshot) -> None:
        with correlation(snap.correlation_id):
            try:
                self.handler(snap)
                self.agent.beat(AgentStatus.RUNNING, ran=True)
            except Exception as exc:  # noqa: BLE001 - isolation boundary
                self.agent.degrade(exc, f"handling {type(snap).__name__}")

    def cancel(self) -> None:
        for t in self._timers.values():
            t.cancel()
        self._timers.clear()
        self._pending.clear()


class BaseAgent:
    name = "agent"

    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx
        self.settings = ctx.settings
        self.bus = ctx.bus
        self.db = ctx.db
        self.log = get_logger(self.name)
        self.status = AgentStatus.STARTING
        self.last_error: Optional[str] = None
        self.last_heartbeat: Optional[datetime] = None
        self.last_run: Optional[datetime] = None
        self.restarts = 0

    async def run(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    async def close(self) -> None:
        """Release external resources on shutdown."""

    def beat(self, status: Optional[AgentStatus] = None, error: Optional[str] = None, ran: bool = False) -> None:
        if status is not None:
            self.status = status
        if error is not None:
            self.last_error = error
        now = datetime.now(timezone.utc)
        self.last_heartbeat = now
        if ran:
            self.last_run = now
        self.db.record_health(self.name, self.status.value, error, ran=ran)
        self.bus.publish(Heartbeat(agent=self.name, status=self.status, last_error=self.last_error))

    def degrade(self, exc: BaseException, context: str = "") -> None:
        msg = f"{context}: {exc}" if context else str(exc)
        self.log.warning("degraded: %s", msg, exc_info=exc)
        self.beat(AgentStatus.DEGRADED, error=msg[:500])

    async def next_event(self, sub: Subscription) -> Optional[Event]:
        """Wait for an event; on idle timeout emit a heartbeat and return None."""
        try:
            return await sub.get(timeout=self.settings.heartbeat_interval_seconds)
        except asyncio.TimeoutError:
            self.beat(AgentStatus.RUNNING if self.status is not AgentStatus.ERROR else None)
            return None

    async def handle_safely(self, handler: Callable, event: Event) -> None:
        """Run an event handler under the event's correlation id; one bad event never kills the agent."""
        with correlation(event.correlation_id):
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
                self.beat(AgentStatus.RUNNING, ran=True)
            except Exception as exc:  # noqa: BLE001 - isolation boundary
                self.degrade(exc, f"handling {type(event).__name__}")
