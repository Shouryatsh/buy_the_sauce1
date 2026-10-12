"""BaseAgent: one job per agent, communication only through the bus/DB."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Callable, Optional

from ..bus import EventBus, Subscription
from ..db import Database
from ..events import AgentStatus, Event, Heartbeat
from ..logs import correlation, get_logger
from ..settings import Settings


@dataclass
class Context:
    settings: Settings
    bus: EventBus
    db: Database
    today: Callable[[], date] = date.today


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
