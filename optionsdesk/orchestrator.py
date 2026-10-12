"""Orchestrator: supervises agents (isolation, restart with backoff) and reports health."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional, Sequence

from .agents.base import BaseAgent, Context
from .events import AgentStatus, Heartbeat
from .logs import get_logger

log = get_logger("orchestrator")


class Orchestrator:
    def __init__(self, ctx: Context, agents: Sequence[BaseAgent], backoff: tuple[float, float] = (1.0, 60.0),
                 stable_after: float = 60.0) -> None:
        self.ctx = ctx
        self.agents = list(agents)
        self._min_backoff, self._max_backoff = backoff
        self._stable_after = stable_after
        self._tasks: dict[str, asyncio.Task] = {}

    async def start(self) -> None:
        for agent in self.agents:
            self._tasks[agent.name] = asyncio.create_task(self._supervise(agent), name=f"agent:{agent.name}")

    async def stop(self) -> None:
        for t in self._tasks.values():
            t.cancel()
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        for agent in self.agents:
            try:
                await agent.close()
            except Exception:  # noqa: BLE001
                log.exception("error closing %s", agent.name)
            agent.beat(AgentStatus.STOPPED)
        self._tasks.clear()

    async def _supervise(self, agent: BaseAgent) -> None:
        delay = self._min_backoff
        loop = asyncio.get_running_loop()
        while True:
            started = loop.time()
            agent.beat(AgentStatus.STARTING)
            try:
                await agent.run()
                agent.beat(AgentStatus.STOPPED)  # a run() that returns is a clean stop
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - isolation boundary
                agent.restarts += 1
                log.error("agent %s crashed: %s", agent.name, exc, exc_info=exc)
                agent.status, agent.last_error = AgentStatus.ERROR, f"{type(exc).__name__}: {exc}"[:500]
                agent.db.record_health(agent.name, AgentStatus.ERROR.value, agent.last_error, restarted=True)
                agent.bus.publish(Heartbeat(agent=agent.name, status=agent.status, last_error=agent.last_error))
            if loop.time() - started >= self._stable_after:
                delay = self._min_backoff
            await asyncio.sleep(delay)
            delay = min(delay * 2, self._max_backoff)

    def health(self) -> list[dict]:
        now = datetime.now(timezone.utc)
        stale_after = 3 * self.ctx.settings.heartbeat_interval_seconds
        out = []
        for a in self.agents:
            age: Optional[float] = (now - a.last_heartbeat).total_seconds() if a.last_heartbeat else None
            out.append({
                "agent": a.name, "status": a.status.value, "last_heartbeat": a.last_heartbeat.isoformat() if a.last_heartbeat else None,
                "last_run": a.last_run.isoformat() if a.last_run else None, "last_error": a.last_error,
                "restarts": a.restarts, "heartbeat_age_seconds": age,
                "stale": age is None or age > stale_after,
            })
        return out
