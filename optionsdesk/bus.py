"""In-process typed pub/sub over asyncio queues.

Each subscriber gets its own bounded queue so a slow consumer can never block a
publisher; when full the oldest event is dropped and counted.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import AsyncIterator, Optional, TypeVar

from .events import Event

E = TypeVar("E", bound=Event)


class Subscription:
    def __init__(self, bus: "EventBus", types: tuple[type[Event], ...], maxsize: int) -> None:
        self._bus = bus
        self.types = types
        self.queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=maxsize)
        self.dropped = 0

    def _offer(self, event: Event) -> None:
        if self.queue.full():
            try:
                self.queue.get_nowait()
                self.dropped += 1
            except asyncio.QueueEmpty:
                pass
        self.queue.put_nowait(event)

    async def get(self, timeout: Optional[float] = None) -> Event:
        if timeout is None:
            return await self.queue.get()
        return await asyncio.wait_for(self.queue.get(), timeout)

    def __aiter__(self) -> AsyncIterator[Event]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[Event]:
        while True:
            yield await self.queue.get()

    def close(self) -> None:
        self._bus._unsubscribe(self)


class EventBus:
    def __init__(self, history: int = 500, history_exclude: tuple[type[Event], ...] = ()) -> None:
        self._subs: dict[type[Event], list[Subscription]] = defaultdict(list)
        self._history: list[Event] = []
        self._history_max = history
        self._history_exclude = history_exclude

    def subscribe(self, *types: type[Event], maxsize: int = 1000) -> Subscription:
        if not types:
            types = (Event,)
        sub = Subscription(self, types, maxsize)
        for t in types:
            self._subs[t].append(sub)
        return sub

    def _unsubscribe(self, sub: Subscription) -> None:
        for t in sub.types:
            if sub in self._subs.get(t, []):
                self._subs[t].remove(sub)

    def publish(self, event: Event) -> None:
        if not isinstance(event, self._history_exclude):
            self._history.append(event)
        if len(self._history) > self._history_max:
            del self._history[: len(self._history) - self._history_max]
        delivered: set[int] = set()
        for cls in type(event).__mro__:
            for sub in self._subs.get(cls, []):  # type: ignore[arg-type]
                if id(sub) not in delivered:
                    delivered.add(id(sub))
                    sub._offer(event)

    def recent(self, event_type: type[E] = Event, limit: int = 100) -> list[E]:  # type: ignore[assignment]
        return [e for e in self._history if isinstance(e, event_type)][-limit:]  # type: ignore[return-value]
