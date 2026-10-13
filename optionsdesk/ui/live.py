"""Background SSE consumer so the UI can show a live event feed without blocking Streamlit."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from typing import Iterable, Iterator, Optional

import httpx

from . import client


def parse_sse_lines(lines: Iterable[str]) -> Iterator[tuple[str, dict]]:
    """Yield (event_name, data_dict) from raw SSE lines. Ignores comments/pings and bad JSON."""
    event, data = "message", []
    for raw in lines:
        line = raw.rstrip("\r\n")
        if not line:
            if data:
                try:
                    yield event, json.loads("\n".join(data))
                except ValueError:
                    pass
            event, data = "message", []
        elif line.startswith(":"):
            continue
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())


class EventFeed:
    def __init__(self, maxlen: int = 200) -> None:
        self.events: deque[dict] = deque(maxlen=maxlen)
        self.connected = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="optionsdesk-sse", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        while True:
            try:
                with httpx.stream("GET", f"{client.API_URL}/stream", headers=client._headers(), timeout=None) as r:
                    self.connected = r.status_code == 200
                    for name, data in parse_sse_lines(r.iter_lines()):
                        self.events.appendleft({"event": name, "data": data})
            except Exception:  # noqa: BLE001 - reconnect forever
                pass
            self.connected = False
            time.sleep(3)

    def snapshot(self, n: int = 30) -> list[dict]:
        return list(self.events)[:n]
