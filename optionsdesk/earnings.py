"""Event calendar from earnings_calendar.csv (symbol,earnings_date)."""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path
from typing import Optional


def load_earnings(path: str) -> dict[str, list[date]]:
    p = Path(path)
    out: dict[str, list[date]] = {}
    if not p.exists():
        return out
    with open(p, newline="") as f:
        for row in csv.DictReader(f):
            sym = (row.get("symbol") or "").strip().upper()
            raw = (row.get("earnings_date") or "").strip()
            if not sym or not raw:
                continue
            try:
                out.setdefault(sym, []).append(date.fromisoformat(raw))
            except ValueError:
                continue
    for v in out.values():
        v.sort()
    return out


def next_event(events: dict[str, list[date]], symbol: str, asof: date) -> Optional[date]:
    return next((d for d in events.get(symbol, []) if d >= asof), None)
