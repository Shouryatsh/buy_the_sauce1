"""
stockscreen/universe.py — manual exclusions are absolute.

A ticker in wheel_exclusions.json must never be scanned, shown, scored or alerted on,
whatever path asks for it (universe list, custom list, CLI, stored results from an
older scan, alerts already raised). This module is the single place that enforces it:

    excluded()          the current exclusion set (read fresh each call; hand edits to the file count)
    drop_excluded()     filter any list of symbols
    apply_exclusions()  purge stored results, cached SEC data, alerts and alert state of excluded tickers
                        (idempotent and cheap — safe to call on every page render)
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Optional

from . import store


def excluded() -> set[str]:
    from wheel_watchlist import load_exclusions
    return {t.upper() for t in load_exclusions()}


def signature() -> str:
    """Changes whenever the exclusion set changes (used in cache keys)."""
    return hashlib.md5(",".join(sorted(excluded())).encode()).hexdigest()[:12]


def drop_excluded(symbols: Iterable[str]) -> tuple[list[str], list[str]]:
    """(kept, dropped), preserving order and removing duplicates."""
    ex = excluded()
    kept, dropped, seen = [], [], set()
    for s in symbols:
        u = s.upper()
        if u in seen:
            continue
        seen.add(u)
        (dropped if u in ex else kept).append(s)
    return kept, dropped


def apply_exclusions(path: Optional[str] = None) -> dict:
    """Remove every trace of excluded tickers from the screener's tables."""
    ex = excluded()
    if not ex:
        return {"results": 0, "cache": 0, "alerts": 0, "states": 0}
    return store.purge_symbols(ex, path)
