"""
wheelcli/data/cache.py — Persistent disk cache using diskcache (SQLite-backed).

Cache keys are plain strings; values can be any picklable Python object,
including lists of Pydantic models.

Usage
-----
  cache = WheelCache(cache_dir=".wheelcache", ttl=900)
  cache.set("spot:AAPL", 182.50)
  price = cache.get("spot:AAPL")     # → 182.50
  cache.invalidate("spot:AAPL")
  cache.close()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import diskcache

logger = logging.getLogger(__name__)


class WheelCache:
    """
    Thin wrapper around ``diskcache.Cache`` with a configurable TTL and
    convenience methods for key-pattern invalidation.
    """

    def __init__(self, cache_dir: str = ".wheelcache", ttl: int = 900) -> None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self._cache = diskcache.Cache(cache_dir)
        self._ttl = ttl
        logger.debug("Cache opened at %s (TTL=%ds)", cache_dir, ttl)

    # ── Read / write ───────────────────────────────────────────────────────

    def get(self, key: str) -> Optional[Any]:
        """Return cached value or None if missing / expired."""
        value = self._cache.get(key, default=None)
        if value is None:
            logger.debug("Cache MISS: %s", key)
        else:
            logger.debug("Cache HIT:  %s", key)
        return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store *value* under *key* with optional per-entry TTL override."""
        self._cache.set(key, value, expire=ttl if ttl is not None else self._ttl)
        logger.debug("Cache SET:  %s", key)

    def invalidate(self, key: str) -> bool:
        """Delete a single entry. Returns True if it existed."""
        existed = key in self._cache
        self._cache.delete(key)
        return existed

    def invalidate_prefix(self, prefix: str) -> int:
        """Delete all entries whose key starts with *prefix*. Returns count removed."""
        keys = [k for k in self._cache if isinstance(k, str) and k.startswith(prefix)]
        for k in keys:
            self._cache.delete(k)
        logger.debug("Invalidated %d cache entries with prefix '%s'", len(keys), prefix)
        return len(keys)

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def clear_all(self) -> None:
        """Wipe the entire cache."""
        self._cache.clear()
        logger.info("Cache cleared.")

    def close(self) -> None:
        """Release the cache file handle."""
        self._cache.close()

    # ── Context manager ───────────────────────────────────────────────────

    def __enter__(self) -> "WheelCache":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
