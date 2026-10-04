"""
wheelcli/data/earnings.py — Pluggable earnings-date provider.

Architecture
------------
EarningsProvider (ABC)
  ├── CSVEarningsProvider     — reads earnings_calendar.csv (recommended)
  └── ExternalEarningsProviderStub — override to plug in a real API

Usage
-----
  provider = CSVEarningsProvider("earnings_calendar.csv")
  next_date = provider.next_earnings("AAPL", as_of=date.today())
  if provider.is_tracked("AAPL"):
      ...

earnings_calendar.csv format
----------------------------
  symbol,earnings_date
  AAPL,2026-07-31
  MSFT,2026-07-28

macro_events.csv format
-----------------------
  date,description
  2026-06-11,FOMC Meeting
  2026-07-30,FOMC Meeting
"""

from __future__ import annotations

import csv
import logging
from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Abstract base
# =============================================================================


class EarningsProvider(ABC):
    """Interface for earnings-date data sources."""

    @abstractmethod
    def next_earnings(self, symbol: str, as_of: date) -> Optional[date]:
        """
        Return the earliest earnings date for *symbol* that is >= *as_of*.

        Returns None only when the symbol has no known future earnings dates
        (i.e. the provider *does* track the symbol but all known dates are
        in the past).  If the symbol is not tracked at all, callers should
        check ``is_tracked`` first.
        """
        ...

    def is_tracked(self, symbol: str) -> bool:
        """
        Return True if the provider has *any* earnings data for *symbol*,
        regardless of whether a future date is available.

        The default implementation returns True (assume all symbols tracked).
        CSV-backed implementations override this for accurate "unknown" flags.
        """
        return True


# =============================================================================
# CSV implementation
# =============================================================================


class CSVEarningsProvider(EarningsProvider):
    """
    Load earnings dates from a plain CSV file maintained by the user.

    The CSV must have at minimum two columns:
      ``symbol``         — ticker (case-insensitive)
      ``earnings_date``  — ISO-8601 date string (YYYY-MM-DD)

    Extra columns are silently ignored.
    Duplicate rows for the same symbol are accumulated and sorted.
    """

    def __init__(self, csv_path: str) -> None:
        self._dates: dict[str, list[date]] = {}   # symbol → sorted list of dates
        p = Path(csv_path)
        if not p.exists():
            logger.warning(
                "Earnings calendar not found at '%s'. "
                "All symbols will be treated as 'unknown earnings'.",
                csv_path,
            )
            return

        with open(p, newline="", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f), start=2):
                sym = row.get("symbol", "").strip().upper()
                raw_date = row.get("earnings_date", "").strip()
                if not sym or not raw_date:
                    continue
                try:
                    d = date.fromisoformat(raw_date)
                except ValueError:
                    logger.warning(
                        "%s row %d: invalid date '%s', skipping", csv_path, i, raw_date
                    )
                    continue
                self._dates.setdefault(sym, []).append(d)

        for sym in self._dates:
            self._dates[sym].sort()

        logger.info(
            "Loaded earnings calendar: %d symbols from '%s'",
            len(self._dates),
            csv_path,
        )

    # ── Interface ──────────────────────────────────────────────────────────

    def is_tracked(self, symbol: str) -> bool:
        return symbol.upper() in self._dates

    def next_earnings(self, symbol: str, as_of: date) -> Optional[date]:
        dates = self._dates.get(symbol.upper(), [])
        future = [d for d in dates if d >= as_of]
        return future[0] if future else None


# =============================================================================
# Stub for external providers
# =============================================================================


class ExternalEarningsProviderStub(EarningsProvider):
    """
    Stub interface for plugging in a real external earnings-data source.

    To implement:
      1. Subclass this class.
      2. Override ``next_earnings`` to call your data provider.
      3. Pass API credentials via environment variables — do NOT hardcode.

    Example external sources:
      • Financial Modeling Prep  (env var: FMP_API_KEY)
      • Alpha Vantage            (env var: ALPHA_VANTAGE_KEY)
      • Nasdaq Data Link         (env var: NASDAQ_API_KEY)
      • Seeking Alpha via RapidAPI

    Example skeleton::

        import os, requests
        from datetime import date

        class FMPEarningsProvider(ExternalEarningsProviderStub):
            BASE = "https://financialmodelingprep.com/api/v3"

            def __init__(self):
                self._key = os.environ["FMP_API_KEY"]

            def next_earnings(self, symbol, as_of):
                url = f"{self.BASE}/historical/earning_calendar/{symbol}"
                resp = requests.get(url, params={"apikey": self._key}, timeout=10)
                resp.raise_for_status()
                for item in resp.json():
                    d = date.fromisoformat(item["date"])
                    if d >= as_of:
                        return d
                return None
    """

    def next_earnings(self, symbol: str, as_of: date) -> Optional[date]:
        raise NotImplementedError(
            "ExternalEarningsProviderStub.next_earnings() must be overridden. "
            "See the docstring for implementation guidance."
        )


# =============================================================================
# Macro events loader
# =============================================================================


def load_macro_events(csv_path: str) -> list[date]:
    """
    Load a list of macro event dates from *csv_path*.

    CSV format::
      date,description
      2026-06-11,FOMC Meeting
      2026-07-30,FOMC Meeting

    Returns a sorted list of ``datetime.date`` objects.
    Unknown or malformed dates are skipped with a warning.
    """
    events: list[date] = []
    p = Path(csv_path)
    if not p.exists():
        logger.debug("Macro events file not found at '%s', ignoring.", csv_path)
        return events

    with open(p, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f), start=2):
            raw = row.get("date", "").strip()
            if not raw:
                continue
            try:
                events.append(date.fromisoformat(raw))
            except ValueError:
                logger.warning(
                    "%s row %d: invalid date '%s', skipping", csv_path, i, raw
                )

    events.sort()
    logger.debug("Loaded %d macro events from '%s'", len(events), csv_path)
    return events
