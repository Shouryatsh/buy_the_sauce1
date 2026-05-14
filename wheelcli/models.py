"""
wheelcli/models.py — Pydantic data models.

UniverseEntry   : one row from universe.csv (symbol + manual valuations)
OptionContract  : raw data for a single option contract from IBKR
CandidatePut    : fully scored CSP candidate (output of the analytics pipeline)
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, model_validator


# =============================================================================
# Universe
# =============================================================================


class UniverseEntry(BaseModel):
    """One row from universe.csv."""

    symbol: str
    ibkr_fair_value: Optional[float] = None
    morningstar_fair_value: Optional[float] = None

    @model_validator(mode="before")
    @classmethod
    def normalise_symbol(cls, data: dict) -> dict:
        if "symbol" in data:
            data["symbol"] = str(data["symbol"]).strip().upper()
        return data

    @classmethod
    def load_csv(cls, path: str) -> list["UniverseEntry"]:
        """Load all entries from a universe CSV file."""
        p = Path(path)
        if not p.exists():
            return []
        entries: list[UniverseEntry] = []
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                sym = row.get("symbol", "").strip().upper()
                if not sym:
                    continue
                ibkr = row.get("ibkr_fair_value", "").strip()
                ms = row.get("morningstar_fair_value", "").strip()
                entries.append(
                    cls(
                        symbol=sym,
                        ibkr_fair_value=float(ibkr) if ibkr else None,
                        morningstar_fair_value=float(ms) if ms else None,
                    )
                )
        return entries


# =============================================================================
# Raw option contract
# =============================================================================


class OptionContract(BaseModel):
    """
    Minimal market-data snapshot for one option contract, as returned by IBKR.

    ``delta`` and ``iv`` come from the model greeks (IBKR theoretical values).
    Both may be None if IBKR hasn't computed them yet.
    """

    symbol: str
    expiry: date
    strike: float
    right: str = "P"  # "P" = put  |  "C" = call

    # Market data
    delta: Optional[float] = None  # absolute value (sign stripped)
    iv: Optional[float] = None  # annualised, decimal (e.g. 0.30 = 30 %)
    bid: Optional[float] = None
    ask: Optional[float] = None
    mid: Optional[float] = None
    open_interest: Optional[int] = None
    volume: Optional[int] = None
    dte: int = 0  # calendar days to expiry

    @property
    def spread(self) -> Optional[float]:
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None

    @property
    def spread_pct(self) -> Optional[float]:
        if self.mid and self.mid > 0 and self.spread is not None:
            return self.spread / self.mid
        return None


# =============================================================================
# Scored candidate
# =============================================================================


class CandidatePut(BaseModel):
    """
    Fully-scored CSP candidate after running all analytics modules.

    Output columns match the spec:
      symbol, expiry, strike, delta, iv, sigma_distance, bid, ask, mid,
      annualized_roc, spread_pct, skew_ratio, event_flag, final_score.
    """

    # ── Core option data ───────────────────────────────────────────────────
    symbol: str
    expiry: date
    strike: float
    delta: Optional[float] = None
    iv: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    mid: Optional[float] = None
    open_interest: Optional[int] = None
    volume: Optional[int] = None
    dte: int

    # ── Analytics ─────────────────────────────────────────────────────────
    sigma_distance: Optional[float] = None

    # Skew
    skew_ratio: Optional[float] = None
    skew_diff: Optional[float] = None
    skew_bonus: float = 0.0  # 0.0 or 1.0

    # Score components
    annualized_roc: Optional[float] = None
    spread_pct: Optional[float] = None
    liquidity_factor: float = 1.0

    # Event risk
    event_flag: str = ""          # e.g. "EARNINGS", "MACRO", "UNKNOWN_EARNINGS"
    event_multiplier: float = 1.0

    # ── Final composite score ─────────────────────────────────────────────
    final_score: float = 0.0

    # ── Context from universe CSV ─────────────────────────────────────────
    ibkr_fair_value: Optional[float] = None
    morningstar_fair_value: Optional[float] = None

    # ── Diagnostic warnings ───────────────────────────────────────────────
    warnings: list[str] = Field(default_factory=list)
