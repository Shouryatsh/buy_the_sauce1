"""
wheel_capital.py — Resolves the capital figures used for exposure and sizing
math: either the manual targets in wheel_config.py, or, once connected to
IBKR, figures derived from the live account itself.

Manual targets remain the default so the dashboard is usable before any
IBKR connection exists. Once connected, "live" mode treats:
  invested_capital = gross position value (stock + option market value)
  spare_capital    = total cash, not full margin buying power -- a
                      cash-secured-put strategy shouldn't size new trades
                      against leverage
  total_book       = net liquidation value
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import wheel_config as wc
from wheel_ibkr import AccountSummary


@dataclass
class CapitalTargets:
    invested_capital: float
    spare_capital: float
    total_book: float
    source: str   # "manual" or "live"


def resolve_capital_targets(account_summary: Optional[AccountSummary]) -> CapitalTargets:
    if account_summary is None:
        return CapitalTargets(
            invested_capital=wc.INVESTED_CAPITAL,
            spare_capital=wc.SPARE_CAPITAL,
            total_book=wc.TOTAL_BOOK,
            source="manual",
        )
    return CapitalTargets(
        invested_capital=account_summary.gross_position_value,
        spare_capital=account_summary.total_cash,
        total_book=account_summary.net_liquidation,
        source="live",
    )
