"""
wheel_risk.py — Exposure and concentration checks for the options wheel.

Phase 1 implements all three exposure measures using the Black-Scholes proxy
delta from wheel_options.py (the same proxy the richness score uses when
IBKR-reported greeks aren't available):

1. Raw exposure          — full assignment liability, no probability weighting
2. Delta-adjusted exposure — probability-weighted by |delta|
3. Stress exposure        — delta recomputed after an underlying/IV shock,
                             showing tail exposure between the other two

Phase 4 upgrades this to use IBKR's live broker-reported greeks wherever an
OPRA subscription makes them available (falling back to this same proxy
otherwise), and adds portfolio-level net delta/theta/vega aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import wheel_config as wc
from wheel_ibkr import PositionRecord
from wheel_options import bs_delta


@dataclass
class ExposureReport:
    raw_exposure: float = 0.0
    delta_adjusted_exposure: float = 0.0
    stress_exposure: float = 0.0
    per_name_concentration: dict = None   # symbol -> pct of total book
    concentration_breaches: list = None   # symbols over MAX_NAME_CONCENTRATION

    def __post_init__(self):
        if self.per_name_concentration is None:
            self.per_name_concentration = {}
        if self.concentration_breaches is None:
            self.concentration_breaches = []


def _short_puts(positions: list[PositionRecord]) -> list[PositionRecord]:
    return [
        p for p in positions
        if p.sec_type == "OPT" and p.right == "P" and p.quantity < 0
    ]


def compute_exposure(
    positions: list[PositionRecord],
    underlying_prices: dict,          # symbol -> current price
    implied_vols: dict,               # symbol -> IV to use for the short puts on that name
    total_book: float = wc.TOTAL_BOOK,
) -> ExposureReport:
    """Compute raw / delta-adjusted / stress exposure across all open short puts.

    *underlying_prices* and *implied_vols* must have an entry for every
    symbol with an open short put, else that position is skipped from the
    delta-adjusted/stress figures (it still counts in raw exposure).
    """
    report = ExposureReport()
    exposure_by_symbol: dict = {}

    for p in _short_puts(positions):
        contracts = abs(p.quantity)
        strike = p.strike or 0.0
        raw = strike * 100 * contracts
        report.raw_exposure += raw
        exposure_by_symbol[p.symbol] = exposure_by_symbol.get(p.symbol, 0.0) + raw

        S = underlying_prices.get(p.symbol)
        sigma = implied_vols.get(p.symbol)
        if S is None or sigma is None or not p.expiry:
            continue

        from datetime import datetime
        try:
            dte = (datetime.strptime(p.expiry, "%Y%m%d").date() - datetime.now().date()).days
        except ValueError:
            continue
        if dte <= 0:
            continue
        T = dte / 365.0

        delta = abs(bs_delta(S, strike, T, wc.RISK_FREE_RATE, sigma, option_type="put"))
        report.delta_adjusted_exposure += delta * strike * 100 * contracts

        stressed_S = S * (1 + wc.STRESS_UNDERLYING_SHOCK_PCT)
        stressed_sigma = sigma * (1 + wc.STRESS_IV_SHOCK_PCT)
        stressed_delta = abs(bs_delta(stressed_S, strike, T, wc.RISK_FREE_RATE, stressed_sigma, option_type="put"))
        report.stress_exposure += stressed_delta * strike * 100 * contracts

    if total_book > 0:
        report.per_name_concentration = {
            sym: value / total_book for sym, value in exposure_by_symbol.items()
        }
        report.concentration_breaches = [
            sym for sym, pct in report.per_name_concentration.items()
            if pct > wc.MAX_NAME_CONCENTRATION
        ]

    return report


def capital_check(
    proposed_capital_required: float,
    spare_capital_remaining: float,
    symbol: str,
    existing_exposure_by_symbol: dict,
    total_book: float = wc.TOTAL_BOOK,
) -> dict:
    """Check a proposed new trade against spare capital and concentration limits.

    Returns a dict with `ok: bool` and human-readable `reasons` for any
    breach, so the dashboard can show why a recommendation was flagged.
    """
    reasons = []
    if proposed_capital_required > spare_capital_remaining:
        reasons.append(
            f"requires ${proposed_capital_required:,.0f} > "
            f"${spare_capital_remaining:,.0f} spare capital remaining"
        )

    existing = existing_exposure_by_symbol.get(symbol, 0.0)
    projected_pct = (existing + proposed_capital_required) / total_book if total_book > 0 else 0.0
    if projected_pct > wc.MAX_NAME_CONCENTRATION:
        reasons.append(
            f"would bring {symbol} to {projected_pct:.1%} of book > "
            f"MAX_NAME_CONCENTRATION={wc.MAX_NAME_CONCENTRATION:.0%}"
        )

    return {"ok": not reasons, "reasons": reasons, "projected_concentration": projected_pct}
