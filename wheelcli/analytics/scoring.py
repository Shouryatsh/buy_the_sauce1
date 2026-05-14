"""
wheelcli/analytics/scoring.py — Composite score for a CSP candidate.

Score formula
-------------

  roc              = mid / strike
  annualized_roc   = roc × (365 / DTE)

  spread_pct       = (ask − bid) / mid
  liquidity_factor = clamp(1 − spread_pct / spread_cap, min_lf, 1.0)

  final_score      = annualized_roc
                     × liquidity_factor
                     × (1 + skew_bonus_weight × skew_bonus)
                     × event_multiplier

A higher score is better.  Scores are ranked descending in the scan output.

Public API
----------
score_candidate(contract, spot, skew_data, earnings_provider,
                macro_events, config, universe_entry) → CandidatePut | None

Returns None if the contract fails the delta/sigma filter or has no valid mid.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from ..analytics.events import compute_event_multiplier
from ..analytics.sigma import compute_sigma_distance, passes_filter
from ..analytics.skew import compute_skew_bonus
from ..config import WheelConfig
from ..data.earnings import EarningsProvider
from ..models import CandidatePut, OptionContract, UniverseEntry

logger = logging.getLogger(__name__)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def score_candidate(
    contract: OptionContract,
    spot: float,
    skew_data: dict,  # dict[float, SkewEntry] from analytics.skew.compute_skew
    earnings_provider: EarningsProvider,
    macro_events: list[date],
    config: WheelConfig,
    universe_entry: Optional[UniverseEntry] = None,
) -> Optional[CandidatePut]:
    """
    Score a single put option contract for the wheel strategy.

    Parameters
    ----------
    contract        : raw option data from IBKR
    spot            : current underlying price
    skew_data       : output of compute_skew() for the same expiry
    earnings_provider: EarningsProvider for this run
    macro_events    : list of macro event dates
    config          : WheelConfig instance
    universe_entry  : optional UniverseEntry to attach valuation context

    Returns
    -------
    CandidatePut if the contract passes all filters, otherwise None.
    """
    warnings: list[str] = []

    # ── Basic validity ─────────────────────────────────────────────────────
    if not contract.mid or contract.mid <= 0:
        return None
    if contract.dte <= 0:
        return None

    # ── Sigma distance ─────────────────────────────────────────────────────
    T = contract.dte / 365.0
    iv_for_sigma = contract.iv or 0.0
    sigma_distance = compute_sigma_distance(spot, contract.strike, iv_for_sigma, T)

    if sigma_distance is None and contract.iv is None:
        warnings.append("no_iv_for_sigma")

    # ── OTM filter ─────────────────────────────────────────────────────────
    delta_abs = abs(contract.delta) if contract.delta is not None else None

    if not passes_filter(delta_abs, sigma_distance, config.max_delta, config.sigma_threshold):
        return None  # too close to the money — skip silently

    # ── Annualised return on capital ───────────────────────────────────────
    # roc = premium_per_share / capital_per_share
    #     = mid / strike   (equivalent to (mid × 100) / (strike × 100))
    roc = contract.mid / contract.strike
    annualized_roc = roc * (365.0 / contract.dte)

    # ── Spread / liquidity ─────────────────────────────────────────────────
    spread_pct: Optional[float] = None
    liquidity_factor = 1.0

    if (
        contract.bid is not None
        and contract.ask is not None
        and contract.mid
        and contract.mid > 0
    ):
        raw_spread = contract.ask - contract.bid
        spread_pct = raw_spread / contract.mid
        liquidity_factor = _clamp(
            1.0 - spread_pct / config.spread_cap,
            config.min_liquidity_factor,
            1.0,
        )
    else:
        warnings.append("no_spread_data")

    # ── Skew bonus ─────────────────────────────────────────────────────────
    entry = skew_data.get(contract.strike, {})
    skew_ratio: Optional[float] = entry.get("skew_ratio")
    skew_diff: Optional[float] = entry.get("skew_diff")
    skew_warn: str = entry.get("warning", "")
    if skew_warn:
        warnings.append(f"skew:{skew_warn}")

    skew_bonus = compute_skew_bonus(
        skew_ratio,
        skew_diff,
        config.skew_ratio_threshold,
        config.skew_diff_threshold,
    )

    # ── Event-risk discount ────────────────────────────────────────────────
    today = date.today()
    event_multiplier, event_flag = compute_event_multiplier(
        symbol=contract.symbol,
        expiry=contract.expiry,
        date_opened=today,
        earnings_provider=earnings_provider,
        macro_events=macro_events,
        event_window_days=config.event_window_days,
        earnings_penalty=config.earnings_penalty,
        macro_penalty=config.macro_penalty,
        unknown_earnings_penalty=config.unknown_earnings_penalty,
    )

    # ── Composite score ────────────────────────────────────────────────────
    final_score = (
        annualized_roc
        * liquidity_factor
        * (1.0 + config.skew_bonus_weight * skew_bonus)
        * event_multiplier
    )

    logger.debug(
        "%s %s $%.0f P  σ=%.2f  δ=%.4f  ROC=%.1f%%  liq=%.2f  skew_bonus=%.0f"
        "  event=%.2f  score=%.4f",
        contract.symbol,
        contract.expiry,
        contract.strike,
        sigma_distance if sigma_distance is not None else float("nan"),
        delta_abs if delta_abs is not None else float("nan"),
        annualized_roc * 100,
        liquidity_factor,
        skew_bonus,
        event_multiplier,
        final_score,
    )

    return CandidatePut(
        symbol=contract.symbol,
        expiry=contract.expiry,
        strike=contract.strike,
        delta=delta_abs,
        iv=contract.iv,
        bid=contract.bid,
        ask=contract.ask,
        mid=contract.mid,
        open_interest=contract.open_interest,
        volume=contract.volume,
        dte=contract.dte,
        sigma_distance=sigma_distance,
        skew_ratio=skew_ratio,
        skew_diff=skew_diff,
        skew_bonus=skew_bonus,
        annualized_roc=annualized_roc,
        spread_pct=spread_pct,
        liquidity_factor=liquidity_factor,
        event_flag=event_flag,
        event_multiplier=event_multiplier,
        final_score=final_score,
        ibkr_fair_value=(
            universe_entry.ibkr_fair_value if universe_entry else None
        ),
        morningstar_fair_value=(
            universe_entry.morningstar_fair_value if universe_entry else None
        ),
        warnings=warnings,
    )
