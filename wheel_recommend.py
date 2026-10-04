"""
wheel_recommend.py — Ranks option candidates that have cleared the
fundamentals screen, and sizes each suggestion against spare capital,
per-name concentration, and progress toward the monthly income target.

Composite rank is a weighted geometric mean of three 0-1 normalized scores
(quality, richness, capital efficiency) so that a name which is weak on any
one dimension can't be masked by strength on another, while
wheel_config.RECOMMENDATION_WEIGHTS still lets you tilt the ranking.
"""

from __future__ import annotations

from dataclasses import dataclass

import wheel_config as wc
from wheel_fundamentals import WheelFundamentalProfile
from wheel_options import OptionCandidate
from wheel_risk import capital_check


@dataclass
class Recommendation:
    symbol: str
    strike: float
    expiry: str
    dte: int
    delta: float
    premium: float
    annualized_yield: float
    quality_score: float
    richness_ratio: float | None
    capital_efficiency_score: float
    composite_rank: float
    capital_required: float
    contracts_suggested: int
    capital_ok: bool
    capital_notes: list


def _capital_efficiency_score(candidate: OptionCandidate, spare_capital_remaining: float) -> float:
    """0-1 score rewarding high annualized yield without requiring more
    capital than is actually available."""
    if spare_capital_remaining <= 0:
        return 0.0
    yield_score = min(candidate.annualized_yield / 0.30, 1.0)   # 30% annualized -> max score
    capital_fit = min(spare_capital_remaining / candidate.capital_required, 1.0) if candidate.capital_required > 0 else 0.0
    return 0.7 * yield_score + 0.3 * capital_fit


def _composite_rank(quality_norm: float, richness_norm: float, capital_eff_norm: float) -> float:
    w = wc.RECOMMENDATION_WEIGHTS
    # Weighted geometric mean; add a tiny floor so one zero score doesn't
    # collapse the whole product to zero for an otherwise-strong candidate.
    eps = 0.01
    score = (
        (quality_norm + eps) ** w["quality"]
        * (richness_norm + eps) ** w["richness"]
        * (capital_eff_norm + eps) ** w["capital_efficiency"]
    )
    return round(score * 100, 1)


def build_recommendation(
    profile: WheelFundamentalProfile,
    candidate: OptionCandidate,
    spare_capital_remaining: float,
    existing_exposure_by_symbol: dict,
    total_book: float = wc.TOTAL_BOOK,
) -> Recommendation:
    quality_norm = (profile.quality_score or 0) / 100.0
    richness_norm = min((candidate.richness_ratio or 1.0) / wc.RICHNESS_OVERPRICED_RATIO, 1.5) / 1.5
    capital_eff = _capital_efficiency_score(candidate, spare_capital_remaining)

    composite = _composite_rank(quality_norm, richness_norm, capital_eff)

    check = capital_check(
        proposed_capital_required=candidate.capital_required,
        spare_capital_remaining=spare_capital_remaining,
        symbol=candidate.symbol,
        existing_exposure_by_symbol=existing_exposure_by_symbol,
        total_book=total_book,
    )
    contracts_suggested = 1 if check["ok"] else 0

    return Recommendation(
        symbol=candidate.symbol,
        strike=candidate.strike,
        expiry=candidate.expiry,
        dte=candidate.dte,
        delta=candidate.delta,
        premium=candidate.premium,
        annualized_yield=candidate.annualized_yield,
        quality_score=profile.quality_score or 0.0,
        richness_ratio=candidate.richness_ratio,
        capital_efficiency_score=round(capital_eff * 100, 1),
        composite_rank=composite,
        capital_required=candidate.capital_required,
        contracts_suggested=contracts_suggested,
        capital_ok=check["ok"],
        capital_notes=check["reasons"],
    )


def rank_recommendations(
    profiles_by_symbol: dict,                  # symbol -> WheelFundamentalProfile
    candidates_by_symbol: dict,                # symbol -> list[OptionCandidate]
    spare_capital_remaining: float,
    existing_exposure_by_symbol: dict,
    month_income_so_far: float = 0.0,
    total_book: float = wc.TOTAL_BOOK,
) -> list[Recommendation]:
    """Build and sort recommendations across every candidate for every
    fundamentally-passing symbol. Only the single best candidate per symbol
    is kept (highest premium efficiency, already sorted upstream) to avoid
    the list being dominated by near-duplicate strikes on one name."""
    recs: list[Recommendation] = []
    for symbol, profile in profiles_by_symbol.items():
        if not profile.passes:
            continue
        candidates = candidates_by_symbol.get(symbol) or []
        if not candidates:
            continue
        best = candidates[0]
        recs.append(build_recommendation(profile, best, spare_capital_remaining, existing_exposure_by_symbol, total_book=total_book))

    recs.sort(key=lambda r: r.composite_rank, reverse=True)

    income_progress = month_income_so_far / wc.TARGET_MONTHLY_INCOME if wc.TARGET_MONTHLY_INCOME else 0.0
    if income_progress >= 1.0:
        for r in recs:
            r.capital_notes = r.capital_notes + [
                f"note: monthly income target already met "
                f"(${month_income_so_far:,.0f} / ${wc.TARGET_MONTHLY_INCOME:,.0f})"
            ]

    return recs
