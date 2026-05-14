"""
wheelcli/analytics/skew.py — Put-skew / tail-pricing analysis.

Purpose
-------
Elevated put skew signals that the market is paying a premium for downside
protection.  When we sell puts into elevated skew we collect richer premium
relative to the theoretical ATM volatility — that's a bonus for the seller.

Definitions
-----------
  ATM IV   : implied volatility of the strike closest to the current spot
             (falls back to the nearest ~25-delta put if ATM has no IV)
  OTM IV   : implied volatility of the candidate put strike (< spot)
  skew_ratio = IV_OTM / IV_ATM        (> 1.10 → bonus)
  skew_diff  = IV_OTM − IV_ATM        (> 0.03 → bonus)
  skew_bonus = 1.0 if EITHER condition met, else 0.0

Public API
----------
  find_atm_iv(contracts, spot)  → (iv_atm, method_string)
  compute_skew(contracts, spot) → dict[strike, SkewEntry]
  compute_skew_bonus(skew_ratio, skew_diff, ...)  → float
"""

from __future__ import annotations

import logging
from typing import Optional, TypedDict

from ..models import OptionContract

logger = logging.getLogger(__name__)


# =============================================================================
# Types
# =============================================================================


class SkewEntry(TypedDict):
    skew_ratio: Optional[float]
    skew_diff: Optional[float]
    atm_iv: Optional[float]
    method: str          # "atm" | "25d_approx" | "unavailable"
    warning: str         # "" | "no_atm_iv" | "no_otm_iv"


# =============================================================================
# ATM IV
# =============================================================================


def find_atm_iv(
    contracts: list[OptionContract],
    spot: float,
) -> tuple[Optional[float], str]:
    """
    Locate the ATM implied volatility for a set of put contracts at one expiry.

    Strategy
    --------
    1. Sort contracts by |strike − spot|.
    2. Return the first contract in that sorted order that has a valid IV.
       Label it "atm".
    3. If no close-to-spot contract has IV, fall back to the contract whose
       delta is closest to 0.25 (the conventional 25-delta put).  Label "25d_approx".
    4. If still nothing, return (None, "unavailable").

    Returns
    -------
    (iv_atm, method) where method is one of "atm" | "25d_approx" | "unavailable".
    """
    if not contracts:
        return None, "unavailable"

    # Strategy 1: nearest-to-spot contracts
    by_distance = sorted(contracts, key=lambda c: abs(c.strike - spot))
    for c in by_distance[:5]:  # check up to 5 nearest strikes
        if c.iv and c.iv > 0:
            return c.iv, "atm"

    # Strategy 2: nearest ~25-delta put
    with_delta = [c for c in contracts if c.delta is not None]
    if with_delta:
        nearest_25d = min(with_delta, key=lambda c: abs((c.delta or 0) - 0.25))
        if nearest_25d.iv and nearest_25d.iv > 0:
            return nearest_25d.iv, "25d_approx"

    return None, "unavailable"


# =============================================================================
# Per-strike skew
# =============================================================================


def compute_skew(
    contracts: list[OptionContract],
    spot: float,
) -> dict[float, SkewEntry]:
    """
    Compute skew_ratio and skew_diff for every OTM put contract in *contracts*.

    ITM puts (strike ≥ spot) are excluded from the output dict — they are
    not sell candidates in the wheel strategy.

    Parameters
    ----------
    contracts : all put contracts for a single expiry date
    spot      : current underlying price

    Returns
    -------
    Dict mapping strike price → SkewEntry.
    """
    iv_atm, method = find_atm_iv(contracts, spot)

    result: dict[float, SkewEntry] = {}

    for c in contracts:
        if c.strike >= spot:
            continue  # skip ATM / ITM puts

        if iv_atm is None:
            result[c.strike] = SkewEntry(
                skew_ratio=None,
                skew_diff=None,
                atm_iv=None,
                method=method,
                warning="no_atm_iv",
            )
            continue

        iv_otm = c.iv
        if not iv_otm or iv_otm <= 0:
            result[c.strike] = SkewEntry(
                skew_ratio=None,
                skew_diff=None,
                atm_iv=iv_atm,
                method=method,
                warning="no_otm_iv",
            )
            continue

        result[c.strike] = SkewEntry(
            skew_ratio=iv_otm / iv_atm,
            skew_diff=iv_otm - iv_atm,
            atm_iv=iv_atm,
            method=method,
            warning="",
        )

    return result


# =============================================================================
# Skew bonus
# =============================================================================


def compute_skew_bonus(
    skew_ratio: Optional[float],
    skew_diff: Optional[float],
    ratio_threshold: float = 1.10,
    diff_threshold: float = 0.03,
) -> float:
    """
    Return 1.0 if put skew is elevated (seller receives a premium), 0.0 otherwise.

    Elevated skew means the market is paying above-ATM volatility for OTM puts,
    which is a positive signal for a CSP seller.

    A bonus (1.0) is awarded if EITHER:
      • skew_ratio ≥ ratio_threshold  (default 1.10), OR
      • skew_diff  ≥ diff_threshold   (default 0.03)

    Both None → 0.0 (conservative: no bonus when data is unavailable).
    """
    if skew_ratio is not None and skew_ratio >= ratio_threshold:
        return 1.0
    if skew_diff is not None and skew_diff >= diff_threshold:
        return 1.0
    return 0.0
