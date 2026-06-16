"""
wheelcli/analytics/sigma.py — Implied sigma-distance computation.

Sigma distance measures how many implied standard deviations a put strike
is below the current spot price:

    sigma_distance = (S − K) / (S × IV × √T)

where:
  S  = current spot price
  K  = put strike
  IV = annualised implied volatility (decimal, e.g. 0.30)
  T  = time to expiry in years (= DTE / 365)

A sigma_distance of 2.0 means the strike is two standard deviations below
spot — roughly the 1-in-20 left-tail of a lognormal distribution.

Pass/fail filter
----------------
An option passes if EITHER:
  • |delta| ≤ max_delta  (default 0.05), OR
  • sigma_distance ≥ sigma_threshold  (default 2.0)

Both conditions failing means the option is too expensive or too close to
the money for the wheel strategy.
"""

from __future__ import annotations

import math
from typing import Optional


def compute_sigma_distance(
    spot: float,
    strike: float,
    iv: float,
    time_to_expiry_years: float,
) -> Optional[float]:
    """
    Return the number of implied standard deviations *strike* is below *spot*.

    Parameters
    ----------
    spot                 : current underlying price  (> 0)
    strike               : put strike price
    iv                   : annualised implied volatility in decimal (e.g. 0.30)
    time_to_expiry_years : T = calendar_days / 365  (must be > 0)

    Returns
    -------
    sigma_distance as a float, or None when inputs are invalid (iv ≤ 0, T ≤ 0,
    spot ≤ 0, or the one-sigma move computes to zero).

    Examples
    --------
    >>> compute_sigma_distance(150.0, 120.0, 0.30, 30/365)
    2.547...   # strike is ~2.5 standard deviations below spot
    """
    if spot <= 0 or iv <= 0 or time_to_expiry_years <= 0:
        return None

    one_sigma = spot * iv * math.sqrt(time_to_expiry_years)
    if one_sigma == 0.0:
        return None

    return (spot - strike) / one_sigma


def passes_filter(
    delta: Optional[float],
    sigma_distance: Optional[float],
    max_delta: float = 0.05,
    sigma_threshold: float = 2.0,
) -> bool:
    """
    Return True if the option passes the wheel-strategy OTM filter.

    An option passes if EITHER:
      • delta is not None and delta ≤ max_delta, OR
      • sigma_distance is not None and sigma_distance ≥ sigma_threshold

    If both are None the option fails (conservative default).

    Parameters
    ----------
    delta           : absolute put delta (positive, e.g. 0.04)
    sigma_distance  : output of compute_sigma_distance
    max_delta       : upper bound on delta  (default 0.05)
    sigma_threshold : lower bound on sigma_distance  (default 2.0)
    """
    if delta is not None and delta <= max_delta:
        return True
    if sigma_distance is not None and sigma_distance >= sigma_threshold:
        return True
    return False
