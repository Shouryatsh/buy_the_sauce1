"""
volsurface/arbitrage.py — static-arbitrage diagnostics for an implied-volatility surface.

Butterfly (within one expiry): the risk-neutral density must be non-negative. For an SVI slice that is
Durrleman's g(k) >= 0 (exact), and Lee's wing bounds b(1 +/- rho) <= 2 (necessary for finite moments).
Calendar (across expiries): total variance must not fall with maturity at a fixed log-moneyness k,
w(k, T2) >= w(k, T1) for T2 > T1 (else a calendar spread has negative cost).
Quote level (model free): call prices non-increasing and convex in strike, checked on bid/ask so a
violation is a real arbitrage even after crossing the spread.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .svi import LEE_MAX_SLOPE, SVI

G_TOL = -1e-8
W_TOL = -1e-10


def _ranges(x: np.ndarray, bad: np.ndarray) -> list[tuple[float, float]]:
    """Contiguous runs of True in *bad* as (x_start, x_end)."""
    out, start = [], None
    for i, b in enumerate(bad):
        if b and start is None:
            start = i
        if start is not None and (not b or i == len(bad) - 1):
            end = i if b else i - 1
            out.append((float(x[start]), float(x[end])))
            start = None
    return out


@dataclass
class ButterflyReport:
    ok: bool                                   # g >= 0 everywhere on the grid (data range and extrapolated wings)
    min_g: float
    k_at_min: float
    bad_ranges: list = field(default_factory=list)
    lee_right: float = 0.0
    lee_left: float = 0.0
    lee_ok: bool = True
    k: Optional[np.ndarray] = None
    g: Optional[np.ndarray] = None
    data_range: Optional[tuple] = None         # (k_min, k_max) of the quotes the slice was fitted to
    ok_in_data: bool = True                    # no violation where the market actually quotes
    bad_in_data: list = field(default_factory=list)


def _overlap(ranges: list, lo: float, hi: float) -> list:
    return [(max(a, lo), min(b, hi)) for a, b in ranges if b >= lo and a <= hi]


def butterfly(params: SVI, data_range: Optional[tuple] = None, k_lo: float = -1.5, k_hi: float = 1.5, n: int = 601) -> ButterflyReport:
    """Durrleman's g(k) on a grid. A violation inside *data_range* (where the market quotes) is a real problem; one
    outside it is an artefact of extrapolating the parametric form into strikes nobody quotes."""
    k = np.linspace(k_lo, k_hi, n)
    g = params.g(k)
    finite = np.where(np.isfinite(g), g, -1e9)
    i = int(np.argmin(finite))
    bad = finite < G_TOL
    right, left = params.wing_right, params.wing_left
    ranges = _ranges(k, bad)
    inside = _overlap(ranges, *data_range) if data_range else ranges
    return ButterflyReport(ok=not bad.any(), min_g=float(g[i]) if np.isfinite(g[i]) else float("-inf"), k_at_min=float(k[i]),
                           bad_ranges=ranges, lee_right=right, lee_left=left,
                           lee_ok=right <= LEE_MAX_SLOPE + 1e-9 and left <= LEE_MAX_SLOPE + 1e-9, k=k, g=g,
                           data_range=data_range, ok_in_data=not inside, bad_in_data=inside)


@dataclass
class CalendarReport:
    label_short: str
    label_long: str
    ok: bool                                   # w(k, T2) >= w(k, T1) on the whole grid
    min_dw: float
    k_at_min: float
    bad_ranges: list = field(default_factory=list)
    k: Optional[np.ndarray] = None
    dw: Optional[np.ndarray] = None
    common_range: Optional[tuple] = None       # k range where BOTH expiries have quotes
    ok_in_data: bool = True
    bad_in_data: list = field(default_factory=list)


def calendar(slices: list[tuple[str, float, SVI]], ranges: Optional[dict] = None, k_lo: float = -0.8, k_hi: float = 0.6,
             n: int = 281) -> list[CalendarReport]:
    """Consecutive-pair calendar check of fitted slices, ordered by maturity: [(label, T, params), ...].
    *ranges* maps label -> (k_min, k_max) of that expiry's quotes; violations inside the range common to both
    expiries are real, the rest is extrapolation."""
    ordered = sorted(slices, key=lambda s: s[1])
    k = np.linspace(k_lo, k_hi, n)
    out = []
    for (l1, _t1, p1), (l2, _t2, p2) in zip(ordered[:-1], ordered[1:]):
        dw = p2.w(k) - p1.w(k)
        i = int(np.argmin(dw))
        ranges_bad = _ranges(k, dw < W_TOL)
        common = None
        if ranges and l1 in ranges and l2 in ranges:
            lo, hi = max(ranges[l1][0], ranges[l2][0]), min(ranges[l1][1], ranges[l2][1])
            common = (lo, hi) if lo < hi else None
        inside = _overlap(ranges_bad, *common) if common else ([] if ranges else ranges_bad)
        out.append(CalendarReport(l1, l2, not ranges_bad, float(dw[i]), float(k[i]), ranges_bad, k, dw, common, not inside, inside))
    return out


def atm_variance_monotone(slices: list[tuple[str, float, SVI]]) -> tuple[bool, list[tuple[str, str]]]:
    """ATM total variance w(0, T) must be non-decreasing in T. Returns (ok, offending consecutive pairs)."""
    ordered = sorted(slices, key=lambda s: s[1])
    bad = [(a[0], b[0]) for a, b in zip(ordered[:-1], ordered[1:]) if b[2].w(0.0) < a[2].w(0.0) + W_TOL]
    return not bad, bad


# ---------------------------------------------------------------------------
# model-free checks on the quotes themselves
# ---------------------------------------------------------------------------
# Calls and puts are each tested in their OWN prices, so no discount factor or forward is needed (converting a deep
# in-the-money put to a call multiplies any error in D by the whole F - K gap and would flag noise as arbitrage).
#   calls: price non-increasing and convex in K        puts: price non-decreasing and convex in K
# A flag needs the violation to exceed one price tick, so rounding of quotes is not mistaken for an arbitrage.

@dataclass
class QuoteChecks:
    n_strikes: int
    monotone_violations: list = field(default_factory=list)     # (right, K1, K2, excess $)
    convexity_violations: list = field(default_factory=list)    # (right, K1, K2, K3, excess $)

    @property
    def ok(self) -> bool:
        return not self.monotone_violations and not self.convexity_violations

    @property
    def worst(self) -> float:
        ex = [v[-1] for v in self.monotone_violations] + [v[-1] for v in self.convexity_violations]
        return max(ex) if ex else 0.0


def quote_checks(quotes: pd.DataFrame, min_excess: float = 0.05, min_bid: float = 0.05,
                 k_window: Optional[tuple] = (-0.8, 0.5), otm_only: bool = True) -> QuoteChecks:
    """Strike-monotonicity and butterfly-convexity of each side's prices, on bid/ask, so a flag is an arbitrage you could lock in
    even after crossing the spread. Needs columns strike, right, bid, ask (and k for *k_window* / *otm_only*).

    *otm_only* tests each side only where it is out of the money (calls at or above the forward, puts at or below it): that is where
    the market is live. Deep in-the-money quotes are stale and wide (hundreds of dollars from any real price) and are not
    tradeable arbitrage."""
    q = quotes[(quotes["bid"] >= min_bid) & (quotes["ask"] >= quotes["bid"])]
    if k_window is not None and "k" in q:
        q = q[(q["k"] >= k_window[0]) & (q["k"] <= k_window[1])]
    if otm_only and "k" in q:
        q = q[((q["right"] == "C") & (q["k"] >= -0.005)) | ((q["right"] == "P") & (q["k"] <= 0.005))]
    out = QuoteChecks(n_strikes=int(q["strike"].nunique()))
    for right in ("C", "P"):
        g = q[q["right"] == right].drop_duplicates("strike").sort_values("strike")
        K, bid, ask = g["strike"].to_numpy(dtype=float), g["bid"].to_numpy(dtype=float), g["ask"].to_numpy(dtype=float)
        for i in range(len(K) - 1):
            ex = (bid[i + 1] - ask[i]) if right == "C" else (bid[i] - ask[i + 1])
            if ex > min_excess:
                out.monotone_violations.append((right, float(K[i]), float(K[i + 1]), float(ex)))
        for i in range(len(K) - 2):
            lam = (K[i + 2] - K[i + 1]) / (K[i + 2] - K[i])
            ex = bid[i + 1] - (lam * ask[i] + (1 - lam) * ask[i + 2])
            if ex > min_excess:
                out.convexity_violations.append((right, float(K[i]), float(K[i + 1]), float(K[i + 2]), float(ex)))
    return out
