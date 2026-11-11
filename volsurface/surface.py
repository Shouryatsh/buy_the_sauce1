"""
volsurface/surface.py — raw option quotes -> a cleaned, arbitrage-diagnosable implied-vol surface.

Pipeline per expiry
  1. time to expiry T from the settlement moment (SPX standard monthlies settle on the open of expiry
     day; SPXW / equity / ETF options at the close), calendar ACT/365
  2. the FORWARD F and discount factor D from put-call parity: for each strike  C - P = D (F - K), so a
     weighted line fitted to (K, C - P) near the money gives D = -slope and F = intercept / D. No interest
     rate or dividend assumption is needed, and F is exactly the forward the market is quoting.
  3. implied vols from bid, mid and ask with Black-76; k = ln(K / F); total variance w = sigma^2 T
  4. only OTM options are used (puts below the forward, calls above): they carry the information, and ITM
     quotes are mostly intrinsic value plus noise.

Nothing is invented: a quote that fails a quality rule is flagged invalid (not repaired), a slice with too
few usable quotes is dropped with the reason recorded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from . import bs

NY = "America/New_York"
AM_SETTLED_ROOTS = {"SPX", "NDX", "RUT"}          # standard (third-Friday) index contracts settle on the open
QUOTE_COLUMNS = ["expiry", "root", "strike", "right", "bid", "ask", "volume", "open_interest"]


@dataclass
class SurfaceOptions:
    min_bid: float = 0.05              # a bid below this is no bid (price grid of index options is 0.05)
    max_rel_spread: float = 0.50       # (ask - bid) / mid above this is too wide to be informative
    k_min: float = -0.80               # keep log-moneyness inside this window
    k_max: float = 0.50
    min_days: float = 2.0
    max_days: float = 900.0
    min_points: int = 8                # usable OTM quotes needed for a slice
    min_pairs: int = 4                 # call/put pairs needed for a parity forward
    rate_fallback: float = 0.04        # only used when parity cannot be estimated (then flagged)


def settle_time(expiry: str, root: str) -> pd.Timestamp:
    d = pd.Timestamp(expiry)
    hour, minute = (9, 30) if root.upper() in AM_SETTLED_ROOTS else (16, 0)
    return (d + pd.Timedelta(hours=hour, minutes=minute)).tz_localize(NY)


def year_fraction(expiry: str, root: str, asof: pd.Timestamp) -> float:
    asof = asof.tz_localize(NY) if asof.tzinfo is None else asof.tz_convert(NY)
    return max((settle_time(expiry, root) - asof).total_seconds() / (365.0 * 86400.0), 0.0)


@dataclass
class Slice:
    expiry: str
    root: str
    T: float
    F: float
    D: float
    forward_source: str               # "parity" or "fallback"
    n_pairs: int
    parity_rmse: float
    carry: Optional[float]            # ln(F / S) / T = r - q
    rate: float                       # -ln(D) / T
    points: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    @property
    def days(self) -> float:
        return self.T * 365.0

    @property
    def usable(self) -> pd.DataFrame:
        p = self.points
        return p[p["use"]]

    @property
    def dividend_yield(self) -> Optional[float]:
        return None if self.carry is None else self.rate - self.carry


@dataclass
class Surface:
    symbol: str
    asof: pd.Timestamp
    spot: Optional[float]
    source: str
    slices: list
    dropped: dict = field(default_factory=dict)           # expiry -> reason
    options: SurfaceOptions = field(default_factory=SurfaceOptions)

    @property
    def points(self) -> pd.DataFrame:
        frames = [s.points.assign(expiry=s.expiry, T=s.T, F=s.F) for s in self.slices if not s.points.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def slice(self, expiry: str) -> Slice:
        return next(s for s in self.slices if s.expiry == expiry)


# ---------------------------------------------------------------------------
# forward from put-call parity
# ---------------------------------------------------------------------------

def parity_forward(q: pd.DataFrame, spot: Optional[float], T: float, opts: SurfaceOptions) -> tuple[float, float, str, int, float]:
    """(F, D, source, pairs used, rmse of the parity line in price units)."""
    calls = q[q["right"] == "C"].drop_duplicates("strike").set_index("strike")
    puts = q[q["right"] == "P"].drop_duplicates("strike").set_index("strike")
    common = calls.index.intersection(puts.index)
    c, p = calls.loc[common], puts.loc[common]
    ok = (c["bid"] > 0) & (c["ask"] >= c["bid"]) & (p["bid"] > 0) & (p["ask"] >= p["bid"])
    c, p = c[ok], p[ok]
    if len(c) >= opts.min_pairs:
        y = ((c["bid"] + c["ask"]) / 2 - (p["bid"] + p["ask"]) / 2).to_numpy()
        K = c.index.to_numpy(dtype=float)
        near = np.argsort(np.abs(y))[:14]                       # the strikes where C ~ P are the at-the-money ones
        K, y = K[near], y[near]
        spread = ((c["ask"] - c["bid"]) + (p["ask"] - p["bid"])).to_numpy()[near]
        wt = 1.0 / np.maximum(spread, 0.01)
        for _ in range(2):                                      # weighted line, then drop > 3 MAD outliers and refit
            A = np.vstack([np.ones_like(K), K]).T * np.sqrt(wt)[:, None]
            coef, *_ = np.linalg.lstsq(A, y * np.sqrt(wt), rcond=None)
            res = y - (coef[0] + coef[1] * K)
            mad = np.median(np.abs(res - np.median(res))) * 1.4826
            keep = np.abs(res) <= max(3 * mad, 1e-6)
            if keep.all() or keep.sum() < max(opts.min_pairs, 3):
                break
            K, y, wt = K[keep], y[keep], wt[keep]
        alpha, beta = coef
        D = -beta
        if 0.3 < D < 1.1 and alpha > 0:
            F = alpha / D
            if spot is None or 0.7 < F / spot < 1.4:
                rmse = float(np.sqrt(np.mean((y - (alpha + beta * K)) ** 2)))
                return float(F), float(D), "parity", int(len(K)), rmse
    D = float(np.exp(-opts.rate_fallback * T))
    if spot:
        return float(spot * np.exp(opts.rate_fallback * T)), D, "fallback", 0, float("nan")
    k_mid = float(np.median(q["strike"]))
    return k_mid, D, "fallback", 0, float("nan")


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def _points(q: pd.DataFrame, F: float, D: float, T: float, opts: SurfaceOptions) -> pd.DataFrame:
    p = q.copy()
    p["K"] = p["strike"].astype(float)
    p["cp"] = np.where(p["right"] == "C", 1.0, -1.0)
    p["k"] = np.log(p["K"] / F)
    p["mid"] = (p["bid"] + p["ask"]) / 2
    p["rel_spread"] = np.where(p["mid"] > 0, (p["ask"] - p["bid"]) / p["mid"], np.inf)
    p["otm"] = ((p["K"] < F) & (p["cp"] < 0)) | ((p["K"] >= F) & (p["cp"] > 0))
    p["iv"] = bs.implied_vol(p["mid"], F, p["K"], T, p["cp"], D)
    bid_ok = np.where(p["bid"] >= opts.min_bid, p["bid"], np.nan)
    p["iv_bid"] = bs.implied_vol(bid_ok, F, p["K"], T, p["cp"], D)
    p["iv_ask"] = bs.implied_vol(p["ask"], F, p["K"], T, p["cp"], D)
    p["w"] = p["iv"] ** 2 * T
    p["vega"] = bs.vega(F, p["K"], T, p["iv"].fillna(0.2), D)
    p["use"] = (p["otm"] & (p["bid"] >= opts.min_bid) & (p["ask"] >= p["bid"]) & (p["rel_spread"] <= opts.max_rel_spread)
                & p["iv"].notna() & (p["k"] >= opts.k_min) & (p["k"] <= opts.k_max))
    return p.sort_values("K").reset_index(drop=True)


def build_surface(quotes: pd.DataFrame, spot: Optional[float], asof, symbol: str = "", source: str = "",
                  opts: Optional[SurfaceOptions] = None) -> Surface:
    opts = opts or SurfaceOptions()
    asof = pd.Timestamp(asof)
    slices, dropped = [], {}
    q = quotes.copy()
    q["expiry"] = q["expiry"].astype(str)
    for expiry, g in q.groupby("expiry"):
        roots = sorted(g["root"].unique(), key=lambda r: (r.upper() in AM_SETTLED_ROOTS, r))     # prefer PM-settled
        root = roots[0]
        g = g[g["root"] == root]
        T = year_fraction(expiry, root, asof)
        days = T * 365.0
        if days < opts.min_days or days > opts.max_days:
            dropped[expiry] = f"{days:.1f} days to expiry is outside {opts.min_days:g}–{opts.max_days:g}"
            continue
        F, D, src, npairs, prmse = parity_forward(g, spot, T, opts)
        pts = _points(g, F, D, T, opts)
        if int(pts["use"].sum()) < opts.min_points:
            dropped[expiry] = f"only {int(pts['use'].sum())} usable OTM quotes (need {opts.min_points})"
            continue
        slices.append(Slice(expiry=expiry, root=root, T=T, F=F, D=D, forward_source=src, n_pairs=npairs, parity_rmse=prmse,
                            carry=float(np.log(F / spot) / T) if spot and T > 0 else None,
                            rate=float(-np.log(D) / T) if T > 0 else float("nan"), points=pts))
    slices.sort(key=lambda s: s.T)
    return Surface(symbol=symbol, asof=asof, spot=spot, source=source, slices=slices, dropped=dropped, options=opts)
