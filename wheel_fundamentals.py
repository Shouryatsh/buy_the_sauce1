"""
wheel_fundamentals.py — Fundamentals screener + Composite Quality Score for
the options wheel strategy.

Data source: SEC EDGAR via edgar.py, the same free/no-key source the
buy-the-dip bot in this repo already uses (yfinance was deliberately dropped
from this project — see requirements.txt). Price for the MAX_ASSIGNMENT_PRICE
check and market cap for FCF yield both come from edgar.get_price_history /
edgar.get_fundamentals.

Any name where EDGAR data looks too thin to score honestly (missing FCF,
margin, or ROE) is flagged via `data_reliable=False` rather than silently
scored as if it failed.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import edgar
import wheel_config as wc

logger = logging.getLogger(__name__)


@dataclass
class WheelFundamentalProfile:
    symbol: str
    price: Optional[float] = None
    free_cash_flow: Optional[float] = None
    fcf_yield: Optional[float] = None
    fcf_trend_score: Optional[float] = None      # 0-1, fraction of up years
    net_margin: Optional[float] = None
    net_margin_prior: Optional[float] = None
    net_margin_trend: Optional[float] = None     # current - prior, in absolute pp terms
    return_on_equity: Optional[float] = None
    revenue_growth: Optional[float] = None       # YoY, most recent
    is_elite: bool = False
    quality_score: Optional[float] = None        # 0-100
    passes: bool = True
    fail_reasons: list = field(default_factory=list)
    data_reliable: bool = True
    data_notes: list = field(default_factory=list)


def _safe_float(value) -> Optional[float]:
    try:
        v = float(value)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _fcf_trend_score(fcf_history: list) -> Optional[float]:
    """Fraction of year-over-year periods where FCF increased (0-1).

    *fcf_history* is newest -> oldest, as returned by edgar.py. Returns None
    when there isn't enough history to judge a trend.
    """
    if not fcf_history or len(fcf_history) < 2:
        return None
    pairs = len(fcf_history) - 1
    up_years = sum(1 for i in range(pairs) if fcf_history[i] > fcf_history[i + 1])
    return up_years / pairs


def _net_margin_history(net_income_hist: list, revenue_hist: list) -> list:
    """Zip net income / revenue histories into a margin history (newest -> oldest)."""
    n = min(len(net_income_hist), len(revenue_hist))
    margins = []
    for i in range(n):
        if revenue_hist[i]:
            margins.append(net_income_hist[i] / revenue_hist[i])
    return margins


def screen_wheel_candidate(symbol: str, info: Optional[dict] = None) -> WheelFundamentalProfile:
    """Screen one symbol for the options wheel and compute its Quality Score.

    Parameters
    ----------
    symbol: ticker
    info: pre-fetched edgar.get_fundamentals() dict, or None to fetch it here.
    """
    if info is None:
        info = edgar.get_fundamentals(symbol)

    profile = WheelFundamentalProfile(symbol=symbol)

    # --- Price (for MAX_ASSIGNMENT_PRICE gate) ---
    # edgar.get_fundamentals() already fetched this for market cap — reuse it
    # rather than hitting Stooq/IBKR again for the same symbol.
    profile.price = _safe_float(info.get("_latest_price"))
    if profile.price is None:
        price_hist, _src = edgar.get_price_history(symbol, period_years=1)
        if price_hist is not None and not price_hist.empty:
            profile.price = _safe_float(price_hist["Close"].iloc[-1])
    if profile.price is None:
        profile.data_reliable = False
        profile.data_notes.append("no price data")
    elif profile.price > wc.MAX_ASSIGNMENT_PRICE:
        profile.passes = False
        profile.fail_reasons.append(
            f"price={profile.price:.2f} > MAX_ASSIGNMENT_PRICE={wc.MAX_ASSIGNMENT_PRICE}"
        )

    # --- Free cash flow (hard filter: must be positive) ---
    fcf_history = info.get("_fcf_history", [])
    fcf = _safe_float(info.get("freeCashflow"))
    profile.free_cash_flow = fcf
    if fcf is None:
        profile.passes = False
        profile.data_reliable = False
        profile.data_notes.append("no FCF data")
        profile.fail_reasons.append("FCF unavailable")
    elif fcf <= 0:
        profile.passes = False
        profile.fail_reasons.append(f"FCF={fcf:,.0f} <= 0")

    # --- FCF yield ---
    market_cap = _safe_float(info.get("marketCap"))
    if fcf is not None and market_cap:
        profile.fcf_yield = fcf / market_cap

    # --- FCF trend (score only; not itself a hard filter) ---
    profile.fcf_trend_score = _fcf_trend_score(fcf_history)

    # --- Net margin + trend ---
    net_income_hist = info.get("_net_income_history", [])
    revenue_hist = info.get("_revenue_history", [])
    margin_hist = _net_margin_history(net_income_hist, revenue_hist)
    if margin_hist:
        profile.net_margin = margin_hist[0]
    if len(margin_hist) >= 2:
        profile.net_margin_prior = margin_hist[1]
        profile.net_margin_trend = profile.net_margin - profile.net_margin_prior
    if profile.net_margin is None:
        profile.data_reliable = False
        profile.data_notes.append("no net margin data")

    # --- ROE ---
    profile.return_on_equity = _safe_float(info.get("returnOnEquity"))
    if profile.return_on_equity is None:
        profile.data_reliable = False
        profile.data_notes.append("no ROE data")

    # --- Revenue growth (most recent YoY) ---
    if len(revenue_hist) >= 2 and revenue_hist[1]:
        profile.revenue_growth = (revenue_hist[0] - revenue_hist[1]) / abs(revenue_hist[1])

    # --- Elite bar ---
    profile.is_elite = (
        (profile.return_on_equity or 0) > wc.ELITE_MIN_ROE
        and (profile.net_margin or 0) > wc.ELITE_MIN_NET_MARGIN
        and (profile.revenue_growth or 0) > wc.ELITE_MIN_REVENUE_GROWTH
    )

    # --- FCF yield gate: standard bar, or the lower elite bar for exceptional names ---
    if profile.fcf_yield is not None:
        min_yield = wc.FCF_YIELD_MIN_ELITE if profile.is_elite else wc.FCF_YIELD_MIN
        if profile.fcf_yield <= min_yield:
            profile.passes = False
            bar_name = "FCF_YIELD_MIN_ELITE" if profile.is_elite else "FCF_YIELD_MIN"
            profile.fail_reasons.append(
                f"FCF yield={profile.fcf_yield:.2%} <= {bar_name}={min_yield:.2%}"
            )
    else:
        profile.data_reliable = False
        profile.data_notes.append("no FCF yield (missing market cap or FCF)")

    # --- Composite Quality Score (0-100) ---
    profile.quality_score = _quality_score(profile)

    if profile.passes:
        logger.info("%s: PASS wheel fundamental screen (quality=%.1f)", symbol, profile.quality_score)
    else:
        logger.info("%s: FAIL wheel fundamental screen — %s", symbol, "; ".join(profile.fail_reasons))
    if not profile.data_reliable:
        logger.warning("%s: EDGAR data incomplete — %s", symbol, "; ".join(profile.data_notes))

    return profile


def _quality_score(p: WheelFundamentalProfile) -> float:
    """Weighted composite 0-100 score. Missing inputs score 0 for that
    component (not benefit-of-the-doubt) since this score ranks candidates
    against each other rather than gating pass/fail."""
    w = wc.QUALITY_WEIGHTS

    net_margin_trend_norm = (
        _clamp01(0.5 + 0.5 * (p.net_margin_trend / wc.NET_MARGIN_TREND_NORM_CAP))
        if p.net_margin_trend is not None
        else 0.0
    )
    fcf_trend_norm = p.fcf_trend_score if p.fcf_trend_score is not None else 0.0
    fcf_yield_norm = (
        _clamp01(p.fcf_yield / wc.FCF_YIELD_NORM_CAP) if p.fcf_yield is not None else 0.0
    )
    roe_norm = (
        _clamp01(p.return_on_equity / wc.ROE_NORM_CAP) if p.return_on_equity is not None else 0.0
    )
    revenue_growth_norm = (
        _clamp01(p.revenue_growth / wc.REVENUE_GROWTH_NORM_CAP)
        if p.revenue_growth is not None
        else 0.0
    )

    score = (
        w["net_margin_trend"] * net_margin_trend_norm
        + w["fcf_trend"] * fcf_trend_norm
        + w["fcf_yield"] * fcf_yield_norm
        + w["roe"] * roe_norm
        + w["revenue_growth"] * revenue_growth_norm
    )
    return round(score * 100, 1)


def screen_watchlist(symbols: list[str]) -> list[WheelFundamentalProfile]:
    """Screen every symbol in *symbols*, returning profiles sorted by quality
    score (passing candidates first, highest score first)."""
    profiles = [screen_wheel_candidate(s) for s in symbols]
    profiles.sort(key=lambda p: (p.passes, p.quality_score or 0), reverse=True)
    return profiles
