"""
screener.py — Fundamental quality filter using yfinance data.

Applies the thresholds defined in config.py to eliminate stocks that do
not meet baseline quality criteria before dip-detection is run.  This
keeps the buy-the-dip signals focused on fundamentally sound companies.

Criteria checked (all from yfinance ``info`` dict)
---------------------------------------------------
1. Forward P/E     : MIN_PE_RATIO < trailingPE (or forwardPE) ≤ MAX_PE_RATIO
2. Profit margin   : profitMargins ≥ MIN_PROFIT_MARGIN
3. Debt/equity     : debtToEquity / 100 ≤ MAX_DEBT_TO_EQUITY
                     (yfinance reports D/E as a percentage, e.g. 150 = 1.5×)
4. Revenue growth  : revenueGrowth ≥ MIN_REVENUE_GROWTH

Any metric that is unavailable (None / NaN) is treated as a warning
and the stock is retained (benefit of the doubt) so that data gaps
don't silently drop good companies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import yfinance as yf

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
class FundamentalProfile:
    symbol: str
    pe_ratio: float | None = None
    profit_margin: float | None = None
    debt_to_equity: float | None = None   # already normalised (not ×100)
    revenue_growth: float | None = None
    passes: bool = True
    fail_reasons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def screen_fundamental(symbol: str, info: dict | None = None) -> FundamentalProfile:
    """Return a FundamentalProfile indicating whether the stock passes the
    fundamental quality filter.

    Parameters
    ----------
    symbol:
        Ticker symbol.
    info:
        Pre-fetched ``yf.Ticker(symbol).info`` dict.  If None, the function
        fetches it automatically (useful in production; pass it explicitly
        in tests to avoid network calls).
    """
    if info is None:
        try:
            info = yf.Ticker(symbol).info
        except Exception as exc:
            logger.warning("%s: could not fetch info — %s", symbol, exc)
            info = {}

    profile = FundamentalProfile(symbol=symbol)

    # --- P/E ratio ---
    pe = info.get("forwardPE") or info.get("trailingPE")
    if pe is not None:
        profile.pe_ratio = float(pe)
        if profile.pe_ratio <= config.MIN_PE_RATIO:
            profile.passes = False
            profile.fail_reasons.append(
                f"PE={profile.pe_ratio:.1f} ≤ MIN_PE_RATIO={config.MIN_PE_RATIO}"
            )
        elif profile.pe_ratio > config.MAX_PE_RATIO:
            profile.passes = False
            profile.fail_reasons.append(
                f"PE={profile.pe_ratio:.1f} > MAX_PE_RATIO={config.MAX_PE_RATIO}"
            )

    # --- Profit margin ---
    margin = info.get("profitMargins")
    if margin is not None:
        profile.profit_margin = float(margin)
        if profile.profit_margin < config.MIN_PROFIT_MARGIN:
            profile.passes = False
            profile.fail_reasons.append(
                f"profitMargin={profile.profit_margin:.1%} < "
                f"MIN={config.MIN_PROFIT_MARGIN:.1%}"
            )

    # --- Debt / equity ---
    de_raw = info.get("debtToEquity")
    if de_raw is not None:
        # yfinance expresses D/E as a percentage (e.g., 150.0 means 1.50×)
        profile.debt_to_equity = float(de_raw) / 100.0
        if profile.debt_to_equity > config.MAX_DEBT_TO_EQUITY:
            profile.passes = False
            profile.fail_reasons.append(
                f"D/E={profile.debt_to_equity:.2f} > MAX={config.MAX_DEBT_TO_EQUITY}"
            )

    # --- Revenue growth ---
    rev_growth = info.get("revenueGrowth")
    if rev_growth is not None:
        profile.revenue_growth = float(rev_growth)
        if profile.revenue_growth < config.MIN_REVENUE_GROWTH:
            profile.passes = False
            profile.fail_reasons.append(
                f"revenueGrowth={profile.revenue_growth:.1%} < "
                f"MIN={config.MIN_REVENUE_GROWTH:.1%}"
            )

    if profile.passes:
        logger.info("%s: PASS fundamental screen", symbol)
    else:
        logger.info("%s: FAIL fundamental screen — %s", symbol, "; ".join(profile.fail_reasons))

    return profile
