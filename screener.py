"""
screener.py — Fundamental quality filter using SEC EDGAR data.

Applies the thresholds defined in config.py to eliminate stocks that do
not meet baseline quality criteria before dip-detection is run.  This
keeps the buy-the-dip signals focused on fundamentally sound companies.

Data source: SEC EDGAR XBRL API via edgar.py (no API key, free, official).
Price history for market-cap / P/E: Stooq via pandas_datareader.

Criteria checked
----------------
1. Trailing P/E         : MIN_PE_RATIO < trailingPE ≤ MAX_PE_RATIO
2. Profit margin        : profitMargins ≥ MIN_PROFIT_MARGIN
3. Debt / equity        : debtToEquity / 100 ≤ MAX_DEBT_TO_EQUITY
4. Free cash flow       : freeCashflow > MIN_FREE_CASH_FLOW  (must be positive)
5. FCF yield            : freeCashflow / marketCap ≥ MIN_FCF_YIELD  (≥ 2%)
6. Increasing FCF       : annual FCF trend increasing over FCF_GROWTH_LOOKBACK_YEARS
7. Return on equity     : returnOnEquity ≥ MIN_RETURN_ON_EQUITY  (≥ 10%)
8. Capex-to-FCF ratio   : |capitalExpenditures| / freeCashflow ≤ MAX_CAPEX_TO_FCF

Any metric that is unavailable (None) is treated with benefit of the doubt
so that EDGAR data gaps don't silently drop good companies.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import config
import edgar

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
class FundamentalProfile:
    symbol: str
    pe_ratio: Optional[float] = None
    profit_margin: Optional[float] = None
    debt_to_equity: Optional[float] = None   # normalised (not ×100)
    free_cash_flow: Optional[float] = None   # absolute FCF in USD
    fcf_yield: Optional[float] = None        # FCF / market cap
    fcf_increasing: Optional[bool] = None    # True if FCF trended up over lookback
    return_on_equity: Optional[float] = None
    capex_to_fcf: Optional[float] = None
    # ── New quality metrics ──────────────────────────────────────────────────
    roic: Optional[float] = None             # Return on Invested Capital (current yr)
    roic_prior: Optional[float] = None       # ROIC prior year
    fcf_margin: Optional[float] = None       # FCF / Revenue (current yr)
    fcf_margin_prior: Optional[float] = None # FCF / Revenue (prior yr)
    cash_conversion: Optional[float] = None  # FCF / Net Income (current yr)
    cash_conversion_prior: Optional[float] = None
    accruals_ratio: Optional[float] = None   # (NI − FCF) / avg assets — lower = better
    accruals_ratio_prior: Optional[float] = None
    receivables_growth: Optional[float] = None  # YoY AR growth
    revenue_growth: Optional[float] = None      # YoY revenue growth
    revenue_current: Optional[float] = None     # raw revenue this year
    revenue_prior: Optional[float] = None       # raw revenue last year
    passes: bool = True
    fail_reasons: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value) -> Optional[float]:
    """Return float or None, silently swallowing conversion errors."""
    try:
        v = float(value)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def _check_fcf_increasing(fcf_history: list, lookback: int, min_growth_years: int) -> Optional[bool]:
    """Return True if FCF grew in at least *min_growth_years* of the last *lookback* periods.

    This is softer than requiring every single year to grow — it tolerates
    one-off capex spikes or a single bad year while still requiring an
    overall upward trend.

    *fcf_history* is ordered newest → oldest (as returned by edgar.py).
    Returns None when there is insufficient history (benefit of the doubt).
    """
    if not fcf_history or len(fcf_history) < 2:
        return None
    values = fcf_history[: lookback + 1]   # up to lookback+1 points → lookback pairs
    if len(values) < 2:
        return None
    growth_years = sum(1 for i in range(len(values) - 1) if values[i] > values[i + 1])
    available_pairs = len(values) - 1
    required = min(min_growth_years, available_pairs)  # don't require more than available
    return growth_years >= required


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def screen_fundamental(
    symbol: str,
    info: Optional[dict] = None,
    # _ticker kept for test compatibility (ignored in production)
    _ticker=None,
) -> FundamentalProfile:
    """Return a FundamentalProfile indicating whether the stock passes the
    fundamental quality filter.

    Parameters
    ----------
    symbol:
        Ticker symbol.
    info:
        Pre-fetched fundamentals dict (from edgar.get_fundamentals or a test
        mock).  If None, fetched automatically from SEC EDGAR.
    """
    if info is None:
        info = edgar.get_fundamentals(symbol)

    # Pre-populate the ML predictor's fundamental cache so predict() won't
    # make a second EDGAR network call for the same symbol.
    try:
        import ml_predictor
        ml_predictor.prime_fundamental_cache(symbol, info)
    except Exception:
        pass  # ml_predictor may not be imported yet — safe to ignore

    profile = FundamentalProfile(symbol=symbol)

    # --- 1. P/E ratio ---
    # Guard against 0.0 being falsy — use explicit None check
    fwd_pe = _safe_float(info.get("forwardPE"))
    trail_pe = _safe_float(info.get("trailingPE"))
    pe = fwd_pe if fwd_pe is not None else trail_pe
    if pe is not None:
        profile.pe_ratio = pe
        if pe <= config.MIN_PE_RATIO:
            profile.passes = False
            profile.fail_reasons.append(
                f"PE={pe:.1f} <= MIN_PE_RATIO={config.MIN_PE_RATIO}"
            )
        elif pe > config.MAX_PE_RATIO:
            profile.passes = False
            profile.fail_reasons.append(
                f"PE={pe:.1f} > MAX_PE_RATIO={config.MAX_PE_RATIO}"
            )

    # --- 2. Profit margin ---
    margin = _safe_float(info.get("profitMargins"))
    if margin is not None:
        profile.profit_margin = margin
        if margin < config.MIN_PROFIT_MARGIN:
            profile.passes = False
            profile.fail_reasons.append(
                f"profitMargin={margin:.1%} < MIN={config.MIN_PROFIT_MARGIN:.1%}"
            )

    # --- 3. Debt / equity (sector-aware) ---
    de_raw = _safe_float(info.get("debtToEquity"))
    if de_raw is not None:
        profile.debt_to_equity = de_raw / 100.0
        # Use a higher cap for financial companies (banks, insurers) — their leverage
        # is structural and regulated, not a sign of distress.
        sic = str(info.get("_sic", "") or "")
        is_financial = any(sic.startswith(p) for p in config.FINANCIAL_SIC_PREFIXES)
        de_cap = config.MAX_DEBT_TO_EQUITY_FINANCIAL if is_financial else config.MAX_DEBT_TO_EQUITY
        if profile.debt_to_equity > de_cap:
            profile.passes = False
            profile.fail_reasons.append(
                f"D/E={profile.debt_to_equity:.2f} > MAX={de_cap} "
                f"({'financial' if is_financial else 'non-financial'})"
            )

    # --- 4. Free cash flow — must be positive ---
    fcf = _safe_float(info.get("freeCashflow"))
    if fcf is not None:
        profile.free_cash_flow = fcf
        if fcf <= config.MIN_FREE_CASH_FLOW:
            profile.passes = False
            profile.fail_reasons.append(
                f"FCF={fcf:,.0f} <= MIN={config.MIN_FREE_CASH_FLOW:,.0f} (negative/zero)"
            )

    # --- 5. FCF yield >= 2% ---
    market_cap = _safe_float(info.get("marketCap"))
    if fcf is not None and market_cap and market_cap > 0:
        profile.fcf_yield = fcf / market_cap
        if profile.fcf_yield < config.MIN_FCF_YIELD:
            profile.passes = False
            profile.fail_reasons.append(
                f"FCF yield={profile.fcf_yield:.2%} < MIN={config.MIN_FCF_YIELD:.2%}"
            )
    elif fcf is not None:
        logger.debug("%s: marketCap unavailable — skipping FCF yield check", symbol)

    # --- 6. FCF must be increasing in at least FCF_MIN_GROWTH_YEARS of lookback ---
    if config.REQUIRE_INCREASING_FCF:
        fcf_history = info.get("_fcf_history", [])
        fcf_increasing = _check_fcf_increasing(
            fcf_history,
            config.FCF_GROWTH_LOOKBACK_YEARS,
            config.FCF_MIN_GROWTH_YEARS,
        )
        profile.fcf_increasing = fcf_increasing
        if fcf_increasing is False:   # None = insufficient data → benefit of the doubt
            profile.passes = False
            profile.fail_reasons.append(
                f"FCF grew in fewer than {config.FCF_MIN_GROWTH_YEARS} of last "
                f"{config.FCF_GROWTH_LOOKBACK_YEARS} year(s)"
            )

    # --- 7. Return on equity >= 10% ---
    roe = _safe_float(info.get("returnOnEquity"))
    if roe is not None:
        profile.return_on_equity = roe
        if roe < config.MIN_RETURN_ON_EQUITY:
            profile.passes = False
            profile.fail_reasons.append(
                f"ROE={roe:.1%} < MIN={config.MIN_RETURN_ON_EQUITY:.1%}"
            )

    # --- 8. Capex-to-FCF <= 50% ---
    capex_raw = _safe_float(info.get("capitalExpenditures"))
    if capex_raw is not None and fcf is not None and fcf > 0:
        capex = abs(capex_raw)
        profile.capex_to_fcf = capex / fcf
        if profile.capex_to_fcf > config.MAX_CAPEX_TO_FCF:
            profile.passes = False
            profile.fail_reasons.append(
                f"Capex/FCF={profile.capex_to_fcf:.2%} > MAX={config.MAX_CAPEX_TO_FCF:.2%}"
            )

    if profile.passes:
        logger.info("%s: PASS fundamental screen", symbol)
    else:
        logger.info(
            "%s: FAIL fundamental screen — %s", symbol, "; ".join(profile.fail_reasons)
        )

    # ── Populate new quality metrics (informational — do not gate pass/fail) ──
    profile.roic               = _safe_float(info.get("roic"))
    profile.roic_prior         = _safe_float(info.get("roic_prior"))
    profile.fcf_margin         = _safe_float(info.get("fcf_margin"))
    profile.fcf_margin_prior   = _safe_float(info.get("fcf_margin_prior"))
    profile.cash_conversion    = _safe_float(info.get("cash_conversion"))
    profile.cash_conversion_prior = _safe_float(info.get("cash_conversion_prior"))
    profile.accruals_ratio     = _safe_float(info.get("accruals_ratio"))
    profile.accruals_ratio_prior = _safe_float(info.get("accruals_ratio_prior"))
    profile.receivables_growth = _safe_float(info.get("receivables_growth"))
    profile.revenue_growth     = _safe_float(info.get("revenue_growth"))
    profile.revenue_current    = _safe_float(info.get("revenue_current"))
    profile.revenue_prior      = _safe_float(info.get("revenue_prior"))

    return profile
