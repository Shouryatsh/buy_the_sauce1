"""
screener.py — Fundamental quality filter using yfinance data.

Applies the thresholds defined in config.py to eliminate stocks that do
not meet baseline quality criteria before dip-detection is run.  This
keeps the buy-the-dip signals focused on fundamentally sound companies.

Criteria checked
----------------
1. Forward P/E          : MIN_PE_RATIO < forwardPE (or trailingPE) ≤ MAX_PE_RATIO
2. Profit margin        : profitMargins ≥ MIN_PROFIT_MARGIN
3. Debt / equity        : debtToEquity / 100 ≤ MAX_DEBT_TO_EQUITY
                          (yfinance reports D/E as a percentage, e.g. 150 = 1.5×)
4. Free cash flow       : freeCashflow > MIN_FREE_CASH_FLOW  (must be positive)
5. FCF yield            : freeCashflow / marketCap ≥ MIN_FCF_YIELD  (≥ 2%)
6. Increasing FCF       : latest FCF > prior-year FCF  (via cash-flow statement)
                          checked over FCF_GROWTH_LOOKBACK_YEARS years
7. Return on equity     : returnOnEquity ≥ MIN_RETURN_ON_EQUITY  (≥ 10%)
8. Capex-to-FCF ratio   : |capitalExpenditures| / freeCashflow ≤ MAX_CAPEX_TO_FCF

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
    debt_to_equity: float | None = None   # normalised (not ×100)
    free_cash_flow: float | None = None   # absolute FCF in reporting currency
    fcf_yield: float | None = None        # FCF / market cap
    fcf_increasing: bool | None = None    # True if FCF trended up over lookback
    return_on_equity: float | None = None
    capex_to_fcf: float | None = None
    passes: bool = True
    fail_reasons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value) -> float | None:
    """Return float or None, silently swallowing conversion errors."""
    try:
        v = float(value)
        import math
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def _check_fcf_increasing(ticker: yf.Ticker, lookback: int) -> bool | None:
    """Return True if FCF has been growing over *lookback* annual periods.

    Uses the cash-flow statement (annual) from yfinance.  Returns None when
    there is insufficient history to make a determination.
    """
    try:
        cf = ticker.cashflow          # columns = fiscal-year end dates (newest first)
        if cf is None or cf.empty:
            return None

        # Row label varies by yfinance version
        for label in ("Free Cash Flow", "FreeCashFlow"):
            if label in cf.index:
                fcf_series = cf.loc[label].dropna()
                break
        else:
            # Construct FCF manually: Operating CF − Capex
            op_label = next(
                (l for l in cf.index if "Operating" in l and "Cash" in l), None
            )
            cx_label = next(
                (l for l in cf.index if "Capital" in l and "Expenditure" in l), None
            )
            if op_label is None or cx_label is None:
                return None
            fcf_series = (cf.loc[op_label] - cf.loc[cx_label].abs()).dropna()

        if len(fcf_series) < 2:
            return None

        # Take up to (lookback + 1) data points so we can compare 'lookback' pairs
        fcf_values = fcf_series.iloc[: lookback + 1].tolist()   # newest → oldest
        # Check every consecutive pair: newer > older
        return all(fcf_values[i] > fcf_values[i + 1] for i in range(len(fcf_values) - 1))

    except Exception as exc:
        logger.debug("FCF trend check failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def screen_fundamental(
    symbol: str,
    info: dict | None = None,
    _ticker: yf.Ticker | None = None,
) -> FundamentalProfile:
    """Return a FundamentalProfile indicating whether the stock passes the
    fundamental quality filter.

    Parameters
    ----------
    symbol:
        Ticker symbol.
    info:
        Pre-fetched ``yf.Ticker(symbol).info`` dict.  If None, fetched
        automatically (useful in production; pass explicitly in tests to
        avoid network calls).
    _ticker:
        Pre-built ``yf.Ticker`` object.  Only used internally / in tests to
        inject a mock; ignored when *info* is also provided from a mock.
    """
    ticker = _ticker or yf.Ticker(symbol)

    if info is None:
        try:
            info = ticker.info
        except Exception as exc:
            logger.warning("%s: could not fetch info — %s", symbol, exc)
            info = {}

    profile = FundamentalProfile(symbol=symbol)

    # --- 1. P/E ratio ---
    pe = _safe_float(info.get("forwardPE") or info.get("trailingPE"))
    if pe is not None:
        profile.pe_ratio = pe
        if pe <= config.MIN_PE_RATIO:
            profile.passes = False
            profile.fail_reasons.append(
                f"PE={pe:.1f} ≤ MIN_PE_RATIO={config.MIN_PE_RATIO}"
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

    # --- 3. Debt / equity ---
    de_raw = _safe_float(info.get("debtToEquity"))
    if de_raw is not None:
        profile.debt_to_equity = de_raw / 100.0          # yfinance gives percentage
        if profile.debt_to_equity > config.MAX_DEBT_TO_EQUITY:
            profile.passes = False
            profile.fail_reasons.append(
                f"D/E={profile.debt_to_equity:.2f} > MAX={config.MAX_DEBT_TO_EQUITY}"
            )

    # --- 4. Free cash flow — must be positive ---
    fcf = _safe_float(info.get("freeCashflow"))
    if fcf is not None:
        profile.free_cash_flow = fcf
        if fcf <= config.MIN_FREE_CASH_FLOW:
            profile.passes = False
            profile.fail_reasons.append(
                f"FCF={fcf:,.0f} ≤ MIN={config.MIN_FREE_CASH_FLOW:,.0f} (negative/zero)"
            )

    # --- 5. FCF yield ≥ 2% ---
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

    # --- 6. FCF must be increasing (YoY trend) ---
    if config.REQUIRE_INCREASING_FCF:
        fcf_increasing = _check_fcf_increasing(ticker, config.FCF_GROWTH_LOOKBACK_YEARS)
        profile.fcf_increasing = fcf_increasing
        if fcf_increasing is False:          # None = data gap → benefit of the doubt
            profile.passes = False
            profile.fail_reasons.append(
                f"FCF not increasing over last {config.FCF_GROWTH_LOOKBACK_YEARS} year(s)"
            )

    # --- 7. Return on equity ≥ 10% ---
    roe = _safe_float(info.get("returnOnEquity"))
    if roe is not None:
        profile.return_on_equity = roe
        if roe < config.MIN_RETURN_ON_EQUITY:
            profile.passes = False
            profile.fail_reasons.append(
                f"ROE={roe:.1%} < MIN={config.MIN_RETURN_ON_EQUITY:.1%}"
            )

    # --- 8. Capex-to-FCF ≤ 50% ---
    capex_raw = _safe_float(info.get("capitalExpenditures"))
    if capex_raw is not None and fcf is not None and fcf > 0:
        # yfinance reports capex as a negative number; take absolute value
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

    return profile
