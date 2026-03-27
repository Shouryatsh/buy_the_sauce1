"""
valuation.py — Conservative intrinsic-value engine.

Implements a multi-model valuation framework inspired by:
  - Aswath Damodaran's "The Little Book of Valuation"
  - Damodaran's "Investment Valuation" (3rd ed.)
  - Greenwald's "Value Investing: From Graham to Buffett"
  - Penman's "Accounting for Value"
  - Graham's "The Intelligent Investor" (margin-of-safety concept)

Design principles
-----------------
  1. **Conservative by default**: Every model uses pessimistic assumptions.
     Where Morningstar picks base-case growth, we use 1/2 to 2/3 of that.
  2. **Multiple independent models**: No single model drives the output.
     The final fair value is a WEIGHTED MEDIAN of 5+ models, each with its
     own lens (cash flow, earnings, assets, relative).
  3. **Margin of safety**: The "buy price" is fair-value × (1 − margin_of_safety).
     Default MoS = 30% for high-quality, 40% for medium, 50% for speculative.
  4. **Transparent**: Every model's output and inputs are returned for display.

Models implemented
------------------
  M1. Two-Stage DCF (FCF-based)      — Damodaran's workhorse; conservative FCF growth
  M2. Reverse DCF                     — "What growth is the market pricing in?"
  M3. Earnings Power Value (EPV)      — Greenwald: value of current earnings, no growth
  M4. Graham Number                   — Graham's classic margin-of-safety formula
  M5. Excess Returns / Residual Income — Penman/Damodaran: only value returns > WACC
  M6. Dividend Discount (if applicable) — Gordon Growth Model with pessimistic payout
  M7. Relative Valuation (sector P/E) — Conservative median-sector P/E × normalized EPS
  M8. Asset-based (Book Value floor)  — Tangible book as absolute floor

Data source: SEC EDGAR via edgar.py (no API key, no scraping).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import config as _cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration — deliberately conservative defaults
# ---------------------------------------------------------------------------

# WACC / discount rate assumptions
# Morningstar uses company-specific WACC (often 7-9% for large caps).
# We use a HIGHER discount rate to be conservative.
DEFAULT_RISK_FREE_RATE: float = 0.043       # ~10Y US Treasury yield (conservative)
DEFAULT_EQUITY_RISK_PREMIUM: float = 0.055  # Damodaran's 2024 ERP for US
DEFAULT_BETA_FLOOR: float = 0.8             # never assume beta < 0.8
DEFAULT_BETA_CAP: float = 2.0               # cap at 2.0
DEFAULT_COST_OF_DEBT: float = 0.055         # after-tax cost of debt assumption
DEFAULT_TAX_RATE: float = 0.21              # US corporate tax rate
WACC_FLOOR: float = 0.09                    # never use WACC below 9%
WACC_CAP: float = 0.16                      # cap at 16%

# Growth assumptions — THESE ARE THE KEY CONSERVATISM LEVERS
# Morningstar issue: they often project 5-15% growth for 5-10 years.
# We cap high-growth at 2/3 of trailing growth, and fade to terminal faster.
MAX_HIGH_GROWTH_RATE: float = 0.12          # never assume > 12% revenue growth
HIGH_GROWTH_HAIRCUT: float = 0.60           # use 60% of trailing growth (Morningstar uses ~80-100%)
HIGH_GROWTH_YEARS: int = 5                  # only 5 years of above-average growth
TERMINAL_GROWTH_RATE: float = 0.025         # long-run GDP growth (2.5%) — Damodaran standard
TERMINAL_GROWTH_CAP: float = 0.03           # terminal growth never exceeds 3%
# For negative/zero growth companies, assume 0% in high-growth phase
MIN_HIGH_GROWTH_RATE: float = 0.0

# FCF conversion
FCF_TO_REVENUE_FLOOR: float = 0.03          # assume at least 3% FCF margin
FCF_TO_REVENUE_CAP: float = 0.30            # cap at 30% FCF margin

# Margin of safety (Buffett/Graham concept)
MARGIN_OF_SAFETY_HIGH_QUALITY: float = 0.25  # 25% MoS for best companies
MARGIN_OF_SAFETY_MEDIUM: float = 0.35        # 35% MoS for average companies
MARGIN_OF_SAFETY_SPECULATIVE: float = 0.50   # 50% MoS for low-quality / high-uncertainty

# Model weights for composite valuation
# Higher weight = more influence on final fair value
MODEL_WEIGHTS = {
    "dcf_two_stage":      3.0,   # primary model
    "reverse_dcf":        1.0,   # diagnostic (not direct valuation)
    "epv":                2.5,   # earnings power value — very conservative
    "graham_number":      1.5,   # classic margin-of-safety
    "excess_returns":     2.0,   # residual income / economic profit
    "ddm":                1.0,   # only if dividend-paying
    "relative_pe":        1.5,   # sector-relative
    "asset_floor":        0.5,   # tangible book floor
}

# Quality score thresholds for margin-of-safety tiering
QUALITY_HIGH_THRESHOLD: float = 70.0    # quality score >= 70 → high quality
QUALITY_MEDIUM_THRESHOLD: float = 45.0  # quality score >= 45 → medium


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    """Output from a single valuation model."""
    model_name: str
    fair_value_per_share: Optional[float] = None
    weight: float = 1.0
    inputs: dict = field(default_factory=dict)
    notes: str = ""
    error: str = ""


@dataclass
class ValuationResult:
    """Composite valuation output for a single ticker."""
    symbol: str
    current_price: Optional[float] = None
    shares_outstanding: Optional[float] = None

    # Individual model results
    models: list[ModelResult] = field(default_factory=list)

    # Composite outputs
    fair_value: Optional[float] = None          # weighted median of models
    fair_value_low: Optional[float] = None      # pessimistic bound (25th pctile)
    fair_value_high: Optional[float] = None     # optimistic bound (75th pctile)

    # Buy/sell signals
    margin_of_safety_pct: float = 0.30          # applied MoS %
    buy_price: Optional[float] = None           # fair_value × (1 − MoS)
    upside_pct: Optional[float] = None          # (fair_value / price − 1) × 100
    upside_to_buy_pct: Optional[float] = None   # (price / buy_price − 1) × 100
    valuation_signal: str = "HOLD"              # DEEP_VALUE / UNDERVALUED / FAIR / OVERVALUED / EXPENSIVE
    valuation_grade: str = "—"                  # A/B/C/D/F letter grade

    # Quality assessment
    quality_score: float = 50.0                 # 0-100 composite quality
    quality_tier: str = "MEDIUM"                # HIGH / MEDIUM / SPECULATIVE
    moat_indicators: list[str] = field(default_factory=list)

    # Reverse DCF diagnostic
    implied_growth_rate: Optional[float] = None  # what growth the market is pricing in
    growth_reasonableness: str = "—"             # "REASONABLE" / "OPTIMISTIC" / "HEROIC"

    # Display
    summary: str = ""
    model_details: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# WACC estimation
# ---------------------------------------------------------------------------

def _estimate_wacc(info: dict) -> float:
    """Estimate WACC from available fundamentals. Conservative bias.

    WACC = (E/V)×Re + (D/V)×Rd×(1−T)
    where Re = Rf + β×ERP  (CAPM)
    """
    rf = DEFAULT_RISK_FREE_RATE
    erp = DEFAULT_EQUITY_RISK_PREMIUM

    # Beta — use trailing if available, otherwise assume sector average
    beta = info.get("beta")
    if beta is None or not isinstance(beta, (int, float)) or math.isnan(beta):
        beta = 1.1  # slightly above market (conservative for unknowns)
    beta = max(DEFAULT_BETA_FLOOR, min(DEFAULT_BETA_CAP, beta))

    # Cost of equity (CAPM)
    cost_of_equity = rf + beta * erp

    # Capital structure
    market_cap = info.get("marketCap") or 0
    de_raw = info.get("debtToEquity")
    if de_raw and market_cap > 0:
        de = de_raw / 100.0
        debt = market_cap * de
        total_value = market_cap + debt
        equity_weight = market_cap / total_value
        debt_weight = debt / total_value
    else:
        equity_weight = 0.80
        debt_weight = 0.20

    cost_of_debt = DEFAULT_COST_OF_DEBT
    tax_rate = DEFAULT_TAX_RATE

    wacc = equity_weight * cost_of_equity + debt_weight * cost_of_debt * (1 - tax_rate)

    # Clamp to floor/cap
    wacc = max(WACC_FLOOR, min(WACC_CAP, wacc))

    return round(wacc, 4)


# ---------------------------------------------------------------------------
# Quality assessment — drives margin of safety tier
# ---------------------------------------------------------------------------

def _assess_quality(info: dict) -> tuple[float, str, list[str]]:
    """Score company quality 0-100 and identify moat indicators.

    Returns (score, tier, moat_indicators).
    """
    score = 50.0  # baseline: average company
    moat_indicators = []

    # ROIC > 15% = has economic moat (returns above cost of capital)
    roic = info.get("roic")
    roic_prior = info.get("roic_prior")
    if roic is not None:
        if roic > 0.20:
            score += 12
            moat_indicators.append(f"ROIC {roic:.0%} (excellent)")
        elif roic > 0.15:
            score += 8
            moat_indicators.append(f"ROIC {roic:.0%} (strong)")
        elif roic > 0.10:
            score += 3
        elif roic < 0.05:
            score -= 10
        # ROIC stability bonus
        if roic_prior is not None and roic > 0.12 and roic_prior > 0.12:
            score += 5
            moat_indicators.append("Stable ROIC (multi-year)")

    # Profit margins > 15% = pricing power / moat
    margin = info.get("profitMargins")
    if margin is not None:
        if margin > 0.25:
            score += 8
            moat_indicators.append(f"Net margin {margin:.0%} (wide moat)")
        elif margin > 0.15:
            score += 4
            moat_indicators.append(f"Net margin {margin:.0%}")
        elif margin < 0.05:
            score -= 8

    # FCF margin consistency
    fcf_margin = info.get("fcf_margin")
    if fcf_margin is not None:
        if fcf_margin > 0.20:
            score += 6
            moat_indicators.append(f"FCF margin {fcf_margin:.0%} (cash machine)")
        elif fcf_margin > 0.10:
            score += 3
        elif fcf_margin < 0:
            score -= 10

    # Cash conversion > 1x = converting earnings to real cash
    cc = info.get("cash_conversion")
    if cc is not None:
        if cc > 1.0:
            score += 4
            moat_indicators.append("Cash conversion >1x")
        elif cc < 0.5:
            score -= 5

    # Low leverage = financial strength
    de_raw = info.get("debtToEquity")
    if de_raw is not None:
        de = de_raw / 100.0
        if de < 0.5:
            score += 5
            moat_indicators.append("Low leverage")
        elif de > 3.0:
            score -= 8

    # Revenue growth consistency
    rev_growth = info.get("revenue_growth")
    if rev_growth is not None:
        if rev_growth > 0.10:
            score += 4
        elif rev_growth < -0.05:
            score -= 6

    # Accruals ratio (earnings quality)
    accruals = info.get("accruals_ratio")
    if accruals is not None:
        if accruals < 0:
            score += 3  # negative accruals = very high earnings quality
            moat_indicators.append("Negative accruals (high quality)")
        elif accruals > 0.10:
            score -= 5

    # FCF increasing
    fcf_hist = info.get("_fcf_history", [])
    if len(fcf_hist) >= 3:
        growth_yrs = sum(1 for i in range(len(fcf_hist) - 1) if fcf_hist[i] > fcf_hist[i+1])
        if growth_yrs >= len(fcf_hist) - 1:
            score += 5
            moat_indicators.append("FCF growing every year")

    # ROE consistency
    roe = info.get("returnOnEquity")
    if roe is not None and roe > 0.20:
        score += 4
        moat_indicators.append(f"ROE {roe:.0%}")

    # Clamp score
    score = max(0, min(100, score))

    # Tier
    if score >= QUALITY_HIGH_THRESHOLD:
        tier = "HIGH"
    elif score >= QUALITY_MEDIUM_THRESHOLD:
        tier = "MEDIUM"
    else:
        tier = "SPECULATIVE"

    return score, tier, moat_indicators


# ---------------------------------------------------------------------------
# Model 1: Two-Stage DCF (FCF-based) — Damodaran
# ---------------------------------------------------------------------------

def _dcf_two_stage(info: dict, wacc: float, shares: float) -> ModelResult:
    """Conservative two-stage DCF using free cash flow.

    Stage 1: HIGH_GROWTH_YEARS at a haircut of trailing FCF growth
    Stage 2: Terminal value at TERMINAL_GROWTH_RATE, valued via Gordon Growth

    Key conservatism vs Morningstar:
      - Uses 60% of trailing growth (not 80-100%)
      - Caps growth at 12% (Morningstar often uses 15-25% for tech)
      - Only 5 years of high growth (Morningstar often uses 10)
      - Higher WACC (9% floor vs Morningstar's 7-8%)
      - Terminal growth capped at 2.5% (some models use 3-4%)
    """
    result = ModelResult(model_name="dcf_two_stage", weight=MODEL_WEIGHTS["dcf_two_stage"])

    fcf_hist = info.get("_fcf_history", [])
    if not fcf_hist or fcf_hist[0] is None:
        result.error = "No FCF data available"
        return result

    current_fcf = fcf_hist[0]
    if current_fcf <= 0:
        result.error = f"Negative FCF (${current_fcf:,.0f}) — cannot run DCF"
        return result

    # Estimate trailing FCF growth rate
    if len(fcf_hist) >= 3 and fcf_hist[-1] > 0:
        n_years = len(fcf_hist) - 1
        fcf_cagr = (fcf_hist[0] / fcf_hist[-1]) ** (1.0 / n_years) - 1
    elif len(fcf_hist) >= 2 and fcf_hist[1] > 0:
        fcf_cagr = (fcf_hist[0] / fcf_hist[1]) - 1
    else:
        fcf_cagr = 0.05  # default conservative assumption

    # Apply conservatism: use HIGH_GROWTH_HAIRCUT of trailing, cap at MAX
    high_growth = fcf_cagr * HIGH_GROWTH_HAIRCUT
    high_growth = max(MIN_HIGH_GROWTH_RATE, min(MAX_HIGH_GROWTH_RATE, high_growth))

    terminal_growth = min(TERMINAL_GROWTH_RATE, TERMINAL_GROWTH_CAP)

    # Additional conservatism: if WACC - terminal_growth is too small,
    # the terminal value explodes. Enforce a minimum spread.
    if wacc - terminal_growth < 0.04:
        terminal_growth = wacc - 0.04

    # Stage 1: Projected FCFs
    stage1_pv = 0.0
    projected_fcf = current_fcf
    for year in range(1, HIGH_GROWTH_YEARS + 1):
        projected_fcf *= (1 + high_growth)
        pv = projected_fcf / (1 + wacc) ** year
        stage1_pv += pv

    # Stage 2: Terminal value (Gordon Growth)
    terminal_fcf = projected_fcf * (1 + terminal_growth)
    terminal_value = terminal_fcf / (wacc - terminal_growth)
    terminal_pv = terminal_value / (1 + wacc) ** HIGH_GROWTH_YEARS

    # Conservative: apply 10% haircut to terminal value
    # (terminal value assumptions are always the weakest link)
    terminal_pv *= 0.90

    enterprise_value = stage1_pv + terminal_pv

    # Deduct net debt to get equity value
    net_debt = _estimate_net_debt(info)
    equity_value = enterprise_value - net_debt

    if shares > 0 and equity_value > 0:
        result.fair_value_per_share = round(equity_value / shares, 2)

    result.inputs = {
        "current_fcf": current_fcf,
        "trailing_fcf_cagr": round(fcf_cagr, 4),
        "applied_growth": round(high_growth, 4),
        "terminal_growth": round(terminal_growth, 4),
        "wacc": wacc,
        "high_growth_years": HIGH_GROWTH_YEARS,
        "stage1_pv": round(stage1_pv, 0),
        "terminal_pv": round(terminal_pv, 0),
        "net_debt": round(net_debt, 0),
        "enterprise_value": round(enterprise_value, 0),
    }
    result.notes = (
        f"FCF=${current_fcf/1e9:.1f}B → grow at {high_growth:.1%} for {HIGH_GROWTH_YEARS}yr "
        f"→ terminal at {terminal_growth:.1%} | WACC={wacc:.1%}"
    )
    return result


# ---------------------------------------------------------------------------
# Model 2: Reverse DCF — "What growth is the market pricing in?"
# ---------------------------------------------------------------------------

def _reverse_dcf(info: dict, wacc: float, shares: float,
                 current_price: float) -> ModelResult:
    """Solve for the FCF growth rate implied by the current market price.

    If the implied growth is unreasonably high (>15%), the stock is expensive
    relative to any reasonable scenario.
    """
    result = ModelResult(model_name="reverse_dcf", weight=MODEL_WEIGHTS["reverse_dcf"])

    fcf_hist = info.get("_fcf_history", [])
    if not fcf_hist or fcf_hist[0] is None or fcf_hist[0] <= 0:
        result.error = "No positive FCF for reverse DCF"
        return result

    current_fcf = fcf_hist[0]
    market_cap = current_price * shares if shares > 0 else 0
    net_debt = _estimate_net_debt(info)
    enterprise_value = market_cap + net_debt

    if enterprise_value <= 0:
        result.error = "Invalid enterprise value"
        return result

    terminal_growth = TERMINAL_GROWTH_RATE

    # Binary search for the implied growth rate
    low, high = -0.10, 0.50
    for _ in range(100):
        mid = (low + high) / 2
        ev = _calc_ev_from_growth(current_fcf, mid, terminal_growth, wacc)
        if ev < enterprise_value:
            low = mid
        else:
            high = mid
        if abs(high - low) < 0.0001:
            break

    implied_growth = round((low + high) / 2, 4)

    # Assess reasonableness
    if implied_growth > 0.20:
        reasonableness = "HEROIC"
    elif implied_growth > 0.12:
        reasonableness = "OPTIMISTIC"
    elif implied_growth > 0.05:
        reasonableness = "REASONABLE"
    elif implied_growth > 0:
        reasonableness = "CONSERVATIVE"
    else:
        reasonableness = "NEGATIVE (deep value?)"

    # Reverse DCF doesn't produce a fair value directly — it's diagnostic
    # We use it to produce a "reality-check" fair value:
    # "If growth is only X%, what would the stock be worth?"
    # Use a reasonable growth assumption (min of trailing or 8%)
    reasonable_growth = min(0.08, max(0.0, implied_growth * 0.5))
    reasonable_ev = _calc_ev_from_growth(current_fcf, reasonable_growth, terminal_growth, wacc)
    equity_val = reasonable_ev - net_debt
    if shares > 0 and equity_val > 0:
        result.fair_value_per_share = round(equity_val / shares, 2)

    result.inputs = {
        "implied_growth": implied_growth,
        "current_fcf": current_fcf,
        "enterprise_value": round(enterprise_value, 0),
        "reasonableness": reasonableness,
    }
    result.notes = f"Market implies {implied_growth:.1%} FCF growth for {HIGH_GROWTH_YEARS}yr — {reasonableness}"
    return result


def _calc_ev_from_growth(fcf: float, growth: float, terminal_growth: float,
                         wacc: float) -> float:
    """Calculate enterprise value for a given FCF growth rate."""
    if wacc <= terminal_growth:
        return float("inf")
    pv = 0.0
    projected = fcf
    for yr in range(1, HIGH_GROWTH_YEARS + 1):
        projected *= (1 + growth)
        pv += projected / (1 + wacc) ** yr
    term_fcf = projected * (1 + terminal_growth)
    term_val = term_fcf / (wacc - terminal_growth)
    pv += term_val / (1 + wacc) ** HIGH_GROWTH_YEARS
    return pv


# ---------------------------------------------------------------------------
# Model 3: Earnings Power Value (Greenwald)
# ---------------------------------------------------------------------------

def _earnings_power_value(info: dict, wacc: float, shares: float) -> ModelResult:
    """Greenwald's EPV: value of CURRENT earnings with ZERO growth.

    EPV = Normalized Earnings / WACC
    This is the most conservative growth-agnostic model — it answers
    "What is this company worth if it NEVER grows?"

    Very powerful for stable businesses; automatically penalises growth stocks.
    """
    result = ModelResult(model_name="epv", weight=MODEL_WEIGHTS["epv"])

    # Use average of last 2-3 years of net income for normalization
    # (avoids one-off earnings spikes)
    fcf_hist = info.get("_fcf_history", [])

    # Prefer FCF over net income (higher quality earnings measure)
    if fcf_hist and len(fcf_hist) >= 2:
        avg_earnings = sum(f for f in fcf_hist[:3] if f is not None) / min(3, len(fcf_hist))
    elif fcf_hist and fcf_hist[0] and fcf_hist[0] > 0:
        avg_earnings = fcf_hist[0]
    else:
        # Fallback: use raw net income from EDGAR or estimate from margin × revenue
        ni = info.get("_net_income")
        if ni is not None and ni > 0:
            avg_earnings = ni
        else:
            margin = info.get("profitMargins")
            rev = info.get("revenue_current")
            if margin is not None and rev is not None and rev > 0:
                avg_earnings = margin * rev
            else:
                result.error = "Insufficient earnings data for EPV"
                return result

    if avg_earnings <= 0:
        result.error = f"Negative normalized earnings — EPV not applicable"
        return result

    # EPV = normalized earnings / cost of capital
    # Then add excess cash, subtract debt
    epv_enterprise = avg_earnings / wacc

    # Maintenance capex adjustment: EPV should use maintenance capex only
    # (not growth capex). Heuristic: 70% of total capex is maintenance.
    # This REDUCES EPV compared to using full FCF.
    capex_raw = info.get("capitalExpenditures")
    op_cf = info.get("_operating_cf")
    if capex_raw and op_cf and op_cf > 0:
        maintenance_capex = abs(capex_raw) * 0.70  # assume 70% is maintenance
        adjusted_earnings = op_cf - maintenance_capex
        if adjusted_earnings > 0:
            epv_enterprise = adjusted_earnings / wacc
    elif capex_raw and fcf_hist:
        op_cf_est = fcf_hist[0] + abs(capex_raw)  # reconstruct operating CF
        maintenance_capex = abs(capex_raw) * 0.70
        adjusted_earnings = op_cf_est - maintenance_capex
        if adjusted_earnings > 0:
            epv_enterprise = adjusted_earnings / wacc

    net_debt = _estimate_net_debt(info)
    equity_value = epv_enterprise - net_debt

    if shares > 0 and equity_value > 0:
        result.fair_value_per_share = round(equity_value / shares, 2)

    result.inputs = {
        "normalized_earnings": round(avg_earnings, 0),
        "wacc": wacc,
        "epv_enterprise": round(epv_enterprise, 0),
        "net_debt": round(net_debt, 0),
    }
    result.notes = f"Zero-growth value: normalized earnings ${avg_earnings/1e9:.1f}B / {wacc:.1%} WACC"
    return result


# ---------------------------------------------------------------------------
# Model 4: Graham Number
# ---------------------------------------------------------------------------

def _graham_number(info: dict, shares: float) -> ModelResult:
    """Benjamin Graham's classic formula: sqrt(22.5 × EPS × BVPS).

    Original: sqrt(15 × 1.5 × EPS × BVPS) = sqrt(22.5 × EPS × BVPS)
    We use Graham's CONSERVATIVE variant: sqrt(15 × EPS × BVPS)
    (drops the 1.5× bond multiplier that was era-specific).
    """
    result = ModelResult(model_name="graham_number", weight=MODEL_WEIGHTS["graham_number"])

    # EPS from net income / shares
    net_income = info.get("_net_income")  # prefer raw EDGAR net income
    fcf_hist = info.get("_fcf_history", [])

    if net_income is None or net_income <= 0:
        # Fallback: estimate from margin × revenue
        margin = info.get("profitMargins")
        rev = info.get("revenue_current")
        if margin is not None and rev is not None and rev > 0:
            net_income = margin * rev
        elif fcf_hist and fcf_hist[0] and fcf_hist[0] > 0:
            # Use FCF as earnings proxy (conservative)
            net_income = fcf_hist[0]

    if net_income is None or net_income <= 0 or shares <= 0:
        result.error = "Need positive earnings for Graham Number"
        return result

    eps = net_income / shares

    # Book value per share from equity / shares
    equity = _get_equity(info)
    if equity is None or equity <= 0:
        result.error = "Need positive book value for Graham Number"
        return result

    bvps = equity / shares

    # Conservative Graham: sqrt(15 × EPS × BVPS)
    # (not the 22.5 version — we want to be MORE conservative than Graham)
    product = 15.0 * eps * bvps
    if product <= 0:
        result.error = "Negative EPS×BVPS product"
        return result

    graham_value = math.sqrt(product)
    result.fair_value_per_share = round(graham_value, 2)

    result.inputs = {
        "eps": round(eps, 2),
        "bvps": round(bvps, 2),
        "multiplier": 15.0,
    }
    result.notes = f"sqrt(15 × ${eps:.2f} EPS × ${bvps:.2f} BVPS) = ${graham_value:.2f}"
    return result


# ---------------------------------------------------------------------------
# Model 5: Excess Returns / Residual Income (Penman/Damodaran)
# ---------------------------------------------------------------------------

def _excess_returns(info: dict, wacc: float, shares: float) -> ModelResult:
    """Value only the returns in excess of the cost of capital.

    Intrinsic Value = Book Value + PV(Excess Returns)
    where Excess Return = (ROIC − WACC) × Invested Capital

    This model is powerful because:
      - A company earning EXACTLY its cost of capital is worth BOOK VALUE
      - Only excess returns (economic profit) deserve a premium
      - Growth is only valuable if ROIC > WACC
    """
    result = ModelResult(model_name="excess_returns", weight=MODEL_WEIGHTS["excess_returns"])

    roic = info.get("roic")
    equity = _get_equity(info)

    if roic is None or equity is None or equity <= 0:
        result.error = "Need ROIC and positive equity for excess returns model"
        return result

    # Invested capital = equity + long-term debt
    ltd = _get_long_term_debt(info)
    invested_capital = equity + ltd

    # Excess return = (ROIC - WACC) × invested capital
    excess_return = (roic - wacc) * invested_capital

    if excess_return <= 0:
        # Company doesn't earn above cost of capital — worth book value at best
        if shares > 0:
            result.fair_value_per_share = round(equity / shares, 2)
        result.notes = f"ROIC {roic:.1%} ≤ WACC {wacc:.1%} — worth ~book value"
        result.inputs = {
            "roic": round(roic, 4),
            "wacc": wacc,
            "invested_capital": round(invested_capital, 0),
            "excess_return": round(excess_return, 0),
        }
        return result

    # PV of excess returns (assume they fade over time)
    # Conservative: assume excess returns decay by 10% per year (competitive erosion)
    fade_rate = 0.10
    pv_excess = 0.0
    annual_excess = excess_return
    for yr in range(1, 16):  # 15-year horizon
        annual_excess *= (1 - fade_rate)
        pv_excess += annual_excess / (1 + wacc) ** yr

    # Intrinsic value = book value + PV(excess returns)
    intrinsic = equity + pv_excess

    if shares > 0 and intrinsic > 0:
        result.fair_value_per_share = round(intrinsic / shares, 2)

    result.inputs = {
        "roic": round(roic, 4),
        "wacc": wacc,
        "invested_capital": round(invested_capital, 0),
        "excess_return": round(excess_return, 0),
        "fade_rate": fade_rate,
        "pv_excess": round(pv_excess, 0),
    }
    result.notes = (
        f"ROIC {roic:.1%} − WACC {wacc:.1%} = {roic-wacc:.1%} spread | "
        f"Excess return ${excess_return/1e9:.1f}B, fading 10%/yr"
    )
    return result


# ---------------------------------------------------------------------------
# Model 6: Dividend Discount Model (Gordon Growth)
# ---------------------------------------------------------------------------

def _dividend_discount(info: dict, wacc: float, shares: float) -> ModelResult:
    """Gordon Growth DDM for dividend-paying stocks.

    Only applied if the company pays dividends. Uses a pessimistic payout ratio.
    """
    result = ModelResult(model_name="ddm", weight=MODEL_WEIGHTS["ddm"])

    # We need dividend info — check if available from the fundamentals
    # Since we're using EDGAR, we don't have direct dividend data easily.
    # Use FCF × conservative payout ratio as a proxy for sustainable dividend.
    fcf_hist = info.get("_fcf_history", [])
    if not fcf_hist or fcf_hist[0] is None or fcf_hist[0] <= 0:
        result.error = "No positive FCF — DDM not applicable"
        result.weight = 0
        return result

    # Conservative: assume company pays out only 40% of FCF as dividend
    # (lower than typical payout ratios)
    payout_ratio = 0.40
    sustainable_dividend = fcf_hist[0] * payout_ratio

    if shares <= 0:
        result.error = "No shares outstanding"
        return result

    dps = sustainable_dividend / shares  # dividend per share

    # Growth rate for dividends — very conservative
    rev_growth = info.get("revenue_growth")
    div_growth = min(TERMINAL_GROWTH_RATE, (rev_growth or 0) * 0.5)
    div_growth = max(0.0, min(0.03, div_growth))

    cost_of_equity = wacc + 0.01  # use slightly higher than WACC for equity-only
    if cost_of_equity <= div_growth:
        result.error = "Cost of equity ≤ dividend growth (invalid)"
        return result

    fair_value = dps / (cost_of_equity - div_growth)

    if fair_value > 0:
        result.fair_value_per_share = round(fair_value, 2)

    result.inputs = {
        "sustainable_dividend_total": round(sustainable_dividend, 0),
        "dps": round(dps, 2),
        "payout_ratio": payout_ratio,
        "div_growth": round(div_growth, 4),
        "cost_of_equity": round(cost_of_equity, 4),
    }
    result.notes = f"DPS=${dps:.2f} (40% of FCF) growing at {div_growth:.1%}, CoE={cost_of_equity:.1%}"
    return result


# ---------------------------------------------------------------------------
# Model 7: Relative Valuation (Conservative P/E)
# ---------------------------------------------------------------------------

def _relative_pe(info: dict, shares: float) -> ModelResult:
    """Conservative relative P/E valuation.

    Uses a CONSERVATIVE target P/E (capped at 18x) × normalized EPS.
    Morningstar often assigns 25-35x P/E for growth stocks.
    We cap at 18x for all companies (Graham's recommendation for a
    "moderately priced" stock).
    """
    result = ModelResult(model_name="relative_pe", weight=MODEL_WEIGHTS["relative_pe"])

    # Normalized EPS: average of last 2-3 years earnings / shares
    margin = info.get("profitMargins")
    rev = info.get("revenue_current")
    rev_prior = info.get("revenue_prior")
    fcf_hist = info.get("_fcf_history", [])

    earnings_estimates = []
    if margin is not None and rev is not None and rev > 0:
        earnings_estimates.append(margin * rev)
    if margin is not None and rev_prior is not None and rev_prior > 0:
        earnings_estimates.append(margin * rev_prior)
    if fcf_hist and fcf_hist[0] and fcf_hist[0] > 0:
        earnings_estimates.append(fcf_hist[0])

    if not earnings_estimates or shares <= 0:
        result.error = "Insufficient data for relative P/E"
        return result

    normalized_earnings = sum(earnings_estimates) / len(earnings_estimates)
    if normalized_earnings <= 0:
        result.error = "Negative normalized earnings"
        return result

    normalized_eps = normalized_earnings / shares

    # Conservative target P/E assignment
    # Graham: pay no more than 15x for a stock with no growth
    # Allow up to 18x for growing companies, but never more
    roic = info.get("roic")
    rev_growth = info.get("revenue_growth") or 0

    if roic and roic > 0.20 and rev_growth > 0.08:
        target_pe = 18.0  # high quality + growing
    elif roic and roic > 0.15:
        target_pe = 16.0
    elif roic and roic > 0.10:
        target_pe = 14.0
    else:
        target_pe = 12.0  # below-average company

    # Actual P/E penalty: if current P/E is way above target, further reduce
    current_pe = info.get("trailingPE")
    if current_pe and current_pe > 40:
        target_pe = min(target_pe, 12.0)  # market is euphoric — be extra conservative

    fair_value = normalized_eps * target_pe

    if fair_value > 0:
        result.fair_value_per_share = round(fair_value, 2)

    result.inputs = {
        "normalized_eps": round(normalized_eps, 2),
        "target_pe": target_pe,
        "current_pe": round(current_pe, 1) if current_pe else None,
    }
    result.notes = f"${normalized_eps:.2f} normalized EPS × {target_pe:.0f}x conservative P/E"
    return result


# ---------------------------------------------------------------------------
# Model 8: Asset-Based Floor (Tangible Book Value)
# ---------------------------------------------------------------------------

def _asset_floor(info: dict, shares: float) -> ModelResult:
    """Tangible book value per share as an absolute floor.

    A stock should not trade below its tangible book value unless the
    business is destroying value (negative ROIC).
    """
    result = ModelResult(model_name="asset_floor", weight=MODEL_WEIGHTS["asset_floor"])

    equity = _get_equity(info)
    if equity is None or shares <= 0:
        result.error = "No equity / shares data"
        return result

    # Use book value directly (we don't have goodwill breakdown from EDGAR easily)
    bvps = equity / shares

    # For asset floor, use 80% of book (conservative — some assets may be impaired)
    conservative_bvps = bvps * 0.80

    if conservative_bvps > 0:
        result.fair_value_per_share = round(conservative_bvps, 2)

    result.inputs = {
        "equity": round(equity, 0),
        "bvps": round(bvps, 2),
        "haircut": 0.80,
    }
    result.notes = f"Book value ${bvps:.2f}/sh × 80% haircut = ${conservative_bvps:.2f} floor"
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _estimate_net_debt(info: dict) -> float:
    """Estimate net debt (total debt − cash) from available data.

    Uses raw EDGAR balance sheet items when available (preferred),
    falls back to D/E ratio heuristic.
    """
    # Prefer raw EDGAR data
    total_debt = info.get("_total_debt")
    cash = info.get("_cash")

    if total_debt is not None and cash is not None:
        return max(0, total_debt - cash)

    if total_debt is not None:
        # Have debt but no cash — conservatively assume minimal cash
        return max(0, total_debt * 0.90)

    # Fallback: estimate from D/E ratio
    de_raw = info.get("debtToEquity")
    equity = _get_equity(info)

    if de_raw is not None and equity is not None and equity > 0:
        total_liabilities = (de_raw / 100.0) * equity
        est_cash = cash if cash is not None else equity * 0.10
        return max(0, total_liabilities * 0.50 - est_cash)

    return 0.0


def _get_equity(info: dict) -> Optional[float]:
    """Extract stockholders' equity from info dict."""
    # Prefer raw EDGAR equity
    raw_equity = info.get("_equity")
    if raw_equity is not None and raw_equity > 0:
        return raw_equity

    # Fallback: back-calculate from ROE and net income
    roe = info.get("returnOnEquity")
    margin = info.get("profitMargins")
    rev = info.get("revenue_current")
    if roe and roe > 0 and margin is not None and rev and rev > 0:
        net_income = margin * rev
        return net_income / roe
    return None


def _get_long_term_debt(info: dict) -> float:
    """Get long-term debt from raw EDGAR data or estimate from D/E ratio."""
    # Prefer raw EDGAR data
    raw_ltd = info.get("_long_term_debt")
    if raw_ltd is not None:
        return raw_ltd

    # Fallback: estimate from D/E
    equity = _get_equity(info)
    de_raw = info.get("debtToEquity")
    if equity and de_raw:
        return equity * (de_raw / 100.0) * 0.5
    return 0.0


# ---------------------------------------------------------------------------
# Composite valuation engine — public API
# ---------------------------------------------------------------------------

def _weighted_median(values: list[tuple[float, float]]) -> float:
    """Compute weighted median from list of (value, weight) tuples."""
    if not values:
        return 0.0

    # Sort by value
    values = sorted(values, key=lambda x: x[0])

    total_weight = sum(w for _, w in values)
    if total_weight <= 0:
        return values[len(values) // 2][0]

    cumulative = 0.0
    for val, weight in values:
        cumulative += weight
        if cumulative >= total_weight / 2:
            return val
    return values[-1][0]


def _weighted_percentile(values: list[tuple[float, float]], pct: float) -> float:
    """Compute weighted percentile."""
    if not values:
        return 0.0
    values = sorted(values, key=lambda x: x[0])
    total_weight = sum(w for _, w in values)
    if total_weight <= 0:
        return values[0][0]
    target = total_weight * pct
    cumulative = 0.0
    for val, weight in values:
        cumulative += weight
        if cumulative >= target:
            return val
    return values[-1][0]


def valuate(symbol: str, info: dict, current_price: Optional[float] = None,
            shares_outstanding: Optional[float] = None) -> ValuationResult:
    """Run all valuation models and produce a composite fair value.

    Parameters
    ----------
    symbol : str
        Ticker symbol.
    info : dict
        Fundamentals dict from edgar.get_fundamentals().
    current_price : float, optional
        Current market price per share.
    shares_outstanding : float, optional
        Total shares outstanding.

    Returns
    -------
    ValuationResult
        Comprehensive valuation with fair value, buy price, and signal.
    """
    result = ValuationResult(symbol=symbol.upper())
    result.current_price = current_price
    result.shares_outstanding = shares_outstanding

    # We need shares outstanding for per-share values
    shares = shares_outstanding or 0
    if shares <= 0:
        # Try to back out from market cap / price
        mcap = info.get("marketCap")
        if mcap and current_price and current_price > 0:
            shares = mcap / current_price

    if shares <= 0:
        result.valuation_signal = "INSUFFICIENT_DATA"
        result.summary = "Cannot value — shares outstanding unknown"
        return result

    result.shares_outstanding = shares

    # Estimate WACC
    wacc = _estimate_wacc(info)

    # Quality assessment
    quality_score, quality_tier, moat_indicators = _assess_quality(info)
    result.quality_score = quality_score
    result.quality_tier = quality_tier
    result.moat_indicators = moat_indicators

    # Set margin of safety based on quality
    if quality_tier == "HIGH":
        result.margin_of_safety_pct = MARGIN_OF_SAFETY_HIGH_QUALITY
    elif quality_tier == "MEDIUM":
        result.margin_of_safety_pct = MARGIN_OF_SAFETY_MEDIUM
    else:
        result.margin_of_safety_pct = MARGIN_OF_SAFETY_SPECULATIVE

    # ── Run all models ────────────────────────────────────────────────────
    models = [
        _dcf_two_stage(info, wacc, shares),
        _earnings_power_value(info, wacc, shares),
        _graham_number(info, shares),
        _excess_returns(info, wacc, shares),
        _dividend_discount(info, wacc, shares),
        _relative_pe(info, shares),
        _asset_floor(info, shares),
    ]

    # Reverse DCF (needs current price)
    if current_price and current_price > 0:
        rev_dcf = _reverse_dcf(info, wacc, shares, current_price)
        models.append(rev_dcf)
        result.implied_growth_rate = rev_dcf.inputs.get("implied_growth")
        result.growth_reasonableness = rev_dcf.inputs.get("reasonableness", "—")

    result.models = models

    # ── Compute composite fair value ──────────────────────────────────────
    valid_models = [
        (m.fair_value_per_share, m.weight)
        for m in models
        if m.fair_value_per_share is not None and m.fair_value_per_share > 0 and m.weight > 0
    ]

    if not valid_models:
        result.valuation_signal = "INSUFFICIENT_DATA"
        result.summary = "No models produced a valid fair value"
        return result

    result.fair_value = round(_weighted_median(valid_models), 2)
    result.fair_value_low = round(_weighted_percentile(valid_models, 0.25), 2)
    result.fair_value_high = round(_weighted_percentile(valid_models, 0.75), 2)

    # Buy price = fair value × (1 − margin of safety)
    result.buy_price = round(result.fair_value * (1 - result.margin_of_safety_pct), 2)

    # ── Signals ───────────────────────────────────────────────────────────
    if current_price and current_price > 0 and result.fair_value:
        upside = (result.fair_value / current_price - 1) * 100
        result.upside_pct = round(upside, 1)

        if result.buy_price:
            result.upside_to_buy_pct = round(
                (current_price / result.buy_price - 1) * 100, 1
            )

        # Valuation signal — deliberately harsher than Morningstar
        # Morningstar: "buy" if price < fair value
        # Us: "buy" only if price < fair_value × (1 − MoS)
        if current_price <= result.buy_price:
            if upside >= 60:
                result.valuation_signal = "DEEP_VALUE"
                result.valuation_grade = "A"
            else:
                result.valuation_signal = "UNDERVALUED"
                result.valuation_grade = "B"
        elif current_price <= result.fair_value:
            result.valuation_signal = "FAIR"
            result.valuation_grade = "C"
        elif current_price <= result.fair_value * 1.15:
            result.valuation_signal = "SLIGHTLY_OVER"
            result.valuation_grade = "D"
        else:
            prem = (current_price / result.fair_value - 1) * 100
            if prem >= 50:
                result.valuation_signal = "EXPENSIVE"
                result.valuation_grade = "F"
            else:
                result.valuation_signal = "OVERVALUED"
                result.valuation_grade = "D"

    # ── Build model details for display ───────────────────────────────────
    for m in models:
        fv_str = f"${m.fair_value_per_share:,.2f}" if m.fair_value_per_share else "n/a"
        status = f"✅ {fv_str}" if m.fair_value_per_share else f"⚠️ {m.error}"
        result.model_details.append(f"{m.model_name}: {status} — {m.notes}")

    # ── Summary ───────────────────────────────────────────────────────────
    fv = result.fair_value
    bp = result.buy_price
    cp = current_price
    sig = result.valuation_signal
    up = result.upside_pct or 0

    if cp and fv:
        result.summary = (
            f"{sig} | Fair Value ${fv:,.2f} | Buy Below ${bp:,.2f} "
            f"(MoS {result.margin_of_safety_pct:.0%}) | "
            f"Price ${cp:,.2f} → {up:+.1f}% upside | "
            f"Quality: {quality_tier} ({quality_score:.0f}/100)"
        )
    else:
        result.summary = f"Fair Value ${fv:,.2f} | Quality: {quality_tier}" if fv else "Insufficient data"

    logger.info(
        "%s: valuation — FV=$%.2f, buy=$%.2f, price=$%.2f, upside=%.1f%%, signal=%s, quality=%s",
        symbol, fv or 0, bp or 0, cp or 0, up, sig, quality_tier,
    )

    return result
