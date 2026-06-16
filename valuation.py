"""
valuation.py — Multi-model, realistic intrinsic-value engine.

Implements a comprehensive valuation framework inspired by:
  - Aswath Damodaran's "The Little Book of Valuation"
  - Damodaran's "Investment Valuation" (3rd ed.)
  - Greenwald's "Value Investing: From Graham to Buffett"
  - Penman's "Accounting for Value"
  - Graham's "The Intelligent Investor" (margin-of-safety concept)

Design Principles
-----------------
  1. **Realistic, not punitive**: Each model uses a single layer of conservatism
     rather than stacking multiple haircuts that compound unrealistically.
  2. **Multiple independent models**: No single model drives the output.
     The final fair value is a WEIGHTED MEDIAN of all models.
  3. **Margin of safety**: The "buy price" is fair_value × (1 − margin_of_safety).
     MoS = 20% HIGH / 30% MEDIUM / 40% SPECULATIVE — applied once on a realistic base.
  4. **Transparent**: Every model's output and inputs are returned for display.

Models Implemented
------------------
  M1. Three-Stage DCF (Morningstar style)  — High growth → fade → terminal
  M2. Two-Stage DCF (Damodaran style)      — Conservative FCF growth
  M3. Reverse DCF                          — "What growth is the market pricing in?"
  M4. Earnings Power Value (Greenwald)     — Value of current earnings, no growth
  M5. Graham Number                        — Graham's classic formula
  M6. Excess Returns / Residual Income     — Penman/Damodaran: only value returns > WACC
  M7. Dividend Discount Model              — Gordon Growth Model with conservative payout
  M8. Relative Valuation (sector P/E)      — Conservative sector-relative multiple
  M9. Asset-based (Book Value floor)       — Tangible book as absolute floor

Composite Output
----------------
  - Fair value: weighted median of all valid models
  - Buy price: fair_value × (1 − margin_of_safety)
  - Zacks Rank: 1–5 based on EPS trends, surprises, momentum (analyst-style)
  - Quality score: 0–100 based on profitability, growth, financial strength
  - Valuation signal: DEEP_VALUE / UNDERVALUED / FAIR / SLIGHTLY_OVER / OVERVALUED / EXPENSIVE

Data Source: SEC EDGAR via edgar.py (no API key, no scraping)
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field
from typing import Optional

import config as _cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration — deliberately realistic defaults
# ---------------------------------------------------------------------------

# WACC / discount rate assumptions
DEFAULT_RISK_FREE_RATE: float = 0.043       # ~10Y US Treasury yield
DEFAULT_EQUITY_RISK_PREMIUM: float = 0.055  # Damodaran's 2024 ERP for US
DEFAULT_BETA_FLOOR: float = 0.8             # never assume beta < 0.8
DEFAULT_BETA_CAP: float = 2.0               # cap at 2.0
DEFAULT_COST_OF_DEBT: float = 0.055         # after-tax cost of debt
DEFAULT_TAX_RATE: float = 0.21              # US corporate tax rate
WACC_FLOOR: float = 0.08                    # floor at 8%
WACC_CAP: float = 0.16                      # cap at 16%

# Growth assumptions — realistic but cautious
MAX_HIGH_GROWTH_RATE: float = 0.15          # cap at 15%
HIGH_GROWTH_HAIRCUT: float = 0.70           # use 70% of trailing growth
HIGH_GROWTH_YEARS: int = 7                  # 7 years of above-average growth
TERMINAL_GROWTH_RATE: float = 0.025         # long-run GDP growth (2.5%)
TERMINAL_GROWTH_CAP: float = 0.03           # terminal growth never exceeds 3%
MIN_HIGH_GROWTH_RATE: float = 0.0

# FCF conversion
FCF_TO_REVENUE_FLOOR: float = 0.03          # assume at least 3% FCF margin
FCF_TO_REVENUE_CAP: float = 0.30            # cap at 30% FCF margin

# Margin of safety (Buffett/Graham concept)
MARGIN_OF_SAFETY_HIGH_QUALITY: float = 0.20  # 20% MoS for best companies
MARGIN_OF_SAFETY_MEDIUM: float = 0.30        # 30% MoS for average companies
MARGIN_OF_SAFETY_SPECULATIVE: float = 0.40   # 40% MoS for low-quality

# Model weights for composite valuation
MODEL_WEIGHTS = {
    "dcf_three_stage":    3.0,   # Morningstar-style 3-stage DCF
    "dcf_two_stage":      2.0,   # classic DCF
    "reverse_dcf":        1.0,   # diagnostic sanity check
    "epv":                1.5,   # zero-growth earnings power
    "graham_number":      1.5,   # Graham's original formula
    "excess_returns":     2.0,   # residual income
    "ddm":                1.5,   # dividend/FCF yield model
    "relative_pe":        2.0,   # sector-relative multiple
    "asset_floor":        0.5,   # tangible book floor
}

# Zacks Rank mapping
ZACKS_RANK_LABELS = {
    1: "Strong Buy",
    2: "Buy",
    3: "Hold",
    4: "Sell",
    5: "Strong Sell",
}

# Quality thresholds
QUALITY_HIGH_THRESHOLD: float = 65.0
QUALITY_MEDIUM_THRESHOLD: float = 35.0


# ---------------------------------------------------------------------------
# Data Classes
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
    margin_of_safety_pct: float = 0.30
    buy_price: Optional[float] = None
    upside_pct: Optional[float] = None
    upside_to_buy_pct: Optional[float] = None
    valuation_signal: str = "HOLD"
    valuation_grade: str = "—"

    # Quality assessment
    quality_score: float = 50.0
    quality_tier: str = "MEDIUM"
    moat_indicators: list[str] = field(default_factory=list)

    # Zacks Rank
    zacks_rank: int = 3
    zacks_label: str = "Hold"

    # Reverse DCF diagnostic
    implied_growth_rate: Optional[float] = None
    growth_reasonableness: str = "—"

    # Display
    summary: str = ""
    model_details: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _estimate_net_debt(info: dict) -> float:
    """Estimate net debt = total debt - cash."""
    # Try new-style keys first
    cash = info.get("cash")
    debt_current = info.get("currentDebt") or 0.0
    debt_long = info.get("longTermDebt") or 0.0
    total_debt = debt_current + debt_long
    
    # Fallback to old-style keys for testing
    if not total_debt and "_total_debt" in info:
        total_debt = info.get("_total_debt") or 0.0
    
    if cash is None:
        # Try old-style key
        cash = info.get("_cash")
    
    if cash is None:
        cash = 0.0
    
    net_debt = max(0.0, total_debt - cash)
    return net_debt


def _get_equity(info: dict) -> float:
    """Get total shareholders' equity."""
    equity = info.get("totalStockholderEquity")
    if equity is None:
        # Try old-style key
        equity = info.get("_equity")
    
    if equity is None:
        # Fallback: assets - liabilities
        assets = info.get("totalAssets")
        liabilities = info.get("totalLiabilities")
        if assets and liabilities:
            equity = assets - liabilities
        else:
            equity = 1.0  # fallback to avoid division by zero
    return equity or 1.0


def _get_long_term_debt(info: dict) -> float:
    """Get long-term debt."""
    return info.get("longTermDebt") or 0.0


def _estimate_wacc(info: dict) -> float:
    """Estimate WACC from available fundamentals.

    WACC = (E/V)×Re + (D/V)×Rd×(1−T)
    where Re = Rf + β×ERP  (CAPM)
    """
    rf = DEFAULT_RISK_FREE_RATE
    erp = DEFAULT_EQUITY_RISK_PREMIUM

    # Estimate beta
    beta = info.get("beta")
    if beta is None or beta <= 0:
        beta = 1.0
    beta = max(DEFAULT_BETA_FLOOR, min(DEFAULT_BETA_CAP, beta))

    # Cost of equity (CAPM)
    re = rf + beta * erp

    # Cost of debt
    rd = DEFAULT_COST_OF_DEBT
    tax_rate = DEFAULT_TAX_RATE

    # Market values (use book values as proxy if market values unavailable)
    equity_value = info.get("marketCap")
    debt_value = _estimate_net_debt(info)

    if equity_value is None or equity_value <= 0:
        # Fallback: use book value
        equity_value = _get_equity(info)

    if equity_value and equity_value > 0 and debt_value and debt_value > 0:
        total_value = equity_value + debt_value
        we = equity_value / total_value
        wd = debt_value / total_value
        wacc = we * re + wd * rd * (1 - tax_rate)
    else:
        # All equity — just use cost of equity
        wacc = re

    # Apply floor and cap
    wacc = max(WACC_FLOOR, min(WACC_CAP, wacc))
    return wacc


def _assess_quality(info: dict) -> tuple[float, str, list[str]]:
    """Composite quality score 0–100 based on profitability, growth, financial strength."""
    score = 50.0  # base score
    moat_indicators = []

    # ROE (return on equity)
    roe = info.get("returnOnEquity")
    if roe is not None:
        if roe > 0.25:
            score += 15
            moat_indicators.append(f"Exceptional ROE {roe:.0%}")
        elif roe > 0.15:
            score += 10
            moat_indicators.append(f"Strong ROE {roe:.0%}")
        elif roe > 0.10:
            score += 5
            moat_indicators.append(f"Adequate ROE {roe:.0%}")
        elif roe > 0:
            score += 0
        else:
            score -= 15
            moat_indicators.append(f"Negative ROE {roe:.0%}")

    # Profit margin (FCF margin preferred)
    fcf_hist = info.get("_fcf_history", [])
    revenue = info.get("revenue_current")
    if fcf_hist and fcf_hist[0] and revenue and revenue > 0:
        fcf_margin = fcf_hist[0] / revenue
        if fcf_margin > 0.20:
            score += 8
            moat_indicators.append(f"Excellent FCF margin {fcf_margin:.0%}")
        elif fcf_margin > 0.10:
            score += 5
        elif fcf_margin > 0.05:
            score += 2
        elif fcf_margin < 0:
            score -= 10

    # Debt/Equity
    de = info.get("debtToEquity")
    if de is not None:
        de_ratio = de / 100.0 if de > 1 else de
        if de_ratio < 0.5:
            score += 8
            moat_indicators.append("Conservative leverage")
        elif de_ratio < 1.0:
            score += 4
        elif de_ratio > 3.0:
            score -= 12

    # Growth consistency
    rev_growth = info.get("revenue_growth")
    if rev_growth is not None:
        if rev_growth > 0.15:
            score += 8
            moat_indicators.append(f"Strong revenue growth {rev_growth:.0%}")
        elif rev_growth > 0.10:
            score += 5
        elif rev_growth > 0.05:
            score += 2
        elif rev_growth < -0.05:
            score -= 8

    # FCF history (consecutive years of growth)
    if len(fcf_hist) >= 3:
        growth_count = sum(1 for i in range(len(fcf_hist) - 1) if fcf_hist[i] > fcf_hist[i+1])
        if growth_count >= 2:
            score += 5
            moat_indicators.append("FCF growing consistently")

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


def _zacks_rank(info: dict) -> tuple[int, str]:
    """Compute Zacks-style rank (1–5) based on EPS trends, surprises, momentum.

    Zacks Rank integrates:
      1. EPS revisions (analyst consensus trending up/down)
      2. EPS surprises (actual beats/misses)
      3. Estimate momentum (revisions accelerating)
      4. Relative valuation (P/E vs. growth)

    Returns: (rank, label)
    """
    rank_score = 50  # 0–100 scale; 50 = neutral (Hold)

    # EPS revisions (direction: are analysts raising or lowering estimates?)
    eps_revision = info.get("epsRevisions")  # e.g., "up 5%", "down 2%"
    if eps_revision:
        # Simplistic: if positive, bump score; if negative, reduce
        # In production, parse the % value
        if isinstance(eps_revision, str):
            if "up" in eps_revision.lower():
                rank_score += 15
            elif "down" in eps_revision.lower():
                rank_score -= 15

    # EPS surprises (actual beats historical expectation)
    eps_surprise = info.get("epsTrailingTwelveMonths")  # e.g., 3.45
    eps_estimate = info.get("epsCurrentYear")  # e.g., 3.50
    if eps_surprise and eps_estimate and eps_estimate > 0:
        surprise_pct = (eps_surprise - eps_estimate) / eps_estimate
        if surprise_pct > 0.05:
            rank_score += 10
        elif surprise_pct < -0.05:
            rank_score -= 10

    # Estimate momentum: forward estimates vs. trailing
    if eps_estimate and eps_surprise:
        if eps_estimate > eps_surprise:
            rank_score += 5  # forward growth expected
        else:
            rank_score -= 5

    # Relative valuation: P/E vs. growth (PEG)
    pe = info.get("trailingPE")
    growth = info.get("revenue_growth")
    if pe and pe > 0 and growth and growth > 0:
        peg = pe / (growth * 100)
        if peg < 1.0:
            rank_score += 8
        elif peg > 2.0:
            rank_score -= 8

    # Momentum: relative to 52-week highs/lows (price action)
    price_52w_high = info.get("fiftyTwoWeekHigh")
    price_52w_low = info.get("fiftyTwoWeekLow")
    current_price = info.get("currentPrice")
    if price_52w_high and price_52w_low and current_price:
        momentum = (current_price - price_52w_low) / (price_52w_high - price_52w_low)
        if momentum > 0.7:
            rank_score += 10
        elif momentum < 0.3:
            rank_score -= 10

    # Clamp and convert to Zacks rank (1–5)
    rank_score = max(0, min(100, rank_score))
    
    # 0–20: Strong Buy (1)
    # 20–40: Buy (2)
    # 40–60: Hold (3)
    # 60–80: Sell (4)
    # 80–100: Strong Sell (5)
    if rank_score < 20:
        zacks_rank = 1
    elif rank_score < 40:
        zacks_rank = 2
    elif rank_score < 60:
        zacks_rank = 3
    elif rank_score < 80:
        zacks_rank = 4
    else:
        zacks_rank = 5

    label = ZACKS_RANK_LABELS.get(zacks_rank, "Hold")
    return zacks_rank, label


def _weighted_median(weighted_values: list[tuple[float, float]]) -> float:
    """Calculate weighted median of values."""
    if not weighted_values:
        return 0.0
    
    # Sort by value
    sorted_vals = sorted(weighted_values, key=lambda x: x[0])
    total_weight = sum(w for _, w in sorted_vals)
    
    if total_weight <= 0:
        return 0.0
    
    cumulative = 0.0
    target = total_weight / 2.0
    
    for val, weight in sorted_vals:
        cumulative += weight
        if cumulative >= target:
            return val
    
    return sorted_vals[-1][0]


def _weighted_percentile(weighted_values: list[tuple[float, float]], percentile: float) -> float:
    """Calculate weighted percentile of values (0.0–1.0)."""
    if not weighted_values:
        return 0.0
    
    sorted_vals = sorted(weighted_values, key=lambda x: x[0])
    total_weight = sum(w for _, w in sorted_vals)
    
    if total_weight <= 0:
        return 0.0
    
    cumulative = 0.0
    target = total_weight * percentile
    
    for val, weight in sorted_vals:
        cumulative += weight
        if cumulative >= target:
            return val
    
    return sorted_vals[-1][0]


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
# Model 1: Three-Stage DCF (Morningstar Style)
# ---------------------------------------------------------------------------

def _dcf_three_stage(info: dict, wacc: float, shares: float) -> ModelResult:
    """Morningstar-style 3-stage DCF: high growth → fade → terminal."""
    result = ModelResult(model_name="dcf_three_stage", weight=MODEL_WEIGHTS["dcf_three_stage"])

    fcf_hist = info.get("_fcf_history", [])
    if not fcf_hist or fcf_hist[0] is None:
        result.error = "No FCF data"
        return result

    current_fcf = fcf_hist[0]
    if current_fcf <= 0:
        result.error = f"Negative FCF"
        return result

    # Stage 1: High growth (5 years)
    analyst_growth = info.get("analyst_fcf_growth")
    sector_growth = info.get("sector_fcf_growth")
    
    trailing_cagr = None
    if len(fcf_hist) >= 3 and fcf_hist[-1] > 0:
        n_years = len(fcf_hist) - 1
        trailing_cagr = (fcf_hist[0] / fcf_hist[-1]) ** (1.0 / n_years) - 1
    elif len(fcf_hist) >= 2 and fcf_hist[1] > 0:
        trailing_cagr = (fcf_hist[0] / fcf_hist[1]) - 1
    else:
        trailing_cagr = 0.07

    high_growth = (
        analyst_growth if analyst_growth is not None else
        sector_growth if sector_growth is not None else
        trailing_cagr if trailing_cagr is not None else
        0.07
    )
    high_growth = max(0.0, min(0.18, high_growth))  # cap at 18%
    
    fade_growth = max(0.03, min(0.08, high_growth * 0.5))  # fade to 3–8%
    terminal_growth = min(TERMINAL_GROWTH_RATE, 0.03)

    stage1_years = 5
    stage2_years = 5

    # Stage 1: Project FCFs
    stage1_pv = 0.0
    projected_fcf = current_fcf
    for year in range(1, stage1_years + 1):
        projected_fcf *= (1 + high_growth)
        pv = projected_fcf / (1 + wacc) ** year
        stage1_pv += pv

    # Stage 2: Fade (linear fade from high_growth to fade_growth)
    stage2_pv = 0.0
    start_growth = high_growth
    for year in range(1, stage2_years + 1):
        # Linear fade
        progress = year / stage2_years
        stage_growth = start_growth + (fade_growth - start_growth) * progress
        projected_fcf *= (1 + stage_growth)
        pv = projected_fcf / (1 + wacc) ** (stage1_years + year)
        stage2_pv += pv

    # Stage 3: Terminal value
    terminal_fcf = projected_fcf * (1 + terminal_growth)
    terminal_value = terminal_fcf / (wacc - terminal_growth)
    stage3_pv = terminal_value / (1 + wacc) ** (stage1_years + stage2_years)

    ev = stage1_pv + stage2_pv + stage3_pv
    net_debt = _estimate_net_debt(info)
    equity_value = ev - net_debt

    if shares > 0 and equity_value > 0:
        result.fair_value_per_share = round(equity_value / shares, 2)

    result.inputs = {
        "current_fcf": round(current_fcf, 0),
        "stage1_growth": high_growth,
        "stage2_fade_to": fade_growth,
        "terminal_growth": terminal_growth,
        "wacc": wacc,
        "ev": round(ev, 0),
    }
    result.notes = f"Stage1: {high_growth:.0%}, Stage2→{fade_growth:.0%}, Terminal: {terminal_growth:.0%}"
    
    return result


# ---------------------------------------------------------------------------
# Model 2: Two-Stage DCF (Damodaran)
# ---------------------------------------------------------------------------

def _dcf_two_stage(info: dict, wacc: float, shares: float) -> ModelResult:
    """Traditional 2-stage DCF: explicit forecast + terminal value."""
    result = ModelResult(model_name="dcf_two_stage", weight=MODEL_WEIGHTS["dcf_two_stage"])

    fcf_hist = info.get("_fcf_history", [])
    if not fcf_hist or fcf_hist[0] is None:
        result.error = "No FCF data"
        return result

    current_fcf = fcf_hist[0]
    if current_fcf <= 0:
        result.error = "Negative FCF"
        return result

    # Estimate high-growth rate
    trailing_cagr = 0.07
    if len(fcf_hist) >= 3 and fcf_hist[-1] and fcf_hist[-1] > 0:
        n_years = len(fcf_hist) - 1
        trailing_cagr = (fcf_hist[0] / fcf_hist[-1]) ** (1.0 / n_years) - 1
    
    high_growth = max(0.0, min(MAX_HIGH_GROWTH_RATE, trailing_cagr * HIGH_GROWTH_HAIRCUT))
    terminal_growth = min(TERMINAL_GROWTH_RATE, 0.03)

    # High-growth phase
    pv_hg = 0.0
    projected_fcf = current_fcf
    explicit_years = 10
    for year in range(1, explicit_years + 1):
        projected_fcf *= (1 + high_growth)
        pv_hg += projected_fcf / (1 + wacc) ** year

    # Terminal value
    terminal_fcf = projected_fcf * (1 + terminal_growth)
    terminal_value = terminal_fcf / (wacc - terminal_growth)
    pv_terminal = terminal_value / (1 + wacc) ** explicit_years

    ev = pv_hg + pv_terminal
    net_debt = _estimate_net_debt(info)
    equity_value = ev - net_debt

    if shares > 0 and equity_value > 0:
        result.fair_value_per_share = round(equity_value / shares, 2)

    result.inputs = {
        "fcf_current": round(current_fcf, 0),
        "growth": high_growth,
        "terminal_growth": terminal_growth,
        "wacc": wacc,
        "ev": round(ev, 0),
    }
    result.notes = f"10-yr explicit, {high_growth:.0%} growth"
    
    return result


# ---------------------------------------------------------------------------
# Model 3: Reverse DCF
# ---------------------------------------------------------------------------

def _reverse_dcf(info: dict, wacc: float, shares: float, current_price: float) -> ModelResult:
    """Reverse DCF: 'What growth rate is the market pricing in?'"""
    result = ModelResult(model_name="reverse_dcf", weight=MODEL_WEIGHTS["reverse_dcf"])

    if not current_price or current_price <= 0:
        result.error = "No current price"
        return result

    fcf_hist = info.get("_fcf_history", [])
    if not fcf_hist or not fcf_hist[0] or fcf_hist[0] <= 0:
        result.error = "No FCF data"
        return result

    current_fcf = fcf_hist[0]
    shares_valid = shares if shares and shares > 0 else 1.0
    market_cap = current_price * shares_valid
    net_debt = _estimate_net_debt(info)
    ev_market = market_cap + net_debt

    # Find implied growth via binary search
    terminal_growth = min(TERMINAL_GROWTH_RATE, 0.03)
    
    low_growth = -0.05
    high_growth = 0.30
    tolerance = 0.0001
    
    for _ in range(50):  # max iterations
        mid_growth = (low_growth + high_growth) / 2.0
        ev_calc = _calc_ev_from_growth(current_fcf, mid_growth, terminal_growth, wacc)
        
        if ev_calc < ev_market:
            low_growth = mid_growth
        else:
            high_growth = mid_growth
        
        if abs(high_growth - low_growth) < tolerance:
            break
    
    implied_growth = (low_growth + high_growth) / 2.0
    implied_growth = max(-0.05, min(0.25, implied_growth))

    # Assess reasonableness
    if implied_growth > 0.20:
        reasonableness = "HEROIC"
    elif implied_growth > 0.12:
        reasonableness = "OPTIMISTIC"
    else:
        reasonableness = "REASONABLE"

    # Use current price as fair value (it's what the market says)
    result.fair_value_per_share = round(current_price, 2)
    
    result.inputs = {
        "implied_growth": implied_growth,
        "reasonableness": reasonableness,
        "current_price": current_price,
        "wacc": wacc,
    }
    result.notes = f"Market implies {implied_growth:.1%} growth — {reasonableness}"
    
    return result


# ---------------------------------------------------------------------------
# Model 4: Earnings Power Value (Greenwald)
# ---------------------------------------------------------------------------

def _earnings_power_value(info: dict, wacc: float, shares: float) -> ModelResult:
    """Greenwald's EPV: value of current earnings with zero growth."""
    result = ModelResult(model_name="epv", weight=MODEL_WEIGHTS["epv"])

    fcf_hist = info.get("_fcf_history", [])
    if fcf_hist and len(fcf_hist) >= 2:
        avg_earnings = sum(f for f in fcf_hist[:3] if f) / min(3, len(fcf_hist))
    elif fcf_hist and fcf_hist[0] and fcf_hist[0] > 0:
        avg_earnings = fcf_hist[0]
    else:
        ni = info.get("_net_income")
        if ni and ni > 0:
            avg_earnings = ni
        else:
            result.error = "Insufficient earnings data"
            return result

    if avg_earnings <= 0:
        result.error = "Negative normalized earnings"
        return result

    epv_enterprise = avg_earnings / wacc
    net_debt = _estimate_net_debt(info)
    equity_value = epv_enterprise - net_debt

    if shares > 0 and equity_value > 0:
        result.fair_value_per_share = round(equity_value / shares, 2)

    result.inputs = {
        "normalized_earnings": round(avg_earnings, 0),
        "wacc": wacc,
        "epv_enterprise": round(epv_enterprise, 0),
    }
    result.notes = f"Zero-growth valuation"
    
    return result


# ---------------------------------------------------------------------------
# Model 5: Graham Number
# ---------------------------------------------------------------------------

def _graham_number(info: dict, shares: float) -> ModelResult:
    """Graham Number: sqrt(22.5 × EPS × book_value_per_share)."""
    result = ModelResult(model_name="graham_number", weight=MODEL_WEIGHTS["graham_number"])

    eps = info.get("trailingEPS")
    if not eps or eps <= 0:
        result.error = "No EPS data"
        return result

    equity = _get_equity(info)
    bvps = equity / shares if shares > 0 else 0.0
    
    if bvps <= 0:
        result.error = "No book value"
        return result

    graham_value = math.sqrt(22.5 * eps * bvps)
    
    if shares > 0:
        result.fair_value_per_share = round(graham_value, 2)

    result.inputs = {
        "eps": eps,
        "book_value_per_share": round(bvps, 2),
    }
    result.notes = f"Graham: sqrt(22.5 × {eps:.2f} × {bvps:.2f})"
    
    return result


# ---------------------------------------------------------------------------
# Model 6: Excess Returns / Residual Income
# ---------------------------------------------------------------------------

def _excess_returns(info: dict, wacc: float, shares: float) -> ModelResult:
    """Penman-style Residual Income: PV of excess returns above WACC."""
    result = ModelResult(model_name="excess_returns", weight=MODEL_WEIGHTS["excess_returns"])

    equity = _get_equity(info)
    fcf_hist = info.get("_fcf_history", [])
    
    if not fcf_hist or not fcf_hist[0] or fcf_hist[0] <= 0:
        result.error = "No FCF data"
        return result

    current_fcf = fcf_hist[0]
    roi = current_fcf / equity if equity > 0 else 0.0
    excess_return = max(0.0, roi - wacc)  # only count returns above WACC
    
    if excess_return <= 0:
        # Company is not earning above its cost of capital
        result.error = "ROI below WACC"
        return result

    # PV of excess returns
    pv_excess = (current_fcf * excess_return / wacc) / (1 + wacc)
    pv_base = equity  # book value
    equity_value = pv_base + pv_excess
    
    if shares > 0:
        result.fair_value_per_share = round(equity_value / shares, 2)

    result.inputs = {
        "equity": round(equity, 0),
        "roi": roi,
        "excess_return": excess_return,
    }
    result.notes = f"ROI {roi:.0%}, Excess {excess_return:.0%}"
    
    return result


# ---------------------------------------------------------------------------
# Model 7: Dividend Discount Model (Gordon Growth)
# ---------------------------------------------------------------------------

def _dividend_discount(info: dict, wacc: float, shares: float) -> ModelResult:
    """Gordon Growth Model: value of dividend/FCF stream."""
    result = ModelResult(model_name="ddm", weight=MODEL_WEIGHTS["ddm"])

    # Use FCF yield as proxy for dividend
    fcf_hist = info.get("_fcf_history", [])
    market_cap = info.get("marketCap")
    
    if not fcf_hist or not fcf_hist[0] or fcf_hist[0] <= 0 or not market_cap or market_cap <= 0:
        result.error = "No FCF or market cap"
        return result

    current_fcf = fcf_hist[0]
    fcf_yield = current_fcf / market_cap
    
    # Terminal growth
    terminal_growth = min(TERMINAL_GROWTH_RATE, 0.03)
    
    if wacc <= terminal_growth:
        result.error = "WACC ≤ terminal growth"
        return result

    # Intrinsic value per share = (current_yield × price) / (wacc - g)
    # Simplifies to: equity_value = (fcf / (wacc - g))
    equity_value = current_fcf / (wacc - terminal_growth)
    
    if shares > 0:
        result.fair_value_per_share = round(equity_value / shares, 2)

    result.inputs = {
        "fcf_current": round(current_fcf, 0),
        "fcf_yield": round(fcf_yield, 4),
        "terminal_growth": terminal_growth,
        "wacc": wacc,
    }
    result.notes = f"FCF yield {fcf_yield:.2%}, {terminal_growth:.0%} growth"
    
    return result


# ---------------------------------------------------------------------------
# Model 8: Relative PE (Sector Comparison)
# ---------------------------------------------------------------------------

def _relative_pe(info: dict, shares: float) -> ModelResult:
    """Sector-relative P/E: conservative median-sector P/E × normalized EPS."""
    result = ModelResult(model_name="relative_pe", weight=MODEL_WEIGHTS["relative_pe"])

    # Use sector median P/E (fallback to S&P 500 median ~21)
    sector_pe = info.get("sector_pe")
    if not sector_pe or sector_pe <= 0:
        sector_pe = 20.0  # conservative default

    eps = info.get("trailingEPS")
    if not eps or eps <= 0:
        result.error = "No EPS data"
        return result

    # Conservative: use 70% of sector P/E
    conservative_pe = sector_pe * 0.70
    equity_value = conservative_pe * eps * shares if shares > 0 else 0.0
    
    if shares > 0 and equity_value > 0:
        result.fair_value_per_share = round(equity_value / shares, 2)

    result.inputs = {
        "sector_pe": round(sector_pe, 2),
        "conservative_pe": round(conservative_pe, 2),
        "eps": eps,
    }
    result.notes = f"Sector P/E {sector_pe:.1f}x, using {conservative_pe:.1f}x"
    
    return result


# ---------------------------------------------------------------------------
# Model 9: Asset Floor (Book Value)
# ---------------------------------------------------------------------------

def _asset_floor(info: dict, shares: float) -> ModelResult:
    """Asset-based floor: tangible book value per share."""
    result = ModelResult(model_name="asset_floor", weight=MODEL_WEIGHTS["asset_floor"])

    equity = _get_equity(info)
    if equity <= 0:
        result.error = "No equity data"
        return result

    # Adjust for intangibles (goodwill, patents)
    goodwill = info.get("goodwill") or 0.0
    intangibles = info.get("intangibleAssets") or 0.0
    tangible_equity = equity - goodwill - intangibles
    
    if tangible_equity <= 0:
        # Use raw equity if tangible is negative
        tangible_equity = equity * 0.5  # conservative haircut

    if shares > 0:
        result.fair_value_per_share = round(tangible_equity / shares, 2)

    result.inputs = {
        "equity": round(equity, 0),
        "goodwill": round(goodwill, 0),
        "tangible_equity": round(tangible_equity, 0),
    }
    result.notes = f"Tangible book value floor"
    
    return result


# ---------------------------------------------------------------------------
# Main Valuation Function
# ---------------------------------------------------------------------------

def valuate(
    symbol: str,
    info: dict,
    current_price: Optional[float] = None,
    shares_outstanding: Optional[float] = None
) -> ValuationResult:
    """Run all valuation models and produce a composite fair value, plus Zacks-style rank."""
    result = ValuationResult(symbol=symbol.upper())
    result.current_price = current_price
    result.shares_outstanding = shares_outstanding

    # Estimate shares outstanding if needed
    shares = shares_outstanding or 0
    if shares <= 0:
        mcap = info.get("marketCap")
        if mcap and current_price and current_price > 0:
            shares = mcap / current_price
        else:
            shares = 1.0  # fallback

    # Estimate WACC
    wacc = _estimate_wacc(info)

    # Quality assessment
    quality_score, quality_tier, moat_indicators = _assess_quality(info)
    result.quality_score = quality_score
    result.quality_tier = quality_tier
    result.moat_indicators = moat_indicators

    # Zacks Rank
    zacks_rank, zacks_label = _zacks_rank(info)
    result.zacks_rank = zacks_rank
    result.zacks_label = zacks_label

    # Set margin of safety based on quality
    if quality_tier == "HIGH":
        result.margin_of_safety_pct = MARGIN_OF_SAFETY_HIGH_QUALITY
    elif quality_tier == "MEDIUM":
        result.margin_of_safety_pct = MARGIN_OF_SAFETY_MEDIUM
    else:
        result.margin_of_safety_pct = MARGIN_OF_SAFETY_SPECULATIVE

    # ── Run all models ────────────────────────────────────────────────────
    models = [
        _dcf_three_stage(info, wacc, shares),
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

        # Valuation signal — Morningstar style
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
            f"Quality: {quality_tier} ({quality_score:.0f}/100) | "
            f"Zacks: {zacks_rank} ({zacks_label})"
        )
    else:
        result.summary = f"Fair Value ${fv:,.2f} | Quality: {quality_tier} | Zacks: {zacks_rank} ({zacks_label})" if fv else "Insufficient data"

    logger.info(
        "%s: valuation — FV=$%.2f, buy=$%.2f, price=$%.2f, upside=%.1f%%, signal=%s, quality=%s, zacks=%s",
        symbol, fv or 0, bp or 0, cp or 0, up, sig, quality_tier, zacks_label,
    )

    return result
