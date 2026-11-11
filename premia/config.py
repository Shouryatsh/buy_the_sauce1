"""
premia/config.py — every tunable number in one place.

Kept deliberately short (working rule: few tunable parameters). Anything chosen by walk-forward search is marked
[WF]; everything else is fixed a priori and documented with its source.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get("PREMIA_DB", ROOT / "premia_data.db"))
RESULTS_DIR = ROOT / "results" / "premia"
EXCLUSIONS_PATH = ROOT / "wheel_exclusions.json"

HISTORY_START = "2006-11-01"          # IBKR implied-vol history starts Nov 2006 for most large names
BACKTEST_START = "2010-01-01"         # leaves 3 years of warm-up for percentiles and model fits
TRADING_DAYS = 252

# ETFs always fetched besides the index members (SPY is the market proxy for beta tests)
ETFS = ["SPY", "QQQ", "IWM", "DIA", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "GLD", "TLT"]
# Yahoo index series: VIX family (SPX implied vol by tenor), CBOE SKEW, VVIX, 13-week T-bill yield (risk-free proxy)
INDEX_SERIES = ["^GSPC", "^VIX", "^VIX9D", "^VIX3M", "^VIX6M", "^SKEW", "^VVIX", "^IRX"]


@dataclass(frozen=True)
class Costs:
    """Trading costs used by the backtester and the edge model (IBKR tiered-ish, conservative)."""
    option_commission: float = 0.65        # $ per contract (IBKR Pro fixed, incl. typical exchange fees)
    stock_commission_per_share: float = 0.005
    stock_min_commission: float = 1.00
    option_half_spread_frac: float = 0.03  # half bid-ask as a fraction of mid when no quote exists (model chains)
    option_min_half_spread: float = 0.025  # $ per share: half of a 5-cent market
    slippage_frac_of_half_spread: float = 0.5   # sell at mid - 0.5 x half spread, buy at mid + 0.5 x half spread (1.0 = cross)
    stock_half_spread_bps: float = 2.0
    borrow_fee_annual: float = 0.0050      # cost of shorting stock for hedges (general collateral for liquid names)


@dataclass(frozen=True)
class Limits:
    """Hard portfolio limits (phase 7). No model can override these."""
    max_net_delta_pct_nav: float = 0.05       # |net dollar delta| <= 5 % of NAV
    max_net_vega_pct_nav: float = 0.004       # |net vega| per vol point <= 0.4 % of NAV
    max_net_vanna_pct_nav: float = 0.002      # $ delta change per vol point, as % of NAV
    max_net_volga_pct_nav: float = 0.002      # $ vega change per vol point, as % of NAV
    max_margin_use: float = 0.50              # cash-secured notional / NAV
    max_sector_weight: float = 0.30           # of short-put notional
    max_name_weight: float = 0.15
    max_loss_per_position_pct_nav: float = 0.02
    max_daily_loss_pct_nav: float = 0.03      # breach trips the kill switch
    max_drawdown_kill_pct: float = 0.15       # from the high-water mark, trips the kill switch
    min_option_open_interest: int = 500
    max_option_spread_frac: float = 0.10      # (ask - bid) / mid
    min_stock_dollar_volume: float = 50e6     # 20-day average, $ per day


@dataclass(frozen=True)
class Screen:
    """Phase 2 hard filters (interpretation documented in SPEC.md, section 'Conflicts and ambiguities')."""
    min_net_margin: float = 0.10              # TTM net income / TTM revenue, strict
    fcf_growth_midcap: float = 0.10           # TTM FCF YoY growth, strict, mid-caps
    fcf_growth_largecap: float = 0.05         # ... large caps
    fcf_increasing_years: int = 2             # TTM FCF higher than 4 and 8 quarters earlier
    midcap_min: float = 2e9                   # USD market cap, point in time
    largecap_min: float = 10e9


@dataclass(frozen=True)
class Params:
    """The few model parameters. [WF] = selected walk-forward on the training window only."""
    put_delta_grid: tuple = (0.10, 0.15, 0.20, 0.25, 0.30)   # [WF] candidate |delta| for short puts
    dte_grid: tuple = (21, 30, 45)                            # candidate days to expiry
    risk_aversion_grid: tuple = (0.0, 0.1, 0.25)              # [WF] lambda on CVaR95 in the wheel objective (edge/CVaR95 of
                                                              # short options is ~0.05-0.3, so lambda >~ 0.3 forbids nearly every trade)
    hedge_band_mult_grid: tuple = (0.5, 1.0, 2.0)             # [WF] x Whalley-Wilmott band width
    straddle_edge_grid: tuple = (0.0, 0.02, 0.04)             # [WF] min forecast-RV minus IV (vol points) to buy
    fhs_paths: int = 4000
    cvar_level: float = 0.95
    train_years: int = 3
    test_months: int = 6


COSTS = Costs()
LIMITS = Limits()
SCREEN = Screen()
PARAMS = Params()


def load_exclusions() -> set[str]:
    """Manual exclusions are absolute: these tickers are never researched as candidates or traded."""
    import json
    try:
        return {s.strip().upper() for s in json.loads(EXCLUSIONS_PATH.read_text())}
    except (OSError, ValueError):
        return set()
