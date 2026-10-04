#!/usr/bin/env python3
"""
wheel_dashboard.py — Options Wheel Strategy Dashboard  (~$50 K portfolio)

Strategy
--------
  Phase 1 — Cash-Secured Put (CSP):
    Sell far-OTM puts (delta ≤ 0.05) on quality stocks near their 52-week
    low.  Collect premium.  Let expire worthless.

  Phase 2 — Covered Call (CC):
    If assigned shares, sell ATM / slightly-OTM calls to earn more premium
    and exit the position at a profit.

Screening Criteria (all must pass)
------------------------------------
  • Net profit margin  > 15 %
  • Free cash flow     > 0  (positive)
  • Debt / assets      < 50 %
  • ROE                ≥ 10 % (moat proxy)
  • Weekly options available  (≥ 3 expirations within 35 days)
  • Wide moat score    ≥ 3 / 5  (based on margin, ROE, FCF, debt)

Entry logic
-----------
  • Stock within 20 % of its 52-week low  (limited downside)
  • Sell the put with the HIGHEST annualised yield where delta ≤ 0.05
    → scans all weekly expirations up to 45 DTE

Portfolio
---------
  Cash available : $50 000 – $60 000
  Monthly target : $2 000 – $4 000

Tabs
----
  1. 📋 Watchlist       Editable candidate list — add / remove tickers
  2. 🔍 Screening       Fundamental quality filter + 52 W-low proximity
  3. 🎯 Wheel Opps      Best CSP per eligible stock (delta ≤ 0.05)
  4. 📂 Open Positions  Track active CSPs & CCs; close / add trades
  5. 📊 P&L Summary     Monthly premium income vs $2 K–$4 K target

Usage
-----
  python3 wheel_dashboard.py              # http://127.0.0.1:8051
  python3 wheel_dashboard.py --port 8052
  python3 wheel_dashboard.py --no-browser
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import threading
import time
import uuid
import warnings
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

import dash
from dash import Input, Output, State, ctx, dash_table, dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("wheel")

# ── Paths ──────────────────────────────────────────────────────────────────
HERE           = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR    = os.path.join(HERE, "results")
WATCHLIST_FILE = os.path.join(HERE, "wheel_watchlist.json")
POSITIONS_FILE = os.path.join(RESULTS_DIR, "wheel_positions.csv")
VALUATIONS_FILE = os.path.join(HERE, "wheel_valuations.json")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Strategy constants ─────────────────────────────────────────────────────
PORTFOLIO_CASH: float       = 50_000   # $50 K cash available
TARGET_MONTHLY_LOW: float   = 2_000    # $2 K/month low target
TARGET_MONTHLY_HIGH: float  = 4_000    # $4 K/month high target
MAX_DELTA: float            = 0.05     # put delta ≤ 0.05
NEAR_52W_LOW_PCT: float     = 0.20     # within 20 % of 52 W low
RISK_FREE_RATE: float       = 0.043    # ~10Y US Treasury

# ── Screening thresholds ───────────────────────────────────────────────────
MIN_NET_MARGIN: float   = 0.15   # net profit margin > 15 %
MAX_DEBT_ASSETS: float  = 0.50   # debt / assets < 50 %
MIN_ROE: float          = 0.10   # ROE ≥ 10 % (moat proxy)
MIN_MOAT_SCORE: int     = 3      # moat score ≥ 3 / 5 for "wide"

# ── Default placeholder watchlist (user edits in the Watchlist tab) ────────
DEFAULT_WATCHLIST: list[str] = [
    # ── Technology / Semiconductors ─────────────────────────────────────
    "AAPL", "MSFT", "NVDA", "AVGO", "GOOGL", "META", "AMZN",
    # ── Financials ──────────────────────────────────────────────────────
    "V", "MA", "JPM",
    # ── Healthcare ──────────────────────────────────────────────────────
    "UNH", "LLY", "ABBV", "TMO", "JNJ",
    # ── Consumer staples / Discretionary ────────────────────────────────
    "KO", "PG", "WMT", "COST", "HD",
]

# ── Moat score component weights ──────────────────────────────────────────
# Each criterion adds 1 point to the moat score (max = 5)
_MOAT_ROE_THRESHOLD       = 0.15   # ROE > 15 %
_MOAT_MARGIN_THRESHOLD    = 0.15   # net margin > 15 %
_MOAT_FCF_MARGIN          = 0.10   # FCF / revenue > 10 %
_MOAT_LOW_DEBT            = 0.30   # debt / assets < 30 %
# 5th point: positive FCF itself

# ── yfinance in-memory cache ───────────────────────────────────────────────
_cache_lock = threading.Lock()
_info_cache: dict[str, tuple[float, dict]] = {}   # symbol → (ts, info)
_CACHE_TTL = 900   # 15 minutes

# ── Global analysis results (written by callbacks, read by other tabs) ─────
_screen_results: list[FundamentalScreen] = []
_wheel_opps: list[PutOpportunity]        = []
_last_run_ts: Optional[str]              = None
_analysis_running: bool                  = False

# =============================================================================
# Data classes
# =============================================================================

@dataclass
class FundamentalScreen:
    symbol: str
    price: Optional[float]         = None
    week52_low: Optional[float]    = None
    week52_high: Optional[float]   = None
    pct_from_52w_low: Optional[float] = None   # positive → above the low
    near_52w_low: bool             = False

    # Fundamental metrics
    net_profit_margin: Optional[float] = None
    free_cash_flow: Optional[float]    = None   # absolute $, positive = good
    debt_to_assets: Optional[float]    = None   # fraction, e.g. 0.35 = 35 %
    roe: Optional[float]               = None
    fcf_margin: Optional[float]        = None   # FCF / revenue

    # Moat
    moat_score: int   = 0      # 0–5
    moat_rating: str  = "None" # Wide / Narrow / None

    # Options eligibility
    has_weekly_options: bool = False
    next_expirations: list   = field(default_factory=list)

    # Manual valuation columns (user fills in from IBKR / Morningstar)
    ibkr_valuation: Optional[float]         = None
    morningstar_valuation: Optional[float]  = None

    # Result
    passes: bool       = False
    fail_reasons: list = field(default_factory=list)


@dataclass
class PutOpportunity:
    symbol: str
    stock_price: float
    strike: float
    expiry: str            # "YYYY-MM-DD"
    dte: int               # days to expiry
    delta: float           # absolute value (0–1)
    bid: float
    ask: float
    mid: float
    iv: float              # annualised implied volatility
    open_interest: int
    volume: int
    annualized_yield_pct: float   # (mid / strike) × (365 / dte) × 100
    max_contracts: int            # floor(PORTFOLIO_CASH / (strike × 100))
    max_premium: float            # max_contracts × mid × 100
    breakeven: float              # strike − mid
    pct_below_current: float      # (stock − strike) / stock × 100
    pct_from_52w_low: Optional[float] = None


# Position column names
_POS_COLS = [
    "id", "symbol", "strategy", "strike", "expiry",
    "premium_received", "contracts", "date_opened",
    "status", "date_closed", "close_price", "realized_pnl", "notes",
]

# =============================================================================
# Storage helpers
# =============================================================================

def load_watchlist() -> list[str]:
    """Load watchlist from JSON, falling back to defaults."""
    if os.path.exists(WATCHLIST_FILE):
        try:
            with open(WATCHLIST_FILE) as f:
                data = json.load(f)
                if isinstance(data, list):
                    return [str(s).upper().strip() for s in data if s]
        except Exception as e:
            logger.warning("Could not load watchlist: %s", e)
    return list(DEFAULT_WATCHLIST)


def save_watchlist(tickers: list[str]) -> None:
    """Persist watchlist to JSON."""
    try:
        with open(WATCHLIST_FILE, "w") as f:
            json.dump([t.upper().strip() for t in tickers if t], f, indent=2)
    except Exception as e:
        logger.error("Could not save watchlist: %s", e)


def load_valuations() -> dict[str, dict]:
    """Load manual IBKR / Morningstar valuations keyed by symbol."""
    if os.path.exists(VALUATIONS_FILE):
        try:
            with open(VALUATIONS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_valuations(vals: dict[str, dict]) -> None:
    """Persist manual valuation overrides."""
    try:
        with open(VALUATIONS_FILE, "w") as f:
            json.dump(vals, f, indent=2)
    except Exception as e:
        logger.error("Could not save valuations: %s", e)


def load_positions() -> pd.DataFrame:
    """Load position journal from CSV, creating empty frame if absent."""
    if os.path.exists(POSITIONS_FILE):
        try:
            df = pd.read_csv(POSITIONS_FILE, dtype=str)
            for col in _POS_COLS:
                if col not in df.columns:
                    df[col] = ""
            return df[_POS_COLS]
        except Exception as e:
            logger.warning("Could not load positions: %s", e)
    return pd.DataFrame(columns=_POS_COLS)


def save_positions(df: pd.DataFrame) -> None:
    """Persist position journal to CSV."""
    try:
        df.to_csv(POSITIONS_FILE, index=False)
    except Exception as e:
        logger.error("Could not save positions: %s", e)


# =============================================================================
# yfinance data fetching with caching
# =============================================================================

def _fetch_info(symbol: str) -> dict:
    """Return yfinance Ticker.info with a 15-minute memory cache."""
    now = time.monotonic()
    with _cache_lock:
        if symbol in _info_cache:
            ts, data = _info_cache[symbol]
            if now - ts < _CACHE_TTL:
                return data

    try:
        t = yf.Ticker(symbol)
        info = t.info or {}
    except Exception as e:
        logger.warning("%s: yfinance info fetch failed — %s", symbol, e)
        info = {}

    with _cache_lock:
        _info_cache[symbol] = (now, info)
    return info


def _fetch_options_expirations(symbol: str) -> tuple[bool, list[str]]:
    """
    Return (has_weekly_options, list_of_next_5_expirations).
    Weekly options = 3+ expirations in the next 35 calendar days.
    """
    try:
        t = yf.Ticker(symbol)
        exps = t.options  # tuple of "YYYY-MM-DD" strings
        if not exps:
            return False, []
        today = date.today()
        near = [
            e for e in exps
            if 1 <= (datetime.strptime(e, "%Y-%m-%d").date() - today).days <= 35
        ]
        has_weekly = len(near) >= 3
        return has_weekly, list(exps[:5])
    except Exception as e:
        logger.debug("%s: options expiration fetch failed — %s", symbol, e)
        return False, []


# =============================================================================
# Fundamental screening
# =============================================================================

def screen_stock(symbol: str) -> FundamentalScreen:
    """
    Screen a single stock for wheel strategy eligibility.
    Uses yfinance for speed; treats missing data permissively.
    """
    result = FundamentalScreen(symbol=symbol)
    info   = _fetch_info(symbol)

    if not info:
        result.fail_reasons.append("No data available")
        return result

    # ── Price and 52-week range ────────────────────────────────────────────
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    if price:
        result.price = float(price)

    w52_low  = info.get("fiftyTwoWeekLow")
    w52_high = info.get("fiftyTwoWeekHigh")
    if w52_low:
        result.week52_low = float(w52_low)
    if w52_high:
        result.week52_high = float(w52_high)

    if result.price and result.week52_low and result.week52_low > 0:
        result.pct_from_52w_low = (
            (result.price - result.week52_low) / result.week52_low * 100
        )
        result.near_52w_low = result.pct_from_52w_low <= NEAR_52W_LOW_PCT * 100

    # ── Net profit margin ──────────────────────────────────────────────────
    npm = info.get("profitMargins")
    if npm is not None:
        try:
            result.net_profit_margin = float(npm)
            if result.net_profit_margin < MIN_NET_MARGIN:
                result.fail_reasons.append(
                    f"Net margin {result.net_profit_margin:.1%} < {MIN_NET_MARGIN:.0%}"
                )
        except (TypeError, ValueError):
            pass

    # ── Free cash flow ─────────────────────────────────────────────────────
    fcf = info.get("freeCashflow")
    if fcf is not None:
        try:
            result.free_cash_flow = float(fcf)
            if result.free_cash_flow <= 0:
                result.fail_reasons.append(
                    f"FCF ${result.free_cash_flow:,.0f} ≤ 0"
                )
        except (TypeError, ValueError):
            pass

    # ── Debt / assets ──────────────────────────────────────────────────────
    total_debt   = info.get("totalDebt")
    total_assets = info.get("totalAssets")
    if total_debt is not None and total_assets and float(total_assets) > 0:
        try:
            result.debt_to_assets = float(total_debt) / float(total_assets)
            if result.debt_to_assets > MAX_DEBT_ASSETS:
                result.fail_reasons.append(
                    f"Debt/Assets {result.debt_to_assets:.1%} > {MAX_DEBT_ASSETS:.0%}"
                )
        except (TypeError, ValueError):
            pass

    # ── Return on equity ──────────────────────────────────────────────────
    roe = info.get("returnOnEquity")
    if roe is not None:
        try:
            result.roe = float(roe)
            if result.roe < MIN_ROE:
                result.fail_reasons.append(
                    f"ROE {result.roe:.1%} < {MIN_ROE:.0%}"
                )
        except (TypeError, ValueError):
            pass

    # ── FCF margin (FCF / revenue) ─────────────────────────────────────────
    revenue = info.get("totalRevenue")
    if revenue and float(revenue) > 0 and result.free_cash_flow:
        try:
            result.fcf_margin = result.free_cash_flow / float(revenue)
        except (TypeError, ValueError):
            pass

    # ── Moat score (0–5, one point per criterion) ─────────────────────────
    score = 0
    if result.roe            and result.roe > _MOAT_ROE_THRESHOLD:       score += 1
    if result.net_profit_margin and result.net_profit_margin > _MOAT_MARGIN_THRESHOLD: score += 1
    if result.fcf_margin     and result.fcf_margin > _MOAT_FCF_MARGIN:   score += 1
    if result.free_cash_flow and result.free_cash_flow > 0:               score += 1
    if result.debt_to_assets and result.debt_to_assets < _MOAT_LOW_DEBT: score += 1
    result.moat_score  = score
    result.moat_rating = "Wide" if score >= 4 else "Narrow" if score >= 2 else "None"

    if result.moat_score < MIN_MOAT_SCORE:
        result.fail_reasons.append(
            f"Moat score {result.moat_score}/5 < {MIN_MOAT_SCORE} (not wide enough)"
        )

    # ── Weekly options ─────────────────────────────────────────────────────
    has_weekly, exps = _fetch_options_expirations(symbol)
    result.has_weekly_options = has_weekly
    result.next_expirations   = exps
    if not has_weekly:
        result.fail_reasons.append("No weekly options available")

    # ── Load persisted manual valuations ─────────────────────────────────
    vals = load_valuations()
    sym_vals = vals.get(symbol, {})
    ibkr_v = sym_vals.get("ibkr_valuation")
    ms_v   = sym_vals.get("morningstar_valuation")
    result.ibkr_valuation        = float(ibkr_v) if ibkr_v else None
    result.morningstar_valuation = float(ms_v)   if ms_v   else None

    # ── Final pass / fail ─────────────────────────────────────────────────
    result.passes = len(result.fail_reasons) == 0
    return result


def run_screening(tickers: list[str]) -> list[FundamentalScreen]:
    """Screen a list of tickers. Returns list sorted: passing first, then by proximity to 52W low."""
    results = []
    for sym in tickers:
        try:
            logger.info("Screening %s …", sym)
            r = screen_stock(sym)
            results.append(r)
        except Exception as e:
            logger.error("%s: screening error — %s", sym, e)
            failed = FundamentalScreen(symbol=sym)
            failed.fail_reasons.append(f"Error: {e}")
            results.append(failed)

    # Sort: passing first; within each group, closest to 52W low first
    def _sort_key(r: FundamentalScreen):
        passing = 0 if r.passes else 1
        proximity = r.pct_from_52w_low if r.pct_from_52w_low is not None else 9999
        return (passing, proximity)

    results.sort(key=_sort_key)
    return results


# =============================================================================
# Black-Scholes delta
# =============================================================================

def _bs_delta_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Compute the absolute value of Black-Scholes delta for a European put.

    Parameters
    ----------
    S     : current stock price
    K     : strike price
    T     : time to expiry in years
    r     : annualised risk-free rate
    sigma : annualised implied volatility

    Returns
    -------
    |delta|  in [0, 1]  — a put delta is inherently negative; we return its
    absolute value so callers can use  delta ≤ 0.05  directly.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        return float(abs(norm.cdf(d1) - 1))   # put delta = N(d1) − 1
    except Exception:
        return 0.0


# =============================================================================
# Options analysis — find best CSP per stock
# =============================================================================

def find_wheel_opportunities(
    screens: list[FundamentalScreen],
    max_delta: float = MAX_DELTA,
    max_dte: int = 45,
) -> list[PutOpportunity]:
    """
    For every stock that passes screening and has weekly options, find the
    single best put option:
        – delta ≤ max_delta (default 0.05)
        – highest annualised yield  =  (mid / strike) × (365 / dte) × 100

    Scans all weekly expirations up to max_dte days out.
    """
    opps: list[PutOpportunity] = []

    eligible = [s for s in screens if s.passes and s.has_weekly_options]
    logger.info("Finding wheel opportunities for %d eligible stocks …", len(eligible))

    for screen in eligible:
        symbol = screen.symbol
        S = screen.price
        if not S or S <= 0:
            logger.debug("%s: no price available, skipping", symbol)
            continue

        try:
            t     = yf.Ticker(symbol)
            exps  = t.options
            today = date.today()

            # Weekly expirations within max_dte days
            near_exps = [
                e for e in exps
                if 4 <= (datetime.strptime(e, "%Y-%m-%d").date() - today).days <= max_dte
            ]

            best: Optional[PutOpportunity] = None

            for exp_str in near_exps[:6]:   # check up to 6 expirations
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                dte      = (exp_date - today).days
                T        = dte / 365.0

                try:
                    chain = t.option_chain(exp_str)
                    puts  = chain.puts
                except Exception as chain_err:
                    logger.debug("%s %s: chain fetch failed — %s", symbol, exp_str, chain_err)
                    continue

                if puts is None or puts.empty:
                    continue

                for _, row in puts.iterrows():
                    K = float(row.get("strike", 0))
                    if K <= 0 or K >= S:
                        continue   # skip ITM and invalid

                    iv_raw = row.get("impliedVolatility")
                    if not iv_raw or iv_raw != iv_raw:   # NaN check
                        continue
                    iv = float(iv_raw)
                    if iv <= 0:
                        continue

                    delta = _bs_delta_put(S, K, T, RISK_FREE_RATE, iv)
                    if delta > max_delta:
                        continue   # too close to the money

                    bid_raw = row.get("bid", 0)
                    ask_raw = row.get("ask", 0)
                    bid = float(bid_raw) if bid_raw and bid_raw == bid_raw else 0.0
                    ask = float(ask_raw) if ask_raw and ask_raw == ask_raw else 0.0
                    last = float(row.get("lastPrice", 0) or 0)
                    mid  = (bid + ask) / 2 if bid > 0 and ask > 0 else last
                    if mid <= 0:
                        continue

                    oi  = int(row.get("openInterest", 0) or 0)
                    vol = int(row.get("volume", 0) or 0)

                    ann_yield     = (mid / K) * (365 / dte) * 100
                    max_contracts = max(1, int(PORTFOLIO_CASH / (K * 100)))
                    max_premium   = max_contracts * mid * 100

                    candidate = PutOpportunity(
                        symbol            = symbol,
                        stock_price       = S,
                        strike            = K,
                        expiry            = exp_str,
                        dte               = dte,
                        delta             = delta,
                        bid               = bid,
                        ask               = ask,
                        mid               = mid,
                        iv                = iv,
                        open_interest     = oi,
                        volume            = vol,
                        annualized_yield_pct = ann_yield,
                        max_contracts     = max_contracts,
                        max_premium       = max_premium,
                        breakeven         = K - mid,
                        pct_below_current = (S - K) / S * 100,
                        pct_from_52w_low  = screen.pct_from_52w_low,
                    )

                    # Keep the highest annualised yield within the delta constraint
                    if best is None or ann_yield > best.annualized_yield_pct:
                        best = candidate

            if best is not None:
                opps.append(best)

        except Exception as e:
            logger.warning("%s: opportunity scan failed — %s", symbol, e)

    # Sort by annualised yield descending (best opportunities first)
    opps.sort(key=lambda x: x.annualized_yield_pct, reverse=True)
    logger.info("Found %d wheel opportunities", len(opps))
    return opps


# =============================================================================
# Dash app — style constants (match existing dashboard theme)
# =============================================================================

BRAND_BG = "#f0eeeb"
CARD_BG  = "#ffffff"
ACCENT   = "#2b2b2b"
GREEN    = "#2e7d32"
RED      = "#c41e1e"
YELLOW   = "#e6a817"
TEXT     = "#333333"
MUTED    = "#999999"
BORDER   = "#ddd8d0"

_TABLE_STYLE = dict(
    style_table={"overflowX": "auto", "borderRadius": "6px"},
    style_header={
        "backgroundColor": ACCENT,
        "color": "#ffffff",
        "fontWeight": "bold",
        "fontSize": "12px",
        "padding": "8px 10px",
    },
    style_cell={
        "backgroundColor": CARD_BG,
        "color": TEXT,
        "fontSize": "12px",
        "padding": "6px 10px",
        "border": f"1px solid {BORDER}",
        "fontFamily": "monospace",
        "whiteSpace": "normal",
    },
    style_data_conditional=[
        {"if": {"row_index": "odd"}, "backgroundColor": "#f7f5f2"},
    ],
)


# =============================================================================
# Helper: build screening DataTable rows
# =============================================================================

def _screen_to_rows(screens: list[FundamentalScreen]) -> list[dict]:
    rows = []
    for s in screens:
        rows.append(
            {
                "Ticker":         s.symbol,
                "Price":          f"${s.price:,.2f}"  if s.price       else "—",
                "52W Low":        f"${s.week52_low:,.2f}" if s.week52_low else "—",
                "52W High":       f"${s.week52_high:,.2f}" if s.week52_high else "—",
                "% vs Low":       f"{s.pct_from_52w_low:+.1f}%"  if s.pct_from_52w_low is not None else "—",
                "Near 52W Low":   "✅ Yes" if s.near_52w_low else "❌ No",
                "Weekly Opts":    "✅ Yes" if s.has_weekly_options else "❌ No",
                "Net Margin":     f"{s.net_profit_margin:.1%}" if s.net_profit_margin is not None else "—",
                "FCF ($B)":       f"{s.free_cash_flow / 1e9:.2f}" if s.free_cash_flow is not None else "—",
                "Debt/Assets":    f"{s.debt_to_assets:.1%}" if s.debt_to_assets is not None else "—",
                "ROE":            f"{s.roe:.1%}" if s.roe is not None else "—",
                "Moat":           s.moat_rating,
                "Moat Score":     f"{s.moat_score}/5",
                "FCF Margin":     f"{s.fcf_margin:.1%}" if s.fcf_margin is not None else "—",
                "Passes":         "✅ PASS" if s.passes else "❌ FAIL",
                "Fail Reasons":   "; ".join(s.fail_reasons) if s.fail_reasons else "",
                # Manual columns — user edits these
                "IBKR Val ($)":           s.ibkr_valuation or "",
                "Morningstar Val ($)":    s.morningstar_valuation or "",
            }
        )
    return rows


def _opps_to_rows(opps: list[PutOpportunity]) -> list[dict]:
    rows = []
    for o in opps:
        monthly_est = o.max_premium * (30 / o.dte)
        rows.append(
            {
                "Ticker":         o.symbol,
                "Stock Price":    f"${o.stock_price:,.2f}",
                "% vs 52W Low":   f"{o.pct_from_52w_low:+.1f}%" if o.pct_from_52w_low is not None else "—",
                "Strike":         f"${o.strike:,.2f}",
                "Expiry":         o.expiry,
                "DTE":            o.dte,
                "Delta":          f"{o.delta:.4f}",
                "IV":             f"{o.iv:.1%}",
                "Bid":            f"${o.bid:.2f}",
                "Ask":            f"${o.ask:.2f}",
                "Mid":            f"${o.mid:.2f}",
                "Open Int.":      o.open_interest,
                "Volume":         o.volume,
                "Ann. Yield%":    f"{o.annualized_yield_pct:.2f}%",
                "Max Contracts":  o.max_contracts,
                "Max Premium":    f"${o.max_premium:,.0f}",
                "Est. Monthly":   f"${monthly_est:,.0f}",
                "Breakeven":      f"${o.breakeven:,.2f}",
                "% Below Stock":  f"{o.pct_below_current:.1f}%",
            }
        )
    return rows


# =============================================================================
# Dash app layout
# =============================================================================

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Wheel Strategy Dashboard",
    suppress_callback_exceptions=True,
    prevent_initial_callbacks="initial_duplicate",
)

# ── Shared stores ─────────────────────────────────────────────────────────
_stores = html.Div(
    [
        dcc.Store(id="store-watchlist",   storage_type="session"),
        dcc.Store(id="store-screen",      storage_type="session"),
        dcc.Store(id="store-opps",        storage_type="session"),
        dcc.Store(id="store-last-run",    storage_type="session"),
        dcc.Store(id="store-valuations",  storage_type="session"),
        dcc.Interval(id="interval-analysis", interval=500, n_intervals=0, disabled=True),
    ]
)

# ── Header ────────────────────────────────────────────────────────────────
_header = dbc.Navbar(
    dbc.Container(
        [
            dbc.NavbarBrand(
                "⚙️  Options Wheel Strategy  —  ~$50 K Portfolio",
                style={"fontWeight": "bold", "fontSize": "18px", "color": "#fff"},
            ),
            dbc.Nav(
                [
                    dbc.NavItem(
                        dbc.Badge(
                            "Target: $2 K–$4 K / month",
                            color="success",
                            pill=True,
                            style={"fontSize": "13px"},
                        )
                    ),
                ],
                className="ms-auto",
            ),
        ],
        fluid=True,
    ),
    color=ACCENT,
    dark=True,
    style={"marginBottom": "0px"},
)

def _metric_card(label: str, value: str, color: str) -> dbc.Col:
    return dbc.Col(
        dbc.Card(
            dbc.CardBody(
                [
                    html.P(label, style={"fontSize": "11px", "color": MUTED, "marginBottom": "2px"}),
                    html.H5(value, style={"fontWeight": "bold", "color": color, "marginBottom": 0}),
                ],
                className="p-2",
            ),
            style={"border": f"1px solid {BORDER}", "borderRadius": "6px"},
        ),
        width="auto",
    )


def _strategy_info_block() -> html.Div:
    rows = [
        ("Phase", "Cash-Secured Put → (if assigned) → Covered Call"),
        ("Cash deployed", f"Up to ${PORTFOLIO_CASH:,.0f}"),
        ("Monthly target", f"${TARGET_MONTHLY_LOW:,.0f} – ${TARGET_MONTHLY_HIGH:,.0f}"),
        ("Put filter", f"Delta ≤ {MAX_DELTA:.0%}  (far OTM, low assignment risk)"),
        ("52W Low trigger", f"Stock ≤ {NEAR_52W_LOW_PCT:.0%} above its 52-week low"),
        ("Assignment plan", "Sell ATM / slightly OTM Covered Call on assigned shares"),
        ("Net margin min", f"{MIN_NET_MARGIN:.0%}"),
        ("FCF required", "Positive (any amount)"),
        ("Debt / assets max", f"{MAX_DEBT_ASSETS:.0%}"),
        ("ROE minimum", f"{MIN_ROE:.0%}"),
        ("Moat minimum", f"{MIN_MOAT_SCORE}/5 (Wide rated)"),
        ("Weekly options", "Required (≥ 3 expirations in 35 days)"),
    ]
    return html.Table(
        [
            html.Tr(
                [
                    html.Td(k, style={"color": MUTED, "fontSize": "12px", "paddingRight": "16px", "paddingBottom": "6px"}),
                    html.Td(v, style={"fontWeight": "bold", "fontSize": "12px", "paddingBottom": "6px"}),
                ]
            )
            for k, v in rows
        ],
        style={"width": "100%"},
    )


# ── Tab 1: Watchlist ──────────────────────────────────────────────────────
_tab_watchlist = dcc.Tab(
    label="📋 Watchlist",
    value="tab-watchlist",
    children=dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("📋 Candidate Stock List", style={"fontWeight": "bold"}),
                                    html.P(
                                        "Edit the list below. One ticker per row. "
                                        "Then click 'Save & Run Analysis' to screen all stocks.",
                                        style={"color": MUTED, "fontSize": "13px"},
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                dbc.Input(
                                                    id="input-add-ticker",
                                                    placeholder="Add ticker, e.g. AAPL",
                                                    type="text",
                                                    debounce=True,
                                                    style={"textTransform": "uppercase"},
                                                ),
                                                width=4,
                                            ),
                                            dbc.Col(
                                                dbc.Button(
                                                    "➕ Add",
                                                    id="btn-add-ticker",
                                                    color="secondary",
                                                    size="sm",
                                                    n_clicks=0,
                                                ),
                                                width="auto",
                                            ),
                                            dbc.Col(
                                                dbc.Button(
                                                    "🗑 Remove Selected",
                                                    id="btn-remove-ticker",
                                                    color="danger",
                                                    size="sm",
                                                    outline=True,
                                                    n_clicks=0,
                                                ),
                                                width="auto",
                                            ),
                                            dbc.Col(
                                                dbc.Button(
                                                    "💾 Save Watchlist",
                                                    id="btn-save-watchlist",
                                                    color="primary",
                                                    size="sm",
                                                    outline=True,
                                                    n_clicks=0,
                                                ),
                                                width="auto",
                                            ),
                                            dbc.Col(
                                                dbc.Button(
                                                    "🔍 Save & Run Analysis",
                                                    id="btn-run-analysis",
                                                    color="success",
                                                    size="sm",
                                                    n_clicks=0,
                                                ),
                                                width="auto",
                                            ),
                                        ],
                                        className="g-2 mb-3",
                                    ),
                                    dash_table.DataTable(
                                        id="tbl-watchlist",
                                        columns=[
                                            {"name": "#", "id": "idx", "editable": False},
                                            {"name": "Ticker", "id": "ticker", "editable": True},
                                        ],
                                        data=[],
                                        row_selectable="multi",
                                        row_deletable=False,
                                        selected_rows=[],
                                        page_size=30,
                                        **_TABLE_STYLE,
                                    ),
                                    html.Div(id="watchlist-feedback", className="mt-2",
                                             style={"color": GREEN, "fontSize": "13px"}),
                                ]
                            ),
                            style={"border": f"1px solid {BORDER}"},
                        ),
                        width=5,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("ℹ️ Strategy Overview", style={"fontWeight": "bold"}),
                                    html.Hr(),
                                    _strategy_info_block(),
                                ]
                            ),
                            style={"border": f"1px solid {BORDER}"},
                        ),
                        width=7,
                    ),
                ],
                className="g-3 mt-2",
            ),
            dbc.Row(
                dbc.Col(
                    dbc.Alert(
                        id="analysis-status-alert",
                        children="Click 'Save & Run Analysis' to screen all stocks.",
                        color="info",
                        is_open=True,
                        style={"fontSize": "13px"},
                    ),
                    width=12,
                ),
                className="mt-2",
            ),
        ],
        fluid=True,
        className="pt-3",
    ),
)


# ── Tab 2: Screening ──────────────────────────────────────────────────────
_tab_screening = dcc.Tab(
    label="🔍 Screening",
    value="tab-screening",
    children=dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.RadioItems(
                            id="screen-filter",
                            options=[
                                {"label": " Show All",          "value": "all"},
                                {"label": " Passing Only",      "value": "pass"},
                                {"label": " Near 52W Low",      "value": "low"},
                                {"label": " Near Low + Passing","value": "low_pass"},
                            ],
                            value="all",
                            inline=True,
                            style={"fontSize": "13px"},
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Button(
                            "💾 Save Manual Valuations",
                            id="btn-save-valuations",
                            color="primary",
                            size="sm",
                            outline=True,
                            n_clicks=0,
                        ),
                        width="auto",
                        className="ms-auto",
                    ),
                ],
                className="g-2 py-2 align-items-center",
            ),
            html.P(
                "✏️ The last two columns (IBKR Val and Morningstar Val) are editable — "
                "enter your own valuations and click 'Save Manual Valuations'.",
                style={"color": MUTED, "fontSize": "12px"},
            ),
            html.Div(id="screening-summary", className="mb-2"),
            dash_table.DataTable(
                id="tbl-screening",
                columns=[
                    {"name": "Ticker",                "id": "Ticker",               "editable": False},
                    {"name": "Price",                 "id": "Price",                "editable": False},
                    {"name": "52W Low",               "id": "52W Low",              "editable": False},
                    {"name": "52W High",              "id": "52W High",             "editable": False},
                    {"name": "% vs Low",              "id": "% vs Low",             "editable": False},
                    {"name": "Near 52W",              "id": "Near 52W Low",         "editable": False},
                    {"name": "Wkly Opts",             "id": "Weekly Opts",          "editable": False},
                    {"name": "Net Margin",            "id": "Net Margin",           "editable": False},
                    {"name": "FCF ($B)",              "id": "FCF ($B)",             "editable": False},
                    {"name": "Debt/Assets",           "id": "Debt/Assets",          "editable": False},
                    {"name": "ROE",                   "id": "ROE",                  "editable": False},
                    {"name": "Moat",                  "id": "Moat",                 "editable": False},
                    {"name": "Score",                 "id": "Moat Score",           "editable": False},
                    {"name": "FCF Margin",            "id": "FCF Margin",           "editable": False},
                    {"name": "Pass?",                 "id": "Passes",               "editable": False},
                    {"name": "Fail Reasons",          "id": "Fail Reasons",         "editable": False},
                    # ── Manual valuation columns ──────────────────────────
                    {"name": "IBKR Val ($)",          "id": "IBKR Val ($)",         "editable": True,  "type": "numeric"},
                    {"name": "Morningstar Val ($)",   "id": "Morningstar Val ($)",  "editable": True,  "type": "numeric"},
                ],
                data=[],
                page_size=25,
                sort_action="native",
                filter_action="native",
                style_data_conditional=[
                    # Green row = passes all criteria
                    {
                        "if": {"filter_query": '{Passes} = "✅ PASS"'},
                        "backgroundColor": "#e8f5e9",
                    },
                    # Red row = fails
                    {
                        "if": {"filter_query": '{Passes} = "❌ FAIL"'},
                        "backgroundColor": "#ffebee",
                    },
                    # Highlight near 52W low
                    {
                        "if": {"filter_query": '{Near 52W Low} = "✅ Yes"'},
                        "fontWeight": "bold",
                    },
                    # Editable cells
                    {
                        "if": {"column_id": ["IBKR Val ($)", "Morningstar Val ($)"]},
                        "backgroundColor": "#fff9c4",
                        "border": "2px solid #f9a825",
                    },
                ],
                **{k: v for k, v in _TABLE_STYLE.items() if k != "style_data_conditional"},
            ),
            html.Div(id="valuation-save-feedback", className="mt-2",
                     style={"color": GREEN, "fontSize": "13px"}),
        ],
        fluid=True,
        className="pt-2",
    ),
)


# ── Tab 3: Wheel Opportunities ────────────────────────────────────────────
_tab_wheel_opps = dcc.Tab(
    label="🎯 Wheel Opps",
    value="tab-opps",
    children=dbc.Container(
        [
            html.Div(id="opps-summary", className="py-2"),
            dash_table.DataTable(
                id="tbl-opps",
                columns=[
                    {"name": c, "id": c}
                    for c in [
                        "Ticker", "Stock Price", "% vs 52W Low",
                        "Strike", "Expiry", "DTE", "Delta", "IV",
                        "Bid", "Ask", "Mid",
                        "Open Int.", "Volume",
                        "Ann. Yield%", "Max Contracts", "Max Premium",
                        "Est. Monthly", "Breakeven", "% Below Stock",
                    ]
                ],
                data=[],
                sort_action="native",
                filter_action="native",
                page_size=20,
                style_data_conditional=[
                    {
                        "if": {"row_index": "odd"},
                        "backgroundColor": "#f7f5f2",
                    },
                    # Highlight high-yield rows
                    {
                        "if": {"filter_query": "{Ann. Yield%} > 10"},
                        "backgroundColor": "#e8f5e9",
                        "fontWeight": "bold",
                    },
                ],
                **{k: v for k, v in _TABLE_STYLE.items() if k != "style_data_conditional"},
            ),
            html.Div(id="opps-disclaimer", className="mt-2",
                     style={"color": MUTED, "fontSize": "11px"}),
        ],
        fluid=True,
        className="pt-2",
    ),
)


# ── Tab 4: Open Positions ─────────────────────────────────────────────────
_tab_positions = dcc.Tab(
    label="📂 Open Positions",
    value="tab-positions",
    children=dbc.Container(
        [
            dbc.Row(
                [
                    # Add position form
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H6("➕ Add New Position", style={"fontWeight": "bold"}),
                                    dbc.Row([
                                        dbc.Col(dbc.Label("Ticker"),    width=4),
                                        dbc.Col(dbc.Input(id="pos-symbol",   type="text",
                                                          placeholder="AAPL",
                                                          style={"textTransform": "uppercase"}), width=8),
                                    ], className="mb-1 align-items-center"),
                                    dbc.Row([
                                        dbc.Col(dbc.Label("Strategy"),  width=4),
                                        dbc.Col(dbc.Select(
                                            id="pos-strategy",
                                            options=[
                                                {"label": "CSP — Cash-Secured Put",  "value": "CSP"},
                                                {"label": "CC  — Covered Call",      "value": "CC"},
                                            ],
                                            value="CSP",
                                        ), width=8),
                                    ], className="mb-1 align-items-center"),
                                    dbc.Row([
                                        dbc.Col(dbc.Label("Strike ($)"), width=4),
                                        dbc.Col(dbc.Input(id="pos-strike",  type="number",
                                                          placeholder="150.00"), width=8),
                                    ], className="mb-1 align-items-center"),
                                    dbc.Row([
                                        dbc.Col(dbc.Label("Expiry"),     width=4),
                                        dbc.Col(dcc.DatePickerSingle(
                                            id="pos-expiry",
                                            min_date_allowed=str(date.today()),
                                            date=str(date.today() + timedelta(days=7)),
                                            display_format="YYYY-MM-DD",
                                        ), width=8),
                                    ], className="mb-1 align-items-center"),
                                    dbc.Row([
                                        dbc.Col(dbc.Label("Premium / contract ($)"), width=4),
                                        dbc.Col(dbc.Input(id="pos-premium", type="number",
                                                          placeholder="0.25"), width=8),
                                    ], className="mb-1 align-items-center"),
                                    dbc.Row([
                                        dbc.Col(dbc.Label("Contracts"),  width=4),
                                        dbc.Col(dbc.Input(id="pos-contracts", type="number",
                                                          min=1, placeholder="1"), width=8),
                                    ], className="mb-1 align-items-center"),
                                    dbc.Row([
                                        dbc.Col(dbc.Label("Notes"),      width=4),
                                        dbc.Col(dbc.Input(id="pos-notes", type="text",
                                                          placeholder="Optional notes"), width=8),
                                    ], className="mb-2 align-items-center"),
                                    dbc.Button("➕ Add Position", id="btn-add-position",
                                               color="success", size="sm", n_clicks=0,
                                               className="w-100"),
                                    html.Div(id="pos-add-feedback", className="mt-2",
                                             style={"fontSize": "12px"}),
                                ]
                            ),
                            style={"border": f"1px solid {BORDER}"},
                        ),
                        width=4,
                    ),
                    # Close position form
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H6("✅ Close / Update Position", style={"fontWeight": "bold"}),
                                    dbc.Row([
                                        dbc.Col(dbc.Label("Position ID"),  width=5),
                                        dbc.Col(dbc.Input(id="close-pos-id", type="text",
                                                          placeholder="Paste position ID"), width=7),
                                    ], className="mb-1 align-items-center"),
                                    dbc.Row([
                                        dbc.Col(dbc.Label("New Status"),   width=5),
                                        dbc.Col(dbc.Select(
                                            id="close-status",
                                            options=[
                                                {"label": "Expired Worthless", "value": "expired"},
                                                {"label": "Closed (buy-back)", "value": "closed"},
                                                {"label": "Assigned",          "value": "assigned"},
                                            ],
                                            value="expired",
                                        ), width=7),
                                    ], className="mb-1 align-items-center"),
                                    dbc.Row([
                                        dbc.Col(dbc.Label("Close price / contract ($)"), width=5),
                                        dbc.Col(dbc.Input(id="close-price", type="number",
                                                          placeholder="0.00 (0 if expired)"), width=7),
                                    ], className="mb-2 align-items-center"),
                                    dbc.Button("✅ Update Position", id="btn-close-position",
                                               color="warning", size="sm", n_clicks=0,
                                               className="w-100"),
                                    html.Div(id="pos-close-feedback", className="mt-2",
                                             style={"fontSize": "12px"}),
                                ]
                            ),
                            style={"border": f"1px solid {BORDER}"},
                        ),
                        width=4,
                    ),
                    # Summary cards
                    dbc.Col(
                        html.Div(id="positions-summary-cards"),
                        width=4,
                    ),
                ],
                className="g-3 mt-1",
            ),
            html.Hr(),
            dbc.Row(
                dbc.Col(
                    [
                        dbc.Button("🔄 Refresh Positions", id="btn-refresh-positions",
                                   color="secondary", size="sm", outline=True, n_clicks=0),
                        html.Span("  "),
                        dbc.RadioItems(
                            id="pos-filter",
                            options=[
                                {"label": " All",    "value": "all"},
                                {"label": " Open",   "value": "open"},
                                {"label": " Closed", "value": "closed"},
                            ],
                            value="open",
                            inline=True,
                            style={"fontSize": "13px", "display": "inline-block", "marginLeft": "12px"},
                        ),
                    ],
                    width=12,
                ),
                className="mb-2",
            ),
            dash_table.DataTable(
                id="tbl-positions",
                columns=[
                    {"name": col.replace("_", " ").title(), "id": col}
                    for col in _POS_COLS
                ],
                data=[],
                page_size=20,
                sort_action="native",
                **_TABLE_STYLE,
            ),
        ],
        fluid=True,
        className="pt-2",
    ),
)


# ── Tab 5: P&L Summary ────────────────────────────────────────────────────
_tab_pnl = dcc.Tab(
    label="📊 P&L Summary",
    value="tab-pnl",
    children=dbc.Container(
        [
            dbc.Row(id="pnl-summary-cards", className="g-3 mt-1"),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="chart-monthly-pnl"), width=8),
                    dbc.Col(dcc.Graph(id="chart-pnl-breakdown"), width=4),
                ],
                className="mt-3",
            ),
            dbc.Row(
                dbc.Col(
                    dbc.Progress(id="progress-monthly-target", value=0, max=100,
                                 label="", color="success",
                                 style={"height": "28px", "fontSize": "14px"}),
                    width=12,
                ),
                className="mt-3",
            ),
            html.P(id="progress-label", className="mt-1",
                   style={"color": MUTED, "fontSize": "12px", "textAlign": "center"}),
        ],
        fluid=True,
        className="pt-2",
    ),
)


# ── Full layout ───────────────────────────────────────────────────────────
app.layout = html.Div(
    [
        _stores,
        _header,
        dbc.Container(
            dbc.Row(
                dbc.Col(
                    [
                        html.P(
                            id="last-run-label",
                            style={"color": MUTED, "fontSize": "12px", "marginBottom": "0"},
                        ),
                    ]
                ),
                className="py-1",
            ),
            fluid=True,
            style={"backgroundColor": "#e8e4df", "borderBottom": f"1px solid {BORDER}"},
        ),
        dbc.Container(
            dbc.Row(
                [
                    _metric_card("💵 Cash",  f"${PORTFOLIO_CASH:,.0f}", "#1565c0"),
                    _metric_card("🎯 Target", "$2 000–$4 000/mo", "#2e7d32"),
                    _metric_card("⚡ Δ ≤",   f"{MAX_DELTA:.0%}",         "#7b1fa2"),
                    _metric_card("📉 Near Low", f"≤ {NEAR_52W_LOW_PCT:.0%} above 52W low", "#e65100"),
                    dbc.Col(html.Div(id="pnl-mtd-card"), width="auto"),
                ],
                className="g-2 py-2",
            ),
            fluid=True,
            style={"backgroundColor": "#e8e4df", "borderBottom": f"2px solid {BORDER}"},
        ),
        dcc.Tabs(
            id="main-tabs",
            value="tab-watchlist",
            children=[
                _tab_watchlist,
                _tab_screening,
                _tab_wheel_opps,
                _tab_positions,
                _tab_pnl,
            ],
            style={"marginTop": "0px"},
        ),
    ],
    style={"backgroundColor": BRAND_BG, "minHeight": "100vh"},
)


# =============================================================================
# Callbacks
# =============================================================================

# ── 1. Initialise watchlist on page load ──────────────────────────────────
@app.callback(
    Output("store-watchlist", "data"),
    Output("tbl-watchlist",   "data"),
    Input("main-tabs",        "value"),   # fires on every tab click; init on first load
    State("store-watchlist",  "data"),
    prevent_initial_call=False,
)
def init_watchlist(tab, stored):
    if stored:
        tickers = stored
    else:
        tickers = load_watchlist()
    rows = [{"idx": i + 1, "ticker": t} for i, t in enumerate(tickers)]
    return tickers, rows


# ── 2. Add a ticker to the watchlist ─────────────────────────────────────
@app.callback(
    Output("store-watchlist",    "data",    allow_duplicate=True),
    Output("tbl-watchlist",      "data",    allow_duplicate=True),
    Output("input-add-ticker",   "value"),
    Output("watchlist-feedback", "children", allow_duplicate=True),
    Input("btn-add-ticker",      "n_clicks"),
    State("input-add-ticker",    "value"),
    State("store-watchlist",     "data"),
    prevent_initial_call=True,
)
def add_ticker(n, new_ticker, stored):
    if not n or not new_ticker:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    tickers = stored or load_watchlist()
    sym = new_ticker.upper().strip()
    if sym and sym not in tickers:
        tickers = tickers + [sym]
        save_watchlist(tickers)
        msg = f"✅ Added {sym} to watchlist."
    else:
        msg = f"ℹ️ {sym} is already in the watchlist." if sym else "⚠️ Empty ticker."

    rows = [{"idx": i + 1, "ticker": t} for i, t in enumerate(tickers)]
    return tickers, rows, "", msg


# ── 3. Remove selected tickers ────────────────────────────────────────────
@app.callback(
    Output("store-watchlist",    "data",    allow_duplicate=True),
    Output("tbl-watchlist",      "data",    allow_duplicate=True),
    Output("watchlist-feedback", "children", allow_duplicate=True),
    Input("btn-remove-ticker",   "n_clicks"),
    State("tbl-watchlist",       "selected_rows"),
    State("tbl-watchlist",       "data"),
    State("store-watchlist",     "data"),
    prevent_initial_call=True,
)
def remove_tickers(n, selected_rows, tbl_data, stored):
    if not n or not selected_rows:
        return dash.no_update, dash.no_update, "⚠️ Select rows first."

    tickers = [row["ticker"] for i, row in enumerate(tbl_data) if i not in selected_rows]
    save_watchlist(tickers)
    rows = [{"idx": i + 1, "ticker": t} for i, t in enumerate(tickers)]
    removed = [tbl_data[i]["ticker"] for i in selected_rows]
    return tickers, rows, f"🗑 Removed: {', '.join(removed)}"


# ── 4. Save watchlist only ────────────────────────────────────────────────
@app.callback(
    Output("watchlist-feedback", "children", allow_duplicate=True),
    Input("btn-save-watchlist",  "n_clicks"),
    State("tbl-watchlist",       "data"),
    prevent_initial_call=True,
)
def save_wl_only(n, tbl_data):
    if not n:
        return dash.no_update
    tickers = [row["ticker"] for row in (tbl_data or []) if row.get("ticker")]
    save_watchlist(tickers)
    return f"💾 Watchlist saved ({len(tickers)} tickers)."


# ── 5. Run full analysis (screening + options scan) ───────────────────────
@app.callback(
    Output("analysis-status-alert", "children"),
    Output("analysis-status-alert", "color"),
    Output("store-screen",           "data"),
    Output("store-opps",             "data"),
    Output("store-last-run",         "data"),
    Output("store-watchlist",        "data",    allow_duplicate=True),
    Input("btn-run-analysis",        "n_clicks"),
    State("tbl-watchlist",           "data"),
    prevent_initial_call=True,
)
def run_analysis(n, tbl_data):
    if not n:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    tickers = [row["ticker"] for row in (tbl_data or []) if row.get("ticker")]
    if not tickers:
        return "⚠️ Watchlist is empty.", "warning", dash.no_update, dash.no_update, dash.no_update, dash.no_update

    save_watchlist(tickers)

    # Run screening
    screens = run_screening(tickers)

    # Run options scan
    opps = find_wheel_opportunities(screens)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Serialise to JSON-safe dicts for Store
    screen_data = [asdict(s) for s in screens]
    opps_data   = [asdict(o) for o in opps]

    passing = sum(1 for s in screens if s.passes)
    near    = sum(1 for s in screens if s.near_52w_low and s.passes)
    msg = (
        f"✅ Analysis complete at {ts}  —  "
        f"{passing}/{len(screens)} stocks pass screening  |  "
        f"{near} near 52W low and eligible  |  "
        f"{len(opps)} wheel opportunities found."
    )
    return msg, "success", screen_data, opps_data, ts, tickers


# ── 6. Update last-run label ──────────────────────────────────────────────
@app.callback(
    Output("last-run-label", "children"),
    Input("store-last-run",  "data"),
)
def update_last_run(ts):
    if not ts:
        return "Analysis not yet run."
    return f"Last analysis: {ts}"


# ── 7. Populate screening table ───────────────────────────────────────────
@app.callback(
    Output("tbl-screening",      "data"),
    Output("screening-summary",  "children"),
    Input("store-screen",        "data"),
    Input("screen-filter",       "value"),
)
def update_screening_table(screen_data, filt):
    if not screen_data:
        return [], html.P("Run analysis first.", style={"color": MUTED})

    screens = [FundamentalScreen(**d) for d in screen_data]

    if filt == "pass":
        screens = [s for s in screens if s.passes]
    elif filt == "low":
        screens = [s for s in screens if s.near_52w_low]
    elif filt == "low_pass":
        screens = [s for s in screens if s.near_52w_low and s.passes]

    rows = _screen_to_rows(screens)

    total   = len(screens)
    passing = sum(1 for s in screens if s.passes)
    near    = sum(1 for s in screens if s.near_52w_low)

    summary = dbc.Row(
        [
            dbc.Col(dbc.Badge(f"Showing: {total}", color="secondary", pill=True), width="auto"),
            dbc.Col(dbc.Badge(f"✅ Pass: {passing}", color="success", pill=True),  width="auto"),
            dbc.Col(dbc.Badge(f"📉 Near 52W Low: {near}", color="warning", pill=True), width="auto"),
        ],
        className="g-2",
    )

    return rows, summary


# ── 8. Save manual IBKR / Morningstar valuations ─────────────────────────
@app.callback(
    Output("valuation-save-feedback", "children"),
    Input("btn-save-valuations",      "n_clicks"),
    State("tbl-screening",            "data"),
    prevent_initial_call=True,
)
def save_manual_valuations(n, tbl_data):
    if not n or not tbl_data:
        return dash.no_update

    vals = load_valuations()
    saved = 0
    for row in tbl_data:
        sym  = row.get("Ticker", "")
        ibkr = row.get("IBKR Val ($)")
        ms   = row.get("Morningstar Val ($)")
        if sym and (ibkr or ms):
            if sym not in vals:
                vals[sym] = {}
            if ibkr:
                vals[sym]["ibkr_valuation"]        = float(ibkr)
            if ms:
                vals[sym]["morningstar_valuation"] = float(ms)
            saved += 1

    save_valuations(vals)
    return f"💾 Manual valuations saved for {saved} stock(s)."


# ── 9. Populate wheel opportunities table ────────────────────────────────
@app.callback(
    Output("tbl-opps",         "data"),
    Output("opps-summary",     "children"),
    Output("opps-disclaimer",  "children"),
    Input("store-opps",        "data"),
    Input("store-last-run",    "data"),
)
def update_opps_table(opps_data, ts):
    if not opps_data:
        msg = "Run analysis from the Watchlist tab first." if not ts else "No wheel opportunities found."
        return [], html.P(msg, style={"color": MUTED}), ""

    opps = [PutOpportunity(**d) for d in opps_data]
    rows = _opps_to_rows(opps)

    total_max_premium = sum(o.max_premium for o in opps)
    est_monthly       = sum(o.max_premium * (30 / o.dte) for o in opps)

    on_target = (
        "✅ Within target range"
        if TARGET_MONTHLY_LOW <= est_monthly <= TARGET_MONTHLY_HIGH
        else (
            "⚠️ Below $2K target — consider more positions or higher-IV stocks"
            if est_monthly < TARGET_MONTHLY_LOW
            else "🔴 Above $4K — concentrated risk, review carefully"
        )
    )

    summary = dbc.Row(
        [
            dbc.Col(dbc.Badge(f"{len(opps)} opportunities", color="primary",  pill=True), width="auto"),
            dbc.Col(dbc.Badge(f"Max premium: ${total_max_premium:,.0f}", color="dark", pill=True), width="auto"),
            dbc.Col(dbc.Badge(f"Est. monthly: ${est_monthly:,.0f}", color="success", pill=True), width="auto"),
            dbc.Col(dbc.Badge(on_target, color="info", pill=True), width="auto"),
        ],
        className="g-2",
    )

    disclaimer = (
        "⚠️  Options data sourced from yfinance; bid/ask may be stale outside market hours.  "
        "Delta computed via Black-Scholes using implied volatility from the option chain.  "
        "'Est. Monthly' = max premium if held through expiry, scaled to 30 days — not a guarantee.  "
        "Always verify live quotes in IBKR before placing trades."
    )

    return rows, summary, disclaimer


# ── 10. Add position ──────────────────────────────────────────────────────
@app.callback(
    Output("pos-add-feedback", "children"),
    Output("tbl-positions",    "data",    allow_duplicate=True),
    Input("btn-add-position",  "n_clicks"),
    State("pos-symbol",        "value"),
    State("pos-strategy",      "value"),
    State("pos-strike",        "value"),
    State("pos-expiry",        "date"),
    State("pos-premium",       "value"),
    State("pos-contracts",     "value"),
    State("pos-notes",         "value"),
    State("pos-filter",        "value"),
    prevent_initial_call=True,
)
def add_position(n, sym, strategy, strike, expiry, premium, contracts, notes, filt):
    if not n:
        return dash.no_update, dash.no_update

    missing = []
    if not sym:       missing.append("Ticker")
    if not strike:    missing.append("Strike")
    if not expiry:    missing.append("Expiry")
    if not premium:   missing.append("Premium")
    if not contracts: missing.append("Contracts")
    if missing:
        return f"⚠️ Fill in: {', '.join(missing)}", dash.no_update

    df = load_positions()
    new_row = {
        "id":               str(uuid.uuid4())[:8],
        "symbol":           sym.upper().strip(),
        "strategy":         strategy,
        "strike":           float(strike),
        "expiry":           str(expiry)[:10],
        "premium_received": float(premium),
        "contracts":        int(contracts),
        "date_opened":      str(date.today()),
        "status":           "open",
        "date_closed":      "",
        "close_price":      "",
        "realized_pnl":     "",
        "notes":            notes or "",
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_positions(df)

    total_collected = float(premium) * int(contracts) * 100
    msg = (
        f"✅ Added {strategy} on {sym.upper()} — "
        f"${strike:.2f} strike, {contracts} contract(s), "
        f"${total_collected:,.0f} total premium collected."
    )

    return msg, _positions_rows(df, filt)


# ── 11. Close / update position ───────────────────────────────────────────
@app.callback(
    Output("pos-close-feedback",  "children"),
    Output("tbl-positions",       "data",    allow_duplicate=True),
    Input("btn-close-position",   "n_clicks"),
    State("close-pos-id",         "value"),
    State("close-status",         "value"),
    State("close-price",          "value"),
    State("pos-filter",           "value"),
    prevent_initial_call=True,
)
def close_position(n, pos_id, new_status, close_price_str, filt):
    if not n or not pos_id:
        return "⚠️ Enter the Position ID.", dash.no_update

    df = load_positions()
    mask = df["id"].astype(str).str.strip() == str(pos_id).strip()
    if not mask.any():
        return f"⚠️ Position ID '{pos_id}' not found.", dash.no_update

    idx          = df[mask].index[0]
    close_price  = float(close_price_str) if close_price_str else 0.0
    premium_rcvd = float(df.at[idx, "premium_received"])
    contracts    = int(df.at[idx, "contracts"])

    # P&L per contract: premium received − close price (cost to buy-back)
    # If expired worthless: close price = 0 → full premium kept
    realized_pnl = (premium_rcvd - close_price) * contracts * 100

    df.at[idx, "status"]       = new_status
    df.at[idx, "date_closed"]  = str(date.today())
    df.at[idx, "close_price"]  = close_price
    df.at[idx, "realized_pnl"] = round(realized_pnl, 2)

    save_positions(df)

    return (
        f"✅ Updated position {pos_id} → {new_status}  |  P&L: ${realized_pnl:+,.2f}",
        _positions_rows(df, filt),
    )


# ── 12. Refresh positions ─────────────────────────────────────────────────
@app.callback(
    Output("tbl-positions",          "data",    allow_duplicate=True),
    Output("positions-summary-cards","children"),
    Input("btn-refresh-positions",   "n_clicks"),
    Input("pos-filter",              "value"),
    prevent_initial_call=False,
)
def refresh_positions(n, filt):
    df = load_positions()
    cards = _build_position_summary_cards(df)
    return _positions_rows(df, filt), cards


def _positions_rows(df: pd.DataFrame, filt: str) -> list[dict]:
    if df.empty:
        return []
    d = df.copy()
    if filt == "open":
        d = d[d["status"] == "open"]
    elif filt == "closed":
        d = d[d["status"] != "open"]
    return d.to_dict("records")


def _build_position_summary_cards(df: pd.DataFrame) -> html.Div:
    if df.empty:
        return html.P("No positions yet.", style={"color": MUTED, "fontSize": "13px"})

    open_pos     = df[df["status"] == "open"]
    closed_pos   = df[df["status"] != "open"]

    def _safe_float_col(d, col):
        try:
            return pd.to_numeric(d[col], errors="coerce").fillna(0)
        except Exception:
            return pd.Series(0.0, index=d.index)

    open_premium = (
        (_safe_float_col(open_pos, "premium_received") * _safe_float_col(open_pos, "contracts") * 100).sum()
        if not open_pos.empty else 0.0
    )
    realized_pnl = _safe_float_col(closed_pos, "realized_pnl").sum() if not closed_pos.empty else 0.0

    return html.Div(
        [
            dbc.Card(
                dbc.CardBody([
                    html.P("Open positions", style={"fontSize": "11px", "color": MUTED, "marginBottom": 0}),
                    html.H5(len(open_pos), style={"fontWeight": "bold", "color": "#1565c0"}),
                ], className="p-2"),
                className="mb-2", style={"border": f"1px solid {BORDER}"},
            ),
            dbc.Card(
                dbc.CardBody([
                    html.P("Open premium at risk", style={"fontSize": "11px", "color": MUTED, "marginBottom": 0}),
                    html.H5(f"${open_premium:,.0f}", style={"fontWeight": "bold", "color": "#e65100"}),
                ], className="p-2"),
                className="mb-2", style={"border": f"1px solid {BORDER}"},
            ),
            dbc.Card(
                dbc.CardBody([
                    html.P("Total realized P&L", style={"fontSize": "11px", "color": MUTED, "marginBottom": 0}),
                    html.H5(
                        f"${realized_pnl:+,.0f}",
                        style={"fontWeight": "bold", "color": GREEN if realized_pnl >= 0 else RED},
                    ),
                ], className="p-2"),
                style={"border": f"1px solid {BORDER}"},
            ),
        ]
    )


# ── 13. P&L tab ───────────────────────────────────────────────────────────
@app.callback(
    Output("chart-monthly-pnl",    "figure"),
    Output("chart-pnl-breakdown",  "figure"),
    Output("pnl-summary-cards",    "children"),
    Output("progress-monthly-target", "value"),
    Output("progress-monthly-target", "label"),
    Output("progress-label",        "children"),
    Output("pnl-mtd-card",          "children"),
    Input("main-tabs",              "value"),
    Input("btn-refresh-positions",  "n_clicks"),
    Input("btn-close-position",     "n_clicks"),
    Input("btn-add-position",       "n_clicks"),
)
def update_pnl_charts(tab, *_):
    df = load_positions()

    empty_fig = go.Figure().update_layout(
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font_color=TEXT,
        margin=dict(l=20, r=20, t=40, b=20),
        title_text="No closed positions yet",
    )

    if df.empty or "realized_pnl" not in df.columns:
        return empty_fig, empty_fig, [], 0, "$0", "No data yet", _mtd_badge(0)

    closed = df[df["status"] != "open"].copy()
    if closed.empty:
        return empty_fig, empty_fig, [], 0, "$0", "No closed positions yet", _mtd_badge(0)

    closed["realized_pnl"] = pd.to_numeric(closed["realized_pnl"], errors="coerce").fillna(0)
    closed["date_closed"]  = pd.to_datetime(closed["date_closed"], errors="coerce")
    closed["month"]        = closed["date_closed"].dt.to_period("M").astype(str)

    monthly = closed.groupby("month")["realized_pnl"].sum().reset_index()
    monthly.columns = ["Month", "P&L"]

    # Monthly bar chart
    bar_fig = px.bar(
        monthly, x="Month", y="P&L",
        color_discrete_sequence=[GREEN],
        title="Monthly Premium Income (Realized P&L)",
        labels={"P&L": "$ Realized"},
    )
    bar_fig.add_hline(y=TARGET_MONTHLY_LOW,  line_dash="dash", line_color=YELLOW,
                      annotation_text=f"${TARGET_MONTHLY_LOW:,.0f} target low",
                      annotation_position="top right")
    bar_fig.add_hline(y=TARGET_MONTHLY_HIGH, line_dash="dot",  line_color=GREEN,
                      annotation_text=f"${TARGET_MONTHLY_HIGH:,.0f} target high",
                      annotation_position="top right")
    bar_fig.update_layout(
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG, font_color=TEXT,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    # Strategy breakdown pie
    by_strategy = closed.groupby("strategy")["realized_pnl"].sum().reset_index()
    pie_fig = px.pie(
        by_strategy, names="strategy", values="realized_pnl",
        title="P&L by Strategy",
        color_discrete_map={"CSP": "#1565c0", "CC": "#2e7d32"},
    )
    pie_fig.update_layout(
        paper_bgcolor=CARD_BG, font_color=TEXT,
        margin=dict(l=10, r=10, t=40, b=10),
    )

    # MTD P&L
    today     = date.today()
    mtd_month = f"{today.year}-{today.month:02d}"
    mtd_pnl   = monthly[monthly["Month"] == mtd_month]["P&L"].sum() if mtd_month in monthly["Month"].values else 0.0

    # Summary cards
    total_pnl   = closed["realized_pnl"].sum()
    total_trades = len(closed)
    winners     = (closed["realized_pnl"] > 0).sum()
    win_rate    = winners / total_trades * 100 if total_trades else 0

    summary_cards = [
        dbc.Col(_stat_card("Total Realized P&L", f"${total_pnl:+,.0f}", GREEN if total_pnl >= 0 else RED), width="auto"),
        dbc.Col(_stat_card("MTD Income",          f"${mtd_pnl:+,.0f}", "#1565c0"), width="auto"),
        dbc.Col(_stat_card("Total Trades",         str(total_trades),   ACCENT),   width="auto"),
        dbc.Col(_stat_card("Win Rate",             f"{win_rate:.0f}%",  GREEN),    width="auto"),
    ]

    progress_pct = min(100, int(mtd_pnl / TARGET_MONTHLY_HIGH * 100)) if TARGET_MONTHLY_HIGH > 0 else 0
    progress_lbl = f"${mtd_pnl:,.0f} / ${TARGET_MONTHLY_HIGH:,.0f}"
    progress_txt = (
        f"Month-to-date: ${mtd_pnl:,.0f}  |  "
        f"Target: ${TARGET_MONTHLY_LOW:,.0f}–${TARGET_MONTHLY_HIGH:,.0f}  |  "
        f"Remaining: ${max(0, TARGET_MONTHLY_LOW - mtd_pnl):,.0f} to low target"
    )

    return (
        bar_fig, pie_fig,
        summary_cards,
        progress_pct, progress_lbl, progress_txt,
        _mtd_badge(mtd_pnl),
    )


def _stat_card(label: str, value: str, color: str) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.P(label, style={"fontSize": "11px", "color": MUTED, "marginBottom": 0}),
                html.H5(value, style={"fontWeight": "bold", "color": color, "marginBottom": 0}),
            ],
            className="p-2",
        ),
        style={"border": f"1px solid {BORDER}", "minWidth": "130px"},
    )


def _mtd_badge(mtd_pnl: float) -> dbc.Col:
    color = "success" if mtd_pnl >= TARGET_MONTHLY_LOW else "warning" if mtd_pnl > 0 else "secondary"
    return dbc.Col(
        dbc.Badge(f"MTD: ${mtd_pnl:+,.0f}", color=color, pill=True,
                  style={"fontSize": "13px"}),
        width="auto",
    )


# =============================================================================
# Entry point
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Options Wheel Strategy Dashboard")
    p.add_argument("--port",       type=int, default=8051,  help="Port (default 8051)")
    p.add_argument("--no-browser", action="store_true",     help="Don't auto-open browser")
    p.add_argument("--debug",      action="store_true",     help="Enable Dash debug mode")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Initialise files
    if not os.path.exists(WATCHLIST_FILE):
        save_watchlist(DEFAULT_WATCHLIST)
        logger.info("Created default watchlist at %s", WATCHLIST_FILE)

    if not os.path.exists(POSITIONS_FILE):
        save_positions(pd.DataFrame(columns=_POS_COLS))
        logger.info("Created empty positions file at %s", POSITIONS_FILE)

    print(
        f"\n  ⚙️  Options Wheel Strategy Dashboard\n"
        f"  ─────────────────────────────────────\n"
        f"  URL  : http://127.0.0.1:{args.port}\n"
        f"  Cash : ${PORTFOLIO_CASH:,.0f}   Target: ${TARGET_MONTHLY_LOW:,.0f}–${TARGET_MONTHLY_HIGH:,.0f}/month\n"
        f"  Delta ≤ {MAX_DELTA:.0%}   |   Near 52W low ≤ {NEAR_52W_LOW_PCT:.0%}\n"
        f"\n"
        f"  1. Go to the 📋 Watchlist tab and edit your stock list.\n"
        f"  2. Click 'Save & Run Analysis' to screen stocks.\n"
        f"  3. Check 🎯 Wheel Opps for best put opportunities.\n"
        f"  4. Log trades in 📂 Open Positions.\n"
        f"  5. Track income in 📊 P&L Summary.\n"
        f"\n"
        f"  Positions : {POSITIONS_FILE}\n"
        f"  Watchlist : {WATCHLIST_FILE}\n"
        f"  Valuations: {VALUATIONS_FILE}\n"
    )

    if not args.no_browser:
        import webbrowser
        webbrowser.open(f"http://127.0.0.1:{args.port}")

    app.run(
        host="127.0.0.1",
        port=args.port,
        debug=args.debug,
        use_reloader=False,
    )
