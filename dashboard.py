"""
dashboard.py — Live dynamic dashboard for the buy-the-sauce trading system.

Tabs
----
  1. 📡 Screener        Live dip detector + ML signal for every watchlist ticker
  2. 💰 Risk & Capital  Position sizing, capital allocation, risk heat-map ($80k budget)
  3. 📈 Back-test       Equity curve + trade log from results/backtest_trades.csv
  4. 🗂 Screen Log      Historical screener runs from results/screen_log.csv
  5. 📒 My Trades       Manual transaction journal — amounts, quantities, P&L
  6. 🏠 System Overview  Portfolio-level summary of system health, recent activity, watchlist

Usage
-----
  python3 dashboard.py              # opens on http://127.0.0.1:8050
  python3 dashboard.py --port 8080  # custom port
  python3 dashboard.py --no-browser # don't auto-open browser tab

The screener tab has a "🔄 Refresh" button that re-runs the full data
fetch on demand (same pipeline as run_screen.py).  A countdown timer
shows the time since the last refresh.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import os
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── ML dependency pre-flight check ────────────────────────────────────────────
# Emit one clear error at startup if ML libraries are missing, rather than a
# cryptic WARNING buried in screener output mid-refresh.
_ML_MISSING: list[str] = []
for _lib in ("lightgbm", "xgboost", "scipy"):
    try:
        __import__(_lib)
    except ImportError:
        _ML_MISSING.append(_lib)
if _ML_MISSING:
    _venv = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python3")
    print(
        f"\n  ⚠️  ML libraries missing from this Python interpreter: {', '.join(_ML_MISSING)}\n"
        f"     Active interpreter: {sys.executable}\n"
        f"     Fix: run the dashboard with the project virtualenv:\n"
        f"       {_venv} dashboard.py\n"
        f"     Or install missing packages:\n"
        f"       {sys.executable} -m pip install {' '.join(_ML_MISSING)}\n"
        f"     ML predictions will show 'n/a' until this is fixed.\n",
        file=sys.stderr,
    )

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc

import config
from risk_manager import (
    calculate_order, size_portfolio, PortfolioAllocation,
    plan_scaled_entry, ScaledEntryPlan, Tranche,
    TrailingStopState, evaluate_partial_exits, check_time_stop,
)
from run_screen import WATCHLIST

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACCOUNT_EQUITY  = config.PORTFOLIO_CAPITAL  # single source of truth → config.py
RESULTS_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SCREEN_CSV      = os.path.join(RESULTS_DIR, "screen_log.csv")
BACKTEST_CSV    = os.path.join(RESULTS_DIR, "backtest_trades.csv")
TRADES_CSV      = os.path.join(RESULTS_DIR, "my_trades.csv")   # manual journal

# ── Light Monochrome + Red theme ────────────────────────────────────────────
# Warm light background, soft white cards, charcoal text. Red is the ONE accent
# colour — reserved for danger, losses, and critical items.  Everything else
# is understated greys so the red pops.
BRAND_BG    = "#f0eeeb"      # warm off-white page background
CARD_BG     = "#ffffff"      # pure white cards
ACCENT      = "#2b2b2b"      # near-black for headers & labels (high contrast)
GREEN       = "#5a5a5a"      # dark grey for "good" / profit (understated)
RED         = "#c41e1e"      # the ONE colour — danger, loss, critical
YELLOW      = "#8c8c8c"      # medium grey for caution / warnings
TEXT        = "#333333"      # charcoal body text
MUTED       = "#999999"      # light grey for secondary / disabled
BORDER      = "#ddd8d0"      # warm light border
ORANGE      = "#7a7a7a"      # neutral grey replacing old orange

_CELL_STYLE = {
    "backgroundColor": CARD_BG,
    "color": TEXT,
    "border": f"1px solid {BORDER}",
    "fontFamily": "monospace",
    "fontSize": "13px",
}
_HDR_STYLE = {
    "backgroundColor": "#f5f3f0",
    "color": ACCENT,
    "fontWeight": "bold",
    "fontFamily": "monospace",
    "fontSize": "13px",
    "border": f"1px solid {BORDER}",
}

# ── My Trades CSV schema ────────────────────────────────────────────────────
_TRADES_COLS = [
    "date", "symbol", "action", "quantity", "entry_price",
    "exit_price", "amount_invested", "realized_pnl", "notes",
    "asset_type",  # "STOCK" or "ETF" — ETFs are held forever (no exit targets)
]

def _ensure_trades_csv():
    """Create the trades CSV with headers if it doesn't exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not os.path.exists(TRADES_CSV):
        with open(TRADES_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_TRADES_COLS)
            writer.writeheader()

def _load_trades() -> pd.DataFrame:
    _ensure_trades_csv()
    try:
        df = pd.read_csv(TRADES_CSV)
        for col in _TRADES_COLS:
            if col not in df.columns:
                df[col] = None
        # coerce numeric cols
        for col in ["quantity", "entry_price", "exit_price", "amount_invested", "realized_pnl"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame(columns=_TRADES_COLS)

def _append_trade(row: dict):
    """Append a single trade dict to the CSV."""
    _ensure_trades_csv()
    with open(TRADES_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_TRADES_COLS)
        writer.writerow({k: row.get(k, "") for k in _TRADES_COLS})

def _delete_trade(idx: int):
    """Delete trade at zero-based index from CSV."""
    df = _load_trades()
    df = df.drop(index=idx).reset_index(drop=True)
    df.to_csv(TRADES_CSV, index=False)


def _fetch_current_prices(symbols: list[str]) -> dict[str, float]:
    """Fetch the latest closing price for each symbol.  Returns {symbol: price}."""
    import edgar as _edgar
    prices: dict[str, float] = {}
    for sym in symbols:
        try:
            hist, _ = _edgar.get_price_history(sym, period_years=1)
            if hist is not None and not hist.empty:
                prices[sym] = float(hist["Close"].iloc[-1])
        except Exception:
            pass
    return prices


def _compute_exit_plan(entry_price: float, current_price: float, symbol: str,
                       hist=None) -> dict:
    """Compute ATR-based exit targets for an open position.

    Returns dict with stop, target, partial-exit prices, trailing stop stage,
    and ATR value.  All prices are rounded to 2dp.
    """
    result = {
        "atr_14": None,
        "stop_price": None,
        "target_price": None,
        "partial_1_price": None,
        "partial_2_price": None,
        "trail_stage": "—",
        "trail_stop": None,
    }
    try:
        import edgar as _edgar
        if hist is None:
            hist, _ = _edgar.get_price_history(symbol, period_years=1)
        if hist is None or hist.empty or len(hist) < 15:
            return result
        # ATR(14)
        h = hist["High"]
        l = hist["Low"]
        c = hist["Close"]
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        atr_abs = float(tr.rolling(14).mean().iloc[-1])
        atr_frac = atr_abs / entry_price if entry_price > 0 else 0

        result["atr_14"] = round(atr_abs, 2)

        # Stop & target (same logic as risk_manager._compute_stops)
        if config.USE_ATR_STOPS and atr_frac > 0:
            raw_stop_pct = config.ATR_STOP_MULTIPLIER * atr_frac
            stop_pct = max(config.ATR_MIN_STOP_PCT, min(raw_stop_pct, config.ATR_MAX_STOP_PCT))
            target_pct = config.ATR_TARGET_MULTIPLIER * atr_frac
        else:
            stop_pct = config.STOP_LOSS_PCT
            target_pct = config.TAKE_PROFIT_PCT

        stop_price = entry_price * (1 - stop_pct)
        target_price = entry_price * (1 + target_pct)
        result["stop_price"] = round(stop_price, 2)
        result["target_price"] = round(target_price, 2)

        # Partial exit prices
        result["partial_1_price"] = round(entry_price + config.PARTIAL_EXIT_1_TRIGGER_ATR * atr_abs, 2)
        result["partial_2_price"] = round(entry_price + config.PARTIAL_EXIT_2_TRIGGER_ATR * atr_abs, 2)

        # Trailing stop state simulation (where are we now?)
        gain_atr = (current_price - entry_price) / atr_abs if atr_abs > 0 else 0
        if gain_atr >= config.TRAILING_STAGE3_TRIGGER_ATR:
            result["trail_stage"] = "TIGHT_TRAIL"
            result["trail_stop"] = round(current_price - config.TRAILING_STAGE3_TRAIL_ATR * atr_abs, 2)
        elif gain_atr >= config.TRAILING_STAGE2_TRIGGER_ATR:
            result["trail_stage"] = "PROFIT_LOCK"
            result["trail_stop"] = round(entry_price + 1.0 * atr_abs, 2)
        elif gain_atr >= config.TRAILING_STAGE1_TRIGGER_ATR:
            result["trail_stage"] = "BREAKEVEN"
            result["trail_stop"] = round(entry_price, 2)
        else:
            result["trail_stage"] = "INITIAL"
            result["trail_stop"] = result["stop_price"]
    except Exception:
        pass
    return result


# Known ETF tickers — used as fallback to auto-classify trades when asset_type
# is not explicitly set.  Extend this list as you add new ETFs.
_KNOWN_ETFS = {
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "VT", "VXUS",
    "BND", "AGG", "TLT", "SHY", "IEF", "LQD", "HYG", "BNDX",
    "VNQ", "SCHD", "VIG", "DGRO", "DVY", "HDV", "NOBL",
    "SMH", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP",
    "XLU", "XLB", "XLRE", "XLC",
    "ARKK", "ARKW", "ARKG", "ARKF", "ARKQ",
    "GLD", "SLV", "IAU", "GLDM",
    "EEM", "VWO", "IEMG", "EFA", "VEA",
    "SCHX", "SCHA", "SCHB", "SCHF", "SCHE", "SCHG", "SCHV",
    "JEPI", "JEPQ", "DIVO",
}


def _infer_asset_type(symbol: str, explicit: str | None = None) -> str:
    """Return 'ETF' or 'STOCK' — uses explicit value if set, else known-ETF list."""
    if explicit and str(explicit).strip().upper() in ("ETF", "STOCK"):
        return str(explicit).strip().upper()
    return "ETF" if symbol.upper() in _KNOWN_ETFS else "STOCK"


# ---------------------------------------------------------------------------
# IBKR connectivity check
# ---------------------------------------------------------------------------

def _check_ibkr_status() -> tuple[bool, str]:
    """Return (connected: bool, message: str) for IBKR TWS/Gateway."""
    try:
        import importlib
        importlib.import_module("ib_insync")
    except ImportError:
        return False, f"⚠️  ib_insync not installed — yfinance fallback active"

    import asyncio, random

    async def _ping():
        from ib_insync import IB
        ib = IB()
        cid = random.randint(200, 299)
        await ib.connectAsync(config.IBKR_HOST, config.IBKR_PORT,
                              clientId=cid, readonly=True, timeout=4)
        connected = ib.isConnected()
        ib.disconnect()
        return connected

    addr = f"{config.IBKR_HOST}:{config.IBKR_PORT}"
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            connected = loop.run_until_complete(_ping())
        finally:
            loop.close()
            asyncio.set_event_loop(None)

        if connected:
            return True, f"✅ IBKR Connected  ({addr})"
        return False, f"❌ IBKR Not Connected  ({addr}) — is TWS / Gateway running?"
    except Exception as exc:
        if "already in use" in str(exc).lower():
            return True, f"✅ IBKR Connected  ({addr})"
        return False, f"❌ IBKR Unavailable ({addr}) — TWS/Gateway not running → yfinance fallback active"


# ---------------------------------------------------------------------------
# Data fetching (runs in-process — same as run_screen.py)
# ---------------------------------------------------------------------------

_last_fetch: dict = {}   # cached results to avoid re-fetching on every callback


def fetch_screen_data() -> list[dict]:
    """Run the full screener pipeline and return a list of record dicts."""
    import traceback
    import edgar
    from screener import screen_fundamental, FundamentalProfile
    from dip_detector import score_dip

    def _fallback_profile(symbol: str, reasons: list) -> "FundamentalProfile":
        p = FundamentalProfile(symbol=symbol)
        p.passes = False
        p.fail_reasons = reasons
        return p

    records = []
    for symbol in WATCHLIST:
        price_source = "error"
        info = {}
        profile = _fallback_profile(symbol, ["not fetched"])
        signal = None
        try:
            # --- price history first (IBKR → yfinance fallback) ---
            hist, price_source = edgar.get_price_history(symbol, period_years=2)
            # --- fundamentals from EDGAR ---
            try:
                info    = edgar.get_fundamentals(symbol)
                profile = screen_fundamental(symbol, info=info)
            except Exception as fund_exc:
                print(f"[dashboard] {symbol} fundamentals error: {fund_exc}")
                profile = _fallback_profile(symbol, [f"fund error: {fund_exc}"])
            # --- dip signal ---
            signal = score_dip(symbol, hist) if hist is not None and not hist.empty else None
        except Exception as exc:
            print(f"[dashboard] {symbol} fetch error: {exc}")
            traceback.print_exc()

        records.append({
            "symbol":       symbol,
            "profile":      profile,
            "signal":       signal,
            "info":         info,
            "price_source": price_source,
        })
    return records


def _records_to_df(records: list[dict]) -> pd.DataFrame:
    """Flatten screener records into a display DataFrame for the screener table."""
    rows = []
    for r in records:
        s   = r["signal"]
        p   = r["profile"]
        sym = r["symbol"]

        price     = round(s.price, 2)       if s else None
        score     = s.score                 if s else None
        rsi       = round(s.rsi, 1)         if s else None
        ma50      = round(s.ma50, 2)        if s else None
        ma200     = round(s.ma200, 2)       if s else None
        rng_pct   = round((s.price - s.week52_low) / max(s.week52_high - s.week52_low, 1e-6) * 100, 0) if s else None
        is_dip    = s.is_dip                if s else False
        ml_dir    = s.ml_direction          if s else None
        ml_prob   = round(s.ml_probability * 100, 0) if s and s.ml_probability else None
        ml_conf   = s.ml_confidence         if s else None
        ml_ok     = s.ml_buy_confirmed      if s else False
        # Show the best AUROC across all three horizons (1W/1M/1Y) so the user
        # understands why a horizon might be n/a (below the 0.51 gate).
        mh_for_auroc = getattr(s, "multi_horizon", None) if s else None
        _horizon_aurocs = []
        if mh_for_auroc:
            for _pred in (mh_for_auroc.week1, mh_for_auroc.month1, mh_for_auroc.year1):
                if _pred and getattr(_pred, "auroc_cv", None) is not None:
                    _horizon_aurocs.append(_pred.auroc_cv)
        # Also check the single-horizon auroc on the signal itself
        _sig_auroc = getattr(s, "ml_auroc_cv", None) if s else None
        if _sig_auroc is not None:
            _horizon_aurocs.append(_sig_auroc)
        ml_auroc  = round(max(_horizon_aurocs), 3) if _horizon_aurocs else None

        fund_pass = p.passes
        de        = round(p.debt_to_equity, 2)         if p.debt_to_equity  is not None else None
        fcf_b     = round(p.free_cash_flow / 1e9, 2)  if p.free_cash_flow  is not None else None
        pe        = round(r["info"].get("trailingPE") or r["info"].get("forwardPE") or 0, 1) or None

        # ── helpers ────────────────────────────────────────────────────────────
        def _pct(v, decimals=1):
            return round(v * 100, decimals) if v is not None else None

        def _yoy(curr, prior, label="%"):
            """Format YoY comparison as 'curr (Δprior)'."""
            if curr is None:
                return "n/a"
            curr_s = f"{curr*100:.1f}{label}"
            if prior is not None:
                delta = curr - prior
                sign  = "+" if delta >= 0 else ""
                return f"{curr_s} ({sign}{delta*100:.1f})"
            return curr_s

        def _yoy_x(curr, prior):
            """Format cash-conversion-style YoY as 'Xx (Δ)'."""
            if curr is None:
                return "n/a"
            curr_s = f"{curr:.2f}x"
            if prior is not None:
                delta  = curr - prior
                sign   = "+" if delta >= 0 else ""
                return f"{curr_s} ({sign}{delta:.2f})"
            return curr_s

        def _yoy_raw(curr, prior, scale=1e9, unit="B"):
            """Format raw-value YoY (e.g. revenue in $B)."""
            if curr is None:
                return "n/a"
            curr_s = f"${curr/scale:.1f}{unit}"
            if prior is not None:
                pct    = (curr / prior - 1) * 100 if prior != 0 else float("nan")
                sign   = "+" if pct >= 0 else ""
                return f"{curr_s} ({sign}{pct:.0f}%)"
            return curr_s

        # ── Multi-horizon cells ───────────────────────────────────────────────
        def _hcell(pred):
            if pred is None: return "n/a"
            arrow = "↑" if pred.direction == "UP" else "↓"
            conf  = (pred.confidence or "")[:3]
            return f"{arrow}{pred.probability:.0%} {conf}"

        mh    = getattr(s, "multi_horizon", None) if s else None
        ml_1w = _hcell(mh.week1  if mh else None)
        ml_1m = _hcell(mh.month1 if mh else None)
        ml_1y = _hcell(mh.year1  if mh else None)

        # ── Swing metrics ─────────────────────────────────────────────────────
        sm = getattr(s, "swing_metrics", None) if s else None

        sell_rec   = sm.sell_recommendation              if sm else None
        sell_score = round(sm.composite_sell_score, 0)  if sm else None
        rsi14      = round(sm.rsi_14, 1)                if sm else None
        rsi_sig    = sm.rsi_signal                       if sm else None
        bb_pos     = round(sm.bb_position, 2)            if sm else None
        bb_sig     = sm.bb_signal                        if sm else None
        macd_sig   = sm.macd_signal                      if sm else None
        vs_ma50    = f"{sm.price_vs_ma50:+.1%}"          if sm and sm.price_vs_ma50  == sm.price_vs_ma50  else None
        vs_ma200   = f"{sm.price_vs_ma200:+.1%}"         if sm and sm.price_vs_ma200 == sm.price_vs_ma200 else None
        atr_stop   = round(sm.atr_stop_price, 2)         if sm else None
        atr_tgt    = round(sm.atr_target_price, 2)       if sm else None
        rr         = round(sm.reward_risk_ratio, 1)      if sm else None
        drawdn     = f"{sm.drawdown_from_high:.1%}"       if sm else None
        days_hi    = sm.days_since_high                   if sm else None
        vol_reg    = sm.vol_regime                        if sm else None

        # ── New fundamentals (TABLE 2) ─────────────────────────────────────────
        # ROIC: current (prior year delta in parens)
        roic_col      = _yoy(getattr(p, "roic", None),            getattr(p, "roic_prior", None))
        fcf_margin_col= _yoy(getattr(p, "fcf_margin", None),      getattr(p, "fcf_margin_prior", None))
        cash_conv_col = _yoy_x(getattr(p, "cash_conversion", None), getattr(p, "cash_conversion_prior", None))
        # FCF raw: current vs prior (from FCF history)
        fcf_vals      = r["info"].get("_fcf_history", [])
        fcf_prior_raw = fcf_vals[1] if isinstance(fcf_vals, list) and len(fcf_vals) > 1 else None
        fcf_col       = _yoy_raw(p.free_cash_flow, fcf_prior_raw)
        de_col        = f"{de:.2f}x" if de is not None else "n/a"
        accruals_col  = _yoy(getattr(p, "accruals_ratio", None),  getattr(p, "accruals_ratio_prior", None), label="")
        # Receivables growth vs revenue growth
        rec_growth = getattr(p, "receivables_growth", None)
        rev_growth = getattr(p, "revenue_growth", None)
        rec_gr  = f"{rec_growth*100:+.1f}%" if rec_growth is not None else "n/a"
        rev_gr  = f"{rev_growth*100:+.1f}%"  if rev_growth is not None else "n/a"
        rec_vs_rev = f"{rec_gr} vs {rev_gr}"          # e.g. "+12.0% vs +8.0%"
        revenue_col   = _yoy_raw(getattr(p, "revenue_current", None), getattr(p, "revenue_prior", None))

        # Signal label
        if is_dip and fund_pass and (not config.ML_GATE_BUY_SIGNAL or not config.ML_ENABLED or ml_ok):
            signal_label = "🟢 BUY"
        elif is_dip and fund_pass and config.ML_GATE_BUY_SIGNAL and ml_dir and not ml_ok:
            signal_label = "🟡 DIP+FUND (ML↓)"
        elif is_dip and not fund_pass:
            signal_label = "🟠 DIP ONLY"
        else:
            signal_label = "⚪ WATCH"

        rows.append({
            # ── TABLE 1: Dip detector ─────────────────────────────────────────
            "Symbol":     sym,
            "Signal":     signal_label,
            "Score":      f"{score}/4" if score is not None else "—",
            "Price":      price,
            "RSI":        rsi,
            "vs MA50%":   round((price - ma50)  / ma50  * 100, 1) if price and ma50  else None,
            "vs MA200%":  round((price - ma200) / ma200 * 100, 1) if price and ma200 else None,
            "52wk%":      rng_pct,
            "Fund":       "✅" if fund_pass else "❌",
            "ML 1W":      ml_1w,
            "ML 1M":      ml_1m,
            "ML 1Y":      ml_1y,
            "ML AUROC":   ml_auroc,
            "P/E":        pe,
            "Src":        r["price_source"],
            # ── TABLE 2: Fundamentals (new) ───────────────────────────────────
            "ROIC":           roic_col,
            "FCF Margin":     fcf_margin_col,
            "FCF":            fcf_col,
            "Cash Conv":      cash_conv_col,
            "D/E":            de_col,
            "Accruals":       accruals_col,
            "AR vs Rev Gr":   rec_vs_rev,
            "Revenue":        revenue_col,
            # ── TABLE 3: Swing sell metrics ───────────────────────────────────
            "Sell Rec":   sell_rec,
            "Sell Score": sell_score,
            "RSI-14":     rsi14,
            "RSI Sig":    rsi_sig,
            "BB%B":       bb_pos,
            "BB Sig":     bb_sig,
            "MACD Sig":   macd_sig,
            "vs MA50":    vs_ma50,
            "vs MA200":   vs_ma200,
            "Stop $":     atr_stop,
            "Target $":   atr_tgt,
            "R/R":        rr,
            "Drawdown":   drawdn,
            "Days@Hi":    days_hi,
            "Vol":        vol_reg,
            # internal — used for risk tab
            "_is_dip":    is_dip,
            "_fund_pass": fund_pass,
            "_ml_ok":     ml_ok,
            "_price":     price,
            # extra swing/fundamental fields for portfolio-level sizing
            "_atr_14":    sm.atr_14              if sm else None,
            "_rr":        sm.reward_risk_ratio    if sm else None,
            "_vol":       sm.vol_regime           if sm else None,
            "_sector":    r["info"].get("sector") or r["info"].get("sectorDisp"),
            "_ml_prob":   (s.ml_probability or 0.0) if s else 0.0,
            "_signal":    signal_label,
        })
    return pd.DataFrame(rows)


def _risk_df(screen_df: pd.DataFrame) -> tuple[pd.DataFrame, "PortfolioAllocation | None"]:
    """Simultaneously size all dip candidates using the portfolio-level allocator.

    Returns (display_df, PortfolioAllocation).
    """
    from risk_manager import size_portfolio

    dip_df = screen_df[screen_df["_is_dip"] == True].copy()
    if dip_df.empty:
        return pd.DataFrame(), None

    # Build candidates list — include all signals (BUY, DIP+FUND, DIP ONLY)
    # so the portfolio allocator can prioritise them correctly.
    candidates = []
    for _, row in dip_df.iterrows():
        candidates.append({
            "symbol":           row["Symbol"],
            "entry_price":      row.get("_price"),
            "signal":           row.get("_signal") or row.get("Signal", ""),
            "atr_14":           row.get("_atr_14"),
            "reward_risk_ratio":row.get("_rr"),
            "vol_regime":       row.get("_vol"),
            "sector":           row.get("_sector"),
            "ml_prob":          row.get("_ml_prob", 0.0),
        })

    alloc = size_portfolio(candidates, account_equity=ACCOUNT_EQUITY)

    # Build display rows — include both accepted and skipped candidates
    rows = []
    accepted = {o.symbol: o for o in alloc.orders}
    skipped  = {s["symbol"]: s["reason"] for s in alloc.skipped}

    for _, row in dip_df.iterrows():
        sym = row["Symbol"]
        price = row.get("_price")
        signal = row.get("_signal") or row.get("Signal", "")

        if sym in accepted:
            o = accepted[sym]
            rows.append({
                "Symbol":     sym,
                "Signal":     signal,
                "Method":     o.stop_method,
                "Sizing":     o.sizing_method,
                "Entry $":    o.entry_price,
                "Qty":        o.quantity,
                "Notional $": round(o.position_value, 0),
                "Stop $":     o.stop_loss_price,
                "Stop %":     f"{o.stop_pct:.1%}",
                "Target $":   o.take_profit_price,
                "Target %":   f"{o.target_pct:.1%}",
                "R/R":        o.reward_risk_ratio,
                "Risk $":     round(o.risk_amount, 0),
                "% Budget":   round(o.position_value / ACCOUNT_EQUITY * 100, 1),
                "Vol Scale":  f"{o.vol_scale:.2f}×",
                "Sect Scale": f"{o.sector_scale:.2f}×",
                "Kelly Qty":  o.kelly_qty or "—",
                "Risk Qty":   o.risk_qty,
                "Status":     "✅ Allocated",
            })
        else:
            reason = skipped.get(sym, "n/a")
            rows.append({
                "Symbol":     sym,
                "Signal":     signal,
                "Method":     "—",
                "Sizing":     "—",
                "Entry $":    price,
                "Qty":        "—",
                "Notional $": "—",
                "Stop $":     "—",
                "Stop %":     "—",
                "Target $":   "—",
                "Target %":   "—",
                "R/R":        "—",
                "Risk $":     "—",
                "% Budget":   "—",
                "Vol Scale":  "—",
                "Sect Scale": "—",
                "Kelly Qty":  "—",
                "Risk Qty":   "—",
                "Status":     f"⚠️ {reason}",
            })

    return pd.DataFrame(rows), alloc


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _card(children, **kwargs):
    return dbc.Card(
        dbc.CardBody(children),
        style={"backgroundColor": CARD_BG, "border": f"1px solid {BORDER}", "borderRadius": "8px", **kwargs},
        className="mb-3",
    )


def _stat_card(label, value, colour=TEXT):
    return dbc.Col(_card([
        html.Div(label, style={"color": MUTED, "fontSize": "12px", "marginBottom": "4px"}),
        html.Div(value, style={"color": colour, "fontSize": "22px", "fontWeight": "bold", "fontFamily": "monospace"}),
    ], padding="12px"), width="auto")


def _make_table(df: pd.DataFrame, tid: str, hide_cols: list[str] | None = None,
                colour_col: str | None = None) -> dash_table.DataTable:
    hide_cols = hide_cols or []
    display_df = df[[c for c in df.columns if c not in hide_cols]]
    columns = [{"name": c, "id": c} for c in display_df.columns]

    style_data_conditional = []
    if "Signal" in display_df.columns:
        style_data_conditional += [
            {"if": {"filter_query": '{Signal} contains "BUY"',  "column_id": "Signal"}, "color": GREEN,  "fontWeight": "bold"},
            {"if": {"filter_query": '{Signal} contains "DIP+FUND"', "column_id": "Signal"}, "color": YELLOW},
            {"if": {"filter_query": '{Signal} contains "DIP ONLY"', "column_id": "Signal"}, "color": ORANGE},
            {"if": {"filter_query": '{Signal} contains "WATCH"', "column_id": "Signal"}, "color": MUTED},
        ]
    if "outcome" in display_df.columns or "Outcome" in display_df.columns:
        col = "outcome" if "outcome" in display_df.columns else "Outcome"
        style_data_conditional += [
            {"if": {"filter_query": f'{{{col}}} = "TP"', "column_id": col}, "color": GREEN},
            {"if": {"filter_query": f'{{{col}}} = "SL"', "column_id": col}, "color": RED},
            {"if": {"filter_query": f'{{{col}}} = "OPEN"', "column_id": col}, "color": YELLOW},
        ]
    if "pct_gain" in display_df.columns:
        style_data_conditional += [
            {"if": {"filter_query": "{pct_gain} > 0", "column_id": "pct_gain"}, "color": GREEN},
            {"if": {"filter_query": "{pct_gain} < 0", "column_id": "pct_gain"}, "color": RED},
        ]
    if "realized_pnl" in display_df.columns or "Realized P&L $" in display_df.columns:
        col = "realized_pnl" if "realized_pnl" in display_df.columns else "Realized P&L $"
        style_data_conditional += [
            {"if": {"filter_query": f"{{{col}}} > 0", "column_id": col}, "color": GREEN, "fontWeight": "bold"},
            {"if": {"filter_query": f"{{{col}}} < 0", "column_id": col}, "color": RED,   "fontWeight": "bold"},
        ]
    if "Exposure %" in display_df.columns:
        style_data_conditional += [
            {"if": {"filter_query": "{Exposure %} <= 10",                     "column_id": "Exposure %"}, "color": GREEN},
            {"if": {"filter_query": "{Exposure %} > 10 && {Exposure %} <= 20","column_id": "Exposure %"}, "color": YELLOW},
            {"if": {"filter_query": "{Exposure %} > 20",                      "column_id": "Exposure %"}, "color": RED, "fontWeight": "bold"},
        ]
    if "Action" in display_df.columns:
        style_data_conditional += [
            {"if": {"filter_query": '{Action} = "BUY"',  "column_id": "Action"}, "color": GREEN},
            {"if": {"filter_query": '{Action} = "SELL"', "column_id": "Action"}, "color": RED},
        ]

    return dash_table.DataTable(
        id=tid,
        columns=columns,
        data=display_df.to_dict("records"),
        style_cell=_CELL_STYLE,
        style_header=_HDR_STYLE,
        style_data_conditional=style_data_conditional,
        style_table={"overflowX": "auto", "borderRadius": "6px"},
        sort_action="native",
        filter_action="native",
        page_size=40,
    )


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="Buy The Sauce — Dashboard",
)
app.layout = html.Div(
    style={"backgroundColor": BRAND_BG, "minHeight": "100vh", "padding": "0"},
    children=[
        # ── top nav ───────────────────────────────────────────────────────────
        dbc.Navbar(
            dbc.Container([
                html.Span("🍅 Buy The Sauce", style={
                    "color": ACCENT, "fontWeight": "bold", "fontSize": "20px", "fontFamily": "monospace",
                }),
                dbc.NavbarToggler(id="navbar-toggler"),
                html.Span(id="ibkr-status-label",
                          style={"color": MUTED, "fontSize": "12px", "fontFamily": "monospace", "marginLeft": "16px"}),
                html.Span(id="last-refresh-label",
                          style={"color": MUTED, "fontSize": "12px", "fontFamily": "monospace", "marginLeft": "auto"}),
            ], fluid=True),
            color=CARD_BG, dark=False,
            style={"borderBottom": f"1px solid {BORDER}", "padding": "8px 24px"},
        ),

        dbc.Container(fluid=True, style={"padding": "24px"}, children=[

            # ── store (holds fetched data as JSON) ────────────────────────────
            dcc.Store(id="screen-store"),
            dcc.Store(id="trades-store"),           # triggers trades table refresh
            dcc.Interval(id="clock-tick", interval=60_000, n_intervals=0),  # 1-min clock

            # ── tabs ──────────────────────────────────────────────────────────
            dbc.Tabs(id="tabs", active_tab="tab-screener", children=[
                dbc.Tab(label="📡  Screener",      tab_id="tab-screener"),
                dbc.Tab(label="💰  Risk & Capital", tab_id="tab-risk"),
                dbc.Tab(label="📈  Back-test",      tab_id="tab-backtest"),
                dbc.Tab(label="🗂  Screen Log",     tab_id="tab-log"),
                dbc.Tab(label="📒  My Trades",      tab_id="tab-trades"),
                dbc.Tab(label="🏠  System Overview", tab_id="tab-overview"),
            ], style={"marginBottom": "20px"}),

            html.Div(id="tab-content"),
        ]),
    ],
)


# ---------------------------------------------------------------------------
# Tab 1 — Screener layout (static skeleton; data injected by callback)
# ---------------------------------------------------------------------------

def _screener_layout():
    return html.Div([
        dbc.Row([
            dbc.Col(
                dbc.Button("🔄 Refresh Data", id="refresh-btn", color="primary", size="sm",
                           style={"fontFamily": "monospace"}),
                width="auto",
            ),
            dbc.Col(
                dcc.Loading(html.Div(id="refresh-status",
                                     style={"color": MUTED, "fontFamily": "monospace", "fontSize": "13px",
                                            "paddingTop": "6px"})),
                width=True,
            ),
        ], className="mb-3"),

        # stat cards row
        dbc.Row(id="stat-cards", className="mb-3"),

        # ── TABLE 1: Dip Detector ─────────────────────────────────────────────
        _card([
            html.H6("📡 TABLE 1 — Dip Detector + ML Outlook",
                    style={"color": ACCENT, "fontFamily": "monospace", "marginBottom": "4px"}),
            html.Div(
                "Sorted by dip score. ML 1W=5d / 1M=21d / 1Y=252d vs SPY.  ↑ = bullish  ↓ = bearish",
                style={"color": MUTED, "fontSize": "11px", "fontFamily": "monospace", "marginBottom": "10px"},
            ),
            html.Div(id="screener-table-container",
                     children=html.Div("Click 🔄 Refresh to load live data.",
                                       style={"color": MUTED, "fontFamily": "monospace"})),
        ]),

        # ── TABLE 2: Fundamentals ─────────────────────────────────────────────
        _card([
            html.H6("📊 TABLE 2 — Fundamental Quality Screen",
                    style={"color": ACCENT, "fontFamily": "monospace", "marginBottom": "4px"}),
            html.Div(
                "ROIC, FCF margin, FCF (YoY), Cash Conversion, D/E, Accruals, AR vs Revenue Growth — PASS = cleared all filters",
                style={"color": MUTED, "fontSize": "11px", "fontFamily": "monospace", "marginBottom": "10px"},
            ),
            html.Div(id="fund-table-container",
                     children=html.Div("Waiting for refresh…",
                                       style={"color": MUTED, "fontFamily": "monospace"})),
        ]),

        # ── TABLE 3: Swing Sell Metrics ───────────────────────────────────────
        _card([
            html.H6("📉 TABLE 3 — Swing Sell Metrics",
                    style={"color": ACCENT, "fontFamily": "monospace", "marginBottom": "4px"}),
            html.Div(
                "Sorted by sell pressure (highest first).  "
                "Stop=price−2×ATR  Target=price+3×ATR  "
                "STRONG_SELL≥75 | CONSIDER_SELL≥45 | HOLD≥20 | ADD=all 4 core signals neutral/bullish",
                style={"color": MUTED, "fontSize": "11px", "fontFamily": "monospace", "marginBottom": "10px"},
            ),
            html.Div(id="swing-table-container",
                     children=html.Div("Waiting for refresh…",
                                       style={"color": MUTED, "fontFamily": "monospace"})),
        ]),
    ])


# ---------------------------------------------------------------------------
# Tab 2 — Risk & Capital layout
# ---------------------------------------------------------------------------

def _build_live_positions_card():
    """Build dashboard cards showing tracked live positions and their sell-side state.

    Returns a Div with:
      1. Summary stat cards (positions, exposure, risk, stages)
      2. Per-position price ladder chart (entry → stop → partials → target)
      3. Trailing stop stage distribution donut
      4. Time-to-expiry / holding-period timeline bars
      5. Unrealized P&L waterfall with risk/reward zones
      6. Detailed positions table
    """
    try:
        from trader import get_open_positions_snapshot
        positions = get_open_positions_snapshot()
    except Exception:
        positions = {}

    if not positions:
        return _card([
            html.H6("🔴 Live Position Management",
                    style={"color": ACCENT, "fontFamily": "monospace", "marginBottom": "4px"}),
            html.Div("No tracked positions.  Positions are registered when orders are placed in live mode.",
                     style={"color": MUTED, "fontFamily": "monospace", "fontSize": "12px"}),
        ])

    # ── Fetch current prices for all tracked symbols ─────────────────────────
    tracked_symbols = list(positions.keys())
    current_prices = _fetch_current_prices(tracked_symbols)

    # ── Build enriched row data for the table + charts ───────────────────────
    rows = []
    chart_data = []  # for price ladder chart
    for sym, state in positions.items():
        holding_days = (datetime.date.today() - state.entry_date).days
        time_left = max(0, config.TIME_STOP_DAYS - holding_days) if config.TIME_STOP_ENABLED else None
        cur_price = current_prices.get(sym)
        entry = state.entry_price
        atr = state.atr_14_abs
        stop = state.trailing_stop.current_stop
        high = state.trailing_stop.highest_price

        # Compute key price levels
        partial_1_price = round(entry + config.PARTIAL_EXIT_1_TRIGGER_ATR * atr, 2)
        partial_2_price = round(entry + config.PARTIAL_EXIT_2_TRIGGER_ATR * atr, 2)
        target_price = round(entry + config.ATR_TARGET_MULTIPLIER * atr, 2)

        # Unrealized P&L
        unreal_dollar = round((cur_price - entry) * state.quantity, 2) if cur_price else None
        unreal_pct = round((cur_price / entry - 1) * 100, 1) if cur_price and entry > 0 else None
        gain_in_atr = round((cur_price - entry) / atr, 2) if cur_price and atr > 0 else None

        # Risk & reward amounts (at current qty)
        risk_at_stop = round((entry - stop) * state.quantity, 2)
        gain_at_p1 = round((partial_1_price - entry) * state.quantity, 2)
        gain_at_p2 = round((partial_2_price - entry) * state.quantity, 2)
        gain_at_target = round((target_price - entry) * state.quantity, 2)

        rows.append({
            "Symbol":        sym,
            "Qty":           state.quantity,
            "Entry $":       round(entry, 2),
            "Current $":     round(cur_price, 2) if cur_price else "n/a",
            "P&L $":         f"{unreal_dollar:+,.2f}" if unreal_dollar is not None else "n/a",
            "P&L %":         f"{unreal_pct:+.1f}%" if unreal_pct is not None else "n/a",
            "Gain ×ATR":     f"{gain_in_atr:+.1f}" if gain_in_atr is not None else "n/a",
            "Entry Date":    str(state.entry_date),
            "Hold Days":     holding_days,
            "Time Left":     f"{time_left}d" if time_left is not None else "—",
            "ATR $":         round(atr, 2),
            "Trail Stage":   state.trailing_stop.stage_label,
            "Stop $":        round(stop, 2),
            "P1 $":          partial_1_price,
            "P2 $":          partial_2_price,
            "Target $":      target_price,
            "High $":        round(high, 2),
            "Partials":      ", ".join(str(x) for x in sorted(state.partial_exits_taken)) or "none",
            "Updates":       state.trailing_stop.n_updates,
        })

        chart_data.append({
            "sym": sym, "entry": entry, "stop": stop, "current": cur_price,
            "high": high, "partial_1": partial_1_price, "partial_2": partial_2_price,
            "target": target_price, "atr": atr, "qty": state.quantity,
            "holding_days": holding_days, "time_left": time_left,
            "stage": state.trailing_stop.stage_label,
            "unreal_dollar": unreal_dollar, "unreal_pct": unreal_pct,
            "gain_in_atr": gain_in_atr,
            "risk_at_stop": risk_at_stop,
            "gain_at_p1": gain_at_p1, "gain_at_p2": gain_at_p2,
            "gain_at_target": gain_at_target,
            "partials_taken": state.partial_exits_taken,
        })

    pos_df = pd.DataFrame(rows)
    n_pos = len(positions)

    # ── Stat cards ────────────────────────────────────────────────────────────
    total_exposure = sum(s.entry_price * s.quantity for s in positions.values())
    total_risk_at_stop = sum(d["risk_at_stop"] for d in chart_data)
    total_unreal = sum(d["unreal_dollar"] for d in chart_data if d["unreal_dollar"] is not None)
    avg_hold = sum(d["holding_days"] for d in chart_data) / n_pos if n_pos else 0
    n_profitable = sum(1 for d in chart_data if d["unreal_dollar"] is not None and d["unreal_dollar"] > 0)
    stage_counts = {}
    for d in chart_data:
        stage_counts[d["stage"]] = stage_counts.get(d["stage"], 0) + 1

    live_stat_cards = dbc.Row([
        _stat_card("Tracked Positions",  str(n_pos),                          ACCENT),
        _stat_card("Total Exposure",     f"${total_exposure:,.0f}",           TEXT),
        _stat_card("Unrealized P&L",     f"${total_unreal:+,.2f}",
                   GREEN if total_unreal >= 0 else RED),
        _stat_card("$ at Risk (stop)",   f"${total_risk_at_stop:,.0f}",       RED),
        _stat_card("Avg Hold",           f"{avg_hold:.0f}d / {config.TIME_STOP_DAYS}d", TEXT),
        _stat_card("Profitable",         f"{n_profitable}/{n_pos}",
                   GREEN if n_profitable > n_pos / 2 else YELLOW),
    ], className="mb-3")

    # ── Chart 1: Per-position price ladder (bullet / range chart) ────────────
    # Shows stop → entry → partial1 → partial2 → target as horizontal ranges
    # with a marker for the current price
    ladder_fig = go.Figure()
    syms = [d["sym"] for d in chart_data]

    # Risk zone: stop → entry (red shading)
    ladder_fig.add_trace(go.Bar(
        y=syms,
        x=[d["entry"] - d["stop"] for d in chart_data],
        base=[d["stop"] for d in chart_data],
        orientation="h",
        name="🔴 Risk (stop → entry)",
        marker_color="rgba(196,30,30,0.20)",
        hovertemplate="%{y}: Stop $%{base:.2f} → Entry $%{x:.2f}<extra>Risk zone</extra>",
    ))
    # Zone: entry → partial 1
    ladder_fig.add_trace(go.Bar(
        y=syms,
        x=[d["partial_1"] - d["entry"] for d in chart_data],
        base=[d["entry"] for d in chart_data],
        orientation="h",
        name=f"� → P1 (+{config.PARTIAL_EXIT_1_TRIGGER_ATR:.0f}×ATR)",
        marker_color="rgba(160,155,148,0.30)",
        hovertemplate="%{y}: Entry → Partial 1 $%{x:.2f}<extra></extra>",
    ))
    # Zone: partial 1 → partial 2
    ladder_fig.add_trace(go.Bar(
        y=syms,
        x=[d["partial_2"] - d["partial_1"] for d in chart_data],
        base=[d["partial_1"] for d in chart_data],
        orientation="h",
        name=f"⚪ → P2 (+{config.PARTIAL_EXIT_2_TRIGGER_ATR:.0f}×ATR)",
        marker_color="rgba(130,125,118,0.25)",
        hovertemplate="%{y}: Partial 1 → Partial 2 $%{x:.2f}<extra></extra>",
    ))
    # Zone: partial 2 → target
    ladder_fig.add_trace(go.Bar(
        y=syms,
        x=[d["target"] - d["partial_2"] for d in chart_data],
        base=[d["partial_2"] for d in chart_data],
        orientation="h",
        name=f"◻ → Target (+{config.ATR_TARGET_MULTIPLIER:.0f}×ATR)",
        marker_color="rgba(100,95,88,0.20)",
        hovertemplate="%{y}: Partial 2 → Target $%{x:.2f}<extra></extra>",
    ))

    # Current price markers
    cur_prices_list = [d["current"] for d in chart_data]
    cur_colours = []
    for d in chart_data:
        if d["current"] is None:
            cur_colours.append(MUTED)
        elif d["current"] >= d["partial_2"]:
            cur_colours.append(GREEN)
        elif d["current"] >= d["entry"]:
            cur_colours.append(YELLOW)
        else:
            cur_colours.append(RED)

    ladder_fig.add_trace(go.Scatter(
        y=syms,
        x=[p if p is not None else 0 for p in cur_prices_list],
        mode="markers+text",
        marker=dict(symbol="diamond", size=14, color=cur_colours,
                    line=dict(color=TEXT, width=1.5)),
        text=[f"${p:.2f}" if p else "n/a" for p in cur_prices_list],
        textposition="top center",
        textfont=dict(color=TEXT, size=10),
        name="◆ Current Price",
        hovertemplate="%{y}: $%{x:.2f}<extra>Current</extra>",
    ))

    # Trailing stop markers (distinct from risk zone)
    ladder_fig.add_trace(go.Scatter(
        y=syms,
        x=[d["stop"] for d in chart_data],
        mode="markers",
        marker=dict(symbol="triangle-left", size=10, color=RED,
                    line=dict(color=RED, width=1)),
        name="◀ Trailing Stop",
        hovertemplate="%{y}: Stop $%{x:.2f}<extra>Trail stop</extra>",
    ))

    ladder_fig.update_layout(
        title="Per-Position Price Ladder — Sell-Side Levels",
        barmode="stack",
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT),
        xaxis=dict(gridcolor=BORDER, color=TEXT, title="Price ($)"),
        yaxis=dict(color=TEXT, gridcolor=BORDER),
        margin=dict(t=40, b=20, l=20, r=20),
        legend=dict(font=dict(color=TEXT, size=10), orientation="h",
                    yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=max(250, 70 * n_pos),
    )

    # ── Chart 2: Trailing stop stage distribution donut ──────────────────────
    stage_order = ["INITIAL", "BREAKEVEN", "PROFIT_LOCK", "TIGHT_TRAIL"]
    stage_colour_map = {"INITIAL": MUTED, "BREAKEVEN": YELLOW,
                        "PROFIT_LOCK": GREEN, "TIGHT_TRAIL": ACCENT}
    stage_labels = [s for s in stage_order if s in stage_counts]
    stage_values = [stage_counts[s] for s in stage_labels]
    stage_colors = [stage_colour_map.get(s, MUTED) for s in stage_labels]

    stage_fig = go.Figure(go.Pie(
        labels=stage_labels, values=stage_values,
        hole=0.55,
        marker=dict(colors=stage_colors),
        textinfo="label+value",
        textfont=dict(color=TEXT, size=12),
        hovertemplate="%{label}: %{value} position(s)  (%{percent})<extra></extra>",
    ))
    stage_fig.update_layout(
        title="Trail Stage Distribution",
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT),
        margin=dict(t=40, b=10, l=10, r=10),
        showlegend=False,
        height=280,
        annotations=[dict(text=f"{n_pos}", x=0.5, y=0.5, font_size=24,
                          showarrow=False, font_color=ACCENT)],
    )

    # ── Chart 3: Holding period timeline (horizontal bars) ───────────────────
    hold_syms = [d["sym"] for d in chart_data]
    hold_days_vals = [d["holding_days"] for d in chart_data]
    time_left_vals = [d["time_left"] if d["time_left"] is not None else 0 for d in chart_data]

    hold_fig = go.Figure()
    hold_fig.add_trace(go.Bar(
        y=hold_syms, x=hold_days_vals,
        orientation="h",
        name="Days Held",
        marker_color=[RED if hd >= config.TIME_STOP_DAYS * 0.8
                      else YELLOW if hd >= config.TIME_STOP_DAYS * 0.5
                      else GREEN for hd in hold_days_vals],
        text=[f"{d}d" for d in hold_days_vals],
        textposition="inside",
        hovertemplate="%{y}: %{x}d held<extra></extra>",
    ))
    hold_fig.add_trace(go.Bar(
        y=hold_syms, x=time_left_vals,
        orientation="h",
        name="Days Remaining",
        marker_color="rgba(200,195,188,0.35)",
        text=[f"{d}d left" if d > 0 else "⚠️" for d in time_left_vals],
        textposition="inside",
        textfont=dict(color=MUTED),
        hovertemplate="%{y}: %{x}d remaining<extra></extra>",
    ))
    hold_fig.add_vline(x=config.TIME_STOP_DAYS, line_dash="dash", line_color=RED,
                       annotation_text=f"Time Stop ({config.TIME_STOP_DAYS}d)",
                       annotation_font_color=RED)
    hold_fig.update_layout(
        title="Holding Period vs Time Stop",
        barmode="stack",
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT),
        xaxis=dict(gridcolor=BORDER, color=TEXT, title="Calendar Days"),
        yaxis=dict(color=TEXT, gridcolor=BORDER),
        margin=dict(t=40, b=20, l=20, r=20),
        showlegend=True,
        legend=dict(font=dict(color=TEXT, size=10)),
        height=max(220, 50 * n_pos),
    )

    # ── Chart 4: Unrealized P&L waterfall ────────────────────────────────────
    pnl_syms = [d["sym"] for d in chart_data]
    pnl_vals = [d["unreal_dollar"] if d["unreal_dollar"] is not None else 0 for d in chart_data]

    pnl_fig = go.Figure(go.Bar(
        x=pnl_syms, y=pnl_vals,
        marker_color=[GREEN if v >= 0 else RED for v in pnl_vals],
        text=[f"${v:+,.0f}" for v in pnl_vals],
        textposition="outside",
        hovertemplate="%{x}: $%{y:+,.2f}<extra></extra>",
    ))
    pnl_fig.add_hline(y=0, line_color=BORDER, line_dash="dash")
    pnl_fig.update_layout(
        title="Unrealized P&L by Position",
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT),
        xaxis=dict(color=TEXT, gridcolor=BORDER),
        yaxis=dict(color=TEXT, gridcolor=BORDER, title="P&L ($)"),
        margin=dict(t=40, b=20, l=20, r=20),
        height=280,
    )

    # ── Chart 5: Risk / reward breakdown per position ────────────────────────
    # Stacked bar: negative (risk at stop) + positive (gain at P1, P2, target)
    rr_syms = [d["sym"] for d in chart_data]
    risk_vals = [-d["risk_at_stop"] for d in chart_data]  # negative = downside
    p1_gain = [d["gain_at_p1"] for d in chart_data]
    p2_incremental = [d["gain_at_p2"] - d["gain_at_p1"] for d in chart_data]
    tgt_incremental = [d["gain_at_target"] - d["gain_at_p2"] for d in chart_data]

    rr_fig = go.Figure()
    rr_fig.add_trace(go.Bar(
        x=rr_syms, y=risk_vals,
        name="⬇ Risk @ Stop",
        marker_color="rgba(196,30,30,0.55)",
        text=[f"-${abs(v):,.0f}" for v in risk_vals],
        textposition="outside",
        hovertemplate="%{x}: -$%{y:,.0f} at stop<extra></extra>",
    ))
    rr_fig.add_trace(go.Bar(
        x=rr_syms, y=p1_gain,
        name=f"⬆ Gain @ P1 (+{config.PARTIAL_EXIT_1_TRIGGER_ATR:.0f}×ATR)",
        marker_color="rgba(140,135,128,0.55)",
        hovertemplate="%{x}: +$%{y:,.0f} at P1<extra></extra>",
    ))
    rr_fig.add_trace(go.Bar(
        x=rr_syms, y=p2_incremental,
        name=f"⬆ Gain @ P2 (+{config.PARTIAL_EXIT_2_TRIGGER_ATR:.0f}×ATR)",
        marker_color="rgba(110,105,98,0.55)",
        hovertemplate="%{x}: +$%{y:,.0f} incremental at P2<extra></extra>",
    ))
    rr_fig.add_trace(go.Bar(
        x=rr_syms, y=tgt_incremental,
        name=f"⬆ Gain @ Target (+{config.ATR_TARGET_MULTIPLIER:.0f}×ATR)",
        marker_color="rgba(80,76,70,0.50)",
        hovertemplate="%{x}: +$%{y:,.0f} incremental at target<extra></extra>",
    ))
    rr_fig.add_hline(y=0, line_color=BORDER, line_dash="dash")
    rr_fig.update_layout(
        title="Risk/Reward Breakdown — $ at Each Exit Level",
        barmode="relative",
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT),
        xaxis=dict(color=TEXT, gridcolor=BORDER),
        yaxis=dict(color=TEXT, gridcolor=BORDER, title="$ Risk (−) / Reward (+)"),
        margin=dict(t=40, b=20, l=20, r=20),
        legend=dict(font=dict(color=TEXT, size=10), orientation="h",
                    yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=300,
    )

    # ── Chart 6: ATR Efficiency Scatter (Gain ×ATR vs Holding Days) ──────────
    # Shows which positions are converting holding time into ATR-multiple gains
    scatter_syms = [d["sym"] for d in chart_data]
    scatter_hold = [d["holding_days"] for d in chart_data]
    scatter_gain_atr = [d["gain_in_atr"] if d["gain_in_atr"] is not None else 0 for d in chart_data]
    scatter_stages = [d["stage"] for d in chart_data]
    scatter_exposure = [d["entry"] * d["qty"] for d in chart_data]

    # Marker colours by trail stage
    _stage_scatter_colours = {
        "INITIAL": MUTED, "BREAKEVEN": YELLOW,
        "PROFIT_LOCK": GREEN, "TIGHT_TRAIL": ACCENT,
    }
    scatter_colours = [_stage_scatter_colours.get(s, MUTED) for s in scatter_stages]
    # Marker size proportional to exposure (normalised 10–40)
    max_exp = max(scatter_exposure) if scatter_exposure else 1
    scatter_sizes = [max(10, min(40, int(e / max_exp * 30 + 10))) for e in scatter_exposure]

    efficiency_fig = go.Figure()
    efficiency_fig.add_trace(go.Scatter(
        x=scatter_hold,
        y=scatter_gain_atr,
        mode="markers+text",
        marker=dict(size=scatter_sizes, color=scatter_colours,
                    line=dict(color=TEXT, width=1),
                    opacity=0.85),
        text=scatter_syms,
        textposition="top center",
        textfont=dict(color=TEXT, size=10),
        hovertemplate=(
            "%{text}<br>"
            "Hold: %{x}d<br>"
            "Gain: %{y:+.2f}×ATR<br>"
            "<extra></extra>"
        ),
    ))
    # Reference lines for trail stage triggers
    efficiency_fig.add_hline(y=0, line_color=BORDER, line_dash="dash")
    efficiency_fig.add_hline(y=config.TRAILING_STAGE1_TRIGGER_ATR,
                             line_color=YELLOW, line_dash="dot",
                             annotation_text=f"BE ({config.TRAILING_STAGE1_TRIGGER_ATR:.0f}×ATR)",
                             annotation_font_color=YELLOW, annotation_position="top left")
    efficiency_fig.add_hline(y=config.TRAILING_STAGE2_TRIGGER_ATR,
                             line_color=GREEN, line_dash="dot",
                             annotation_text=f"Lock ({config.TRAILING_STAGE2_TRIGGER_ATR:.0f}×ATR)",
                             annotation_font_color=GREEN, annotation_position="top left")
    efficiency_fig.add_hline(y=config.TRAILING_STAGE3_TRIGGER_ATR,
                             line_color=ACCENT, line_dash="dot",
                             annotation_text=f"Tight ({config.TRAILING_STAGE3_TRIGGER_ATR:.0f}×ATR)",
                             annotation_font_color=ACCENT, annotation_position="top left")
    # Time stop vertical line
    efficiency_fig.add_vline(x=config.TIME_STOP_DAYS, line_color=RED, line_dash="dash",
                             annotation_text=f"Time Stop ({config.TIME_STOP_DAYS}d)",
                             annotation_font_color=RED)
    efficiency_fig.update_layout(
        title="ATR Efficiency — Gain ×ATR vs Holding Days",
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT),
        xaxis=dict(color=TEXT, gridcolor=BORDER, title="Days Held"),
        yaxis=dict(color=TEXT, gridcolor=BORDER, title="Gain (×ATR)"),
        margin=dict(t=40, b=20, l=20, r=20),
        showlegend=False,
        height=320,
    )

    # ── Chart 7: Exit Proximity Radar — how close to each trigger ────────────
    # For each position, show % distance to next partial, target, and time stop
    prox_categories = ["P1 Exit", "P2 Exit", "Target", "Time Stop", "Trail Stop Hit"]
    prox_data = []  # list of dicts with sym + proximity values
    for d in chart_data:
        cur = d["current"]
        if cur is None:
            continue
        entry = d["entry"]
        atr = d["atr"]

        # Distance to P1 (% of the way from entry to P1)
        p1_dist = d["partial_1"] - entry
        p1_progress = min(100, max(0, (cur - entry) / p1_dist * 100)) if p1_dist > 0 else 0

        # Distance to P2 (% of the way from entry to P2)
        p2_dist = d["partial_2"] - entry
        p2_progress = min(100, max(0, (cur - entry) / p2_dist * 100)) if p2_dist > 0 else 0

        # Distance to target
        tgt_dist = d["target"] - entry
        tgt_progress = min(100, max(0, (cur - entry) / tgt_dist * 100)) if tgt_dist > 0 else 0

        # Time stop progress
        time_progress = min(100, d["holding_days"] / config.TIME_STOP_DAYS * 100) if config.TIME_STOP_ENABLED else 0

        # Trail stop proximity: how close is current price to the trail stop
        # (100% = sitting on stop, 0% = far above)
        stop = d["stop"]
        if cur > stop and entry > stop:
            trail_prox = max(0, min(100, (1 - (cur - stop) / (entry - stop + atr)) * 100))
        elif cur <= stop:
            trail_prox = 100
        else:
            trail_prox = 0

        prox_data.append({
            "sym": d["sym"],
            "P1 Exit": round(p1_progress, 1),
            "P2 Exit": round(p2_progress, 1),
            "Target": round(tgt_progress, 1),
            "Time Stop": round(time_progress, 1),
            "Trail Stop Hit": round(trail_prox, 1),
        })

    if prox_data:
        prox_fig = go.Figure()
        for pd_item in prox_data:
            prox_fig.add_trace(go.Scatterpolar(
                r=[pd_item[c] for c in prox_categories],
                theta=prox_categories,
                fill="toself",
                name=pd_item["sym"],
                opacity=0.6,
                line=dict(width=2),
            ))
        prox_fig.update_layout(
            title="Exit Proximity Radar — % Progress to Each Trigger",
            polar=dict(
                radialaxis=dict(
                    visible=True, range=[0, 100],
                    gridcolor=BORDER, color=MUTED,
                    ticksuffix="%",
                ),
                angularaxis=dict(gridcolor=BORDER, color=TEXT),
                bgcolor=CARD_BG,
            ),
            paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
            font=dict(color=TEXT),
            margin=dict(t=50, b=20, l=40, r=40),
            showlegend=True,
            legend=dict(font=dict(color=TEXT, size=10)),
            height=380,
        )
    else:
        prox_fig = go.Figure()
        prox_fig.update_layout(
            paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
            annotations=[dict(text="No price data", x=0.5, y=0.5,
                              showarrow=False, font=dict(color=MUTED, size=14))],
            height=300,
        )

    # ── Chart 8: Trailing Stop Progression — how far each stop has moved ─────
    # Bullet-style: initial_stop → current_stop → entry → current_price
    trail_syms = [d["sym"] for d in chart_data]
    trail_initial = []
    trail_current_stop = []
    trail_max_possible = []
    for sym, state in positions.items():
        ts = state.trailing_stop
        trail_initial.append(ts.initial_stop)
        trail_current_stop.append(ts.current_stop)
        # Max possible trail = entry + best case (stage 3 trail would move up to)
        trail_max_possible.append(ts.highest_price - config.TRAILING_STAGE3_TRAIL_ATR * ts.atr_14_abs
                                  if ts.highest_price > ts.entry_price else ts.entry_price)

    # Stop moved: current_stop - initial_stop (positive = good, stop has tightened)
    stop_moved = [c - i for c, i in zip(trail_current_stop, trail_initial)]

    trail_fig = go.Figure()
    trail_fig.add_trace(go.Bar(
        y=trail_syms, x=stop_moved,
        orientation="h",
        name="Stop Moved $",
        marker_color=[GREEN if m > 0 else (YELLOW if m == 0 else RED) for m in stop_moved],
        text=[f"${m:+.2f}" for m in stop_moved],
        textposition="outside",
        hovertemplate="%{y}: Stop moved %{x:+$.2f}<br>Initial: $%{customdata[0]:.2f}<br>Current: $%{customdata[1]:.2f}<extra></extra>",
        customdata=list(zip(trail_initial, trail_current_stop)),
    ))
    trail_fig.add_vline(x=0, line_color=BORDER, line_dash="dash")
    trail_fig.update_layout(
        title="Trailing Stop Progression — $ Moved from Initial",
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT),
        xaxis=dict(color=TEXT, gridcolor=BORDER, title="$ Stop Moved (+ = tightened)"),
        yaxis=dict(color=TEXT, gridcolor=BORDER),
        margin=dict(t=40, b=20, l=20, r=20),
        showlegend=False,
        height=max(220, 50 * n_pos),
    )

    # ── Chart 9: Position Health Heatmap ─────────────────────────────────────
    # Colour-coded grid: rows = symbols, cols = health metrics
    health_metrics = ["P&L %", "Gain ×ATR", "Hold %", "Trail Stage", "Partials Done"]
    health_z = []   # numeric matrix for heatmap
    health_text = []  # text annotations
    for d in chart_data:
        pnl_pct = d["unreal_pct"] if d["unreal_pct"] is not None else 0
        gain_atr_val = d["gain_in_atr"] if d["gain_in_atr"] is not None else 0
        hold_pct = d["holding_days"] / config.TIME_STOP_DAYS * 100 if config.TIME_STOP_ENABLED else 0
        stage_num = {"INITIAL": 0, "BREAKEVEN": 1, "PROFIT_LOCK": 2, "TIGHT_TRAIL": 3}.get(d["stage"], 0)
        partials_done = len(d["partials_taken"])

        # Normalise each metric to 0–100 for colour mapping
        # P&L%: -10% = 0, +10% = 100 (clamped)
        pnl_norm = max(0, min(100, (pnl_pct + 10) / 20 * 100))
        # Gain×ATR: -2 = 0, +3 = 100
        atr_norm = max(0, min(100, (gain_atr_val + 2) / 5 * 100))
        # Hold%: 0 = 100 (good), 100 = 0 (bad) — inverted: more time left = healthier
        hold_norm = max(0, min(100, 100 - hold_pct))
        # Stage: 0→25, 1→50, 2→75, 3→100
        stage_norm = stage_num / 3 * 100
        # Partials: 0→33, 1→66, 2→100
        partial_norm = partials_done / 2 * 100

        health_z.append([pnl_norm, atr_norm, hold_norm, stage_norm, partial_norm])
        health_text.append([
            f"{pnl_pct:+.1f}%",
            f"{gain_atr_val:+.1f}×",
            f"{d['holding_days']}d/{config.TIME_STOP_DAYS}d",
            d["stage"],
            f"{partials_done}/2",
        ])

    health_fig = go.Figure(go.Heatmap(
        z=health_z,
        x=health_metrics,
        y=[d["sym"] for d in chart_data],
        text=health_text,
        texttemplate="%{text}",
        textfont=dict(size=11, color=TEXT),
        colorscale=[
            [0.0, "#c41e1e"],     # RED — critical / poor
            [0.25, "#e0b0b0"],    # faded rose — below average
            [0.5, "#e0ddd8"],     # warm neutral — average
            [0.75, "#c8c4bc"],    # warm grey — good
            [1.0, "#8a8580"],     # dark warm grey — excellent
        ],
        showscale=False,
        hovertemplate="%{y} — %{x}: %{text}<extra></extra>",
    ))
    health_fig.update_layout(
        title="Position Health Heatmap",
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT),
        xaxis=dict(color=TEXT, gridcolor=BORDER, side="top"),
        yaxis=dict(color=TEXT, gridcolor=BORDER, autorange="reversed"),
        margin=dict(t=60, b=20, l=20, r=20),
        height=max(200, 55 * n_pos + 80),
    )

    # ── Chart 10: Portfolio P&L Distribution (box + strip) ───────────────────
    pnl_pct_vals = [d["unreal_pct"] if d["unreal_pct"] is not None else 0 for d in chart_data]

    dist_fig = go.Figure()
    dist_fig.add_trace(go.Box(
        y=pnl_pct_vals,
        name="P&L %",
        marker_color=ACCENT,
        line_color=ACCENT,
        fillcolor="rgba(90,90,90,0.08)",
        boxpoints="all",
        jitter=0.4,
        pointpos=-1.5,
        text=[d["sym"] for d in chart_data],
        hovertemplate="%{text}: %{y:+.1f}%<extra></extra>",
        marker=dict(
            color=[GREEN if v >= 0 else RED for v in pnl_pct_vals],
            size=10,
            line=dict(color=TEXT, width=1),
        ),
    ))
    dist_fig.add_hline(y=0, line_color=BORDER, line_dash="dash")

    # Add mean line
    mean_pnl = sum(pnl_pct_vals) / len(pnl_pct_vals) if pnl_pct_vals else 0
    dist_fig.add_hline(y=mean_pnl, line_color=YELLOW, line_dash="dot",
                       annotation_text=f"Avg: {mean_pnl:+.1f}%",
                       annotation_font_color=YELLOW)
    dist_fig.update_layout(
        title="Unrealized P&L Distribution",
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT),
        yaxis=dict(color=TEXT, gridcolor=BORDER, title="P&L %"),
        margin=dict(t=40, b=20, l=20, r=20),
        showlegend=False,
        height=280,
    )

    # ── Positions table with colour-coding ───────────────────────────────────
    stage_colours = [
        {"if": {"filter_query": '{Trail Stage} = "INITIAL"',      "column_id": "Trail Stage"}, "color": MUTED},
        {"if": {"filter_query": '{Trail Stage} = "BREAKEVEN"',    "column_id": "Trail Stage"}, "color": YELLOW},
        {"if": {"filter_query": '{Trail Stage} = "PROFIT_LOCK"',  "column_id": "Trail Stage"}, "color": GREEN},
        {"if": {"filter_query": '{Trail Stage} = "TIGHT_TRAIL"',  "column_id": "Trail Stage"}, "color": GREEN, "fontWeight": "bold"},
        # P&L colouring
        {"if": {"filter_query": '{P&L %} contains "+"', "column_id": "P&L %"}, "color": GREEN, "fontWeight": "bold"},
        {"if": {"filter_query": '{P&L %} contains "-"', "column_id": "P&L %"}, "color": RED,   "fontWeight": "bold"},
        {"if": {"filter_query": '{P&L $} contains "+"', "column_id": "P&L $"}, "color": GREEN},
        {"if": {"filter_query": '{P&L $} contains "-"', "column_id": "P&L $"}, "color": RED},
        {"if": {"filter_query": '{Gain ×ATR} contains "+"', "column_id": "Gain ×ATR"}, "color": GREEN},
        {"if": {"filter_query": '{Gain ×ATR} contains "-"', "column_id": "Gain ×ATR"}, "color": RED},
    ]

    pos_table = dash_table.DataTable(
        id="live-positions-table",
        columns=[{"name": c, "id": c} for c in pos_df.columns],
        data=pos_df.to_dict("records"),
        style_cell=_CELL_STYLE,
        style_header=_HDR_STYLE,
        style_data_conditional=stage_colours,
        style_table={"overflowX": "auto", "borderRadius": "6px"},
        sort_action="native",
        page_size=20,
    )

    # ── Assemble all sell-side cards ─────────────────────────────────────────
    return html.Div([
        _card([
            html.H6("🟢 Live Position Management — Sell-Side State",
                    style={"color": ACCENT, "fontFamily": "monospace", "marginBottom": "4px"}),
            html.Div(
                f"Tracked positions with trailing stops, partial exits, and time stops.  "
                f"Time stop: {config.TIME_STOP_DAYS}d max hold.  "
                f"Partial exits at +{config.PARTIAL_EXIT_1_TRIGGER_ATR:.0f}×ATR and +{config.PARTIAL_EXIT_2_TRIGGER_ATR:.0f}×ATR.  "
                f"Trail stages: INITIAL → BREAKEVEN → PROFIT_LOCK → TIGHT_TRAIL.",
                style={"color": MUTED, "fontSize": "11px", "fontFamily": "monospace", "marginBottom": "10px"},
            ),
            live_stat_cards,
        ]),

        # Position health heatmap (full width — at-a-glance overview)
        _card([
            html.H6("🩺 Position Health Heatmap",
                    style={"color": ACCENT, "fontFamily": "monospace", "marginBottom": "4px"}),
            html.Div(
                "Colour-coded snapshot of each position across key health dimensions.  "
                "Red = poor/at-risk, Yellow = caution, Green/Blue = healthy/advanced.  "
                "P&L normalised ±10%, ATR ±2→+3×, Hold inverted (more time = better), Stage 0–3, Partials 0–2.",
                style={"color": MUTED, "fontSize": "11px", "fontFamily": "monospace", "marginBottom": "10px"},
            ),
            dcc.Graph(figure=health_fig, config={"displayModeBar": False}),
        ]),

        # Price ladder (full width — the main sell-side visualization)
        _card([
            html.H6("🎯 Price Ladder — Entry, Stops, Partials & Targets",
                    style={"color": ACCENT, "fontFamily": "monospace", "marginBottom": "4px"}),
            html.Div(
                "Horizontal bars show the risk zone (stop→entry, red), "
                "profit-taking zones (P1 at +2×ATR, P2 at +3×ATR), and full target.  "
                "◆ = current price.  ◀ = trailing stop level.",
                style={"color": MUTED, "fontSize": "11px", "fontFamily": "monospace", "marginBottom": "10px"},
            ),
            dcc.Graph(figure=ladder_fig, config={"displayModeBar": False}),
        ]),

        # Row: stage donut + holding period + unrealized P&L
        dbc.Row([
            dbc.Col(_card([
                dcc.Graph(figure=stage_fig, config={"displayModeBar": False}),
            ]), width=3),
            dbc.Col(_card([
                dcc.Graph(figure=hold_fig, config={"displayModeBar": False}),
            ]), width=4),
            dbc.Col(_card([
                dcc.Graph(figure=pnl_fig, config={"displayModeBar": False}),
            ]), width=5),
        ], className="mb-3"),

        # Row: ATR efficiency scatter + Exit proximity radar
        dbc.Row([
            dbc.Col(_card([
                html.H6("📈 ATR Efficiency — Gain vs Time",
                        style={"color": ACCENT, "fontFamily": "monospace", "marginBottom": "4px"}),
                html.Div(
                    "Each position plotted by days held vs ATR multiples gained.  "
                    "Bubble size ∝ exposure.  Colour = trail stage.  "
                    "Dotted lines mark trail-stage triggers; red vertical = time stop.",
                    style={"color": MUTED, "fontSize": "11px", "fontFamily": "monospace", "marginBottom": "10px"},
                ),
                dcc.Graph(figure=efficiency_fig, config={"displayModeBar": False}),
            ]), width=6),
            dbc.Col(_card([
                html.H6("🎯 Exit Proximity Radar",
                        style={"color": ACCENT, "fontFamily": "monospace", "marginBottom": "4px"}),
                html.Div(
                    "Radar chart showing % progress toward each exit trigger.  "
                    "P1/P2/Target = price progress from entry.  "
                    "Time Stop = days held as % of max.  Trail Stop Hit = closeness to stop.",
                    style={"color": MUTED, "fontSize": "11px", "fontFamily": "monospace", "marginBottom": "10px"},
                ),
                dcc.Graph(figure=prox_fig, config={"displayModeBar": False}),
            ]), width=6),
        ], className="mb-3"),

        # Risk/reward breakdown (full width)
        _card([
            html.H6("⚖️ Risk / Reward Breakdown",
                    style={"color": ACCENT, "fontFamily": "monospace", "marginBottom": "4px"}),
            html.Div(
                f"Red bar = downside risk to trailing stop.  "
                f"Yellow = gain at Partial 1 ({config.PARTIAL_EXIT_1_FRACTION:.0%} @ +{config.PARTIAL_EXIT_1_TRIGGER_ATR:.0f}×ATR).  "
                f"Green = incremental at Partial 2 ({config.PARTIAL_EXIT_2_FRACTION:.0%} @ +{config.PARTIAL_EXIT_2_TRIGGER_ATR:.0f}×ATR).  "
                f"Blue = incremental to full target (+{config.ATR_TARGET_MULTIPLIER:.0f}×ATR).",
                style={"color": MUTED, "fontSize": "11px", "fontFamily": "monospace", "marginBottom": "10px"},
            ),
            dcc.Graph(figure=rr_fig, config={"displayModeBar": False}),
        ]),

        # Row: trailing stop progression + P&L distribution
        dbc.Row([
            dbc.Col(_card([
                html.H6("📐 Trailing Stop Progression",
                        style={"color": ACCENT, "fontFamily": "monospace", "marginBottom": "4px"}),
                html.Div(
                    "How far each position's trailing stop has moved from its initial level.  "
                    "Green = stop has tightened (protecting gains).  Grey = no movement yet.",
                    style={"color": MUTED, "fontSize": "11px", "fontFamily": "monospace", "marginBottom": "10px"},
                ),
                dcc.Graph(figure=trail_fig, config={"displayModeBar": False}),
            ]), width=7),
            dbc.Col(_card([
                html.H6("📊 P&L Distribution",
                        style={"color": ACCENT, "fontFamily": "monospace", "marginBottom": "4px"}),
                html.Div(
                    "Box plot showing the spread of unrealized P&L across all positions.  "
                    "Individual dots show each position.  Dotted line = average.",
                    style={"color": MUTED, "fontSize": "11px", "fontFamily": "monospace", "marginBottom": "10px"},
                ),
                dcc.Graph(figure=dist_fig, config={"displayModeBar": False}),
            ]), width=5),
        ], className="mb-3"),

        # Detailed table
        _card([
            html.H6("📋 Position Detail Table",
                    style={"color": ACCENT, "fontFamily": "monospace", "marginBottom": "4px"}),
            html.Div(
                "Sortable/filterable.  P&L columns show unrealized gain/loss.  "
                "Gain ×ATR = how many ATR units above/below entry.  "
                "Partials = which staged exits have been taken (1 and/or 2).",
                style={"color": MUTED, "fontSize": "11px", "fontFamily": "monospace", "marginBottom": "10px"},
            ),
            pos_table,
        ]),
    ])


def _risk_layout(screen_df: pd.DataFrame | None):
    if screen_df is None or screen_df.empty:
        return _card(html.Div("Run the screener first (📡 Screener → 🔄 Refresh).",
                               style={"color": MUTED, "fontFamily": "monospace"}))

    risk_df, alloc = _risk_df(screen_df)
    if risk_df.empty or alloc is None:
        return _card(html.Div("No dip candidates found — nothing to size.",
                               style={"color": MUTED, "fontFamily": "monospace"}))

    n_candidates  = len(alloc.orders)
    total_deployed = alloc.total_notional
    total_risk     = alloc.total_risk
    cash_remaining = alloc.cash_remaining
    pct_deployed   = alloc.pct_deployed * 100
    max_deploy_cap = ACCOUNT_EQUITY * config.MAX_CAPITAL_DEPLOYED_PCT

    stat_cards = dbc.Row([
        _stat_card("Budget",              f"${ACCOUNT_EQUITY:,.0f}",  ACCENT),
        _stat_card("Max Deployable",      f"${max_deploy_cap:,.0f}  ({config.MAX_CAPITAL_DEPLOYED_PCT:.0%})",  MUTED),
        _stat_card("Deployed",            f"${total_deployed:,.0f}  ({pct_deployed:.1f}%)",
                   GREEN if pct_deployed <= config.MAX_CAPITAL_DEPLOYED_PCT * 100 else YELLOW),
        _stat_card("Cash Remaining",      f"${cash_remaining:,.0f}",  TEXT),
        _stat_card("Total $ at Risk",     f"${total_risk:,.0f}  ({total_risk/ACCOUNT_EQUITY*100:.1f}%)", RED),
        _stat_card("Positions Allocated", str(n_candidates),          ACCENT),
        _stat_card("Skipped",             str(len(alloc.skipped)),    MUTED),
    ], className="mb-3")

    pos_table = _make_table(risk_df, "risk-table")

    # ── capital allocation pie (use alloc.orders for accepted positions) ─────
    pie_labels = [o.symbol for o in alloc.orders] + ["Cash"]
    pie_values = [o.position_value for o in alloc.orders] + [max(cash_remaining, 0)]

    pie_fig = go.Figure(go.Pie(
        labels=pie_labels,
        values=pie_values,
        hole=0.55,
        marker=dict(colors=px.colors.qualitative.Plotly),
        textinfo="label+percent",
        textfont=dict(color=TEXT, size=12),
    ))
    pie_fig.update_layout(
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT),
        margin=dict(t=30, b=10, l=10, r=10),
        showlegend=False,
        annotations=[dict(text=f"${ACCOUNT_EQUITY/1e3:.0f}k", x=0.5, y=0.5,
                          font_size=18, showarrow=False, font_color=ACCENT)],
    )

    # ── risk heat-map — % of budget + $ at risk for each accepted position ───
    hm_syms = [o.symbol       for o in alloc.orders]
    hm_pct  = [o.position_value / ACCOUNT_EQUITY * 100 for o in alloc.orders]
    hm_risk = [o.risk_amount  for o in alloc.orders]
    hm_rr   = [o.reward_risk_ratio for o in alloc.orders]

    heat_fig = go.Figure()
    heat_fig.add_trace(go.Bar(
        name="% of Budget", x=hm_syms, y=hm_pct,
        marker_color=ACCENT, text=[f"{v:.1f}%" for v in hm_pct],
        textposition="outside",
    ))
    heat_fig.add_trace(go.Bar(
        name="$ at Risk (÷100)", x=hm_syms,
        y=[r / 100 for r in hm_risk],
        marker_color=RED,
        text=[f"${r:,.0f}" for r in hm_risk],
        textposition="outside",
    ))
    heat_fig.add_hline(
        y=config.MAX_POSITION_PCT * 100, line_dash="dash",
        line_color=YELLOW, annotation_text=f"Max position {config.MAX_POSITION_PCT:.0%}",
        annotation_font_color=YELLOW,
    )
    heat_fig.add_hline(
        y=config.MAX_CAPITAL_DEPLOYED_PCT * 100 / max(len(alloc.orders), 1),
        line_dash="dot", line_color=MUTED,
        annotation_text="Equal-weight", annotation_font_color=MUTED,
    )
    heat_fig.update_layout(
        barmode="group",
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT), legend=dict(font=dict(color=TEXT)),
        xaxis=dict(gridcolor=BORDER, color=TEXT),
        yaxis=dict(gridcolor=BORDER, color=TEXT, title="% of Budget / $Risk÷100"),
        margin=dict(t=20, b=10, l=10, r=10),
    )

    # ── R/R bar (shows actual per-position reward:risk ratio) ─────────────────
    rr_fig = go.Figure(go.Bar(
        x=hm_syms, y=hm_rr,
        marker_color=[GREEN if r >= 2.0 else YELLOW if r >= 1.5 else RED for r in hm_rr],
        text=[f"{r:.2f}" for r in hm_rr],
        textposition="outside",
    ))
    rr_fig.add_hline(y=2.0, line_dash="dash", line_color=GREEN,
                     annotation_text="2:1 target", annotation_font_color=GREEN)
    rr_fig.add_hline(y=1.0, line_dash="dot",  line_color=RED,
                     annotation_text="1:1 min",   annotation_font_color=RED)
    rr_fig.update_layout(
        title="Reward : Risk by Position",
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT),
        xaxis=dict(color=TEXT), yaxis=dict(color=TEXT, gridcolor=BORDER, title="R/R Ratio"),
        margin=dict(t=40, b=10, l=10, r=10),
    )

    # ── risk parameters card ──────────────────────────────────────────────────
    avg_rr   = sum(hm_rr) / len(hm_rr) if hm_rr else 0
    kelly_enabled = config.KELLY_FRACTION > 0
    params_card = _card([
        html.H6("Risk Parameters", style={"color": ACCENT, "fontFamily": "monospace"}),
        html.Hr(style={"borderColor": BORDER}),
        dbc.Row([
            dbc.Col([
                html.Div(f"Stop method:     {'ATR-based' if config.USE_ATR_STOPS else 'Fixed %'}",
                         style={"fontFamily": "monospace", "fontSize": "13px", "color": ACCENT}),
                html.Div(f"ATR stop mult:   {config.ATR_STOP_MULTIPLIER}×  (clamp {config.ATR_MIN_STOP_PCT:.0%}–{config.ATR_MAX_STOP_PCT:.0%})",
                         style={"fontFamily": "monospace", "fontSize": "13px"}),
                html.Div(f"ATR target mult: {config.ATR_TARGET_MULTIPLIER}×",
                         style={"fontFamily": "monospace", "fontSize": "13px"}),
                html.Div(f"Fallback stop:   {config.STOP_LOSS_PCT:.0%}  fixed",
                         style={"fontFamily": "monospace", "fontSize": "13px", "color": MUTED}),
                html.Div(f"Portfolio avg RR: {avg_rr:.2f}:1",
                         style={"fontFamily": "monospace", "fontSize": "13px", "color": GREEN}),
            ], width=4),
            dbc.Col([
                html.Div(f"Risk per trade:  {config.RISK_PER_TRADE_PCT:.1%}  =  ${ACCOUNT_EQUITY * config.RISK_PER_TRADE_PCT:,.0f}",
                         style={"fontFamily": "monospace", "fontSize": "13px"}),
                html.Div(f"Max position:    {config.MAX_POSITION_PCT:.0%}  =  ${ACCOUNT_EQUITY * config.MAX_POSITION_PCT:,.0f}",
                         style={"fontFamily": "monospace", "fontSize": "13px"}),
                html.Div(f"Max deployed:    {config.MAX_CAPITAL_DEPLOYED_PCT:.0%}  =  ${ACCOUNT_EQUITY * config.MAX_CAPITAL_DEPLOYED_PCT:,.0f}",
                         style={"fontFamily": "monospace", "fontSize": "13px"}),
                html.Div(f"Max positions:   {config.MAX_POSITIONS}  concurrent",
                         style={"fontFamily": "monospace", "fontSize": "13px"}),
            ], width=4),
            dbc.Col([
                html.Div(f"Kelly sizing:    {'✅ ON  (' + str(config.KELLY_FRACTION) + '×)' if kelly_enabled else '❌ OFF'}",
                         style={"fontFamily": "monospace", "fontSize": "13px",
                                "color": GREEN if kelly_enabled else MUTED}),
                html.Div(f"Kelly win rate:  {config.KELLY_WIN_RATE:.0%}  (assumed)",
                         style={"fontFamily": "monospace", "fontSize": "13px"}),
                html.Div(f"Vol-scale HIGH:  {config.VOL_SCALE_HIGH:.2f}×  |  LOW: {config.VOL_SCALE_LOW:.2f}×",
                         style={"fontFamily": "monospace", "fontSize": "13px"}),
                html.Div(f"Sector penalty:  {config.CORRELATION_SAME_SECTOR_SCALE:.0%}  on same-sector add",
                         style={"fontFamily": "monospace", "fontSize": "13px"}),
                html.Div(f"Max portfolio risk: ${ACCOUNT_EQUITY * config.RISK_PER_TRADE_PCT * config.MAX_POSITIONS:,.0f}  ({config.RISK_PER_TRADE_PCT * config.MAX_POSITIONS:.0%})",
                         style={"fontFamily": "monospace", "fontSize": "13px", "color": RED}),
            ], width=4),
        ]),
    ])

    # ── Trailing stop & exit parameters card ──────────────────────────────────
    trailing_enabled = config.TRAILING_STOP_ENABLED
    partial_enabled  = config.PARTIAL_EXIT_ENABLED
    time_stop_enabled = config.TIME_STOP_ENABLED
    scaled_enabled   = config.SCALED_ENTRY_ENABLED

    exit_card = _card([
        html.H6("🛡️ Exit Strategy & Trailing Stop", style={"color": ACCENT, "fontFamily": "monospace"}),
        html.Hr(style={"borderColor": BORDER}),
        dbc.Row([
            dbc.Col([
                html.Div("Trailing Stop (3-stage adaptive)",
                         style={"fontFamily": "monospace", "fontSize": "13px", "color": ACCENT, "fontWeight": "bold"}),
                html.Div(f"  Enabled:         {'✅ ON' if trailing_enabled else '❌ OFF'}",
                         style={"fontFamily": "monospace", "fontSize": "13px",
                                "color": GREEN if trailing_enabled else MUTED}),
                html.Div(f"  Stage 1 → BE:    +{config.TRAILING_STAGE1_TRIGGER_ATR:.1f}×ATR → trail = entry",
                         style={"fontFamily": "monospace", "fontSize": "13px"}),
                html.Div(f"  Stage 2 → Lock:  +{config.TRAILING_STAGE2_TRIGGER_ATR:.1f}×ATR → trail = entry + 1×ATR",
                         style={"fontFamily": "monospace", "fontSize": "13px"}),
                html.Div(f"  Stage 3 → Tight: +{config.TRAILING_STAGE3_TRIGGER_ATR:.1f}×ATR → trail = high − {config.TRAILING_STAGE3_TRAIL_ATR:.1f}×ATR",
                         style={"fontFamily": "monospace", "fontSize": "13px"}),
            ], width=4),
            dbc.Col([
                html.Div("Partial Profit-Taking",
                         style={"fontFamily": "monospace", "fontSize": "13px", "color": ACCENT, "fontWeight": "bold"}),
                html.Div(f"  Enabled:    {'✅ ON' if partial_enabled else '❌ OFF'}",
                         style={"fontFamily": "monospace", "fontSize": "13px",
                                "color": GREEN if partial_enabled else MUTED}),
                html.Div(f"  Exit 1:     sell {config.PARTIAL_EXIT_1_FRACTION:.0%} @ +{config.PARTIAL_EXIT_1_TRIGGER_ATR:.1f}×ATR",
                         style={"fontFamily": "monospace", "fontSize": "13px"}),
                html.Div(f"  Exit 2:     sell {config.PARTIAL_EXIT_2_FRACTION:.0%} @ +{config.PARTIAL_EXIT_2_TRIGGER_ATR:.1f}×ATR",
                         style={"fontFamily": "monospace", "fontSize": "13px"}),
                html.Div(f"  Remainder:  rides trailing stop ({1 - config.PARTIAL_EXIT_1_FRACTION - config.PARTIAL_EXIT_2_FRACTION:.0%})",
                         style={"fontFamily": "monospace", "fontSize": "13px", "color": MUTED}),
            ], width=4),
            dbc.Col([
                html.Div("Time Stop",
                         style={"fontFamily": "monospace", "fontSize": "13px", "color": ACCENT, "fontWeight": "bold"}),
                html.Div(f"  Enabled:    {'✅ ON' if time_stop_enabled else '❌ OFF'}",
                         style={"fontFamily": "monospace", "fontSize": "13px",
                                "color": GREEN if time_stop_enabled else MUTED}),
                html.Div(f"  Max hold:   {config.TIME_STOP_DAYS} calendar days",
                         style={"fontFamily": "monospace", "fontSize": "13px"}),
                html.Div("",  style={"height": "8px"}),
                html.Div("Scaled Entry (Buy Ladder)",
                         style={"fontFamily": "monospace", "fontSize": "13px", "color": ACCENT, "fontWeight": "bold"}),
                html.Div(f"  Enabled:    {'✅ ON' if scaled_enabled else '❌ OFF'}",
                         style={"fontFamily": "monospace", "fontSize": "13px",
                                "color": GREEN if scaled_enabled else MUTED}),
                html.Div(f"  Tranches:   {config.SCALED_ENTRY_N_TRANCHES}  "
                         f"({', '.join(f'{f:.0%}' for f in config.SCALED_ENTRY_FRACTIONS)})",
                         style={"fontFamily": "monospace", "fontSize": "13px"}),
                html.Div(f"  ATR offsets: {', '.join(f'{o:.1f}×' for o in config.SCALED_ENTRY_ATR_OFFSETS)}",
                         style={"fontFamily": "monospace", "fontSize": "13px"}),
                html.Div(f"  RSI abort:  >{config.SCALED_ENTRY_RSI_ABORT_LEVEL:.0f}  "
                         f"| Turn req: {'yes' if config.SCALED_ENTRY_RSI_TURN_REQUIRED else 'no'}  "
                         f"| Expiry: {config.SCALED_ENTRY_EXPIRY_DAYS}d",
                         style={"fontFamily": "monospace", "fontSize": "13px"}),
            ], width=4),
        ]),
    ])

    # ── Scaled entry tranches table (per-symbol buy ladder) ──────────────────
    scaled_section = html.Div()
    if alloc and alloc.scaled_entries:
        se_rows = []
        for sym, plan in alloc.scaled_entries.items():
            for row in plan.summary_table():
                se_rows.append({"Symbol": sym, **row})

        if se_rows:
            se_df = pd.DataFrame(se_rows)

            # Colour-code status
            se_colours = [
                {"if": {"filter_query": '{Status} = "PENDING"',   "column_id": "Status"}, "color": YELLOW},
                {"if": {"filter_query": '{Status} = "FILLED"',    "column_id": "Status"}, "color": GREEN,  "fontWeight": "bold"},
                {"if": {"filter_query": '{Status} = "CANCELLED"', "column_id": "Status"}, "color": RED},
                {"if": {"filter_query": '{Status} = "EXPIRED"',   "column_id": "Status"}, "color": MUTED},
            ]

            se_table = dash_table.DataTable(
                id="scaled-entry-table",
                columns=[{"name": c, "id": c} for c in se_df.columns],
                data=se_df.to_dict("records"),
                style_cell=_CELL_STYLE,
                style_header=_HDR_STYLE,
                style_data_conditional=se_colours,
                style_table={"overflowX": "auto", "borderRadius": "6px"},
                sort_action="native",
                filter_action="native",
                page_size=40,
            )

            # Build a visual ladder chart
            chart_traces = []
            symbols_in_plan = list(alloc.scaled_entries.keys())
            for sym in symbols_in_plan:
                plan = alloc.scaled_entries[sym]
                for t in plan.tranches:
                    chart_traces.append({
                        "symbol": sym,
                        "label": f"T{t.tranche_id}",
                        "price": t.limit_price,
                        "qty": t.quantity,
                    })
                # Add stop line
                chart_traces.append({
                    "symbol": sym,
                    "label": "STOP",
                    "price": plan.stop_loss_price,
                    "qty": 0,
                })

            ladder_fig = go.Figure()
            for sym in symbols_in_plan:
                sym_data = [d for d in chart_traces if d["symbol"] == sym and d["label"] != "STOP"]
                stop_data = [d for d in chart_traces if d["symbol"] == sym and d["label"] == "STOP"]

                ladder_fig.add_trace(go.Bar(
                    name=sym,
                    x=[f"{sym}\n{d['label']}" for d in sym_data],
                    y=[d["price"] for d in sym_data],
                    text=[f"{d['qty']}sh" for d in sym_data],
                    textposition="outside",
                ))

                # Add stop line as a scatter marker
                if stop_data:
                    for sd in stop_data:
                        ladder_fig.add_trace(go.Scatter(
                            x=[f"{sym}\nT1", f"{sym}\nT{len(sym_data)}"],
                            y=[sd["price"], sd["price"]],
                            mode="lines",
                            line=dict(color=RED, width=2, dash="dash"),
                            name=f"{sym} stop",
                            showlegend=False,
                            hovertemplate=f"{sym} STOP: ${sd['price']:.2f}<extra></extra>",
                        ))

            ladder_fig.update_layout(
                title="Buy Ladder — Tranche Limit Prices",
                paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
                font=dict(color=TEXT),
                xaxis=dict(gridcolor=BORDER, color=TEXT),
                yaxis=dict(gridcolor=BORDER, color=TEXT, title="Price ($)"),
                margin=dict(t=40, b=20, l=20, r=20),
                barmode="group",
                showlegend=True,
                legend=dict(font=dict(color=TEXT)),
            )

            scaled_section = _card([
                html.H6("🪜 Scaled Entry — Buy Ladder",
                        style={"color": ACCENT, "fontFamily": "monospace", "marginBottom": "4px"}),
                html.Div(
                    f"Split entry into {config.SCALED_ENTRY_N_TRANCHES} tranches at progressively lower prices.  "
                    f"T1 = immediate.  T2/T3 fill only if price dips further AND RSI confirms.  "
                    f"Unfilled tranches expire after {config.SCALED_ENTRY_EXPIRY_DAYS}d or abort if RSI > {config.SCALED_ENTRY_RSI_ABORT_LEVEL:.0f}.",
                    style={"color": MUTED, "fontSize": "11px", "fontFamily": "monospace", "marginBottom": "10px"},
                ),
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=ladder_fig, config={"displayModeBar": False}), width=6),
                    dbc.Col(se_table, width=6),
                ]),
            ])

    # ── Live position management state  ──────────────────────────────────────
    live_positions_section = _build_live_positions_card()

    return html.Div([
        stat_cards,
        params_card,
        exit_card,
        _card([
            html.H6("📊 Capital Allocation", style={"color": ACCENT, "fontFamily": "monospace", "marginBottom": "12px"}),
            dbc.Row([
                dbc.Col(_card(dcc.Graph(figure=pie_fig,  config={"displayModeBar": False})), width=4),
                dbc.Col(_card(dcc.Graph(figure=heat_fig, config={"displayModeBar": False})), width=5),
                dbc.Col(_card(dcc.Graph(figure=rr_fig,   config={"displayModeBar": False})), width=3),
            ], className="mb-0"),
        ]),
        _card([
            html.H6("💼 Position Sizing — Simultaneous Portfolio Allocation",
                    style={"color": ACCENT, "fontFamily": "monospace", "marginBottom": "4px"}),
            html.Div(
                "Candidates ranked by signal quality (BUY→DIP+FUND→DIP ONLY) then ML prob.  "
                "Stop & Target are ATR-based where available, else fixed %.  "
                "Vol Scale = volatility-regime multiplier.  Sect Scale = same-sector penalty.",
                style={"color": MUTED, "fontSize": "11px", "fontFamily": "monospace", "marginBottom": "10px"},
            ),
            pos_table,
        ]),
        scaled_section,
        live_positions_section,
    ])


# ---------------------------------------------------------------------------
# Tab 3 — Back-test layout
# ---------------------------------------------------------------------------

def _backtest_layout():
    if not os.path.exists(BACKTEST_CSV):
        return _card(html.Div([
            html.Div("No back-test data found.", style={"color": MUTED, "fontFamily": "monospace"}),
            html.Div("Run:  python3 backtest.py  (or  python3 run_screen.py --backtest)",
                     style={"color": MUTED, "fontFamily": "monospace", "fontSize": "12px", "marginTop": "8px"}),
        ]))

    df = pd.read_csv(BACKTEST_CSV)
    df = df[df["outcome"].notna()]

    # ── equity curve (cumulative return over entry date, per run) ─────────────
    latest_run = df["backtest_date"].max()
    run_df     = df[df["backtest_date"] == latest_run].copy()
    run_df["pct_gain"] = pd.to_numeric(run_df["pct_gain"], errors="coerce")
    run_df = run_df.dropna(subset=["pct_gain"]).sort_values("entry_date")
    run_df["cum_return"] = (1 + run_df["pct_gain"] / 100).cumprod() - 1

    eq_fig = go.Figure()
    eq_fig.add_trace(go.Scatter(
        x=run_df["entry_date"], y=run_df["cum_return"] * 100,
        mode="lines+markers",
        line=dict(color=ACCENT, width=2),
        marker=dict(color=[GREEN if v >= 0 else RED for v in run_df["cum_return"]], size=7),
        name="Cumulative Return %",
        hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
    ))
    eq_fig.add_hline(y=0, line_color=BORDER, line_dash="dash")
    eq_fig.update_layout(
        title="Equity Curve (latest back-test run)",
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT),
        xaxis=dict(gridcolor=BORDER, color=TEXT, title="Entry Date"),
        yaxis=dict(gridcolor=BORDER, color=TEXT, title="Cumulative Return %"),
        margin=dict(t=40, b=20, l=20, r=20),
    )

    # ── outcome breakdown bar ─────────────────────────────────────────────────
    outcome_counts = run_df["outcome"].value_counts().reset_index()
    outcome_counts.columns = ["outcome", "count"]
    colour_map = {"TP": GREEN, "SL": RED, "OPEN": YELLOW}

    bar_fig = go.Figure(go.Bar(
        x=outcome_counts["outcome"],
        y=outcome_counts["count"],
        marker_color=[colour_map.get(o, MUTED) for o in outcome_counts["outcome"]],
        text=outcome_counts["count"],
        textposition="outside",
    ))
    bar_fig.update_layout(
        title="Trade Outcomes",
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT),
        xaxis=dict(color=TEXT), yaxis=dict(color=TEXT, gridcolor=BORDER),
        margin=dict(t=40, b=20, l=20, r=20),
        showlegend=False,
    )

    # ── gain distribution histogram ───────────────────────────────────────────
    hist_fig = go.Figure(go.Histogram(
        x=run_df["pct_gain"], nbinsx=30,
        marker_color=ACCENT, opacity=0.8,
        hovertemplate="Gain: %{x:.1f}%  Count: %{y}<extra></extra>",
    ))
    hist_fig.add_vline(x=0, line_color=BORDER, line_dash="dash")
    hist_fig.update_layout(
        title="Trade Gain Distribution",
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT),
        xaxis=dict(color=TEXT, title="% Gain"), yaxis=dict(color=TEXT, gridcolor=BORDER, title="Count"),
        margin=dict(t=40, b=20, l=20, r=20),
    )

    # ── per-symbol stats ──────────────────────────────────────────────────────
    sym_stats = (
        run_df.groupby("symbol")["pct_gain"]
        .agg(trades="count", avg_gain="mean", win_rate=lambda x: (x > 0).mean())
        .reset_index()
        .round(2)
    )
    sym_stats.columns = ["Symbol", "Trades", "Avg Gain %", "Win Rate"]
    sym_stats["Win Rate"] = (sym_stats["Win Rate"] * 100).round(1).astype(str) + "%"

    sym_bar = go.Figure(go.Bar(
        x=sym_stats["Symbol"], y=sym_stats["Avg Gain %"],
        marker_color=[GREEN if v >= 0 else RED for v in sym_stats["Avg Gain %"]],
        text=[f"{v:.1f}%" for v in sym_stats["Avg Gain %"]],
        textposition="outside",
    ))
    sym_bar.add_hline(y=0, line_color=BORDER, line_dash="dash")
    sym_bar.update_layout(
        title="Avg Gain % by Symbol",
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT),
        xaxis=dict(color=TEXT), yaxis=dict(color=TEXT, gridcolor=BORDER),
        margin=dict(t=40, b=20, l=20, r=20),
    )

    # ── summary stats row ─────────────────────────────────────────────────────
    n_closed = len(run_df[run_df["outcome"] != "OPEN"])
    n_wins   = len(run_df[run_df["pct_gain"] > 0])
    wr       = n_wins / n_closed if n_closed else 0
    avg_g    = run_df["pct_gain"].mean()
    cum_ret  = run_df["cum_return"].iloc[-1] * 100 if not run_df.empty else 0

    stat_row = dbc.Row([
        _stat_card("Back-test Date",    latest_run,                    MUTED),
        _stat_card("Total Trades",      str(len(run_df)),              TEXT),
        _stat_card("Win Rate",          f"{wr:.1%}",                   GREEN if wr >= 0.5 else RED),
        _stat_card("Avg Gain/Trade",    f"{avg_g:+.2f}%",              GREEN if avg_g >= 0 else RED),
        _stat_card("Cumulative Return", f"{cum_ret:+.1f}%",            GREEN if cum_ret >= 0 else RED),
    ], className="mb-3")

    return html.Div([
        stat_row,
        dbc.Row([
            dbc.Col(_card(dcc.Graph(figure=eq_fig,   config={"displayModeBar": False})), width=12),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(_card(dcc.Graph(figure=bar_fig,  config={"displayModeBar": False})), width=4),
            dbc.Col(_card(dcc.Graph(figure=hist_fig, config={"displayModeBar": False})), width=4),
            dbc.Col(_card(dcc.Graph(figure=sym_bar,  config={"displayModeBar": False})), width=4),
        ], className="mb-3"),
        _card([
            html.H6("Trade Log (latest run)", style={"color": ACCENT, "fontFamily": "monospace", "marginBottom": "12px"}),
            _make_table(
                run_df[["symbol","entry_date","entry_price","exit_date","exit_price","pct_gain","outcome","hold_days","score"]],
                "backtest-table",
            ),
        ]),
        _card([
            html.H6("Per-Symbol Summary", style={"color": ACCENT, "fontFamily": "monospace", "marginBottom": "12px"}),
            _make_table(sym_stats, "sym-stats-table"),
        ]),
    ])


# ---------------------------------------------------------------------------
# Tab 4 — Screen Log layout
# ---------------------------------------------------------------------------

def _log_layout():
    if not os.path.exists(SCREEN_CSV):
        return _card(html.Div("No screen log found. Run the screener first.",
                               style={"color": MUTED, "fontFamily": "monospace"}))

    # on_bad_lines='skip' tolerates rows written by older versions of run_screen.py
    # that had a different column count than the current schema.
    df = pd.read_csv(SCREEN_CSV, on_bad_lines="skip")
    df["dip_score"]   = pd.to_numeric(df["dip_score"],   errors="coerce")
    df["rsi"]         = pd.to_numeric(df["rsi"],         errors="coerce")
    df["price"]       = pd.to_numeric(df["price"],       errors="coerce")

    # ── dip score over time per symbol ───────────────────────────────────────
    score_fig = px.line(
        df.sort_values("date"), x="date", y="dip_score", color="symbol",
        title="Dip Score Over Time by Symbol",
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Plotly,
    )
    score_fig.update_layout(
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT), legend=dict(font=dict(color=TEXT)),
        xaxis=dict(gridcolor=BORDER), yaxis=dict(gridcolor=BORDER, title="Dip Score"),
        margin=dict(t=40, b=20, l=20, r=20),
    )

    # ── RSI heat-map (symbol × date) ─────────────────────────────────────────
    rsi_pivot = df.pivot_table(index="symbol", columns="date", values="rsi", aggfunc="mean")
    rsi_fig = go.Figure(go.Heatmap(
        z=rsi_pivot.values,
        x=rsi_pivot.columns.tolist(),
        y=rsi_pivot.index.tolist(),
        colorscale=[[0, RED], [0.35, YELLOW], [0.65, ACCENT], [1, GREEN]],
        zmin=0, zmax=100,
        colorbar=dict(
            title=dict(text="RSI", font=dict(color=TEXT)),
            tickfont=dict(color=TEXT),
        ),
        hovertemplate="Date: %{x}<br>Symbol: %{y}<br>RSI: %{z:.1f}<extra></extra>",
    ))
    rsi_fig.add_hline(y=-0.5, annotation_text="Oversold < 35",
                      line_color=RED, line_dash="dot", annotation_font_color=RED)
    rsi_fig.update_layout(
        title="RSI Heat-map (symbol × date)",
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT),
        xaxis=dict(color=TEXT), yaxis=dict(color=TEXT),
        margin=dict(t=40, b=20, l=20, r=20),
    )

    return html.Div([
        dbc.Row([
            dbc.Col(_card(dcc.Graph(figure=score_fig, config={"displayModeBar": False})), width=12),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(_card(dcc.Graph(figure=rsi_fig,   config={"displayModeBar": False})), width=12),
        ], className="mb-3"),
        _card([
            html.H6("Full Screen Log", style={"color": ACCENT, "fontFamily": "monospace", "marginBottom": "12px"}),
            _make_table(df, "log-table"),
        ]),
    ])


# ---------------------------------------------------------------------------
# Tab 5 — My Trades (manual transaction journal)
# ---------------------------------------------------------------------------

_INPUT_STYLE = {
    "backgroundColor": "#faf9f7",
    "color": TEXT,
    "border": f"1px solid {BORDER}",
    "borderRadius": "4px",
    "fontFamily": "monospace",
    "fontSize": "13px",
    "width": "100%",
    "padding": "6px 10px",
}

def _trades_layout():
    """Build the full My Trades tab layout (form + summary + chart + table)."""
    df = _load_trades()

    # ── Ensure asset_type is populated ────────────────────────────────────────
    if not df.empty:
        if "asset_type" not in df.columns:
            df["asset_type"] = None
        df["asset_type"] = df.apply(
            lambda r: _infer_asset_type(r["symbol"], r.get("asset_type")), axis=1
        )

    # ── Fetch live prices for all held symbols ────────────────────────────────
    open_buys = df[df["action"] == "BUY"] if not df.empty else pd.DataFrame()
    held_symbols = list(open_buys["symbol"].unique()) if not open_buys.empty else []
    current_prices = _fetch_current_prices(held_symbols) if held_symbols else {}

    # ── Build per-symbol position summary (aggregate multiple buys) ───────────
    # Net position per symbol: sum of BUY qty minus sum of SELL qty
    pos_map: dict[str, dict] = {}  # sym → {qty, cost_basis, invested, asset_type}
    if not df.empty:
        for _, row in df.iterrows():
            sym = row["symbol"]
            if sym not in pos_map:
                pos_map[sym] = {"qty": 0, "total_cost": 0.0, "invested": 0.0,
                                "asset_type": _infer_asset_type(sym, row.get("asset_type"))}
            if row["action"] == "BUY":
                q = float(row["quantity"] or 0)
                p = float(row["entry_price"] or 0)
                pos_map[sym]["qty"] += q
                pos_map[sym]["total_cost"] += q * p
                pos_map[sym]["invested"] += float(row["amount_invested"] or 0)
            elif row["action"] == "SELL":
                pos_map[sym]["qty"] -= float(row["quantity"] or 0)

    # Remove closed positions (qty ≤ 0)
    pos_map = {s: v for s, v in pos_map.items() if v["qty"] > 0}

    # ── Compute unrealized P&L & exit plans ───────────────────────────────────
    total_market_value = 0.0
    total_unrealized = 0.0
    total_invested_open = 0.0

    position_rows = []
    for sym, pos in pos_map.items():
        cur_price = current_prices.get(sym)
        avg_entry = pos["total_cost"] / pos["qty"] if pos["qty"] > 0 else 0
        mkt_val = cur_price * pos["qty"] if cur_price else None
        unreal = (cur_price - avg_entry) * pos["qty"] if cur_price else None
        unreal_pct = ((cur_price / avg_entry) - 1) * 100 if cur_price and avg_entry > 0 else None

        if mkt_val is not None:
            total_market_value += mkt_val
        if unreal is not None:
            total_unrealized += unreal
        total_invested_open += pos["invested"]

        is_etf = pos["asset_type"] == "ETF"

        # Exit plan (only for stocks — ETFs are hold-forever)
        if is_etf:
            ep = {"atr_14": None, "stop_price": "∞ HOLD",
                  "target_price": "∞ HOLD",
                  "partial_1_price": "—", "partial_2_price": "—",
                  "trail_stage": "HOLD FOREVER", "trail_stop": "—"}
        else:
            ep = _compute_exit_plan(avg_entry, cur_price or avg_entry, sym)

        position_rows.append({
            "Symbol":       sym,
            "Type":         "📦 ETF" if is_etf else "📈 Stock",
            "Qty":          int(pos["qty"]),
            "Avg Entry $":  round(avg_entry, 2),
            "Current $":    round(cur_price, 2) if cur_price else "n/a",
            "Mkt Value $":  round(mkt_val, 0) if mkt_val else "n/a",
            "Unreal P&L $": round(unreal, 2) if unreal is not None else "n/a",
            "Unreal %":     f"{unreal_pct:+.1f}%" if unreal_pct is not None else "n/a",
            "ATR(14) $":    ep["atr_14"] or "—",
            "Stop $":       ep["stop_price"] or "—",
            "Partial 1 $":  ep["partial_1_price"] or "—",
            "Partial 2 $":  ep["partial_2_price"] or "—",
            "Target $":     ep["target_price"] or "—",
            "Trail Stage":  ep["trail_stage"],
            "Trail Stop $": ep["trail_stop"] or "—",
            "% Budget":     round((mkt_val / ACCOUNT_EQUITY * 100), 1) if mkt_val else "—",
        })

    pos_df = pd.DataFrame(position_rows) if position_rows else pd.DataFrame()

    # ── Summary stats ─────────────────────────────────────────────────────────
    total_invested  = df["amount_invested"].sum() if not df.empty else 0
    realized_pnl    = df["realized_pnl"].dropna().sum() if not df.empty else 0
    n_trades        = len(df)
    n_wins          = int((df["realized_pnl"] > 0).sum()) if not df.empty else 0
    win_rate        = n_wins / n_trades if n_trades > 0 else 0
    roi_pct         = realized_pnl / total_invested * 100 if total_invested else 0
    cash_available  = max(ACCOUNT_EQUITY - total_market_value, 0)
    pct_utilised    = total_market_value / ACCOUNT_EQUITY * 100 if ACCOUNT_EQUITY else 0

    # ETF vs Stock breakdown
    etf_value = sum(
        (current_prices.get(s, 0) * pos_map[s]["qty"])
        for s in pos_map if pos_map[s]["asset_type"] == "ETF"
    )
    stock_value = total_market_value - etf_value

    stat_row = dbc.Row([
        _stat_card("Total Trades",      str(n_trades),                    ACCENT),
        _stat_card("Total Mkt Value",   f"${total_market_value:,.0f}",    ACCENT),
        _stat_card("% Utilised",        f"{pct_utilised:.1f}%",
                   GREEN if pct_utilised <= config.MAX_CAPITAL_DEPLOYED_PCT * 100 else RED),
        _stat_card("💵 Cash Available", f"${cash_available:,.0f}",        GREEN if cash_available > 0 else RED),
        _stat_card("Unrealized P&L",    f"${total_unrealized:+,.2f}",     GREEN if total_unrealized >= 0 else RED),
        _stat_card("Realized P&L",      f"${realized_pnl:+,.2f}",        GREEN if realized_pnl >= 0 else RED),
        _stat_card("Win Rate",          f"{win_rate:.0%}  ({n_wins}/{n_trades})",
                   GREEN if win_rate >= 0.5 else (YELLOW if n_trades else MUTED)),
    ], className="mb-3")

    # Second row: ETF vs Stock breakdown
    stat_row2 = dbc.Row([
        _stat_card("📦 ETF Holdings",    f"${etf_value:,.0f}", ACCENT),
        _stat_card("📈 Stock Holdings",   f"${stock_value:,.0f}", YELLOW),
        _stat_card("Budget",              f"${ACCOUNT_EQUITY:,.0f}", MUTED),
        _stat_card("ROI % (realized)",    f"{roi_pct:+.2f}%", GREEN if roi_pct >= 0 else RED),
    ], className="mb-3")

    # ── Open Positions with Exit Plan table ───────────────────────────────────
    if not pos_df.empty:
        exit_colours = [
            # Type colouring
            {"if": {"filter_query": '{Type} contains "ETF"',   "column_id": "Type"},       "color": ACCENT},
            {"if": {"filter_query": '{Type} contains "Stock"', "column_id": "Type"},       "color": YELLOW},
            # Unrealised P&L
            {"if": {"filter_query": '{Unreal %} contains "+"', "column_id": "Unreal %"},  "color": GREEN, "fontWeight": "bold"},
            {"if": {"filter_query": '{Unreal %} contains "-"', "column_id": "Unreal %"},  "color": RED,   "fontWeight": "bold"},
            {"if": {"filter_query": '{Unreal P&L $} > 0',     "column_id": "Unreal P&L $"}, "color": GREEN},
            {"if": {"filter_query": '{Unreal P&L $} < 0',     "column_id": "Unreal P&L $"}, "color": RED},
            # Trail stage
            {"if": {"filter_query": '{Trail Stage} = "INITIAL"',      "column_id": "Trail Stage"}, "color": MUTED},
            {"if": {"filter_query": '{Trail Stage} = "BREAKEVEN"',    "column_id": "Trail Stage"}, "color": YELLOW},
            {"if": {"filter_query": '{Trail Stage} = "PROFIT_LOCK"',  "column_id": "Trail Stage"}, "color": GREEN},
            {"if": {"filter_query": '{Trail Stage} = "TIGHT_TRAIL"',  "column_id": "Trail Stage"}, "color": GREEN, "fontWeight": "bold"},
            {"if": {"filter_query": '{Trail Stage} = "HOLD FOREVER"', "column_id": "Trail Stage"}, "color": ACCENT, "fontWeight": "bold"},
        ]

        exit_table = dash_table.DataTable(
            id="exit-plan-table",
            columns=[{"name": c, "id": c} for c in pos_df.columns],
            data=pos_df.to_dict("records"),
            style_cell=_CELL_STYLE,
            style_header=_HDR_STYLE,
            style_data_conditional=exit_colours,
            style_table={"overflowX": "auto", "borderRadius": "6px"},
            sort_action="native",
            page_size=40,
        )

        exit_plan_section = _card([
            html.H6("🎯 Open Positions — Live Prices & Scaled Exit Plan",
                    style={"color": ACCENT, "fontFamily": "monospace", "marginBottom": "4px"}),
            html.Div(
                f"Current prices fetched live.  "
                f"Stocks: ATR-based exits — "
                f"Partial 1 = sell {config.PARTIAL_EXIT_1_FRACTION:.0%} @ +{config.PARTIAL_EXIT_1_TRIGGER_ATR:.0f}×ATR,  "
                f"Partial 2 = sell {config.PARTIAL_EXIT_2_FRACTION:.0%} @ +{config.PARTIAL_EXIT_2_TRIGGER_ATR:.0f}×ATR,  "
                f"remainder rides trailing stop.  "
                f"ETFs = hold forever (no exit targets).",
                style={"color": MUTED, "fontSize": "11px", "fontFamily": "monospace", "marginBottom": "10px"},
            ),
            exit_table,
        ])
    else:
        exit_plan_section = _card(
            html.Div("No open positions. Log a BUY trade to see exit targets.",
                     style={"color": MUTED, "fontFamily": "monospace"})
        )

    # ── Utilisation donut: ETF / Stock / Cash ─────────────────────────────────
    util_fig = go.Figure(go.Pie(
        labels=["📦 ETFs", "📈 Stocks", "💵 Cash"],
        values=[max(etf_value, 0), max(stock_value, 0), max(cash_available, 0)],
        hole=0.55,
        marker=dict(colors=[ACCENT, YELLOW, GREEN]),
        textinfo="label+percent",
        textfont=dict(color=TEXT, size=12),
        hovertemplate="%{label}: $%{value:,.0f}  (%{percent})<extra></extra>",
    ))
    util_fig.update_layout(
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT),
        margin=dict(t=30, b=10, l=10, r=10),
        showlegend=True,
        legend=dict(font=dict(color=TEXT)),
        annotations=[dict(text=f"${ACCOUNT_EQUITY/1e3:.0f}k", x=0.5, y=0.5,
                          font_size=18, showarrow=False, font_color=ACCENT)],
    )

    # ── Per-symbol unrealized P&L bar ─────────────────────────────────────────
    if position_rows:
        sym_names = [r["Symbol"] for r in position_rows]
        sym_unreal = []
        for r in position_rows:
            v = r["Unreal P&L $"]
            sym_unreal.append(float(v) if isinstance(v, (int, float)) else 0)

        unreal_fig = go.Figure(go.Bar(
            x=sym_names, y=sym_unreal,
            marker_color=[GREEN if v >= 0 else RED for v in sym_unreal],
            text=[f"${v:+,.0f}" for v in sym_unreal],
            textposition="outside",
        ))
        unreal_fig.add_hline(y=0, line_color=BORDER, line_dash="dash")
        unreal_fig.update_layout(
            title="Unrealized P&L by Position",
            paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
            font=dict(color=TEXT),
            xaxis=dict(color=TEXT), yaxis=dict(color=TEXT, gridcolor=BORDER, title="P&L ($)"),
            margin=dict(t=40, b=20, l=20, r=20),
        )

        # Per-symbol % of budget bar
        sym_budget_pcts = []
        for r in position_rows:
            v = r["% Budget"]
            sym_budget_pcts.append(float(v) if isinstance(v, (int, float)) else 0)

        budget_fig = go.Figure(go.Bar(
            x=sym_names, y=sym_budget_pcts,
            marker_color=[
                ACCENT if pos_map.get(s, {}).get("asset_type") == "ETF" else YELLOW
                for s in sym_names
            ],
            text=[f"{v:.1f}%" for v in sym_budget_pcts],
            textposition="outside",
        ))
        budget_fig.add_hline(
            y=config.MAX_POSITION_PCT * 100, line_dash="dash", line_color=RED,
            annotation_text=f"Max {config.MAX_POSITION_PCT:.0%}", annotation_font_color=RED,
        )
        budget_fig.update_layout(
            title="Position Size — % of Budget",
            paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
            font=dict(color=TEXT),
            xaxis=dict(color=TEXT), yaxis=dict(color=TEXT, gridcolor=BORDER, title="% Budget"),
            margin=dict(t=40, b=20, l=20, r=20),
        )

        charts_row1 = dbc.Row([
            dbc.Col(_card(dcc.Graph(figure=util_fig,   config={"displayModeBar": False})), width=4),
            dbc.Col(_card(dcc.Graph(figure=unreal_fig, config={"displayModeBar": False})), width=4),
            dbc.Col(_card(dcc.Graph(figure=budget_fig, config={"displayModeBar": False})), width=4),
        ], className="mb-3")
    else:
        charts_row1 = dbc.Row([
            dbc.Col(_card(dcc.Graph(figure=util_fig, config={"displayModeBar": False})), width=12),
        ], className="mb-3")

    # ── Cumulative realized P&L chart (existing) ──────────────────────────────
    if not df.empty and df["realized_pnl"].notna().any():
        pnl_df = df[df["realized_pnl"].notna()].copy()
        pnl_df = pnl_df.sort_values("date")
        pnl_df["cum_pnl"] = pnl_df["realized_pnl"].cumsum()

        pnl_fig = go.Figure()
        pnl_fig.add_trace(go.Scatter(
            x=pnl_df["date"], y=pnl_df["cum_pnl"],
            mode="lines+markers",
            line=dict(color=ACCENT, width=2),
            marker=dict(
                color=[GREEN if v >= 0 else RED for v in pnl_df["cum_pnl"]],
                size=8,
            ),
            fill="tozeroy",
            fillcolor="rgba(90,90,90,0.06)",
            hovertemplate="%{x}<br>Cum P&L: $%{y:+,.2f}<extra></extra>",
        ))
        pnl_fig.add_hline(y=0, line_color=BORDER, line_dash="dash")
        pnl_fig.update_layout(
            title="Cumulative Realized P&L",
            paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
            font=dict(color=TEXT),
            xaxis=dict(gridcolor=BORDER, color=TEXT, title="Date"),
            yaxis=dict(gridcolor=BORDER, color=TEXT, title="Cumulative P&L ($)"),
            margin=dict(t=40, b=20, l=20, r=20),
        )
        realized_charts = dbc.Row([
            dbc.Col(_card(dcc.Graph(figure=pnl_fig, config={"displayModeBar": False})), width=12),
        ], className="mb-3")
    else:
        realized_charts = html.Div()

    # ── trade entry form ──────────────────────────────────────────────────────
    form_card = _card([
        html.H6("➕ Log a New Trade", style={"color": ACCENT, "fontFamily": "monospace", "marginBottom": "12px"}),
        dbc.Row([
            dbc.Col([
                html.Label("Date", style={"color": MUTED, "fontSize": "12px"}),
                dcc.Input(
                    id="trade-date", type="text",
                    placeholder=datetime.date.today().isoformat(),
                    value=datetime.date.today().isoformat(),
                    debounce=True, style=_INPUT_STYLE,
                ),
            ], width=2),
            dbc.Col([
                html.Label("Symbol", style={"color": MUTED, "fontSize": "12px"}),
                dcc.Input(id="trade-symbol", type="text", placeholder="AAPL",
                          debounce=True, style=_INPUT_STYLE),
            ], width=2),
            dbc.Col([
                html.Label("Type", style={"color": MUTED, "fontSize": "12px"}),
                dcc.Dropdown(
                    id="trade-asset-type",
                    options=[
                        {"label": "📈 Stock", "value": "STOCK"},
                        {"label": "📦 ETF",   "value": "ETF"},
                    ],
                    value="STOCK",
                    style={"backgroundColor": "#faf9f7", "color": TEXT, "fontFamily": "monospace"},
                    clearable=False,
                ),
            ], width=1),
            dbc.Col([
                html.Label("Action", style={"color": MUTED, "fontSize": "12px"}),
                dcc.Dropdown(
                    id="trade-action",
                    options=[{"label": "BUY", "value": "BUY"}, {"label": "SELL", "value": "SELL"}],
                    value="BUY",
                    style={"backgroundColor": "#faf9f7", "color": TEXT, "fontFamily": "monospace"},
                    clearable=False,
                ),
            ], width=1),
            dbc.Col([
                html.Label("Quantity", style={"color": MUTED, "fontSize": "12px"}),
                dcc.Input(id="trade-qty", type="number", placeholder="10",
                          debounce=True, style=_INPUT_STYLE),
            ], width=1),
            dbc.Col([
                html.Label("Entry Price $", style={"color": MUTED, "fontSize": "12px"}),
                dcc.Input(id="trade-entry", type="number", placeholder="150.00",
                          debounce=True, style=_INPUT_STYLE),
            ], width=2),
            dbc.Col([
                html.Label("Exit Price $ (opt)", style={"color": MUTED, "fontSize": "12px"}),
                dcc.Input(id="trade-exit", type="number", placeholder="165.00",
                          debounce=True, style=_INPUT_STYLE),
            ], width=1),
            dbc.Col([
                html.Label("Notes (opt)", style={"color": MUTED, "fontSize": "12px"}),
                dcc.Input(id="trade-notes", type="text", placeholder="Screener signal",
                          debounce=True, style=_INPUT_STYLE),
            ], width=2),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(
                dbc.Button("💾 Save Trade", id="save-trade-btn", color="success", size="sm",
                           style={"fontFamily": "monospace"}),
                width="auto",
            ),
            dbc.Col(
                html.Div(id="trade-save-status",
                         style={"color": MUTED, "fontFamily": "monospace", "fontSize": "13px",
                                "paddingTop": "6px"}),
                width=True,
            ),
        ]),
        html.Div([
            html.Hr(style={"borderColor": BORDER, "marginTop": "16px"}),
            html.Div([
                html.Span("💡 Auto-detects ETFs (SMH, SCHD, SPY, QQQ, …). "
                          "ETFs are excluded from swing exit logic and counted as long-term holds. "
                          "Realized P&L auto-calculated on SELL trades. ",
                          style={"color": MUTED, "fontSize": "11px", "fontFamily": "monospace"}),
                html.Span("CSV: ", style={"color": MUTED, "fontSize": "11px"}),
                html.Code("results/my_trades.csv",
                          style={"color": ACCENT, "fontSize": "11px"}),
            ]),
        ]),
    ])

    # ── trade log table ───────────────────────────────────────────────────────
    if not df.empty:
        display_df = df.copy()
        # Add computed columns
        display_df["exposure_pct"] = (
            display_df["amount_invested"] / ACCOUNT_EQUITY * 100
        ).round(2)
        # Add current price & unrealized P&L
        display_df["current_price"] = display_df["symbol"].map(current_prices)
        display_df["unrealized_pnl"] = display_df.apply(
            lambda r: round((r["current_price"] - r["entry_price"]) * r["quantity"], 2)
            if pd.notna(r.get("current_price")) and r["action"] == "BUY" else None,
            axis=1,
        )

        display_df.columns = [
            "Date", "Symbol", "Action", "Qty", "Entry $",
            "Exit $", "Invested $", "Realized P&L $", "Notes", "Type",
            "Exposure %", "Current $", "Unrealized P&L $",
        ]

        # Reorder columns
        display_df = display_df[[
            "Date", "Symbol", "Type", "Action", "Qty", "Entry $", "Current $",
            "Exit $", "Invested $", "Exposure %",
            "Unrealized P&L $", "Realized P&L $", "Notes",
        ]]

        table_section = _card([
            html.H6(f"📒 Trade Journal  ({n_trades} entries)",
                    style={"color": ACCENT, "fontFamily": "monospace", "marginBottom": "12px"}),
            _make_table(display_df, "trades-table"),
            html.Div([
                html.Hr(style={"borderColor": BORDER}),
                html.Span("✏️  Edit directly in ",
                          style={"color": MUTED, "fontSize": "11px", "fontFamily": "monospace"}),
                html.Code("results/my_trades.csv",
                          style={"color": ACCENT, "fontSize": "11px"}),
                html.Span(" and refresh the page to reload.",
                          style={"color": MUTED, "fontSize": "11px", "fontFamily": "monospace"}),
            ]),
        ])
    else:
        table_section = _card(
            html.Div("No trades logged yet. Use the form above to add your first trade.",
                     style={"color": MUTED, "fontFamily": "monospace"})
        )

    return html.Div([stat_row, stat_row2, charts_row1, exit_plan_section,
                      realized_charts, form_card, table_section])


# ---------------------------------------------------------------------------
# Tab 6 — System Overview (portfolio / system health)
# ---------------------------------------------------------------------------

def _overview_layout():
    """Build the 🏠 System Overview tab — a single-screen snapshot of system health."""
    now = datetime.datetime.now()

    # ── 1. Connectivity & ML status ──────────────────────────────────────────
    ibkr_ok, ibkr_msg = _check_ibkr_status()
    ibkr_colour = GREEN if ibkr_ok else RED

    ml_status = "✅ Enabled" if config.ML_ENABLED else "❌ Disabled"
    ml_colour = GREEN if config.ML_ENABLED else MUTED
    if _ML_MISSING:
        ml_status = f"⚠️ Missing: {', '.join(_ML_MISSING)}"
        ml_colour = YELLOW

    trailing_status = "✅ ON" if config.TRAILING_STOP_ENABLED else "❌ OFF"
    partial_status  = "✅ ON" if config.PARTIAL_EXIT_ENABLED else "❌ OFF"
    time_stop_status = "✅ ON" if config.TIME_STOP_ENABLED else "❌ OFF"
    scaled_status   = "✅ ON" if config.SCALED_ENTRY_ENABLED else "❌ OFF"
    atr_status      = "✅ ATR-based" if config.USE_ATR_STOPS else "⚠️ Fixed %"

    status_card = _card([
        html.H6("🖥️ System Status", style={"color": ACCENT, "fontFamily": "monospace", "marginBottom": "12px"}),
        dbc.Row([
            dbc.Col([
                html.Div("Connectivity", style={"color": ACCENT, "fontFamily": "monospace", "fontWeight": "bold", "fontSize": "13px", "marginBottom": "6px"}),
                html.Div(ibkr_msg, style={"fontFamily": "monospace", "fontSize": "13px", "color": ibkr_colour}),
                html.Div(f"ML Ensemble:  {ml_status}", style={"fontFamily": "monospace", "fontSize": "13px", "color": ml_colour}),
                html.Div(f"Python:  {sys.executable}", style={"fontFamily": "monospace", "fontSize": "11px", "color": MUTED, "marginTop": "4px"}),
                html.Div(f"Dashboard started:  {now:%Y-%m-%d %H:%M}", style={"fontFamily": "monospace", "fontSize": "11px", "color": MUTED}),
            ], width=6),
            dbc.Col([
                html.Div("Feature Flags", style={"color": ACCENT, "fontFamily": "monospace", "fontWeight": "bold", "fontSize": "13px", "marginBottom": "6px"}),
                html.Div(f"Stop method:       {atr_status}", style={"fontFamily": "monospace", "fontSize": "13px"}),
                html.Div(f"Trailing stop:     {trailing_status}", style={"fontFamily": "monospace", "fontSize": "13px"}),
                html.Div(f"Partial exits:     {partial_status}", style={"fontFamily": "monospace", "fontSize": "13px"}),
                html.Div(f"Time stop:         {time_stop_status}", style={"fontFamily": "monospace", "fontSize": "13px"}),
                html.Div(f"Scaled entry:      {scaled_status}", style={"fontFamily": "monospace", "fontSize": "13px"}),
            ], width=6),
        ]),
    ])

    # ── 2. Open positions summary ────────────────────────────────────────────
    try:
        from trader import get_open_positions_snapshot
        positions = get_open_positions_snapshot()
    except Exception:
        positions = {}

    n_positions = len(positions)
    total_exposure = sum(s.entry_price * s.quantity for s in positions.values())
    pct_deployed = total_exposure / ACCOUNT_EQUITY * 100 if ACCOUNT_EQUITY else 0
    total_risk_est = sum(
        abs(s.entry_price - s.trailing_stop.current_stop) * s.quantity
        for s in positions.values()
    )
    oldest_hold = 0
    if positions:
        oldest_hold = max((datetime.date.today() - s.entry_date).days for s in positions.values())

    position_cards = dbc.Row([
        _stat_card("Open Positions", str(n_positions), GREEN if n_positions > 0 else MUTED),
        _stat_card("Total Exposure", f"${total_exposure:,.0f}", ACCENT),
        _stat_card("% Deployed", f"{pct_deployed:.1f}%",
                   GREEN if pct_deployed <= config.MAX_CAPITAL_DEPLOYED_PCT * 100 else RED),
        _stat_card("Est. $ at Risk", f"${total_risk_est:,.0f}", RED if total_risk_est > 0 else MUTED),
        _stat_card("Cash Available", f"${max(ACCOUNT_EQUITY - total_exposure, 0):,.0f}", TEXT),
        _stat_card("Oldest Hold", f"{oldest_hold}d" if n_positions else "—",
                   YELLOW if oldest_hold > config.TIME_STOP_DAYS * 0.7 else TEXT),
        _stat_card("Budget", f"${ACCOUNT_EQUITY:,.0f}", ACCENT),
    ], className="mb-3")

    # ── 3. Portfolio heat gauge (exposure donut) ──────────────────────────────
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct_deployed,
        title={"text": "Portfolio Utilisation %", "font": {"color": TEXT, "size": 14}},
        number={"suffix": "%", "font": {"color": TEXT, "size": 28}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": MUTED, "tickfont": {"color": MUTED}},
            "bar": {"color": GREEN if pct_deployed <= 60 else YELLOW if pct_deployed <= 80 else RED},
            "bgcolor": CARD_BG,
            "bordercolor": BORDER,
            "steps": [
                {"range": [0, 60],  "color": "#f0f0ed"},
                {"range": [60, 80], "color": "#e8e5df"},
                {"range": [80, 100],"color": "#f0dada"},
            ],
            "threshold": {
                "line": {"color": YELLOW, "width": 3},
                "thickness": 0.8,
                "value": config.MAX_CAPITAL_DEPLOYED_PCT * 100,
            },
        },
    ))
    gauge_fig.update_layout(
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font={"color": TEXT},
        margin=dict(t=50, b=20, l=30, r=30),
        height=220,
    )

    # Position-level exposure mini bar (if positions exist)
    if positions:
        pos_syms = list(positions.keys())
        pos_exposures = [positions[s].entry_price * positions[s].quantity for s in pos_syms]
        pos_pcts = [e / ACCOUNT_EQUITY * 100 for e in pos_exposures]
        pos_stops = [positions[s].trailing_stop.stage_label for s in pos_syms]

        pos_bar = go.Figure(go.Bar(
            x=pos_syms, y=pos_pcts,
            marker_color=[GREEN if p <= config.MAX_POSITION_PCT * 100 else RED for p in pos_pcts],
            text=[f"{p:.1f}%" for p in pos_pcts],
            textposition="outside",
            hovertemplate="%{x}: %{y:.1f}% of budget<extra></extra>",
        ))
        pos_bar.add_hline(
            y=config.MAX_POSITION_PCT * 100, line_dash="dash", line_color=YELLOW,
            annotation_text=f"Max {config.MAX_POSITION_PCT:.0%}", annotation_font_color=YELLOW,
        )
        pos_bar.update_layout(
            title="Position Exposure (% of Budget)",
            paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
            font=dict(color=TEXT),
            xaxis=dict(color=TEXT, gridcolor=BORDER),
            yaxis=dict(color=TEXT, gridcolor=BORDER, title="% of Budget"),
            margin=dict(t=40, b=20, l=20, r=20),
            height=250,
        )
        exposure_chart = dbc.Col(_card(dcc.Graph(figure=pos_bar, config={"displayModeBar": False})), width=8)
    else:
        exposure_chart = dbc.Col(_card(
            html.Div("No open positions — exposure charts will appear once trades are placed.",
                     style={"color": MUTED, "fontFamily": "monospace", "padding": "40px 0", "textAlign": "center"})
        ), width=8)

    # ── 4. Config summary card ────────────────────────────────────────────────
    config_card = _card([
        html.H6("⚙️ Config Summary", style={"color": ACCENT, "fontFamily": "monospace", "marginBottom": "12px"}),
        dbc.Row([
            dbc.Col([
                html.Div("Risk Management", style={"color": ACCENT, "fontFamily": "monospace", "fontWeight": "bold", "fontSize": "13px", "marginBottom": "4px"}),
                html.Div(f"Risk/trade:      {config.RISK_PER_TRADE_PCT:.1%} = ${ACCOUNT_EQUITY * config.RISK_PER_TRADE_PCT:,.0f}",
                         style={"fontFamily": "monospace", "fontSize": "12px"}),
                html.Div(f"Max position:    {config.MAX_POSITION_PCT:.0%} = ${ACCOUNT_EQUITY * config.MAX_POSITION_PCT:,.0f}",
                         style={"fontFamily": "monospace", "fontSize": "12px"}),
                html.Div(f"Max deployed:    {config.MAX_CAPITAL_DEPLOYED_PCT:.0%} = ${ACCOUNT_EQUITY * config.MAX_CAPITAL_DEPLOYED_PCT:,.0f}",
                         style={"fontFamily": "monospace", "fontSize": "12px"}),
                html.Div(f"Max positions:   {config.MAX_POSITIONS}",
                         style={"fontFamily": "monospace", "fontSize": "12px"}),
                html.Div(f"Max portfolio $: ${ACCOUNT_EQUITY * config.RISK_PER_TRADE_PCT * config.MAX_POSITIONS:,.0f} at risk",
                         style={"fontFamily": "monospace", "fontSize": "12px", "color": RED}),
            ], width=3),
            dbc.Col([
                html.Div("Stop & Target", style={"color": ACCENT, "fontFamily": "monospace", "fontWeight": "bold", "fontSize": "13px", "marginBottom": "4px"}),
                html.Div(f"ATR stop mult:   {config.ATR_STOP_MULTIPLIER}×",
                         style={"fontFamily": "monospace", "fontSize": "12px"}),
                html.Div(f"ATR target mult: {config.ATR_TARGET_MULTIPLIER}×",
                         style={"fontFamily": "monospace", "fontSize": "12px"}),
                html.Div(f"Stop clamp:      {config.ATR_MIN_STOP_PCT:.0%}–{config.ATR_MAX_STOP_PCT:.0%}",
                         style={"fontFamily": "monospace", "fontSize": "12px"}),
                html.Div(f"Time stop:       {config.TIME_STOP_DAYS}d",
                         style={"fontFamily": "monospace", "fontSize": "12px"}),
                html.Div(f"Trailing stages: 3-stage adaptive",
                         style={"fontFamily": "monospace", "fontSize": "12px"}),
            ], width=3),
            dbc.Col([
                html.Div("Sizing", style={"color": ACCENT, "fontFamily": "monospace", "fontWeight": "bold", "fontSize": "13px", "marginBottom": "4px"}),
                html.Div(f"Kelly fraction:  {config.KELLY_FRACTION}×{'  ✅' if config.KELLY_FRACTION > 0 else '  ❌'}",
                         style={"fontFamily": "monospace", "fontSize": "12px"}),
                html.Div(f"Kelly win rate:  {config.KELLY_WIN_RATE:.0%}",
                         style={"fontFamily": "monospace", "fontSize": "12px"}),
                html.Div(f"Vol scale HIGH:  {config.VOL_SCALE_HIGH:.2f}×",
                         style={"fontFamily": "monospace", "fontSize": "12px"}),
                html.Div(f"Vol scale LOW:   {config.VOL_SCALE_LOW:.2f}×",
                         style={"fontFamily": "monospace", "fontSize": "12px"}),
                html.Div(f"Sector penalty:  {config.CORRELATION_SAME_SECTOR_SCALE:.0%}",
                         style={"fontFamily": "monospace", "fontSize": "12px"}),
            ], width=3),
            dbc.Col([
                html.Div("Scaled Entry", style={"color": ACCENT, "fontFamily": "monospace", "fontWeight": "bold", "fontSize": "13px", "marginBottom": "4px"}),
                html.Div(f"Tranches:    {config.SCALED_ENTRY_N_TRANCHES}  ({', '.join(f'{f:.0%}' for f in config.SCALED_ENTRY_FRACTIONS)})",
                         style={"fontFamily": "monospace", "fontSize": "12px"}),
                html.Div(f"ATR offsets: {', '.join(f'{o:.1f}×' for o in config.SCALED_ENTRY_ATR_OFFSETS)}",
                         style={"fontFamily": "monospace", "fontSize": "12px"}),
                html.Div(f"RSI abort:   > {config.SCALED_ENTRY_RSI_ABORT_LEVEL:.0f}",
                         style={"fontFamily": "monospace", "fontSize": "12px"}),
                html.Div(f"Expiry:      {config.SCALED_ENTRY_EXPIRY_DAYS}d",
                         style={"fontFamily": "monospace", "fontSize": "12px"}),
                html.Div(f"Partial 1:   {config.PARTIAL_EXIT_1_FRACTION:.0%} @ +{config.PARTIAL_EXIT_1_TRIGGER_ATR:.0f}×ATR",
                         style={"fontFamily": "monospace", "fontSize": "12px"}),
                html.Div(f"Partial 2:   {config.PARTIAL_EXIT_2_FRACTION:.0%} @ +{config.PARTIAL_EXIT_2_TRIGGER_ATR:.0f}×ATR",
                         style={"fontFamily": "monospace", "fontSize": "12px"}),
            ], width=3),
        ]),
    ])

    # ── Recent activity summary ────────────────────────────────────────────
    # Last screener run (check for newest screen_*.txt in results/)
    last_screen = "—"
    try:
        screen_files = sorted(
            [f for f in os.listdir(RESULTS_DIR) if f.startswith("screen_") and f.endswith(".txt")],
            reverse=True,
        )
        if screen_files:
            last_screen = screen_files[0].replace("screen_", "").replace(".txt", "")
    except Exception:
        pass

    # Last backtest run
    last_backtest = "—"
    try:
        if os.path.exists(BACKTEST_CSV):
            bt_df = pd.read_csv(BACKTEST_CSV)
            if "backtest_date" in bt_df.columns:
                last_backtest = bt_df["backtest_date"].max()
    except Exception:
        pass

    # Trade journal stats
    trades_df = _load_trades()
    n_trades = len(trades_df)
    total_pnl = trades_df["realized_pnl"].dropna().sum() if not trades_df.empty else 0
    last_trade = trades_df["date"].iloc[-1] if not trades_df.empty else "—"

    # Screen log latest entry count
    n_screen_log = 0
    last_screen_log_date = "—"
    try:
        if os.path.exists(SCREEN_CSV):
            sl_df = pd.read_csv(SCREEN_CSV, on_bad_lines="skip")
            n_screen_log = len(sl_df)
            if "date" in sl_df.columns and not sl_df.empty:
                last_screen_log_date = sl_df["date"].iloc[-1]
    except Exception:
        pass

    activity_card = _card([
        html.H6("📋 Recent Activity", style={"color": ACCENT, "fontFamily": "monospace", "marginBottom": "12px"}),
        dbc.Row([
            dbc.Col([
                html.Div(f"Last screener run:    {last_screen}",
                         style={"fontFamily": "monospace", "fontSize": "13px"}),
                html.Div(f"Last backtest run:    {last_backtest}",
                         style={"fontFamily": "monospace", "fontSize": "13px"}),
                html.Div(f"Screen log entries:   {n_screen_log}  (latest: {last_screen_log_date})",
                         style={"fontFamily": "monospace", "fontSize": "13px"}),
            ], width=6),
            dbc.Col([
                html.Div(f"Trade journal:   {n_trades} entries  (last: {last_trade})",
                         style={"fontFamily": "monospace", "fontSize": "13px"}),
                html.Div(f"Realized P&L:    ${total_pnl:+,.2f}",
                         style={"fontFamily": "monospace", "fontSize": "13px",
                                "color": GREEN if total_pnl >= 0 else RED}),
                html.Div(f"Positions file:  {os.path.basename(os.path.join(RESULTS_DIR, 'positions.json'))}  ({n_positions} tracked)",
                         style={"fontFamily": "monospace", "fontSize": "13px", "color": MUTED}),
            ], width=6),
        ]),
    ])

    # ── 6.5 Mini sell-side summary for overview tab ─────────────────────────
    sellside_summary_card = html.Div()  # default empty
    if positions:
        # Fetch current prices for mini summary
        overview_prices = _fetch_current_prices(list(positions.keys()))

        # Build mini data for each position
        ov_syms = []
        ov_pnl_pcts = []
        ov_stages = []
        ov_hold_days = []
        ov_gain_atrs = []
        for sym, state in positions.items():
            cur = overview_prices.get(sym)
            entry = state.entry_price
            atr = state.atr_14_abs
            hd = (datetime.date.today() - state.entry_date).days
            pnl_pct = ((cur / entry - 1) * 100) if cur and entry > 0 else 0
            gain_atr = ((cur - entry) / atr) if cur and atr > 0 else 0

            ov_syms.append(sym)
            ov_pnl_pcts.append(round(pnl_pct, 1))
            ov_stages.append(state.trailing_stop.stage_label)
            ov_hold_days.append(hd)
            ov_gain_atrs.append(round(gain_atr, 2))

        # Mini P&L bar
        mini_pnl_fig = go.Figure(go.Bar(
            x=ov_syms, y=ov_pnl_pcts,
            marker_color=[GREEN if v >= 0 else RED for v in ov_pnl_pcts],
            text=[f"{v:+.1f}%" for v in ov_pnl_pcts],
            textposition="outside",
            hovertemplate="%{x}: %{y:+.1f}%<extra></extra>",
        ))
        mini_pnl_fig.add_hline(y=0, line_color=BORDER, line_dash="dash")
        mini_pnl_fig.update_layout(
            title="Open Positions — Unrealized P&L %",
            paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
            font=dict(color=TEXT),
            xaxis=dict(color=TEXT, gridcolor=BORDER),
            yaxis=dict(color=TEXT, gridcolor=BORDER, title="P&L %"),
            margin=dict(t=40, b=20, l=20, r=20),
            height=220,
        )

        # Mini trail stage indicator
        _stage_colour = {"INITIAL": MUTED, "BREAKEVEN": YELLOW,
                         "PROFIT_LOCK": GREEN, "TIGHT_TRAIL": ACCENT}
        mini_stage_fig = go.Figure(go.Bar(
            x=ov_syms,
            y=[{"INITIAL": 1, "BREAKEVEN": 2, "PROFIT_LOCK": 3, "TIGHT_TRAIL": 4}.get(s, 0)
               for s in ov_stages],
            marker_color=[_stage_colour.get(s, MUTED) for s in ov_stages],
            text=ov_stages,
            textposition="inside",
            hovertemplate="%{x}: %{text}<br>Hold: " +
                          "".join("") +  # placeholder
                          "<extra></extra>",
            customdata=list(zip(ov_hold_days, ov_gain_atrs)),
        ))
        mini_stage_fig.update_layout(
            title="Trail Stage + Holding Days",
            paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
            font=dict(color=TEXT),
            xaxis=dict(color=TEXT, gridcolor=BORDER),
            yaxis=dict(
                color=TEXT, gridcolor=BORDER,
                tickvals=[1, 2, 3, 4],
                ticktext=["INITIAL", "BREAKEVEN", "PROFIT_LOCK", "TIGHT_TRAIL"],
                title="Trail Stage",
            ),
            margin=dict(t=40, b=20, l=20, r=20),
            height=220,
        )

        # Summary metrics
        total_unreal_pct = sum(ov_pnl_pcts) / len(ov_pnl_pcts) if ov_pnl_pcts else 0
        best_pos = max(zip(ov_syms, ov_pnl_pcts), key=lambda x: x[1]) if ov_pnl_pcts else ("—", 0)
        worst_pos = min(zip(ov_syms, ov_pnl_pcts), key=lambda x: x[1]) if ov_pnl_pcts else ("—", 0)
        n_in_profit = sum(1 for v in ov_pnl_pcts if v > 0)
        avg_gain_atr = sum(ov_gain_atrs) / len(ov_gain_atrs) if ov_gain_atrs else 0

        sellside_summary_card = _card([
            html.H6("📊 Sell-Side Summary — Open Positions",
                    style={"color": ACCENT, "fontFamily": "monospace", "marginBottom": "4px"}),
            html.Div(
                "Quick view of open position performance.  See 💰 Risk & Capital tab for full sell-side dashboard.",
                style={"color": MUTED, "fontSize": "11px", "fontFamily": "monospace", "marginBottom": "10px"},
            ),
            dbc.Row([
                _stat_card("Avg P&L", f"{total_unreal_pct:+.1f}%",
                           GREEN if total_unreal_pct >= 0 else RED),
                _stat_card("Best", f"{best_pos[0]} {best_pos[1]:+.1f}%", GREEN),
                _stat_card("Worst", f"{worst_pos[0]} {worst_pos[1]:+.1f}%", RED),
                _stat_card("In Profit", f"{n_in_profit}/{len(ov_pnl_pcts)}",
                           GREEN if n_in_profit > len(ov_pnl_pcts) / 2 else YELLOW),
                _stat_card("Avg ×ATR", f"{avg_gain_atr:+.1f}×",
                           GREEN if avg_gain_atr > 0 else RED),
            ], className="mb-2"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=mini_pnl_fig, config={"displayModeBar": False}), width=6),
                dbc.Col(dcc.Graph(figure=mini_stage_fig, config={"displayModeBar": False}), width=6),
            ]),
        ])

    # ── 7. Watchlist grid ─────────────────────────────────────────────────────
    wl_rows = []
    for i, sym in enumerate(WATCHLIST):
        is_tracked = sym in positions
        hold_days = (datetime.date.today() - positions[sym].entry_date).days if is_tracked else None
        wl_rows.append({
            "#":       i + 1,
            "Symbol":  sym,
            "Status":  "🟢 OPEN" if is_tracked else "⚪ Watching",
            "Qty":     positions[sym].quantity if is_tracked else "—",
            "Entry $": round(positions[sym].entry_price, 2) if is_tracked else "—",
            "Stop $":  round(positions[sym].trailing_stop.current_stop, 2) if is_tracked else "—",
            "Trail":   positions[sym].trailing_stop.stage_label if is_tracked else "—",
            "Hold":    f"{hold_days}d" if hold_days is not None else "—",
        })

    wl_df = pd.DataFrame(wl_rows)
    wl_colours = [
        {"if": {"filter_query": '{Status} = "🟢 OPEN"',    "column_id": "Status"}, "color": GREEN, "fontWeight": "bold"},
        {"if": {"filter_query": '{Status} = "⚪ Watching"', "column_id": "Status"}, "color": MUTED},
        {"if": {"filter_query": '{Trail} = "BREAKEVEN"',    "column_id": "Trail"},  "color": YELLOW},
        {"if": {"filter_query": '{Trail} = "PROFIT_LOCK"',  "column_id": "Trail"},  "color": GREEN},
        {"if": {"filter_query": '{Trail} = "TIGHT_TRAIL"',  "column_id": "Trail"},  "color": GREEN, "fontWeight": "bold"},
    ]

    watchlist_card = _card([
        html.H6(f"📋 Watchlist  ({len(WATCHLIST)} tickers)", style={"color": ACCENT, "fontFamily": "monospace", "marginBottom": "4px"}),
        html.Div("All tracked tickers and their current state.  🟢 = open position, ⚪ = watching for dip.",
                 style={"color": MUTED, "fontSize": "11px", "fontFamily": "monospace", "marginBottom": "10px"}),
        dash_table.DataTable(
            id="overview-watchlist-table",
            columns=[{"name": c, "id": c} for c in wl_df.columns],
            data=wl_df.to_dict("records"),
            style_cell=_CELL_STYLE,
            style_header=_HDR_STYLE,
            style_data_conditional=wl_colours,
            style_table={"overflowX": "auto", "borderRadius": "6px"},
            sort_action="native",
            page_size=50,
        ),
    ])

    # ── Assemble the tab ──────────────────────────────────────────────────────
    return html.Div([
        status_card,
        position_cards,
        dbc.Row([
            dbc.Col(_card(dcc.Graph(figure=gauge_fig, config={"displayModeBar": False})), width=4),
            exposure_chart,
        ], className="mb-3"),
        sellside_summary_card,
        config_card,
        activity_card,
        watchlist_card,
    ])


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab"),
    State("screen-store", "data"),
    State("trades-store", "data"),
)
def render_tab(active_tab, store_data, trades_store):
    if active_tab == "tab-screener":
        return _screener_layout()
    if active_tab == "tab-risk":
        if store_data:
            screen_df = pd.DataFrame(store_data)
        else:
            screen_df = None
        return _risk_layout(screen_df)
    if active_tab == "tab-backtest":
        return _backtest_layout()
    if active_tab == "tab-log":
        return _log_layout()
    if active_tab == "tab-trades":
        return _trades_layout()
    if active_tab == "tab-overview":
        return _overview_layout()
    return html.Div("Unknown tab")


@app.callback(
    Output("screen-store",            "data"),
    Output("screener-table-container","children"),
    Output("fund-table-container",    "children"),
    Output("swing-table-container",   "children"),
    Output("stat-cards",              "children"),
    Output("refresh-status",          "children"),
    Output("last-refresh-label",      "children"),
    Output("ibkr-status-label",       "children"),
    Input("refresh-btn",              "n_clicks"),
    prevent_initial_call=True,
)
def refresh_screener(n_clicks):
    """Re-run the screener pipeline and update all three tables + store."""
    global _last_fetch

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ibkr_ok, ibkr_msg = _check_ibkr_status()
    ibkr_colour = GREEN if ibkr_ok else YELLOW

    try:
        records  = fetch_screen_data()
        df       = _records_to_df(records)
        _last_fetch = {"records": records, "df": df}
    except Exception as exc:
        err = html.Span(f"❌ Error: {exc}", style={"color": RED})
        no = dash.no_update
        return no, no, no, no, no, err, f"Last refresh: {ts}", html.Span(ibkr_msg, style={"color": ibkr_colour})

    hide_internal = ["_is_dip", "_fund_pass", "_ml_ok", "_price"]

    # ── TABLE 1: Dip Detector — entry signals ────────────────────────────────
    t1_cols = ["Symbol","Signal","Score","Price","RSI","vs MA50%","vs MA200%",
               "52wk%","Fund","ML 1W","ML 1M","ML 1Y","ML AUROC","Src"]
    t1_df = df[t1_cols].sort_values(
        "Score", ascending=False,
        key=lambda s: s.str.extract(r"(\d)")[0].astype(float),
    )
    t1 = _make_table(t1_df, "screener-table")

    # ── TABLE 2: Fundamentals ─────────────────────────────────────────────────
    t2_cols = ["Symbol","Fund","ROIC","FCF Margin","FCF","Cash Conv","D/E","Accruals","AR vs Rev Gr","Revenue","P/E","Signal"]
    t2_df = df[t2_cols].copy()

    # Colour-code ROIC and accruals cells
    fund_colours = [
        {"if": {"filter_query": '{Fund} = "✅"', "column_id": "Fund"}, "color": GREEN, "fontWeight": "bold"},
        {"if": {"filter_query": '{Fund} = "❌"', "column_id": "Fund"}, "color": RED,   "fontWeight": "bold"},
    ]
    t2 = dash_table.DataTable(
        id="fund-table",
        columns=[{"name": c, "id": c} for c in t2_df.columns],
        data=t2_df.to_dict("records"),
        style_cell=_CELL_STYLE,
        style_header=_HDR_STYLE,
        style_data_conditional=fund_colours,
        style_table={"overflowX": "auto", "borderRadius": "6px"},
        sort_action="native",
        filter_action="native",
        page_size=40,
    )

    # ── TABLE 3: Swing Sell Metrics — sorted by sell pressure ─────────────────
    t3_cols = ["Symbol","Sell Rec","Sell Score","RSI-14","RSI Sig",
               "BB%B","BB Sig","MACD Sig","vs MA50","vs MA200",
               "Stop $","Target $","R/R","Drawdown","Days@Hi","Vol"]
    t3_df = df[t3_cols].copy()
    # Sort: highest sell score first (numeric sort)
    t3_df = t3_df.sort_values("Sell Score", ascending=False, na_position="last")

    # Colour-code sell recommendation cells
    sell_colours = [
        {"if": {"filter_query": '{Sell Rec} = "STRONG_SELL"',   "column_id": "Sell Rec"}, "color": RED,    "fontWeight": "bold"},
        {"if": {"filter_query": '{Sell Rec} = "CONSIDER_SELL"', "column_id": "Sell Rec"}, "color": ORANGE, "fontWeight": "bold"},
        {"if": {"filter_query": '{Sell Rec} = "HOLD"',          "column_id": "Sell Rec"}, "color": YELLOW},
        {"if": {"filter_query": '{Sell Rec} = "ADD"',           "column_id": "Sell Rec"}, "color": GREEN},
        {"if": {"filter_query": '{RSI Sig} = "OVERBOUGHT"',     "column_id": "RSI Sig"},  "color": RED},
        {"if": {"filter_query": '{RSI Sig} = "OVERSOLD"',       "column_id": "RSI Sig"},  "color": GREEN},
        {"if": {"filter_query": '{BB Sig} = "EXTENDED"',        "column_id": "BB Sig"},   "color": RED},
        {"if": {"filter_query": '{BB Sig} = "COMPRESSED"',      "column_id": "BB Sig"},   "color": GREEN},
        {"if": {"filter_query": '{MACD Sig} = "BEARISH_CROSS"', "column_id": "MACD Sig"}, "color": RED},
        {"if": {"filter_query": '{MACD Sig} = "BULLISH"',       "column_id": "MACD Sig"}, "color": GREEN},
        {"if": {"filter_query": "{Sell Score} >= 75",           "column_id": "Sell Score"}, "color": RED,   "fontWeight": "bold"},
        {"if": {"filter_query": "{Sell Score} >= 45 && {Sell Score} < 75", "column_id": "Sell Score"}, "color": ORANGE},
        {"if": {"filter_query": "{Sell Score} < 20",            "column_id": "Sell Score"}, "color": GREEN},
        {"if": {"filter_query": '{Vol} = "HIGH"',               "column_id": "Vol"},       "color": RED},
        {"if": {"filter_query": '{Vol} = "LOW"',                "column_id": "Vol"},       "color": GREEN},
    ]
    t3 = dash_table.DataTable(
        id="swing-table",
        columns=[{"name": c, "id": c} for c in t3_df.columns],
        data=t3_df.to_dict("records"),
        style_cell=_CELL_STYLE,
        style_header=_HDR_STYLE,
        style_data_conditional=sell_colours,
        style_table={"overflowX": "auto", "borderRadius": "6px"},
        sort_action="native",
        filter_action="native",
        page_size=40,
    )

    # ── stat cards ────────────────────────────────────────────────────────────
    src_counts  = df["Src"].value_counts().to_dict() if "Src" in df.columns else {}
    src_summary = "  ".join(f"{src}:{cnt}" for src, cnt in src_counts.items())
    n_buy   = df["Signal"].str.contains("BUY").sum()
    n_dip   = df["Signal"].str.contains("DIP").sum()
    n_watch = df["Signal"].str.contains("WATCH").sum()
    n_strong_sell   = (df["Sell Rec"] == "STRONG_SELL").sum()
    n_consider_sell = (df["Sell Rec"] == "CONSIDER_SELL").sum()

    stat_cards = dbc.Row([
        _stat_card("🟢 Buy Signals",    str(n_buy),               GREEN),
        _stat_card("🟡 Dip Alerts",     str(n_dip),               YELLOW),
        _stat_card("⚪ Watch",           str(n_watch),             MUTED),
        _stat_card("🔴 Strong Sell",    str(n_strong_sell),       RED),
        _stat_card("🟠 Consider Sell",  str(n_consider_sell),     ORANGE),
        _stat_card("Tickers Scanned",   str(len(df)),             ACCENT),
        _stat_card("Price Sources",     src_summary or "—",       MUTED),
    ])

    return (
        df.to_dict("records"),
        t1, t2, t3,
        stat_cards,
        html.Span(f"✅ Updated {ts}", style={"color": GREEN}),
        f"Last refresh: {ts}",
        html.Span(ibkr_msg, style={"color": ibkr_colour}),
    )


@app.callback(
    Output("trades-store",      "data"),
    Output("trade-save-status", "children"),
    Output("trade-symbol",      "value"),
    Output("trade-qty",         "value"),
    Output("trade-entry",       "value"),
    Output("trade-exit",        "value"),
    Output("trade-notes",       "value"),
    Input("save-trade-btn",     "n_clicks"),
    State("trade-date",         "value"),
    State("trade-symbol",       "value"),
    State("trade-asset-type",   "value"),
    State("trade-action",       "value"),
    State("trade-qty",          "value"),
    State("trade-entry",        "value"),
    State("trade-exit",         "value"),
    State("trade-notes",        "value"),
    prevent_initial_call=True,
)
def save_trade(n_clicks, date, symbol, asset_type, action, qty, entry, exit_p, notes):
    """Validate inputs, auto-calculate P&L, append to CSV, refresh display."""
    errors = []
    if not symbol or not str(symbol).strip():
        errors.append("Symbol is required.")
    if not qty or float(qty) <= 0:
        errors.append("Quantity must be > 0.")
    if not entry or float(entry) <= 0:
        errors.append("Entry Price must be > 0.")
    if errors:
        return (
            dash.no_update,
            html.Span("⚠️ " + "  ".join(errors), style={"color": YELLOW}),
            dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
        )

    qty_f    = float(qty)
    entry_f  = float(entry)
    exit_f   = float(exit_p) if exit_p else None
    sym_upper = str(symbol).strip().upper()

    # Auto-compute realized P&L for SELL trades if exit is provided
    realized = None
    if action == "SELL" and exit_f is not None:
        realized = (exit_f - entry_f) * qty_f
    amount_invested = round(qty_f * entry_f, 2)

    # Resolve asset type (form dropdown → auto-detect fallback)
    resolved_type = _infer_asset_type(sym_upper, asset_type)

    row = {
        "date":            date or datetime.date.today().isoformat(),
        "symbol":          sym_upper,
        "action":          action or "BUY",
        "quantity":        qty_f,
        "entry_price":     entry_f,
        "exit_price":      exit_f if exit_f is not None else "",
        "amount_invested": amount_invested,
        "realized_pnl":    round(realized, 2) if realized is not None else "",
        "notes":           (notes or "").strip(),
        "asset_type":      resolved_type,
    }
    _append_trade(row)

    type_label = "📦 ETF" if resolved_type == "ETF" else "📈 Stock"
    pnl_msg = f"  →  P&L: ${realized:+,.2f}" if realized is not None else ""
    status_msg = html.Span(
        f"✅ Saved {row['action']} {qty_f:.0f}x {sym_upper} ({type_label}) @ ${entry_f:.2f}{pnl_msg}",
        style={"color": GREEN},
    )
    # Return a timestamp as store data to trigger re-render of trades tab
    return (
        {"saved_at": datetime.datetime.now().isoformat()},
        status_msg,
        "",    # clear symbol
        None,  # clear qty
        None,  # clear entry
        None,  # clear exit
        "",    # clear notes
    )


@app.callback(
    Output("last-refresh-label", "children", allow_duplicate=True),
    Input("clock-tick", "n_intervals"),
    State("last-refresh-label", "children"),
    prevent_initial_call=True,
)
def tick_clock(_, current_label):
    """Keep the navbar label fresh without re-fetching data."""
    return current_label or "Not refreshed yet"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="Buy-the-sauce live dashboard")
    parser.add_argument("--port", type=int, default=8055)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    _ensure_trades_csv()   # make sure journal exists on startup
    args = _parse_args()
    print(f"\n  🍅 Buy The Sauce Dashboard")
    print(f"  Open → http://127.0.0.1:{args.port}\n")
    print(f"  IBKR config: {config.IBKR_HOST}:{config.IBKR_PORT}  (clientId {config.IBKR_CLIENT_ID})")
    print(f"  Trade journal: {TRADES_CSV}\n")
    ibkr_ok, ibkr_msg = _check_ibkr_status()
    print(f"  {ibkr_msg}\n")
    app.run(
        debug=False,
        port=args.port,
        use_reloader=False,
    )
