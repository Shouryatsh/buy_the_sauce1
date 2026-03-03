"""
trader.py — Orchestrates the full pipeline for one scan cycle.

Pipeline
--------
1. For each ticker in the watchlist:
   a. Fetch fundamental data (SEC EDGAR via edgar.py).
   b. Apply fundamental quality screen (screener.py).
   c. Fetch price history (Stooq via edgar.py).
   d. Score for dip signals (dip_detector.py).
   e. If score >= MIN_DIP_SCORE -> candidate for buying.
2. For each dip candidate:
   a. Calculate order parameters (risk_manager.py).
   b. Place bracket order via IBKR (broker.py).

The scan is intentionally stateless — running it daily via run.py
or a cron job is how it achieves the "run daily, buy only on real
opportunities" requirement.
"""

from __future__ import annotations

import logging
from typing import Optional

import config
import edgar
from broker import IBKRBroker
from dip_detector import DipSignal, score_dip
from risk_manager import OrderSpec, calculate_order
from screener import FundamentalProfile, screen_fundamental
from watchlist import WATCHLIST

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data fetching helpers
# ---------------------------------------------------------------------------

def _fetch_info(symbol: str) -> dict:
    """Fetch fundamentals from SEC EDGAR (free, no API key)."""
    try:
        return edgar.get_fundamentals(symbol)
    except Exception as exc:
        logger.warning("%s: EDGAR fetch failed — %s", symbol, exc)
        return {}


def _fetch_history(symbol: str, period_years: int = 2):
    """Fetch price history (IBKR if available, else Stooq).

    Returns the DataFrame only; source label is logged automatically by edgar.py.
    """
    try:
        hist, source = edgar.get_price_history(symbol, period_years=period_years)
        if hist is None or hist.empty:
            raise ValueError("empty price history")
        logger.info("%s: price history source = %s", symbol, source)
        return hist
    except Exception as exc:
        logger.warning("%s: price history fetch failed — %s", symbol, exc)
        return None


# ---------------------------------------------------------------------------
# Single-stock evaluation
# ---------------------------------------------------------------------------

def evaluate_symbol(symbol: str) -> tuple:
    """Run the full evaluation for one ticker.

    Returns
    -------
    (FundamentalProfile, DipSignal) — either may be None on data failure.
    """
    logger.info("-- Evaluating %s --", symbol)

    info = _fetch_info(symbol)
    fundamental = screen_fundamental(symbol, info=info)

    if not fundamental.passes:
        logger.info("%s: failed fundamental screen, skipping dip detection", symbol)
        return fundamental, None

    history = _fetch_history(symbol)
    if history is None:
        logger.warning("%s: no price history — skipping dip detection", symbol)
        return fundamental, None

    dip = score_dip(symbol, history)
    return fundamental, dip


# ---------------------------------------------------------------------------
# Full scan
# ---------------------------------------------------------------------------

def run_scan(dry_run: bool = False) -> list:
    """Scan all watchlist tickers and optionally place orders.

    Parameters
    ----------
    dry_run:
        If True, evaluate signals and sizes but do NOT connect to IBKR
        or place any real orders.  Useful for testing and paper-review.

    Returns
    -------
    List of OrderSpec objects that were (or would be) submitted.
    """
    logger.info("=== Starting scan — %d tickers, dry_run=%s ===", len(WATCHLIST), dry_run)

    # Collect dip candidates
    candidates: list = []
    for symbol in WATCHLIST:
        _, dip = evaluate_symbol(symbol)
        if dip is not None and dip.is_dip:
            candidates.append(dip)
            logger.info("%s: DIP CANDIDATE (score=%d)", symbol, dip.score)

    logger.info("=== %d dip candidate(s) found ===", len(candidates))

    if not candidates:
        logger.info("No opportunities — no orders placed")
        return []

    if dry_run:
        PLACEHOLDER_EQUITY = 100_000.0
        specs = []
        for dip in candidates:
            spec = calculate_order(
                symbol=dip.symbol,
                entry_price=dip.price,
                account_equity=PLACEHOLDER_EQUITY,
                open_positions=0,
            )
            if spec:
                specs.append(spec)
                logger.info("[DRY RUN] Would place: %s", spec)
        return specs

    # --- Live execution ---
    submitted: list = []
    with IBKRBroker() as broker:
        equity = broker.get_account_equity()
        positions = broker.get_open_positions()
        open_count = len([p for p in positions if p["quantity"] > 0])
        logger.info("Account equity: $%.2f | Open positions: %d", equity, open_count)

        for dip in candidates:
            spec = calculate_order(
                symbol=dip.symbol,
                entry_price=dip.price,
                account_equity=equity,
                open_positions=open_count,
            )
            if spec is None:
                continue

            trades = broker.place_bracket_order(spec)
            if trades:
                submitted.append(spec)
                open_count += 1

    logger.info("=== Scan complete — %d order(s) submitted ===", len(submitted))
    return submitted
