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

import datetime
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

import config
import edgar
from broker import IBKRBroker
from dip_detector import DipSignal, score_dip
from risk_manager import (
    OrderSpec, calculate_order, plan_scaled_entry,
    ScaledEntryPlan, TrailingStopState,
    evaluate_partial_exits, check_time_stop,
)
from screener import FundamentalProfile, screen_fundamental
from watchlist import WATCHLIST

logger = logging.getLogger(__name__)

# Path for persisting position state across restarts
_STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "position_state.json")


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

    # Surface multi-horizon outlook + swing metrics in logs
    if dip is not None:
        _log_outlook(dip)

    return fundamental, dip


def _log_outlook(dip: DipSignal) -> None:
    """Log multi-horizon ML outlook and swing sell metrics for a ticker."""
    mh = dip.multi_horizon
    if mh is not None:
        logger.info(
            "%s — Multi-horizon outlook:\n"
            "  1W : %-4s  p=%-4s  [%s]%s\n"
            "  1M : %-4s  p=%-4s  [%s]%s\n"
            "  1Y : %-4s  p=%-4s  [%s]%s",
            dip.symbol,
            _pred_dir(mh.week1),  _pred_prob(mh.week1),  _pred_conf(mh.week1),  _pred_auroc(mh.week1),
            _pred_dir(mh.month1), _pred_prob(mh.month1), _pred_conf(mh.month1), _pred_auroc(mh.month1),
            _pred_dir(mh.year1),  _pred_prob(mh.year1),  _pred_conf(mh.year1),  _pred_auroc(mh.year1),
        )

    sm = dip.swing_metrics
    if sm is not None:
        logger.info(
            "%s — Swing sell metrics:\n"
            "  RSI-14        : %.1f  [%s]\n"
            "  Bollinger %%B  : %.2f  [%s]\n"
            "  MACD hist     : %+.6f  [%s]\n"
            "  vs MA-50      : %+.1f%%\n"
            "  vs MA-200     : %+.1f%%\n"
            "  Trend strength: %.2f   Vol regime: %s\n"
            "  ATR-14 (norm) : %.2f%%\n"
            "  Stop  (2×ATR) : $%.2f\n"
            "  Target (3×ATR): $%.2f   R/R = %.1f\n"
            "  Days since 52w high: %d   Drawdown: %.1f%%\n"
            "  ─── Sell score: %.0f/100  →  %s",
            sm.symbol,
            sm.rsi_14, sm.rsi_signal,
            sm.bb_position, sm.bb_signal,
            sm.macd_hist, sm.macd_signal,
            sm.price_vs_ma50  * 100 if not (sm.price_vs_ma50  != sm.price_vs_ma50) else float("nan"),
            sm.price_vs_ma200 * 100 if not (sm.price_vs_ma200 != sm.price_vs_ma200) else float("nan"),
            sm.trend_strength, sm.vol_regime,
            sm.atr_14 * 100,
            sm.atr_stop_price,
            sm.atr_target_price, sm.reward_risk_ratio,
            sm.days_since_high, sm.drawdown_from_high * 100,
            sm.composite_sell_score, sm.sell_recommendation,
        )


# ── Helper formatters ──────────────────────────────────────────────────────

def _pred_dir(p) -> str:
    return p.direction if p else "n/a"

def _pred_prob(p) -> str:
    return f"{p.probability:.0%}" if p else "—"

def _pred_conf(p) -> str:
    return p.confidence if p else "—"

def _pred_auroc(p) -> str:
    return f"  AUROC={p.auroc_cv:.3f}" if (p and p.auroc_cv is not None) else ""


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
            logger.info(
                "%s: DIP CANDIDATE (score=%d) | ML=%s | Sell=%s",
                symbol, dip.score,
                dip.ml_direction or "n/a",
                dip.swing_metrics.sell_recommendation if dip.swing_metrics else "n/a",
            )

    logger.info("=== %d dip candidate(s) found ===", len(candidates))

    if not candidates:
        logger.info("No opportunities — no orders placed")
        return []

    if dry_run:
        PLACEHOLDER_EQUITY = 100_000.0
        specs = []
        for dip in candidates:
            sm = dip.swing_metrics
            spec = calculate_order(
                symbol=dip.symbol,
                entry_price=dip.price,
                account_equity=PLACEHOLDER_EQUITY,
                open_positions=0,
                atr_14=sm.atr_14 if sm else None,
                reward_risk_ratio=sm.reward_risk_ratio if sm else None,
                vol_regime=sm.vol_regime if sm else None,
            )
            if spec:
                specs.append(spec)
                logger.info("[DRY RUN] Would place: %s", spec)

                # Show scaled entry plan
                if config.SCALED_ENTRY_ENABLED:
                    plan = plan_scaled_entry(
                        spec,
                        atr_14=sm.atr_14 if sm else None,
                        current_rsi=sm.rsi_14 if sm else None,
                    )
                    logger.info("[DRY RUN] Scaled entry plan:\n%s", plan)

        return specs

    # --- Live execution ---
    submitted: list = []
    with IBKRBroker() as broker:
        equity = broker.get_account_equity()
        positions = broker.get_open_positions()
        open_count = len([p for p in positions if p["quantity"] > 0])
        logger.info("Account equity: $%.2f | Open positions: %d", equity, open_count)

        for dip in candidates:
            sm = dip.swing_metrics
            spec = calculate_order(
                symbol=dip.symbol,
                entry_price=dip.price,
                account_equity=equity,
                open_positions=open_count,
                atr_14=sm.atr_14 if sm else None,
                reward_risk_ratio=sm.reward_risk_ratio if sm else None,
                vol_regime=sm.vol_regime if sm else None,
            )
            if spec is None:
                continue

            # ── Pre-trade safety check ─────────────────────────────────────
            ok, reason = broker.pre_trade_check(spec)
            if not ok:
                logger.warning("%s: skipping — %s", dip.symbol, reason)
                continue

            # ── Scaled entry or single bracket ─────────────────────────────
            atr_abs = (sm.atr_14 * dip.price) if sm and sm.atr_14 else (spec.stop_pct * dip.price)

            if config.SCALED_ENTRY_ENABLED and sm and sm.atr_14 > 0:
                plan = plan_scaled_entry(
                    spec,
                    atr_14=sm.atr_14,
                    current_rsi=sm.rsi_14 if sm else None,
                )
                trades = broker.place_scaled_entry(plan, spec)
                if trades:
                    submitted.append(spec)
                    open_count += 1
                    # Register for sell-side management
                    register_position(
                        symbol=dip.symbol,
                        entry_price=spec.entry_price,
                        quantity=spec.quantity,
                        atr_14_abs=atr_abs,
                        stop_loss_price=spec.stop_loss_price,
                        scaled_plan=plan,
                    )
                    logger.info(
                        "%s: placed %d-tranche scaled entry (%d orders)",
                        dip.symbol, plan.n_tranches, len(trades),
                    )
            else:
                # Fallback to single bracket order
                trades = broker.place_bracket_order(spec)
                if trades:
                    submitted.append(spec)
                    open_count += 1
                    # Register for sell-side management
                    register_position(
                        symbol=dip.symbol,
                        entry_price=spec.entry_price,
                        quantity=spec.quantity,
                        atr_14_abs=atr_abs,
                        stop_loss_price=spec.stop_loss_price,
                    )

        # ── Manage existing positions (trailing stops, partials, time) ────
        mgmt = manage_open_positions(broker=broker, dry_run=False)
        logger.info("Position management result: %s", mgmt)

    logger.info("=== Scan complete — %d order(s) submitted ===", len(submitted))
    return submitted


# ---------------------------------------------------------------------------
# Position management only  (lightweight — no buy scan)
# ---------------------------------------------------------------------------

def run_manage_only(dry_run: bool = False) -> dict:
    """Run sell-side position management without scanning for new buys.

    This is the lightweight loop that should run every 15 minutes:
    - Fetches current prices for all tracked positions
    - Updates trailing stops (and pushes to IBKR)
    - Fires partial exits at +2×ATR and +3×ATR
    - Closes time-stopped positions
    - Checks scaled entry RSI abort / expiry

    Connects to IBKR only if there are tracked positions and dry_run=False.
    """
    if not _open_positions:
        logger.debug("[MANAGE] No tracked positions — nothing to do")
        return {"trailing_updates": [], "partial_exits": [], "time_stops": [], "full_exits": []}

    logger.info("[MANAGE] === Position management cycle — %d position(s) ===", len(_open_positions))

    if dry_run:
        result = manage_open_positions(broker=None, dry_run=True)
        logger.info("[MANAGE] Dry-run result: %s", result)
        return result

    try:
        with IBKRBroker() as broker:
            result = manage_open_positions(broker=broker, dry_run=False)
            logger.info("[MANAGE] Result: %s", result)
            _save_state()
            return result
    except Exception as exc:
        logger.error("[MANAGE] IBKR connection failed — %s.  Stops are still active at broker level.", exc)
        # Even if we can't connect, run the logic in dry-run mode to log state
        result = manage_open_positions(broker=None, dry_run=True)
        return result


# ---------------------------------------------------------------------------
# Position management state  (persisted to JSON across scan cycles)
# ---------------------------------------------------------------------------

@dataclass
class PositionState:
    """Tracks the sell-side state for one open position."""
    symbol:             str
    entry_price:        float
    entry_date:         datetime.date
    quantity:           int
    atr_14_abs:         float     # ATR(14) in dollar terms at entry
    trailing_stop:      TrailingStopState
    partial_exits_taken: set      = field(default_factory=set)  # {1, 2}
    scaled_plan:        ScaledEntryPlan | None = None

    def __str__(self) -> str:
        return (
            f"{self.symbol}: {self.quantity}sh @ ${self.entry_price:.2f}"
            f" | {self.trailing_stop.stage_label}"
            f" | partials: {self.partial_exits_taken or 'none'}"
        )

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dict."""
        return {
            "symbol":              self.symbol,
            "entry_price":         self.entry_price,
            "entry_date":          self.entry_date.isoformat(),
            "quantity":            self.quantity,
            "atr_14_abs":          self.atr_14_abs,
            "trailing_stop": {
                "entry_price":    self.trailing_stop.entry_price,
                "atr_14_abs":     self.trailing_stop.atr_14_abs,
                "initial_stop":   self.trailing_stop.initial_stop,
                "current_stop":   self.trailing_stop.current_stop,
                "highest_price":  self.trailing_stop.highest_price,
                "stage":          self.trailing_stop.stage,
                "n_updates":      self.trailing_stop.n_updates,
            },
            "partial_exits_taken": list(self.partial_exits_taken),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PositionState":
        """Deserialise from a JSON-safe dict."""
        ts_d = d["trailing_stop"]
        ts = TrailingStopState(
            entry_price=ts_d["entry_price"],
            atr_14_abs=ts_d["atr_14_abs"],
            initial_stop=ts_d["initial_stop"],
        )
        ts.current_stop = ts_d["current_stop"]
        ts.highest_price = ts_d["highest_price"]
        ts.stage = ts_d["stage"]
        ts.n_updates = ts_d["n_updates"]
        return cls(
            symbol=d["symbol"],
            entry_price=d["entry_price"],
            entry_date=datetime.date.fromisoformat(d["entry_date"]),
            quantity=d["quantity"],
            atr_14_abs=d["atr_14_abs"],
            trailing_stop=ts,
            partial_exits_taken=set(d.get("partial_exits_taken", [])),
            scaled_plan=None,   # scaled plans are transient; not persisted
        )


# Module-level state store — loaded from disk on startup.
_open_positions: dict[str, PositionState] = {}


def _save_state() -> None:
    """Persist position state to disk as JSON."""
    try:
        os.makedirs(os.path.dirname(_STATE_FILE), exist_ok=True)
        data = {sym: state.to_dict() for sym, state in _open_positions.items()}
        with open(_STATE_FILE, "w") as f:
            json.dump(data, f, indent=2)
        logger.debug("Position state saved (%d positions) → %s", len(data), _STATE_FILE)
    except Exception as exc:
        logger.warning("Failed to save position state: %s", exc)


def _load_state() -> None:
    """Load position state from disk on startup."""
    global _open_positions
    if not os.path.exists(_STATE_FILE):
        return
    try:
        with open(_STATE_FILE) as f:
            data = json.load(f)
        _open_positions = {sym: PositionState.from_dict(d) for sym, d in data.items()}
        logger.info("Loaded position state: %d positions from %s", len(_open_positions), _STATE_FILE)
    except Exception as exc:
        logger.warning("Failed to load position state: %s", exc)
        _open_positions = {}


# Load persisted state on module import
_load_state()


def register_position(
    symbol: str,
    entry_price: float,
    quantity: int,
    atr_14_abs: float,
    stop_loss_price: float,
    scaled_plan: ScaledEntryPlan | None = None,
) -> PositionState:
    """Register a newly entered position for sell-side management."""
    ts = TrailingStopState(
        entry_price=entry_price,
        atr_14_abs=atr_14_abs,
        initial_stop=stop_loss_price,
    )
    state = PositionState(
        symbol=symbol,
        entry_price=entry_price,
        entry_date=datetime.date.today(),
        quantity=quantity,
        atr_14_abs=atr_14_abs,
        trailing_stop=ts,
        scaled_plan=scaled_plan,
    )
    _open_positions[symbol] = state
    _save_state()
    logger.info("Registered position for sell management: %s", state)
    return state


def unregister_position(symbol: str) -> None:
    """Remove a position from the sell-side tracker (after full exit)."""
    if symbol in _open_positions:
        del _open_positions[symbol]
        _save_state()
        logger.info("Unregistered position: %s", symbol)


def get_open_positions_snapshot() -> dict[str, PositionState]:
    """Return a shallow copy of the tracked positions (for dashboard use)."""
    return dict(_open_positions)


# ---------------------------------------------------------------------------
# Sell-side management loop  (called every scan cycle)
# ---------------------------------------------------------------------------

def manage_open_positions(broker: IBKRBroker | None = None, dry_run: bool = False) -> dict:
    """Evaluate all tracked positions for sell-side actions.

    For each open position, in order:
    1. Time stop — if holding > TIME_STOP_DAYS, close at market.
    2. Trailing stop — update stop level; if price ≤ stop, close.
    3. Partial exits — if +2×ATR or +3×ATR hit, sell the configured fraction.
    4. Scaled entry abort — if RSI has reversed, cancel unfilled tranches.

    Parameters
    ----------
    broker : IBKRBroker | None
        Live broker connection.  If None, run in dry-run/audit mode.
    dry_run : bool
        Log actions but don't place real orders.

    Returns
    -------
    dict with keys:
        trailing_updates : list of (symbol, old_stop, new_stop)
        partial_exits    : list of (symbol, exit_id, qty_sold, price)
        time_stops       : list of symbol
        full_exits       : list of (symbol, reason)
    """
    result = {
        "trailing_updates": [],
        "partial_exits":    [],
        "time_stops":       [],
        "full_exits":       [],
    }

    if not _open_positions:
        logger.info("No tracked positions — nothing to manage")
        return result

    logger.info("=== Managing %d open position(s) ===", len(_open_positions))

    # Get current prices for all tracked symbols
    current_prices: dict[str, float] = {}
    for symbol in list(_open_positions.keys()):
        try:
            hist = _fetch_history(symbol, period_years=1)
            if hist is not None and not hist.empty:
                current_prices[symbol] = float(hist["Close"].iloc[-1])
        except Exception as exc:
            logger.warning("%s: failed to get current price — %s", symbol, exc)

    for symbol, state in list(_open_positions.items()):
        price = current_prices.get(symbol)
        if price is None:
            logger.warning("%s: no current price — skipping management", symbol)
            continue

        logger.info(
            "%s: price=$%.2f  entry=$%.2f  stop=$%.2f [%s]  qty=%d",
            symbol, price, state.entry_price,
            state.trailing_stop.current_stop, state.trailing_stop.stage_label,
            state.quantity,
        )

        # ── 1. Time stop ─────────────────────────────────────────────────
        if check_time_stop(state.entry_date):
            logger.warning(
                "%s: TIME STOP triggered — held %d days (max %d)",
                symbol,
                (datetime.date.today() - state.entry_date).days,
                config.TIME_STOP_DAYS,
            )
            if not dry_run and broker:
                broker.close_position(symbol)
            result["time_stops"].append(symbol)
            result["full_exits"].append((symbol, "time_stop"))
            unregister_position(symbol)
            continue

        # ── 2. Trailing stop update ───────────────────────────────────────
        old_stop = state.trailing_stop.current_stop
        new_stop = state.trailing_stop.update(price)

        if new_stop > old_stop:
            logger.info(
                "%s: trailing stop updated $%.2f → $%.2f [%s]",
                symbol, old_stop, new_stop, state.trailing_stop.stage_label,
            )
            result["trailing_updates"].append((symbol, old_stop, new_stop))

            # Push the new stop to the broker
            if not dry_run and broker:
                broker.update_stop_order(symbol, new_stop)

        # Check if current price has hit the trailing stop
        if price <= state.trailing_stop.current_stop:
            logger.warning(
                "%s: TRAILING STOP HIT — price $%.2f ≤ stop $%.2f [%s]",
                symbol, price, state.trailing_stop.current_stop,
                state.trailing_stop.stage_label,
            )
            if not dry_run and broker:
                broker.close_position(symbol)
            result["full_exits"].append((symbol, f"trailing_stop_{state.trailing_stop.stage_label}"))
            unregister_position(symbol)
            continue

        # ── 3. Partial exits ──────────────────────────────────────────────
        exits = evaluate_partial_exits(
            current_price=price,
            entry_price=state.entry_price,
            atr_14_abs=state.atr_14_abs,
            exits_already_taken=state.partial_exits_taken,
        )

        for sig in exits:
            sell_qty = max(1, int(state.quantity * sig.fraction))
            sell_qty = min(sell_qty, state.quantity - 1)  # keep ≥1 share for remainder

            if sell_qty <= 0:
                continue

            logger.info(
                "%s: PARTIAL EXIT #%d — sell %d/%d shares @ $%.2f (+%.1f×ATR)",
                symbol, sig.exit_id, sell_qty, state.quantity,
                sig.exit_price, sig.trigger_atr,
            )

            if not dry_run and broker:
                broker.sell_partial(symbol, sell_qty, limit_price=sig.exit_price)

            state.quantity -= sell_qty
            state.partial_exits_taken.add(sig.exit_id)
            result["partial_exits"].append((symbol, sig.exit_id, sell_qty, sig.exit_price))

        # ── 4. Scaled entry abort (RSI reversal check) ───────────────────
        if state.scaled_plan:
            try:
                from ml_predictor import compute_swing_metrics
                hist = _fetch_history(symbol, period_years=1)
                if hist is not None and not hist.empty:
                    sm = compute_swing_metrics(symbol, hist)
                    if sm:
                        state.scaled_plan.check_rsi_abort(sm.rsi_14)
                        state.scaled_plan.check_expiry()
            except Exception as exc:
                logger.debug("%s: scaled entry RSI check skipped — %s", symbol, exc)

    logger.info(
        "=== Position management complete — %d trailing updates, %d partial exits, "
        "%d time stops, %d full exits ===",
        len(result["trailing_updates"]), len(result["partial_exits"]),
        len(result["time_stops"]), len(result["full_exits"]),
    )
    return result
