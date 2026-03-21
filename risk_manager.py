"""
risk_manager.py — Position sizing, stop-loss, take-profit, trailing stops,
                   partial exits, and scaled entry (buy ladder).

Design — Simultaneous Portfolio Sizing
---------------------------------------
The module exposes these main interfaces:

1. calculate_order()  — sizes a SINGLE position given the current portfolio state.
   Used by both the live trader and the dashboard for "what-if" per-ticker sizing.

2. size_portfolio()   — the PORTFOLIO-LEVEL entry point used by the dashboard's
   Risk & Capital tab.  It processes all dip candidates simultaneously, tracks
   cumulative capital deployment, and returns the full allocation table.

3. plan_scaled_entry() — given an OrderSpec plus ATR, generates a ScaledEntryPlan
   with N Tranche objects at progressively lower prices.

4. TrailingStopState — tracks the adaptive three-stage trailing stop for a live
   position (breakeven lock → profit lock → tight trail).

5. evaluate_partial_exits() — given current price and position state, returns
   which partial-exit triggers have been hit.

6. check_time_stop() — returns True if the holding period exceeds TIME_STOP_DAYS.

Position-size logic (applied in order, most conservative wins)
---------------------------------------------------------------
A. Fixed-risk sizing
   qty_risk = floor( (equity × RISK_PER_TRADE_PCT) / risk_per_share )
   Uses ATR-based stop when USE_ATR_STOPS=True, else fixed STOP_LOSS_PCT.

B. Fractional Kelly sizing (optional, KELLY_FRACTION > 0)
   edge  = win_rate − (1 − win_rate) / reward_risk_ratio
   full_kelly  = edge / win_rate                  (fraction of equity)
   kelly_notional = KELLY_FRACTION × full_kelly × equity
   qty_kelly = floor( kelly_notional / entry_price )

C. Max-position cap
   qty_cap = floor( equity × MAX_POSITION_PCT / entry_price )

D. Remaining-capital cap  (portfolio-level only)
   qty_cash = floor( remaining_cash / entry_price )

   Final qty = min(qty_risk, qty_kelly [if enabled], qty_cap, qty_cash)

E. Volatility-regime scaling
   scale = VOL_SCALE_{HIGH|NORMAL|LOW}  (from SwingMetrics.vol_regime)
   qty = floor( qty × scale )            — applied after min() above

F. Sector-correlation penalty
   If a same-sector position is already allocated, scale qty by
   CORRELATION_SAME_SECTOR_SCALE (default 0.75 = 25% haircut).

Stop-loss placement
-------------------
  ATR mode  : stop  = entry − ATR_STOP_MULTIPLIER × ATR(14)
              target = entry + ATR_TARGET_MULTIPLIER × ATR(14)
              Clamped to [ATR_MIN_STOP_PCT, ATR_MAX_STOP_PCT] of entry.
  Fixed mode: stop  = entry × (1 − STOP_LOSS_PCT)
              target = entry × (1 + TAKE_PROFIT_PCT)

Trailing stop (adaptive three-stage)
-------------------------------------
  Stage 1: price reaches entry + 1×ATR  → trail = entry (breakeven)
  Stage 2: price reaches entry + 2×ATR  → trail = entry + 1×ATR
  Stage 3: price reaches entry + 3×ATR  → trail = high  − 1×ATR (tight)
  Trail can only move UP, never down.

Scaled entry (buy ladder)
-------------------------
  Split full qty into N tranches at progressively lower prices:
    T1: at current dip price (immediate)
    T2: −1×ATR below T1
    T3: −2×ATR below T1
  Abort unfilled tranches if RSI > 50 (momentum reversal) or after 5 days.

Additional guardrails
---------------------
* Open positions are capped at MAX_POSITIONS.
* Total deployed capital is capped at equity × MAX_CAPITAL_DEPLOYED_PCT.
* Any position that would require allocating more than remaining cash is
  scaled down rather than skipped (unless it would round to 0 shares).
"""

from __future__ import annotations

import datetime
import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------

@dataclass
class OrderSpec:
    """All the parameters needed to place a bracket order."""
    symbol:           str
    quantity:         int
    entry_price:      float   # limit price for the entry leg
    stop_loss_price:  float   # stop price for the protective stop
    take_profit_price: float  # limit price for the take-profit leg
    risk_amount:      float   # dollar risk for this trade (= qty × stop_dist)
    position_value:   float   # notional value of the position
    stop_method:      str     # "ATR" | "FIXED"
    stop_pct:         float   # actual stop distance as fraction of entry
    target_pct:       float   # actual target distance as fraction of entry
    reward_risk_ratio: float  # target_dist / stop_dist
    kelly_qty:        int     # Kelly-optimal qty (before vol/correlation scaling)
    risk_qty:         int     # fixed-risk-optimal qty
    vol_scale:        float   # vol-regime multiplier applied
    sector_scale:     float   # sector-correlation multiplier applied
    sizing_method:    str     # which constraint was binding

    def __str__(self) -> str:
        return (
            f"{self.symbol}: qty={self.quantity} ({self.sizing_method})"
            f" entry={self.entry_price:.2f}"
            f" stop={self.stop_loss_price:.2f} ({self.stop_pct:.1%}, {self.stop_method})"
            f" target={self.take_profit_price:.2f} ({self.target_pct:.1%})"
            f" RR={self.reward_risk_ratio:.2f}"
            f" risk=${self.risk_amount:.0f}"
            f" notional=${self.position_value:.0f}"
            f" vol_scale={self.vol_scale:.2f}"
            f" sector_scale={self.sector_scale:.2f}"
        )


# ---------------------------------------------------------------------------
# Scaled entry (buy ladder)
# ---------------------------------------------------------------------------

@dataclass
class Tranche:
    """One leg of a scaled entry plan."""
    tranche_id:   int        # 1-based (T1, T2, T3…)
    limit_price:  float      # limit order price for this tranche
    quantity:     int         # shares to buy in this tranche
    fraction:     float      # fraction of total position (e.g. 0.40)
    atr_offset:   float      # ATR multiples below T1 price
    status:       str = "PENDING"  # PENDING | FILLED | CANCELLED | EXPIRED

    def notional(self) -> float:
        return self.quantity * self.limit_price

    def __str__(self) -> str:
        return (
            f"T{self.tranche_id}: {self.quantity}sh @ ${self.limit_price:.2f}"
            f" ({self.fraction:.0%}, −{self.atr_offset:.1f}×ATR)"
            f" [{self.status}]"
        )


@dataclass
class ScaledEntryPlan:
    """Complete scaled entry (buy ladder) plan for a position.

    Attributes
    ----------
    symbol          : ticker
    total_quantity  : full position size (from OrderSpec.quantity)
    tranches        : list of Tranche objects
    atr_14_abs      : ATR(14) in dollar terms
    entry_price     : T1 price (= OrderSpec.entry_price)
    stop_loss_price : hard stop for the entire position
    take_profit_price : target for the entire position
    expiry_date     : date after which unfilled tranches auto-cancel
    rsi_abort_level : RSI above which unfilled T2+ tranches are aborted
    rsi_turn_required : if True, T2+ require RSI to be declining bar/bar
    created_at      : timestamp when the plan was generated
    """
    symbol:             str
    total_quantity:     int
    tranches:           list[Tranche]
    atr_14_abs:         float
    entry_price:        float
    stop_loss_price:    float
    take_profit_price:  float
    expiry_date:        datetime.date
    rsi_abort_level:    float
    rsi_turn_required:  bool
    created_at:         datetime.datetime = field(
        default_factory=datetime.datetime.now
    )

    @property
    def n_tranches(self) -> int:
        return len(self.tranches)

    @property
    def filled_qty(self) -> int:
        return sum(t.quantity for t in self.tranches if t.status == "FILLED")

    @property
    def pending_qty(self) -> int:
        return sum(t.quantity for t in self.tranches if t.status == "PENDING")

    @property
    def avg_fill_price(self) -> float:
        fills = [(t.quantity, t.limit_price) for t in self.tranches if t.status == "FILLED"]
        if not fills:
            return 0.0
        total_cost = sum(q * p for q, p in fills)
        total_qty  = sum(q for q, _ in fills)
        return total_cost / total_qty if total_qty > 0 else 0.0

    @property
    def total_committed_notional(self) -> float:
        return sum(t.notional() for t in self.tranches if t.status in ("PENDING", "FILLED"))

    def abort_unfilled(self, reason: str = "ABORTED") -> int:
        """Cancel all unfilled tranches.  Returns count of cancelled."""
        n = 0
        for t in self.tranches:
            if t.status == "PENDING":
                t.status = "CANCELLED"
                n += 1
        if n > 0:
            logger.info("%s: aborted %d unfilled tranche(s) — %s", self.symbol, n, reason)
        return n

    def check_expiry(self, today: datetime.date | None = None) -> int:
        """Expire unfilled tranches past the expiry date.  Returns count."""
        today = today or datetime.date.today()
        n = 0
        for t in self.tranches:
            if t.status == "PENDING" and today >= self.expiry_date:
                t.status = "EXPIRED"
                n += 1
        if n > 0:
            logger.info("%s: expired %d unfilled tranches past %s",
                        self.symbol, n, self.expiry_date)
        return n

    def check_rsi_abort(self, current_rsi: float, prior_rsi: float | None = None) -> int:
        """Abort unfilled T2+ tranches if RSI has risen above abort level.

        If rsi_turn_required, also checks that RSI is declining (current < prior).
        Returns count of aborted tranches.
        """
        if current_rsi <= self.rsi_abort_level:
            return 0  # RSI still in oversold zone — keep tranches alive

        n = 0
        for t in self.tranches:
            if t.tranche_id >= 2 and t.status == "PENDING":
                t.status = "CANCELLED"
                n += 1
        if n > 0:
            logger.info("%s: RSI abort (%.1f > %.1f) — cancelled %d tranche(s)",
                        self.symbol, current_rsi, self.rsi_abort_level, n)
        return n

    def should_fill_tranche(
        self,
        tranche: Tranche,
        current_price: float,
        current_rsi: float | None = None,
        prior_rsi: float | None = None,
    ) -> bool:
        """Check if a pending tranche should be filled given current conditions.

        T1 fills at limit price.
        T2+ also require RSI turn confirmation if configured.
        """
        if tranche.status != "PENDING":
            return False
        if current_price > tranche.limit_price:
            return False  # price hasn't dipped enough

        # T1 always fills at limit
        if tranche.tranche_id == 1:
            return True

        # T2+ require RSI confirmation if enabled
        if self.rsi_turn_required and current_rsi is not None and prior_rsi is not None:
            if current_rsi >= prior_rsi:
                # RSI not declining → don't fill deeper tranches yet
                return False

        return True

    def summary_table(self) -> list[dict]:
        """Return list of dicts suitable for dashboard display."""
        rows = []
        for t in self.tranches:
            rows.append({
                "Tranche":  f"T{t.tranche_id}",
                "Price $":  round(t.limit_price, 2),
                "Qty":      t.quantity,
                "Fraction": f"{t.fraction:.0%}",
                "ATR Off":  f"−{t.atr_offset:.1f}×",
                "Notional $": round(t.notional(), 0),
                "Status":   t.status,
            })
        rows.append({
            "Tranche":  "TOTAL",
            "Price $":  round(self.avg_fill_price, 2) if self.filled_qty > 0 else "—",
            "Qty":      self.total_quantity,
            "Fraction": "100%",
            "ATR Off":  "—",
            "Notional $": round(self.total_committed_notional, 0),
            "Status":   f"{self.filled_qty}/{self.total_quantity} filled",
        })
        return rows

    def __str__(self) -> str:
        lines = [f"{self.symbol} ScaledEntryPlan ({self.n_tranches} tranches):"]
        for t in self.tranches:
            lines.append(f"  {t}")
        lines.append(
            f"  Expires: {self.expiry_date}  RSI abort: >{self.rsi_abort_level}"
            f"  RSI turn: {'yes' if self.rsi_turn_required else 'no'}"
        )
        return "\n".join(lines)


def plan_scaled_entry(
    spec: OrderSpec,
    atr_14: float | None = None,
    current_rsi: float | None = None,
) -> ScaledEntryPlan:
    """Generate a scaled entry (buy ladder) plan from an OrderSpec.

    Parameters
    ----------
    spec : OrderSpec
        Position sizing result from calculate_order().
    atr_14 : float | None
        14-day ATR as a fraction of price (e.g. 0.018).
        If None, falls back to stop_pct as a proxy.
    current_rsi : float | None
        Current RSI-14 (for informational logging only).

    Returns
    -------
    ScaledEntryPlan with N tranches.
    """
    entry = spec.entry_price
    total_qty = spec.quantity

    if not config.SCALED_ENTRY_ENABLED or config.SCALED_ENTRY_N_TRANCHES <= 1:
        # Single tranche = no laddering
        t = Tranche(
            tranche_id=1, limit_price=entry, quantity=total_qty,
            fraction=1.0, atr_offset=0.0, status="PENDING",
        )
        return ScaledEntryPlan(
            symbol=spec.symbol,
            total_quantity=total_qty,
            tranches=[t],
            atr_14_abs=(atr_14 or spec.stop_pct) * entry,
            entry_price=entry,
            stop_loss_price=spec.stop_loss_price,
            take_profit_price=spec.take_profit_price,
            expiry_date=datetime.date.today() + datetime.timedelta(days=config.SCALED_ENTRY_EXPIRY_DAYS),
            rsi_abort_level=config.SCALED_ENTRY_RSI_ABORT_LEVEL,
            rsi_turn_required=config.SCALED_ENTRY_RSI_TURN_REQUIRED,
        )

    # ATR in dollar terms
    atr_abs = (atr_14 if atr_14 and atr_14 > 0 else spec.stop_pct) * entry

    n = min(config.SCALED_ENTRY_N_TRANCHES, len(config.SCALED_ENTRY_FRACTIONS),
            len(config.SCALED_ENTRY_ATR_OFFSETS))

    fractions = config.SCALED_ENTRY_FRACTIONS[:n]
    offsets   = config.SCALED_ENTRY_ATR_OFFSETS[:n]

    # Normalise fractions to sum to 1.0
    frac_sum = sum(fractions)
    if abs(frac_sum - 1.0) > 0.001:
        fractions = tuple(f / frac_sum for f in fractions)

    tranches: list[Tranche] = []
    remaining_qty = total_qty

    for i in range(n):
        limit_price = round(entry - offsets[i] * atr_abs, 2)
        limit_price = max(limit_price, 0.01)  # safety floor

        if i < n - 1:
            qty = max(1, int(total_qty * fractions[i]))
            remaining_qty -= qty
        else:
            # Last tranche gets whatever is left (avoids rounding gaps)
            qty = max(1, remaining_qty)

        tranches.append(Tranche(
            tranche_id=i + 1,
            limit_price=limit_price,
            quantity=qty,
            fraction=fractions[i],
            atr_offset=offsets[i],
            status="PENDING",
        ))

    plan = ScaledEntryPlan(
        symbol=spec.symbol,
        total_quantity=total_qty,
        tranches=tranches,
        atr_14_abs=atr_abs,
        entry_price=entry,
        stop_loss_price=spec.stop_loss_price,
        take_profit_price=spec.take_profit_price,
        expiry_date=datetime.date.today() + datetime.timedelta(days=config.SCALED_ENTRY_EXPIRY_DAYS),
        rsi_abort_level=config.SCALED_ENTRY_RSI_ABORT_LEVEL,
        rsi_turn_required=config.SCALED_ENTRY_RSI_TURN_REQUIRED,
    )

    logger.info("ScaledEntryPlan:\n%s", plan)
    return plan


# ---------------------------------------------------------------------------
# Trailing stop state machine
# ---------------------------------------------------------------------------

@dataclass
class TrailingStopState:
    """Tracks the adaptive trailing stop for a live position.

    Usage:
        ts = TrailingStopState(entry_price=150.0, atr_14_abs=3.0,
                               initial_stop=144.0)
        # On each bar:
        ts.update(current_price=155.0)
        if current_price <= ts.current_stop:
            # exit position
    """
    entry_price:   float
    atr_14_abs:    float     # ATR(14) in dollar terms
    initial_stop:  float     # hard stop from OrderSpec
    current_stop:  float = 0.0
    highest_price: float = 0.0
    stage:         int   = 0   # 0=initial, 1=breakeven, 2=profit-lock, 3=tight
    n_updates:     int   = 0

    def __post_init__(self):
        if self.current_stop == 0.0:
            self.current_stop = self.initial_stop
        if self.highest_price == 0.0:
            self.highest_price = self.entry_price

    def update(self, current_price: float) -> float:
        """Update trailing stop given the current price.

        Returns the new current_stop (only moves UP, never down).
        """
        if not config.TRAILING_STOP_ENABLED:
            return self.current_stop

        self.n_updates += 1
        self.highest_price = max(self.highest_price, current_price)

        atr = self.atr_14_abs
        entry = self.entry_price

        # Stage 3: tight trail (highest_price − 1×ATR)
        if current_price >= entry + config.TRAILING_STAGE3_TRIGGER_ATR * atr:
            new_stop = self.highest_price - config.TRAILING_STAGE3_TRAIL_ATR * atr
            if new_stop > self.current_stop:
                self.current_stop = round(new_stop, 2)
                if self.stage < 3:
                    self.stage = 3
                    logger.debug("Trailing stop → Stage 3 (tight trail) @ %.2f", self.current_stop)

        # Stage 2: profit lock (entry + 1×ATR)
        elif current_price >= entry + config.TRAILING_STAGE2_TRIGGER_ATR * atr:
            new_stop = entry + 1.0 * atr
            if new_stop > self.current_stop:
                self.current_stop = round(new_stop, 2)
                if self.stage < 2:
                    self.stage = 2
                    logger.debug("Trailing stop → Stage 2 (profit lock) @ %.2f", self.current_stop)

        # Stage 1: breakeven lock
        elif current_price >= entry + config.TRAILING_STAGE1_TRIGGER_ATR * atr:
            new_stop = entry
            if new_stop > self.current_stop:
                self.current_stop = round(new_stop, 2)
                if self.stage < 1:
                    self.stage = 1
                    logger.debug("Trailing stop → Stage 1 (breakeven) @ %.2f", self.current_stop)

        return self.current_stop

    @property
    def stage_label(self) -> str:
        labels = {0: "INITIAL", 1: "BREAKEVEN", 2: "PROFIT_LOCK", 3: "TIGHT_TRAIL"}
        return labels.get(self.stage, "UNKNOWN")

    def __str__(self) -> str:
        return (
            f"TrailingStop(stage={self.stage_label}, stop=${self.current_stop:.2f},"
            f" high=${self.highest_price:.2f}, entry=${self.entry_price:.2f},"
            f" ATR=${self.atr_14_abs:.2f})"
        )


# ---------------------------------------------------------------------------
# Partial exit evaluator
# ---------------------------------------------------------------------------

@dataclass
class PartialExitSignal:
    """Describes a partial exit trigger."""
    exit_id:    int      # 1 or 2
    trigger_atr: float   # ATR multiple above entry that triggered
    fraction:   float    # fraction of remaining position to sell
    exit_price: float    # price at which to exit

    def __str__(self) -> str:
        return (
            f"PartialExit #{self.exit_id}: sell {self.fraction:.0%}"
            f" @ ${self.exit_price:.2f} (+{self.trigger_atr:.1f}×ATR)"
        )


def evaluate_partial_exits(
    current_price: float,
    entry_price: float,
    atr_14_abs: float,
    exits_already_taken: set[int] | None = None,
) -> list[PartialExitSignal]:
    """Check which partial exit triggers are hit at the current price.

    Returns a list of PartialExitSignal for each newly triggered exit.
    """
    if not config.PARTIAL_EXIT_ENABLED:
        return []

    exits_taken = exits_already_taken or set()
    signals: list[PartialExitSignal] = []

    # Exit 1: +2×ATR
    trigger_1 = entry_price + config.PARTIAL_EXIT_1_TRIGGER_ATR * atr_14_abs
    if 1 not in exits_taken and current_price >= trigger_1:
        signals.append(PartialExitSignal(
            exit_id=1,
            trigger_atr=config.PARTIAL_EXIT_1_TRIGGER_ATR,
            fraction=config.PARTIAL_EXIT_1_FRACTION,
            exit_price=round(current_price, 2),
        ))

    # Exit 2: +3×ATR
    trigger_2 = entry_price + config.PARTIAL_EXIT_2_TRIGGER_ATR * atr_14_abs
    if 2 not in exits_taken and current_price >= trigger_2:
        signals.append(PartialExitSignal(
            exit_id=2,
            trigger_atr=config.PARTIAL_EXIT_2_TRIGGER_ATR,
            fraction=config.PARTIAL_EXIT_2_FRACTION,
            exit_price=round(current_price, 2),
        ))

    return signals


# ---------------------------------------------------------------------------
# Time stop
# ---------------------------------------------------------------------------

def check_time_stop(
    entry_date: datetime.date,
    today: datetime.date | None = None,
) -> bool:
    """Return True if the holding period exceeds TIME_STOP_DAYS."""
    if not config.TIME_STOP_ENABLED:
        return False
    today = today or datetime.date.today()
    holding_days = (today - entry_date).days
    return holding_days >= config.TIME_STOP_DAYS


# ---------------------------------------------------------------------------
# Internal helpers  (unchanged from original)
# ---------------------------------------------------------------------------

def _compute_stops(
    entry_price: float,
    atr_14: Optional[float] = None,     # ATR as a fraction of price (e.g. 0.018)
    reward_risk_ratio: Optional[float] = None,
) -> tuple[float, float, str, float, float, float]:
    """Return (stop_price, target_price, method, stop_pct, target_pct, rr).

    Prefers ATR-based stops when config.USE_ATR_STOPS=True and atr_14 is
    available.  Falls back to fixed percentages otherwise.
    """
    if config.USE_ATR_STOPS and atr_14 is not None and atr_14 > 0:
        atr_abs = atr_14 * entry_price       # convert fraction → dollars

        raw_stop_pct = config.ATR_STOP_MULTIPLIER * atr_14
        # Clamp to sensible bounds
        stop_pct = max(config.ATR_MIN_STOP_PCT,
                       min(config.ATR_MAX_STOP_PCT, raw_stop_pct))
        # If RR is available from SwingMetrics, honour the ATR target directly
        # but still clamp the stop.
        if reward_risk_ratio is not None and reward_risk_ratio > 0:
            target_pct = stop_pct * reward_risk_ratio
        else:
            target_pct = config.ATR_TARGET_MULTIPLIER * atr_14
            target_pct = max(target_pct, stop_pct * 1.5)  # always at least 1.5:1 RR

        stop_price   = round(entry_price * (1.0 - stop_pct), 2)
        target_price = round(entry_price * (1.0 + target_pct), 2)
        rr = target_pct / stop_pct if stop_pct > 0 else config.TAKE_PROFIT_PCT / config.STOP_LOSS_PCT
        return stop_price, target_price, "ATR", stop_pct, target_pct, rr
    else:
        stop_pct   = config.STOP_LOSS_PCT
        target_pct = config.TAKE_PROFIT_PCT
        stop_price   = round(entry_price * (1.0 - stop_pct), 2)
        target_price = round(entry_price * (1.0 + target_pct), 2)
        rr = target_pct / stop_pct
        return stop_price, target_price, "FIXED", stop_pct, target_pct, rr


def _kelly_qty(
    entry_price: float,
    account_equity: float,
    rr: float,
    win_rate: float = config.KELLY_WIN_RATE,
    kelly_fraction: float = config.KELLY_FRACTION,
) -> int:
    """Return Kelly-optimal share count (fractional Kelly, floor).

    Returns 0 if Kelly has negative edge or is disabled (kelly_fraction=0).
    """
    if kelly_fraction <= 0:
        return 0
    # Full Kelly fraction of equity
    # edge = p - (1-p)/b   where b = reward/risk ratio
    edge = win_rate - (1.0 - win_rate) / rr if rr > 0 else 0.0
    if edge <= 0:
        return 0
    full_kelly = edge / win_rate  # fraction of equity to bet
    notional = kelly_fraction * full_kelly * account_equity
    return int(notional / entry_price) if entry_price > 0 else 0


def _vol_scale(vol_regime: Optional[str]) -> float:
    """Map volatility regime string to position-size multiplier."""
    if vol_regime == "HIGH":
        return config.VOL_SCALE_HIGH
    if vol_regime == "LOW":
        return config.VOL_SCALE_LOW
    return config.VOL_SCALE_NORMAL


# ---------------------------------------------------------------------------
# Single-position calculator
# ---------------------------------------------------------------------------

def calculate_order(
    symbol: str,
    entry_price: float,
    account_equity: float,
    open_positions: int = 0,
    # Optional rich inputs from SwingMetrics
    atr_14: Optional[float] = None,
    reward_risk_ratio: Optional[float] = None,
    vol_regime: Optional[str] = None,
    sector: Optional[str] = None,
    open_sectors: Optional[list[str]] = None,
    remaining_cash: Optional[float] = None,
    # Override config defaults
    stop_loss_pct: float = config.STOP_LOSS_PCT,
    take_profit_pct: float = config.TAKE_PROFIT_PCT,
    risk_per_trade_pct: float = config.RISK_PER_TRADE_PCT,
    max_position_pct: float = config.MAX_POSITION_PCT,
    max_positions: int = config.MAX_POSITIONS,
) -> OrderSpec | None:
    """Calculate position size and bracket-order prices for a single ticker.

    Parameters
    ----------
    symbol:
        Ticker symbol.
    entry_price:
        Expected entry price (e.g. last close or mid-spread).
    account_equity:
        Total account net liquidation value in USD.
    open_positions:
        Number of positions currently held (to check MAX_POSITIONS).
    atr_14:
        14-day ATR expressed as a fraction of current price (from SwingMetrics).
        Used for ATR-based stop placement when config.USE_ATR_STOPS is True.
    reward_risk_ratio:
        ATR target/stop ratio from SwingMetrics (used to set target price).
    vol_regime:
        "HIGH" | "NORMAL" | "LOW" from SwingMetrics.vol_regime.
    sector:
        GICS sector of the ticker (from EDGAR fundamentals).
    open_sectors:
        List of sectors already allocated in the current portfolio batch.
        Used to apply the same-sector concentration penalty.
    remaining_cash:
        Cash not yet committed to other simultaneous positions.
        If None, assumed to be equity × MAX_CAPITAL_DEPLOYED_PCT.

    Returns
    -------
    OrderSpec or None if the trade should be skipped.
    """
    if entry_price <= 0:
        logger.warning("%s: invalid entry price %.4f", symbol, entry_price)
        return None
    if account_equity <= 0:
        logger.warning("Account equity %.2f is non-positive", account_equity)
        return None
    if open_positions >= max_positions:
        logger.info("%s: skipped — max positions reached (%d/%d)", symbol, open_positions, max_positions)
        return None

    # ── 1. Stop / target prices ───────────────────────────────────────────────
    stop_price, target_price, stop_method, stop_pct, target_pct, rr = _compute_stops(
        entry_price, atr_14, reward_risk_ratio
    )

    # ── 2. Fixed-risk quantity ────────────────────────────────────────────────
    risk_per_share = entry_price - stop_price
    max_risk_dollars = account_equity * risk_per_trade_pct
    qty_risk = int(max_risk_dollars / risk_per_share) if risk_per_share > 0 else 0

    # ── 3. Max-position cap ───────────────────────────────────────────────────
    max_position_value = account_equity * max_position_pct
    qty_cap = int(max_position_value / entry_price) if entry_price > 0 else 0

    # ── 4. Kelly quantity ─────────────────────────────────────────────────────
    qty_kelly = _kelly_qty(entry_price, account_equity, rr)

    # ── 5. Remaining-cash cap ─────────────────────────────────────────────────
    if remaining_cash is None:
        remaining_cash = account_equity * config.MAX_CAPITAL_DEPLOYED_PCT
    qty_cash = int(remaining_cash / entry_price) if entry_price > 0 else 0

    # ── 6. Binding constraint ─────────────────────────────────────────────────
    candidates = {"risk": qty_risk, "cap": qty_cap, "cash": qty_cash}
    if qty_kelly > 0:
        candidates["kelly"] = qty_kelly

    quantity = min(candidates.values())
    sizing_method = min(candidates, key=candidates.get)

    # ── 7. Volatility-regime scaling ──────────────────────────────────────────
    vscale = _vol_scale(vol_regime)
    quantity = int(quantity * vscale)

    # ── 8. Sector-correlation penalty ─────────────────────────────────────────
    sscale = 1.0
    if sector and open_sectors and sector in open_sectors:
        sscale = config.CORRELATION_SAME_SECTOR_SCALE
        quantity = int(quantity * sscale)
        logger.debug("%s: same-sector penalty applied (%.0f%%)", symbol, sscale * 100)

    if quantity <= 0:
        logger.info(
            "%s: skipped — position size rounded to 0 "
            "(equity=%.0f, risk_budget=%.0f, risk_per_share=%.2f, cash=%.0f)",
            symbol, account_equity, max_risk_dollars, risk_per_share, remaining_cash,
        )
        return None

    position_value = quantity * entry_price
    risk_amount    = quantity * risk_per_share

    spec = OrderSpec(
        symbol            = symbol,
        quantity          = quantity,
        entry_price       = round(entry_price, 2),
        stop_loss_price   = stop_price,
        take_profit_price = target_price,
        risk_amount       = round(risk_amount, 2),
        position_value    = round(position_value, 2),
        stop_method       = stop_method,
        stop_pct          = stop_pct,
        target_pct        = target_pct,
        reward_risk_ratio = round(rr, 2),
        kelly_qty         = qty_kelly,
        risk_qty          = qty_risk,
        vol_scale         = vscale,
        sector_scale      = sscale,
        sizing_method     = sizing_method,
    )
    logger.info("OrderSpec: %s", spec)
    return spec


# ---------------------------------------------------------------------------
# Portfolio-level simultaneous sizing
# ---------------------------------------------------------------------------

@dataclass
class PortfolioAllocation:
    """Full portfolio allocation result from size_portfolio()."""
    orders:            list[OrderSpec]     # one per accepted candidate
    skipped:           list[dict]          # {symbol, reason}
    total_notional:    float
    total_risk:        float
    cash_remaining:    float
    pct_deployed:      float
    n_positions:       int
    scaled_entries:    dict = field(default_factory=dict)
    # {symbol: ScaledEntryPlan} — populated when SCALED_ENTRY_ENABLED


def size_portfolio(
    candidates: list[dict],
    account_equity: float = config.PORTFOLIO_CAPITAL,
    max_deployed_pct: float = config.MAX_CAPITAL_DEPLOYED_PCT,
) -> PortfolioAllocation:
    """
    Simultaneously size all dip candidates as a single portfolio.

    Candidates are sorted by signal quality (BUY > DIP+FUND > DIP ONLY)
    then by ML probability (descending).  Capital is allocated greedily from
    the highest-conviction trade down, with each commitment reducing the
    remaining cash pool for subsequent trades.

    Parameters
    ----------
    candidates : list of dicts with keys:
        symbol      (str)
        entry_price (float)
        signal      (str)  — "🟢 BUY" | "🟡 DIP+FUND (ML↓)" | "🟠 DIP ONLY"
        atr_14      (float | None)  — from SwingMetrics.atr_14
        reward_risk_ratio (float | None)
        vol_regime  (str | None)
        sector      (str | None)
        ml_prob     (float | None)  — ML probability for sorting

    account_equity : float
        Total account value.
    max_deployed_pct : float
        Maximum fraction of equity to deploy across all positions.
    """
    # Signal priority: BUY=3, DIP+FUND=2, DIP ONLY=1, else 0
    def _priority(c: dict) -> tuple[int, float]:
        sig  = c.get("signal", "")
        prob = c.get("ml_prob") or 0.0
        if "BUY" in sig and "DIP" not in sig:
            return (3, prob)
        if "DIP+FUND" in sig or ("DIP" in sig and "FUND" in sig):
            return (2, prob)
        if "DIP" in sig:
            return (1, prob)
        return (0, prob)

    sorted_candidates = sorted(candidates, key=_priority, reverse=True)

    max_cash = account_equity * max_deployed_pct
    remaining_cash = max_cash
    open_sectors: list[str] = []
    orders:  list[OrderSpec] = []
    skipped: list[dict]      = []

    for c in sorted_candidates:
        sym   = c.get("symbol", "?")
        price = c.get("entry_price")
        if not price or price <= 0:
            skipped.append({"symbol": sym, "reason": "invalid price"})
            continue

        if remaining_cash < price:
            skipped.append({"symbol": sym, "reason": "insufficient cash"})
            continue

        if len(orders) >= config.MAX_POSITIONS:
            skipped.append({"symbol": sym, "reason": "max positions reached"})
            continue

        spec = calculate_order(
            symbol            = sym,
            entry_price       = price,
            account_equity    = account_equity,
            open_positions    = len(orders),
            atr_14            = c.get("atr_14"),
            reward_risk_ratio = c.get("reward_risk_ratio"),
            vol_regime        = c.get("vol_regime"),
            sector            = c.get("sector"),
            open_sectors      = open_sectors,
            remaining_cash    = remaining_cash,
        )

        if spec is None:
            skipped.append({"symbol": sym, "reason": "sized to 0"})
            continue

        remaining_cash -= spec.position_value
        orders.append(spec)
        if c.get("sector"):
            open_sectors.append(c["sector"])

    total_notional = sum(o.position_value for o in orders)
    total_risk     = sum(o.risk_amount    for o in orders)
    pct_deployed   = total_notional / account_equity if account_equity > 0 else 0.0

    # ── Generate scaled entry plans for each accepted order ───────────────────
    scaled_entries: dict[str, ScaledEntryPlan] = {}
    if config.SCALED_ENTRY_ENABLED:
        # Build a quick lookup: symbol → candidate dict (for atr_14)
        cand_map = {c.get("symbol", "?"): c for c in sorted_candidates}
        for spec in orders:
            c = cand_map.get(spec.symbol, {})
            atr_14 = c.get("atr_14")
            plan = plan_scaled_entry(spec, atr_14=atr_14)
            scaled_entries[spec.symbol] = plan

    return PortfolioAllocation(
        orders         = orders,
        skipped        = skipped,
        total_notional = round(total_notional, 2),
        total_risk     = round(total_risk, 2),
        cash_remaining = round(account_equity - total_notional, 2),
        pct_deployed   = round(pct_deployed, 4),
        n_positions    = len(orders),
        scaled_entries = scaled_entries,
    )
