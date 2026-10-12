"""Wheel strategy: pure state machine + strike selection + management rules.

States:  SEARCHING -> CSP_OPEN -> ASSIGNED -> CC_OPEN -> CALLED_AWAY -> SEARCHING
A CSP that expires/closes without assignment returns straight to SEARCHING.
No I/O here, so live agents and the backtester share identical logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Literal, Optional, Sequence

from pydantic import BaseModel, Field

from ..events import DataQuality, OptionQuote
from ..settings import WheelSettings
from .pricing import bs_greeks, implied_vol

MULT = 100


class WheelState(str, Enum):
    SEARCHING = "SEARCHING"
    CSP_OPEN = "CSP_OPEN"
    ASSIGNED = "ASSIGNED"
    CC_OPEN = "CC_OPEN"
    CALLED_AWAY = "CALLED_AWAY"


class InvalidTransition(Exception):
    pass


class CostBasisViolation(InvalidTransition):
    pass


class OpenLeg(BaseModel):
    right: Literal["C", "P"]
    strike: float
    expiry: date
    premium: float  # per-share credit received when this leg was opened/rolled
    contracts: int
    opened_on: date


class WheelPosition(BaseModel):
    symbol: str
    state: WheelState = WheelState.SEARCHING
    allow_cc_below_basis: bool = False
    shares: int = 0
    assignment_price: Optional[float] = None
    leg: Optional[OpenLeg] = None
    cycle_premium: float = 0.0  # net option premium $ this cycle (credits - debits)
    lifetime_premium: float = 0.0
    realized_pnl: float = 0.0
    cycle_start: Optional[date] = None
    cycle_capital: float = 0.0
    history: list[dict] = Field(default_factory=list)

    # ---- derived --------------------------------------------------------
    @property
    def cost_basis(self) -> Optional[float]:
        """Per-share cost basis net of premium collected this cycle."""
        if self.shares <= 0 or self.assignment_price is None:
            return None
        return self.assignment_price - self.cycle_premium / self.shares

    @property
    def cycles(self) -> int:
        return len(self.history)

    def cycle_annualized_return(self, asof: date) -> Optional[float]:
        if not self.cycle_start or self.cycle_capital <= 0:
            return None
        days = max((asof - self.cycle_start).days, 1)
        return self.cycle_premium / self.cycle_capital * 365.0 / days

    def total_annualized_return(self) -> Optional[float]:
        """Realized P&L per capital-year across completed cycles."""
        capital_days = sum(c["capital"] * max(c["days"], 1) for c in self.history)
        if capital_days <= 0:
            return None
        return sum(c["pnl"] for c in self.history) / (capital_days / 365.0)

    # ---- helpers --------------------------------------------------------
    def _require(self, *states: WheelState) -> None:
        if self.state not in states:
            raise InvalidTransition(f"{self.symbol}: need {[s.value for s in states]}, am {self.state.value}")

    def _add_premium(self, dollars: float) -> None:
        self.cycle_premium += dollars
        self.lifetime_premium += dollars

    def _finish_cycle(self, asof: date, pnl: float, assigned: bool) -> dict:
        start = self.cycle_start or asof
        rec = {
            "start": start.isoformat(), "end": asof.isoformat(), "days": (asof - start).days,
            "capital": self.cycle_capital, "pnl": pnl, "assigned": assigned,
            "premium": self.cycle_premium,
            "return_on_capital": pnl / self.cycle_capital if self.cycle_capital else 0.0,
        }
        days = max(rec["days"], 1)
        rec["annualized_return"] = rec["return_on_capital"] * 365.0 / days
        self.history.append(rec)
        self.realized_pnl += pnl
        return rec

    def _check_floor(self, strike: float) -> None:
        basis = self.cost_basis
        if basis is not None and strike < basis and not self.allow_cc_below_basis:
            raise CostBasisViolation(
                f"{self.symbol}: call strike {strike} below cost basis {basis:.2f} (allow_cc_below_cost_basis=false)"
            )

    # ---- transitions ----------------------------------------------------
    def open_csp(self, strike: float, expiry: date, premium: float, contracts: int, asof: date) -> None:
        self._require(WheelState.SEARCHING)
        self.cycle_premium = 0.0
        self.cycle_start = asof
        self.cycle_capital = strike * MULT * contracts
        self.leg = OpenLeg(right="P", strike=strike, expiry=expiry, premium=premium, contracts=contracts, opened_on=asof)
        self._add_premium(premium * MULT * contracts)
        self.state = WheelState.CSP_OPEN

    def roll(self, new_strike: float, new_expiry: date, new_premium: float, close_cost: float, asof: date) -> None:
        """Buy back the current leg at `close_cost` and sell a new one at `new_premium` (per share)."""
        self._require(WheelState.CSP_OPEN, WheelState.CC_OPEN)
        assert self.leg is not None
        if self.state is WheelState.CC_OPEN:
            self._check_floor(new_strike)
        n = self.leg.contracts
        self._add_premium((new_premium - close_cost) * MULT * n)
        if self.state is WheelState.CSP_OPEN:
            self.cycle_capital = max(self.cycle_capital, new_strike * MULT * n)
        self.leg = OpenLeg(right=self.leg.right, strike=new_strike, expiry=new_expiry,
                           premium=new_premium, contracts=n, opened_on=asof)

    def close_leg(self, close_price: float, asof: date) -> Optional[dict]:
        """Buy to close. CSP: cycle ends. CC: shares kept, back to ASSIGNED."""
        self._require(WheelState.CSP_OPEN, WheelState.CC_OPEN)
        assert self.leg is not None
        self._add_premium(-close_price * MULT * self.leg.contracts)
        if self.state is WheelState.CSP_OPEN:
            rec = self._finish_cycle(asof, self.cycle_premium, assigned=False)
            self.leg, self.state, self.cycle_start = None, WheelState.SEARCHING, None
            return rec
        self.leg, self.state = None, WheelState.ASSIGNED
        return None

    def expire_leg(self, asof: date) -> Optional[dict]:
        return self.close_leg(0.0, asof)

    def assign_put(self, asof: date) -> None:
        self._require(WheelState.CSP_OPEN)
        assert self.leg is not None and self.leg.right == "P"
        self.shares = self.leg.contracts * MULT
        self.assignment_price = self.leg.strike
        self.leg = None
        self.state = WheelState.ASSIGNED

    def open_cc(self, strike: float, expiry: date, premium: float, asof: date, contracts: Optional[int] = None) -> None:
        self._require(WheelState.ASSIGNED)
        self._check_floor(strike)
        n = contracts or self.shares // MULT
        if n * MULT > self.shares or n <= 0:
            raise InvalidTransition(f"{self.symbol}: cannot cover {n} contracts with {self.shares} shares")
        self.leg = OpenLeg(right="C", strike=strike, expiry=expiry, premium=premium, contracts=n, opened_on=asof)
        self._add_premium(premium * MULT * n)
        self.state = WheelState.CC_OPEN

    def call_away(self, asof: date) -> dict:
        self._require(WheelState.CC_OPEN)
        assert self.leg is not None and self.assignment_price is not None
        stock_pnl = (self.leg.strike - self.assignment_price) * self.leg.contracts * MULT
        rec = self._finish_cycle(asof, self.cycle_premium + stock_pnl, assigned=True)
        self.shares -= self.leg.contracts * MULT
        self.leg = None
        self.state = WheelState.CALLED_AWAY
        return rec

    def restart(self) -> None:
        self._require(WheelState.CALLED_AWAY)
        self.assignment_price, self.cycle_start, self.cycle_premium, self.cycle_capital = None, None, 0.0, 0.0
        self.state = WheelState.SEARCHING


# --------------------------------------------------------------------------
# Selection and management (pure)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Candidate:
    quote: OptionQuote
    abs_delta: float
    delta_estimated: bool
    dte: int
    mid: float
    annualized_roc: float
    spread_pct: float


def quote_abs_delta(q: OptionQuote, spot: float, asof: date, r: float = 0.045) -> Optional[tuple[float, bool]]:
    """Broker delta if present, else Black-Scholes from broker IV or IV implied by mid."""
    if q.delta is not None:
        return abs(q.delta), q.quality is DataQuality.ESTIMATED
    T = (q.expiry - asof).days / 365.0
    sigma = q.iv
    if sigma is None and q.mid:
        sigma = implied_vol(q.mid, spot, q.strike, T, r, q.right)
    if not sigma or T <= 0:
        return None
    return abs(bs_greeks(spot, q.strike, T, r, sigma, q.right).delta), True


def select_candidate(
    quotes: Sequence[OptionQuote],
    right: Literal["C", "P"],
    spot: float,
    asof: date,
    delta_range: tuple[float, float],
    dte_range: tuple[int, int],
    min_open_interest: int,
    max_spread_pct: float,
    min_strike: Optional[float] = None,
    r: float = 0.045,
) -> Optional[Candidate]:
    """OTM contract whose |delta| is closest to the middle of `delta_range`, after liquidity filters."""
    target = sum(delta_range) / 2.0
    best: Optional[Candidate] = None
    best_key: Optional[tuple[float, float]] = None
    for q in quotes:
        if q.right != right:
            continue
        dte = (q.expiry - asof).days
        if not dte_range[0] <= dte <= dte_range[1]:
            continue
        if (right == "P" and q.strike >= spot) or (right == "C" and q.strike <= spot):
            continue
        if min_strike is not None and q.strike < min_strike:
            continue
        mid, spread = q.mid, q.spread_pct
        if mid is None or spread is None or spread > max_spread_pct:
            continue
        if q.open_interest is None or q.open_interest < min_open_interest:
            continue
        d = quote_abs_delta(q, spot, asof, r)
        if d is None or not delta_range[0] <= d[0] <= delta_range[1]:
            continue
        roc = mid / q.strike * 365.0 / dte
        key = (abs(d[0] - target), -roc)
        if best_key is None or key < best_key:
            best, best_key = Candidate(q, d[0], d[1], dte, mid, roc, spread), key
    return best


def entry_ok(iv_rank: Optional[float], s: WheelSettings) -> tuple[bool, str]:
    if iv_rank is None:
        if s.require_iv_rank:
            return False, "IV rank unavailable (insufficient history)"
        return True, "IV rank unavailable (insufficient history) - entry allowed, flagged"
    if iv_rank < s.min_iv_rank:
        return False, f"IV rank {iv_rank:.0f} < {s.min_iv_rank:.0f}"
    return True, f"IV rank {iv_rank:.0f} >= {s.min_iv_rank:.0f}"


@dataclass(frozen=True)
class ManagementAction:
    kind: Literal["close_profit", "roll_dte", "roll_tested"]
    reason: str


def management_actions(pos: WheelPosition, spot: float, asof: date, current_mid: Optional[float],
                       s: WheelSettings) -> list[ManagementAction]:
    """Ordered by priority: take profit, then roll for time, then roll if tested."""
    leg = pos.leg
    if leg is None:
        return []
    out: list[ManagementAction] = []
    if current_mid is not None and leg.premium > 0:
        captured = 1.0 - current_mid / leg.premium
        if captured >= s.profit_take_pct:
            out.append(ManagementAction("close_profit", f"{captured:.0%} of premium captured (target {s.profit_take_pct:.0%})"))
    dte = (leg.expiry - asof).days
    if 0 < dte <= s.roll_dte:
        out.append(ManagementAction("roll_dte", f"{dte} DTE <= roll threshold {s.roll_dte}"))
    tested = spot <= leg.strike * (1 + s.tested_buffer_pct) if leg.right == "P" else spot >= leg.strike * (1 - s.tested_buffer_pct)
    if tested:
        out.append(ManagementAction("roll_tested", f"spot {spot:.2f} is testing {leg.right} strike {leg.strike:g}"))
    return out
