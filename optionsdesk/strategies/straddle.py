"""Straddle strategy math (pure): payoff, breakevens, net Greeks, entry filters, exits.

Position value convention: `value = sum(sign * price)` per share with long = +,
so entry value is +debit for a long straddle and -credit for a short one, and
P&L per share is simply `value_now - value_entry`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal, Optional, Sequence

import numpy as np
from pydantic import BaseModel

from ..events import OptionQuote
from ..settings import StraddleSettings
from .pricing import Greeks, bs_greeks, expected_abs_move

MULT = 100


class PayoffLeg(BaseModel):
    right: Literal["C", "P"]
    strike: float
    qty: int  # signed contracts: + long, - short
    price: float  # per-share entry price
    expiry: date
    iv: Optional[float] = None


def net_value_per_share(legs: Sequence[PayoffLeg]) -> float:
    """Signed value at entry prices: + debit paid, - credit received (per share)."""
    return sum(l.qty * l.price for l in legs)


def payoff_at_expiry(legs: Sequence[PayoffLeg], spots) -> np.ndarray:
    """Total P&L in dollars at expiry for each underlying price in `spots`."""
    s = np.asarray(spots, dtype=float)
    total = np.zeros_like(s)
    for l in legs:
        intrinsic = np.maximum(s - l.strike, 0.0) if l.right == "C" else np.maximum(l.strike - s, 0.0)
        total += l.qty * (intrinsic - l.price) * MULT
    return total


def _kinks(legs: Sequence[PayoffLeg]) -> list[float]:
    return sorted({0.0, *(l.strike for l in legs)})


def breakevens(legs: Sequence[PayoffLeg]) -> list[float]:
    """Underlying prices where expiry P&L crosses zero (payoff is piecewise linear)."""
    pts = _kinks(legs)
    top = pts[-1]
    far = max(top * 4.0, top + 1000.0)
    xs = pts + [far]
    ys = payoff_at_expiry(legs, xs)
    out: list[float] = []
    for i in range(len(xs) - 1):
        x0, x1, y0, y1 = xs[i], xs[i + 1], ys[i], ys[i + 1]
        if y0 == 0 and y1 == 0:
            continue
        if y0 == 0:
            out.append(x0)
        elif y0 * y1 < 0:
            out.append(x0 + (x1 - x0) * (-y0) / (y1 - y0))
    if ys[-1] == 0:
        out.append(xs[-1])
    return sorted({round(x, 6) for x in out if x > 0})


def max_profit_loss(legs: Sequence[PayoffLeg]) -> tuple[Optional[float], Optional[float]]:
    """(max_profit, max_loss) in dollars; max_loss is a positive number. None = unbounded."""
    pts = _kinks(legs)
    vals = payoff_at_expiry(legs, pts)
    up_slope = sum(l.qty for l in legs if l.right == "C") * MULT  # P&L slope as S -> infinity
    best, worst = float(vals.max()), float(vals.min())
    max_profit = None if up_slope > 0 else best
    max_loss = None if up_slope < 0 else -worst
    if max_loss is not None:
        max_loss = max(max_loss, 0.0)
    return max_profit, max_loss


def net_greeks(legs: Sequence[PayoffLeg], spot: float, asof: date, r: float = 0.045,
               iv_shift: float = 0.0) -> Greeks:
    """Position Greeks in share-equivalents / dollars (theta $ per day, vega $ per vol point)."""
    d = g = th = v = p = 0.0
    for l in legs:
        T = max((l.expiry - asof).days, 0) / 365.0
        sigma = max((l.iv or 0.3) + iv_shift, 1e-4)
        gr = bs_greeks(spot, l.strike, T, r, sigma, l.right)
        k = l.qty * MULT
        d, g, th, v, p = d + k * gr.delta, g + k * gr.gamma, th + k * gr.theta, v + k * gr.vega, p + k * gr.price
    return Greeks(price=p, delta=d, gamma=g, theta=th, vega=v)


def position_pnl(legs: Sequence[PayoffLeg], spot: float, asof: date, r: float = 0.045,
                 iv_shift: float = 0.0) -> float:
    """Mark-to-model P&L in dollars at `asof` (before expiry) for a given spot and IV shift."""
    value_now = net_greeks(legs, spot, asof, r, iv_shift).price  # sum qty*MULT*price
    return value_now - net_value_per_share(legs) * MULT


def payoff_curve(legs: Sequence[PayoffLeg], spot: float, span_pct: float = 0.30, n: int = 121):
    xs = np.linspace(spot * (1 - span_pct), spot * (1 + span_pct), n)
    return xs, payoff_at_expiry(legs, xs)


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------

def _nearest(values: Sequence[float], target: float) -> float:
    return min(values, key=lambda v: abs(v - target))


def choose_expiry(expiries: Sequence[date], asof: date, dte_range: tuple[int, int],
                  event_date: Optional[date] = None) -> Optional[date]:
    """First expiry after `event_date` within the DTE window, else closest to window middle."""
    ok = sorted(e for e in expiries if dte_range[0] <= (e - asof).days <= dte_range[1])
    if not ok:
        return None
    if event_date is not None:
        after = [e for e in ok if e > event_date]
        if after:
            return after[0]
    mid = sum(dte_range) / 2.0
    return min(ok, key=lambda e: abs((e - asof).days - mid))


def build_legs(
    quotes: Sequence[OptionQuote],
    spot: float,
    asof: date,
    mode: Literal["long", "short"],
    s: StraddleSettings,
    event_date: Optional[date] = None,
    contracts: int = 1,
) -> Optional[list[PayoffLeg]]:
    """ATM straddle at mid prices. With wings: short mode buys OTM wings (iron butterfly),
    long mode sells OTM wings (reverse iron butterfly) - both cap the risk/reward."""
    expiries = {q.expiry for q in quotes}
    expiry = choose_expiry(list(expiries), asof, (s.dte_min, s.dte_max), event_date)
    if expiry is None:
        return None
    chain = [q for q in quotes if q.expiry == expiry and q.mid is not None]
    by_key = {(q.right, q.strike): q for q in chain}
    strikes = sorted({k for (_, k) in by_key})
    both = [k for k in strikes if ("C", k) in by_key and ("P", k) in by_key]
    if not both:
        return None
    atm = _nearest(both, spot)
    sign = 1 if mode == "long" else -1
    legs = [
        PayoffLeg(right="C", strike=atm, qty=sign * contracts, price=by_key[("C", atm)].mid, expiry=expiry,
                  iv=by_key[("C", atm)].iv),
        PayoffLeg(right="P", strike=atm, qty=sign * contracts, price=by_key[("P", atm)].mid, expiry=expiry,
                  iv=by_key[("P", atm)].iv),
    ]
    if s.use_wings:
        calls_otm = [k for (r_, k) in by_key if r_ == "C" and k > atm]
        puts_otm = [k for (r_, k) in by_key if r_ == "P" and k < atm]
        if not calls_otm or not puts_otm:
            return None
        ck = _nearest(calls_otm, atm * (1 + s.wing_width_pct))
        pk = _nearest(puts_otm, atm * (1 - s.wing_width_pct))
        wing_sign = -sign
        legs += [
            PayoffLeg(right="C", strike=ck, qty=wing_sign * contracts, price=by_key[("C", ck)].mid, expiry=expiry,
                      iv=by_key[("C", ck)].iv),
            PayoffLeg(right="P", strike=pk, qty=wing_sign * contracts, price=by_key[("P", pk)].mid, expiry=expiry,
                      iv=by_key[("P", pk)].iv),
        ]
    return legs


# --------------------------------------------------------------------------
# Entry / exit rules
# --------------------------------------------------------------------------

@dataclass
class EntryDecision:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


def evaluate_entry(
    mode: Literal["long", "short"],
    s: StraddleSettings,
    iv_rank: Optional[float],
    iv_over_rv: Optional[float],
    days_to_event: Optional[int],
    days_to_expiry: int,
    implied_move: Optional[float],
    realized_move: Optional[float],
) -> EntryDecision:
    """`implied_move` is the ATM straddle price; `realized_move` the move expected from realized vol."""
    d = EntryDecision(ok=True)

    def fail(msg: str) -> None:
        d.ok = False
        d.reasons.append(msg)

    if iv_rank is None:
        d.flags.append("IV rank unavailable (insufficient history)")
    elif mode == "long" and iv_rank > s.long_max_iv_rank:
        fail(f"IV rank {iv_rank:.0f} > {s.long_max_iv_rank:.0f} (long wants cheap vol)")
    elif mode == "short" and iv_rank < s.short_min_iv_rank:
        fail(f"IV rank {iv_rank:.0f} < {s.short_min_iv_rank:.0f} (short wants rich vol)")

    if iv_over_rv is None:
        d.flags.append("realized vol unavailable")
    elif mode == "long" and iv_over_rv > s.max_iv_over_rv:
        fail(f"IV/RV {iv_over_rv:.2f} > {s.max_iv_over_rv:.2f}")
    elif mode == "short" and iv_over_rv < s.min_iv_over_rv:
        fail(f"IV/RV {iv_over_rv:.2f} < {s.min_iv_over_rv:.2f}")

    if mode == "long":
        if days_to_event is None or not s.event_window_days_min <= days_to_event <= s.event_window_days_max:
            if s.long_requires_event:
                fail("no event within window" if days_to_event is None else f"event in {days_to_event}d outside window")
        elif days_to_event > days_to_expiry:
            fail("event is after expiry")
    elif days_to_event is not None and 0 <= days_to_event <= days_to_expiry:
        fail(f"event in {days_to_event}d falls before expiry (short vol)")

    if implied_move is not None and realized_move:
        edge = (realized_move - implied_move) / realized_move if mode == "long" else (implied_move - realized_move) / realized_move
        if edge < s.min_edge_vs_expected_move:
            fail(f"implied move {implied_move:.2f} vs realized-vol move {realized_move:.2f}: no edge")
    return d


def exit_decision(entry_value: float, current_value: float, dte: int, s: StraddleSettings) -> Optional[str]:
    """Reason string if the position should be closed, else None. Values are signed per share."""
    basis = abs(entry_value)
    if basis <= 0:
        return None
    pnl = current_value - entry_value
    if pnl >= s.profit_target_pct * basis:
        return f"profit target: {pnl / basis:.0%} of premium (target {s.profit_target_pct:.0%})"
    if pnl <= -s.stop_loss_pct * basis:
        return f"stop loss: {pnl / basis:.0%} of premium (limit -{s.stop_loss_pct:.0%})"
    if dte <= s.exit_dte:
        return f"time exit: {dte} DTE <= {s.exit_dte}"
    return None
