"""Portfolio risk: capital at risk, concentration, net Greeks, scenario grid, limit checks.

Pure functions over `RiskPosition`s so the RiskAgent, the API and tests share one
implementation. A signal's legs are *merged* (netted) into the book to evaluate the
post-trade portfolio, so closes and rolls are handled by the same code as opens.
Limits that a trade does not worsen are flagged, never blocked, so risk-reducing
trades always go through.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Optional, Sequence

from pydantic import BaseModel, Field

from .events import Signal, Verdict
from .settings import RiskSettings
from .strategies.pricing import bs_greeks
from .strategies.straddle import MULT, PayoffLeg, max_profit_loss, net_greeks

R = 0.045


class RiskPosition(BaseModel):
    symbol: str
    shares: int = 0
    legs: list[PayoffLeg] = Field(default_factory=list)


@dataclass
class RiskMetrics:
    capital_by_ticker: dict[str, float] = field(default_factory=dict)
    total_capital: float = 0.0
    concentration: dict[str, float] = field(default_factory=dict)  # fraction of max_capital_at_risk
    net_delta: float = 0.0  # share-equivalents
    net_vega: float = 0.0  # $ per vol point
    net_theta: float = 0.0  # $ per day
    scenario: dict[str, dict[float, float]] = field(default_factory=dict)  # ticker|TOTAL -> move -> P&L
    worst_scenario_pnl: float = 0.0
    heatmap: dict[str, dict[str, float]] = field(default_factory=dict)  # ticker -> expiry|STK -> delta-$
    unbounded: list[str] = field(default_factory=list)


def _value(legs: Sequence[PayoffLeg], spot: float, asof: date) -> float:
    return net_greeks(legs, spot, asof, R).price if legs else 0.0


def _uncovered(pos: RiskPosition) -> list[PayoffLeg]:
    """Drop short calls covered by shares (their risk is the stock's)."""
    cover = pos.shares // MULT
    out: list[PayoffLeg] = []
    for l in pos.legs:
        if l.right == "C" and l.qty < 0 and cover > 0:
            covered = min(cover, -l.qty)
            cover -= covered
            if l.qty + covered == 0:
                continue
            l = l.model_copy(update={"qty": l.qty + covered})
        out.append(l)
    return out


def merge_signal(positions: Sequence[RiskPosition], signal: Signal) -> list[RiskPosition]:
    """Net the signal's legs into a copy of the book (same contract -> quantities add)."""
    book = {p.symbol: p.model_copy(deep=True) for p in positions}
    pos = book.setdefault(signal.symbol, RiskPosition(symbol=signal.symbol))
    for sl in signal.legs:
        if sl.right == "STK":
            pos.shares += sl.quantity
            continue
        for existing in pos.legs:
            if existing.right == sl.right and existing.strike == sl.strike and existing.expiry == sl.expiry:
                existing.qty += sl.quantity
                break
        else:
            pos.legs.append(PayoffLeg(right=sl.right, strike=sl.strike, qty=sl.quantity, price=sl.price or 0.0,
                                      expiry=sl.expiry, iv=sl.iv))
        pos.legs = [l for l in pos.legs if l.qty != 0]
    return [p for p in book.values() if p.shares or p.legs]


def compute_metrics(positions: Sequence[RiskPosition], spots: dict[str, float], asof: date,
                    s: RiskSettings) -> RiskMetrics:
    m = RiskMetrics()
    moves = sorted(set(s.scenario_moves) | {0.0})
    widest = max(abs(x) for x in moves) if moves else 0.2
    total_by_move: dict[float, float] = defaultdict(float)

    for pos in positions:
        spot = spots[pos.symbol]
        legs = _uncovered(pos)

        scen: dict[float, float] = {}
        base_all = _value(pos.legs, spot, asof)
        for mv in moves:
            sp = spot * (1 + mv)
            scen[mv] = _value(pos.legs, sp, asof) - base_all + pos.shares * (sp - spot)
            total_by_move[mv] += scen[mv]
        m.scenario[pos.symbol] = scen

        cap = max(0.0, pos.shares * spot)
        if legs:
            _, max_loss = max_profit_loss(legs)
            if max_loss is None:
                m.unbounded.append(pos.symbol)
                base = _value(legs, spot, asof)
                tails = [_value(legs, spot * (1 + sg * widest), asof) - base for sg in (-1, 1)]
                max_loss = max(0.0, -min(tails))
            cap += max_loss
        m.capital_by_ticker[pos.symbol] = cap

        g = net_greeks(pos.legs, spot, asof, R)
        m.net_delta += g.delta + pos.shares
        m.net_vega += g.vega
        m.net_theta += g.theta

        row: dict[str, float] = {}
        if pos.shares:
            row["STK"] = pos.shares * spot
        for l in pos.legs:
            T = max((l.expiry - asof).days, 0) / 365.0
            d = bs_greeks(spot, l.strike, T, R, l.iv or 0.3, l.right).delta
            row[l.expiry.isoformat()] = row.get(l.expiry.isoformat(), 0.0) + l.qty * MULT * d * spot
        m.heatmap[pos.symbol] = row

    m.total_capital = sum(m.capital_by_ticker.values())
    cap_limit = s.max_capital_at_risk or 1.0
    m.concentration = {k: v / cap_limit for k, v in m.capital_by_ticker.items()}
    m.scenario["TOTAL"] = dict(total_by_move)
    m.worst_scenario_pnl = min(total_by_move.values()) if total_by_move else 0.0
    return m


def utilization(m: RiskMetrics, s: RiskSettings) -> dict[str, float]:
    """Fraction of each limit used (1.0 = at limit)."""
    cap = s.max_capital_at_risk or 1.0
    return {
        "capital": m.total_capital / cap,
        "concentration": (max(m.concentration.values()) if m.concentration else 0.0) / s.max_ticker_concentration,
        "net_delta": abs(m.net_delta) / s.max_net_delta,
        "net_vega": abs(m.net_vega) / s.max_net_vega,
        "net_theta": abs(m.net_theta) / s.max_net_theta_abs,
        "scenario_loss": max(0.0, -m.worst_scenario_pnl) / (cap * s.max_scenario_loss_pct),
    }


_LABELS = {
    "capital": "total capital at risk", "concentration": "single-ticker concentration",
    "net_delta": "net delta", "net_vega": "net vega", "net_theta": "net theta",
    "scenario_loss": "worst scenario loss",
}


@dataclass
class CheckResult:
    verdict: Verdict
    reasons: list[str]
    utilization: dict[str, float]


def check_signal(signal: Signal, positions: Sequence[RiskPosition], spots: dict[str, float], asof: date,
                 s: RiskSettings) -> CheckResult:
    if signal.symbol not in spots or not spots[signal.symbol] or spots[signal.symbol] <= 0:
        return CheckResult(Verdict.BLOCK, [f"no price for {signal.symbol}"], {})
    missing = [p.symbol for p in positions if p.symbol not in spots]
    if missing:
        return CheckResult(Verdict.BLOCK, [f"no price for open position(s): {', '.join(missing)}"], {})

    pre = utilization(compute_metrics(positions, spots, asof, s), s)
    post_metrics = compute_metrics(merge_signal(positions, signal), spots, asof, s)
    post = utilization(post_metrics, s)

    blocks: list[str] = []
    flags: list[str] = []
    for k, u in post.items():
        if u > 1.0 and u > pre.get(k, 0.0) + 1e-9:
            blocks.append(f"{_LABELS[k]} {u:.0%} of limit")
        elif u > 1.0:
            flags.append(f"{_LABELS[k]} still over limit ({u:.0%}) - trade does not worsen it")
        elif u >= s.warn_utilization and u > pre.get(k, 0.0) + 1e-9:
            flags.append(f"{_LABELS[k]} at {u:.0%} of limit")
    if signal.symbol in post_metrics.unbounded:
        flags.append(f"{signal.symbol}: unbounded loss position (uncovered short call/straddle)")

    if blocks:
        return CheckResult(Verdict.BLOCK, blocks + flags, post)
    return CheckResult(Verdict.FLAG if flags else Verdict.APPROVE, flags, post)
