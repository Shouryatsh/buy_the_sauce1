"""Historical simulation of the wheel and straddle strategies.

IMPORTANT: no historical option chains are available, so option prices are MODELLED
(Black-Scholes on an IV proxy = trailing realized vol x iv_premium, with an uplift for
expiries that span a known event). Results test strategy mechanics and risk shape, not
real-market edge. The same pure strategy code used live (selection, state machine,
entry/exit rules) drives the simulation.

No lookahead: every decision on bar i uses closes[:i+1] only (vol proxies are backward
rolling), and resulting orders are filled on bar i+1 at that bar's modelled prices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from .events import DataQuality, OptionQuote
from .settings import Settings
from .strategies.pricing import bs_greeks, bs_price, expected_abs_move, iv_rank
from .strategies.straddle import (
    PayoffLeg, build_legs, evaluate_entry, exit_decision, net_value_per_share,
)
from .strategies.wheel import (
    InvalidTransition, WheelPosition, WheelState, entry_ok, management_actions, select_candidate,
)

MULT = 100
SYNTHETIC_OI = 10_000


@dataclass
class BacktestResult:
    strategy: str
    dates: list[date] = field(default_factory=list)
    equity: list[float] = field(default_factory=list)
    drawdown: list[float] = field(default_factory=list)  # fraction below running peak (<= 0)
    trades: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"strategy": self.strategy, "metrics": self.metrics, "trades": self.trades,
                "equity": [[d.isoformat(), e] for d, e in zip(self.dates, self.equity)],
                "drawdown": [[d.isoformat(), x] for d, x in zip(self.dates, self.drawdown)]}


# ---------------------------------------------------------------------------
# Modelled market
# ---------------------------------------------------------------------------

def _step(spot: float) -> float:
    return 0.5 if spot < 25 else 1.0 if spot < 100 else 2.5 if spot < 250 else 5.0


def _fridays(asof: date, dmin: int, dmax: int) -> list[date]:
    return [asof + timedelta(d) for d in range(dmin, dmax + 1) if (asof + timedelta(d)).weekday() == 4]


def _sigma_for(sigma: float, asof: date, expiry: date, events: Sequence[date], uplift: float) -> float:
    return sigma * uplift if any(asof < e <= expiry for e in events) else sigma


def synthetic_chain(symbol: str, spot: float, asof: date, sigma: float, expiries: Sequence[date],
                    rights: str, band: tuple[float, float], r: float, events: Sequence[date],
                    uplift: float) -> list[OptionQuote]:
    step = _step(spot)
    lo, hi = math.floor(spot * band[0] / step) * step, math.ceil(spot * band[1] / step) * step
    strikes = np.arange(lo, hi + step / 2, step)
    out: list[OptionQuote] = []
    for exp in expiries:
        T = (exp - asof).days / 365.0
        sig = _sigma_for(sigma, asof, exp, events, uplift)
        for k in strikes:
            for right in rights:
                g = bs_greeks(spot, float(k), T, r, sig, right)  # type: ignore[arg-type]
                if g.price < 0.01:
                    continue
                half = max(0.025, 0.02 * g.price)
                out.append(OptionQuote(
                    symbol=symbol, expiry=exp, strike=float(k), right=right,  # type: ignore[arg-type]
                    bid=max(g.price - half, 0.0), ask=g.price + half, iv=sig, delta=g.delta,
                    open_interest=SYNTHETIC_OI, quality=DataQuality.ESTIMATED))
    return out


def _mid(spot: float, strike: float, right: str, asof: date, expiry: date, sigma: float, r: float,
         events: Sequence[date], uplift: float) -> float:
    T = max((expiry - asof).days, 0) / 365.0
    return bs_price(spot, strike, T, r, _sigma_for(sigma, asof, expiry, events, uplift), right)  # type: ignore[arg-type]


def _fill(mid: float, side: str, slip: float) -> float:
    half = max(0.025, 0.02 * mid)
    return max(mid - slip * half, 0.01) if side == "sell" else mid + slip * half


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(dates: Sequence[date], equity: Sequence[float], trades: Sequence[dict], initial: float,
                    rf: float, deployed: Optional[Sequence[float]] = None) -> dict:
    eq = np.asarray(equity, dtype=float)
    if len(eq) < 2:
        return {"trades": len(trades)}
    rets = eq[1:] / eq[:-1] - 1.0
    daily_rf = rf / 252.0
    excess = rets - daily_rf
    sd = rets.std(ddof=1) if len(rets) > 1 else 0.0
    downside = np.minimum(excess, 0.0)
    dd_dev = math.sqrt(float((downside**2).mean()))
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    days = max((dates[-1] - dates[0]).days, 1)
    total_ret = eq[-1] / eq[0] - 1.0
    cagr = (eq[-1] / eq[0]) ** (365.0 / days) - 1.0 if eq[-1] > 0 else -1.0
    pnls = [t["pnl"] for t in trades]
    m = {
        "initial_capital": initial, "final_equity": float(eq[-1]), "total_return": float(total_ret),
        "annualized_return": float(cagr), "max_drawdown": float(dd.min()),
        "sharpe": float(excess.mean() / sd * math.sqrt(252)) if sd > 1e-12 else None,
        "sortino": float(excess.mean() / dd_dev * math.sqrt(252)) if dd_dev > 1e-12 else None,
        "trades": len(trades), "win_rate": (sum(p > 0 for p in pnls) / len(pnls)) if pnls else None,
        "avg_trade_pnl": float(np.mean(pnls)) if pnls else None, "total_trade_pnl": float(sum(pnls)),
        "days": days,
    }
    if deployed is not None and len(deployed) and float(np.mean(deployed)) > 0:
        m["avg_capital_deployed"] = float(np.mean(deployed))
        m["annualized_return_on_deployed"] = float((eq[-1] - eq[0]) / np.mean(deployed) * 365.0 / days)
    return m


def _drawdown(equity: Sequence[float]) -> list[float]:
    eq = np.asarray(equity, dtype=float)
    return (eq / np.maximum.accumulate(eq) - 1.0).tolist()


# ---------------------------------------------------------------------------
# Shared prep
# ---------------------------------------------------------------------------

@dataclass
class _Series:
    symbol: str
    closes: list[float]
    iv: list[float]
    rv: list[float]


def _prepare(prices: dict[str, pd.Series], s: Settings) -> tuple[list[date], dict[str, _Series]]:
    df = pd.DataFrame(prices).dropna()
    dates = [d.date() if hasattr(d, "date") else d for d in df.index]
    out: dict[str, _Series] = {}
    for sym in df.columns:
        lr = np.log(df[sym]).diff()
        b = s.backtest
        iv = (lr.rolling(b.iv_window).std() * math.sqrt(252) * b.iv_premium).tolist()
        rv = (lr.rolling(b.rv_window).std() * math.sqrt(252)).tolist()
        out[sym] = _Series(sym, df[sym].tolist(), iv, rv)
    return dates, out


def _iv_rank_at(ser: _Series, i: int) -> Optional[float]:
    hist = [v for v in ser.iv[max(0, i - 251): i + 1] if not math.isnan(v)]
    if not hist or math.isnan(ser.iv[i]):
        return None
    return iv_rank(ser.iv[i], hist)


# ---------------------------------------------------------------------------
# Wheel
# ---------------------------------------------------------------------------

def run_wheel_backtest(prices: dict[str, pd.Series], start: date, end: date, settings: Settings,
                       events: Optional[dict[str, list[date]]] = None) -> BacktestResult:
    s, w, b = settings, settings.wheel, settings.backtest
    events = events or {}
    dates, data = _prepare(prices, s)
    idx = [i for i, d in enumerate(dates) if start <= d <= end]
    res = BacktestResult("wheel")
    if len(idx) < 2:
        return res

    cash = b.initial_capital
    reserved = 0.0
    n = b.contracts
    pos = {sym: WheelPosition(symbol=sym, allow_cc_below_basis=w.allow_cc_below_cost_basis) for sym in data}
    pending: dict[str, Optional[dict]] = {sym: None for sym in data}
    deployed: list[float] = []

    def spot_of(sym: str, i: int) -> float:
        return data[sym].closes[i]

    def record(sym: str, rec: Optional[dict]) -> None:
        if rec:
            res.trades.append({"symbol": sym, **rec})

    for i in idx:
        today = dates[i]
        for sym, ser in data.items():
            p, spot, sigma, ev = pos[sym], ser.closes[i], ser.iv[i], events.get(sym, [])

            # 1) expiry / assignment settlement
            if p.leg is not None and p.leg.expiry <= today:
                lg = p.leg
                if lg.right == "P":
                    if spot < lg.strike:
                        cash -= lg.strike * MULT * lg.contracts
                        reserved -= lg.strike * MULT * lg.contracts
                        p.assign_put(today)
                    else:
                        reserved -= lg.strike * MULT * lg.contracts
                        record(sym, p.expire_leg(today))
                else:
                    if spot > lg.strike:
                        cash += lg.strike * MULT * lg.contracts
                        record(sym, p.call_away(today))
                        p.restart()
                    else:
                        p.expire_leg(today)

            # 2) execute orders decided on the previous bar
            order, pending[sym] = pending[sym], None
            if order is not None and not math.isnan(sigma):
                try:
                    cash, reserved = _wheel_execute(order, p, spot, sigma, ev, today, cash, reserved, n, b, res, sym, record)
                except InvalidTransition:
                    pass

        # 3) mark to market
        equity = cash
        for sym, ser in data.items():
            p, spot = pos[sym], ser.closes[i]
            equity += p.shares * spot
            if p.leg is not None and not math.isnan(ser.iv[i]):
                equity -= _mid(spot, p.leg.strike, p.leg.right, today, p.leg.expiry, ser.iv[i],
                               b.risk_free_rate, events.get(sym, []), b.event_iv_uplift) * MULT * p.leg.contracts
        res.dates.append(today)
        res.equity.append(equity)
        deployed.append(reserved + sum(p.shares * data[sym].closes[i] for sym, p in pos.items()))

        # 4) decisions for the next bar (closes up to and including today only)
        for sym, ser in data.items():
            if math.isnan(ser.iv[i]) or pending[sym] is not None:
                continue
            pending[sym] = _wheel_decide(pos[sym], ser, i, today, w, b, events.get(sym, []), cash, reserved, n)

    res.drawdown = _drawdown(res.equity)
    res.metrics = compute_metrics(res.dates, res.equity, res.trades, b.initial_capital, b.risk_free_rate, deployed)
    return res


def _wheel_decide(p: WheelPosition, ser: _Series, i: int, today: date, w, b, ev, cash: float, reserved: float,
                  n: int) -> Optional[dict]:
    spot, sigma, r = ser.closes[i], ser.iv[i], b.risk_free_rate
    dte_rng = (w.dte_min, w.dte_max)
    exps = _fridays(today, w.dte_min, w.dte_max)

    def pick(right: str, drange: tuple[float, float], min_strike: Optional[float] = None):
        band = (0.70, 1.0) if right == "P" else (1.0, 1.35)
        chain = synthetic_chain(ser.symbol, spot, today, sigma, exps, right, band, r, ev, b.event_iv_uplift)
        return select_candidate(chain, right, spot, today, drange, dte_rng, w.min_open_interest,  # type: ignore[arg-type]
                                w.max_spread_pct, min_strike, r)

    if p.state is WheelState.SEARCHING:
        ok, _ = entry_ok(_iv_rank_at(ser, i), w)
        if not ok:
            return None
        c = pick("P", (w.put_delta_min, w.put_delta_max))
        if c and cash - reserved >= c.quote.strike * MULT * n:
            return {"kind": "open_csp", "strike": c.quote.strike, "expiry": c.quote.expiry}
        return None
    if p.state is WheelState.ASSIGNED:
        floor = None if w.allow_cc_below_cost_basis else p.cost_basis
        c = pick("C", (w.call_delta_min, w.call_delta_max), floor)
        return {"kind": "open_cc", "strike": c.quote.strike, "expiry": c.quote.expiry} if c else None
    if p.leg is not None:
        mid = _mid(spot, p.leg.strike, p.leg.right, today, p.leg.expiry, sigma, r, ev, b.event_iv_uplift)
        acts = management_actions(p, spot, today, mid, w)
        if not acts:
            return None
        if acts[0].kind == "close_profit":
            return {"kind": "close"}
        right = p.leg.right
        floor = (p.cost_basis if not w.allow_cc_below_cost_basis else None) if right == "C" else None
        c = pick(right, (w.put_delta_min, w.put_delta_max) if right == "P" else (w.call_delta_min, w.call_delta_max), floor)
        return {"kind": "roll", "strike": c.quote.strike, "expiry": c.quote.expiry} if c else None
    return None


def _wheel_execute(order: dict, p: WheelPosition, spot: float, sigma: float, ev, today: date, cash: float,
                   reserved: float, n: int, b, res: BacktestResult, sym: str, record: Callable) -> tuple[float, float]:
    r, up, slip, comm = b.risk_free_rate, b.event_iv_uplift, b.slippage_pct_of_spread, b.commission_per_contract
    kind = order["kind"]
    if kind in ("open_csp", "open_cc"):
        right = "P" if kind == "open_csp" else "C"
        k, exp = order["strike"], order["expiry"]
        prem = _fill(_mid(spot, k, right, today, exp, sigma, r, ev, up), "sell", slip)
        if kind == "open_csp":
            if cash - reserved < k * MULT * n:
                return cash, reserved
            p.open_csp(k, exp, prem, n, today)
            reserved += k * MULT * n
        else:
            p.open_cc(k, exp, prem, today)
            n = p.leg.contracts  # type: ignore[union-attr]
        return cash + prem * MULT * n - comm * n, reserved
    lg = p.leg
    if lg is None:
        return cash, reserved
    n = lg.contracts
    close_px = _fill(_mid(spot, lg.strike, lg.right, today, lg.expiry, sigma, r, ev, up), "buy", slip)
    if kind == "close":
        was_csp = p.state is WheelState.CSP_OPEN
        if was_csp:
            reserved -= lg.strike * MULT * n
        record(sym, p.close_leg(close_px, today))
        return cash - close_px * MULT * n - comm * n, reserved
    # roll
    k, exp = order["strike"], order["expiry"]
    new_px = _fill(_mid(spot, k, lg.right, today, exp, sigma, r, ev, up), "sell", slip)
    old_strike = lg.strike
    p.roll(k, exp, new_px, close_px, today)  # may raise CostBasisViolation -> order dropped
    if lg.right == "P":
        reserved += (k - old_strike) * MULT * n
    return cash + (new_px - close_px) * MULT * n - 2 * comm * n, reserved


# ---------------------------------------------------------------------------
# Straddle
# ---------------------------------------------------------------------------

@dataclass
class _OpenStraddle:
    legs: list[PayoffLeg]  # entry prices = fills
    opened: date
    entry_value: float  # signed per share (debit +, credit -)
    mode: str


def run_straddle_backtest(prices: dict[str, pd.Series], start: date, end: date, settings: Settings,
                          events: Optional[dict[str, list[date]]] = None,
                          mode: Optional[str] = None) -> BacktestResult:
    s, sd, b = settings, settings.straddle, settings.backtest
    mode = mode or sd.mode
    events = events or {}
    dates, data = _prepare(prices, s)
    idx = [i for i, d in enumerate(dates) if start <= d <= end]
    res = BacktestResult(f"straddle_{mode}")
    if len(idx) < 2:
        return res

    cash, n = b.initial_capital, b.contracts
    r, up, slip, comm = b.risk_free_rate, b.event_iv_uplift, b.slippage_pct_of_spread, b.commission_per_contract
    open_pos: dict[str, Optional[_OpenStraddle]] = {sym: None for sym in data}
    pending: dict[str, Optional[dict]] = {sym: None for sym in data}

    def leg_mid(l: PayoffLeg, spot: float, sigma: float, today: date, ev) -> float:
        return _mid(spot, l.strike, l.right, today, l.expiry, sigma, r, ev, up)

    for i in idx:
        today = dates[i]
        for sym, ser in data.items():
            spot, sigma, ev = ser.closes[i], ser.iv[i], events.get(sym, [])
            op = open_pos[sym]

            # 1) expiry settlement at intrinsic
            if op is not None and op.legs[0].expiry <= today:
                for l in op.legs:
                    intr = max(spot - l.strike, 0.0) if l.right == "C" else max(l.strike - spot, 0.0)
                    cash += l.qty * intr * MULT
                pnl = _close_pnl(op, [max(spot - l.strike, 0.0) if l.right == "C" else max(l.strike - spot, 0.0)
                                       for l in op.legs], comm, closing_commission=False)
                res.trades.append({"symbol": sym, "mode": op.mode, "opened": op.opened.isoformat(),
                                   "closed": today.isoformat(), "pnl": pnl, "reason": "expired"})
                open_pos[sym], op = None, None

            # 2) execute pending
            order, pending[sym] = pending[sym], None
            if order is not None and not math.isnan(sigma):
                if order["kind"] == "open" and op is None:
                    legs = [l.model_copy(update={"price": _fill(leg_mid(l, spot, sigma, today, ev),
                                                                  "buy" if l.qty > 0 else "sell", slip)})
                            for l in order["legs"]]
                    cash -= sum(l.qty * l.price for l in legs) * MULT + comm * sum(abs(l.qty) for l in legs)
                    open_pos[sym] = _OpenStraddle(legs, today, net_value_per_share(legs), order["mode"])
                elif order["kind"] == "close" and op is not None:
                    px = [_fill(leg_mid(l, spot, sigma, today, ev), "sell" if l.qty > 0 else "buy", slip) for l in op.legs]
                    cash += sum(l.qty * p_ for l, p_ in zip(op.legs, px)) * MULT - comm * sum(abs(l.qty) for l in op.legs)
                    pnl = _close_pnl(op, px, comm, closing_commission=True)
                    res.trades.append({"symbol": sym, "mode": op.mode, "opened": op.opened.isoformat(),
                                       "closed": today.isoformat(), "pnl": pnl, "reason": order["reason"]})
                    open_pos[sym] = None

        # 3) mark
        equity = cash
        for sym, ser in data.items():
            op = open_pos[sym]
            if op is not None and not math.isnan(ser.iv[i]):
                equity += sum(l.qty * leg_mid(l, ser.closes[i], ser.iv[i], today, events.get(sym, [])) for l in op.legs) * MULT
        res.dates.append(today)
        res.equity.append(equity)

        # 4) decisions for next bar
        for sym, ser in data.items():
            if math.isnan(ser.iv[i]) or pending[sym] is not None:
                continue
            spot, sigma, ev = ser.closes[i], ser.iv[i], events.get(sym, [])
            op = open_pos[sym]
            if op is not None:
                cur = sum(l.qty * leg_mid(l, spot, sigma, today, ev) for l in op.legs)
                why = exit_decision(op.entry_value, cur, (op.legs[0].expiry - today).days, sd)
                if why:
                    pending[sym] = {"kind": "close", "reason": why.split(":")[0]}
                continue
            pending[sym] = _straddle_decide(ser, i, today, mode, sd, b, ev, n)

    res.drawdown = _drawdown(res.equity)
    res.metrics = compute_metrics(res.dates, res.equity, res.trades, b.initial_capital, b.risk_free_rate)
    return res


def _close_pnl(op: _OpenStraddle, exit_px: Sequence[float], comm: float, closing_commission: bool) -> float:
    gross = sum(l.qty * (px - l.price) for l, px in zip(op.legs, exit_px)) * MULT
    legs_n = sum(abs(l.qty) for l in op.legs)
    return gross - comm * legs_n * (2 if closing_commission else 1)


def _straddle_decide(ser: _Series, i: int, today: date, mode: str, sd, b, ev: Sequence[date], n: int) -> Optional[dict]:
    spot, sigma, r = ser.closes[i], ser.iv[i], b.risk_free_rate
    rv = ser.rv[i]
    nxt = next((e for e in sorted(ev) if e >= today), None)
    days_to_event = (nxt - today).days if nxt else None
    exps = _fridays(today, sd.dte_min, sd.dte_max)
    if not exps:
        return None
    chain = synthetic_chain(ser.symbol, spot, today, sigma, exps, "CP", (0.85, 1.15), r, ev, b.event_iv_uplift)
    legs = build_legs(chain, spot, today, mode, sd, event_date=nxt, contracts=n)  # type: ignore[arg-type]
    if not legs:
        return None
    dte = (legs[0].expiry - today).days
    implied = legs[0].price + legs[1].price
    realized = expected_abs_move(spot, rv, dte / 365.0) if not math.isnan(rv) else None
    iv_over_rv = (sigma / b.iv_premium) / rv if rv and not math.isnan(rv) else None  # use raw RV20 vs RV60
    d = evaluate_entry(mode, sd, _iv_rank_at(ser, i), iv_over_rv, days_to_event, dte, implied, realized)  # type: ignore[arg-type]
    if not d.ok:
        return None
    return {"kind": "open", "legs": legs, "mode": mode}
