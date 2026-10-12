"""Read models for the API/dashboard. All data comes from the runtime's DB and in-memory state."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date
from typing import Any, Optional

from ..agents.risk import build_book
from ..agents.wheel import WheelAgent, find_quote
from ..earnings import next_event
from ..events import ChainSnapshot
from ..risk import compute_metrics, utilization
from ..runtime import Runtime
from ..strategies.pricing import expected_abs_move
from ..strategies.straddle import (
    MULT, PayoffLeg, breakevens, build_legs, evaluate_entry, max_profit_loss, net_greeks, net_value_per_share,
    payoff_curve,
)
from ..strategies.wheel import WheelPosition, quote_abs_delta


def _clean(x: Any) -> Any:
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if hasattr(x, "tolist"):
        return _clean(x.tolist())
    return x


def spots_for(rt: Runtime, symbols: list[str]) -> dict[str, float]:
    spots = rt.tracker.spots()
    for s in symbols:
        if s not in spots:
            rows = rt.ctx.db.closes(s, limit=1)
            if rows:
                spots[s] = rows[-1][1]
    return spots


def _wheel_positions(rt: Runtime) -> list[WheelPosition]:
    return [WheelPosition.model_validate(r["payload"]) for r in rt.ctx.db.wheel_states()]


def _mid(rt: Runtime, symbol: str, right: str, strike: float, expiry: date) -> Optional[float]:
    snap = rt.tracker.chains.get(symbol)
    q = find_quote(snap.quotes, right, strike, expiry) if snap else None
    return q.mid if q else None


# ---------------------------------------------------------------------------

def overview(rt: Runtime) -> dict:
    ctx, today = rt.ctx, rt.ctx.today()
    s = ctx.settings
    wheel = _wheel_positions(rt)
    spots = spots_for(rt, [p.symbol for p in wheel])
    realized = sum(p.realized_pnl for p in wheel)
    realized_today = sum(c["pnl"] for p in wheel for c in p.history if c["end"] == today.isoformat())
    unrealized, open_premium, complete = 0.0, 0.0, True

    for p in wheel:
        if p.state.value == "SEARCHING":
            continue
        unrealized += p.cycle_premium
        if p.leg is not None:
            open_premium += p.leg.premium * MULT * p.leg.contracts
            m = _mid(rt, p.symbol, p.leg.right, p.leg.strike, p.leg.expiry)
            if m is None:
                complete = False
            else:
                unrealized -= m * MULT * p.leg.contracts
        if p.shares and p.assignment_price is not None:
            if p.symbol in spots:
                unrealized += p.shares * (spots[p.symbol] - p.assignment_price)
            else:
                complete = False

    for pos in ctx.db.open_positions():
        if pos["strategy"] != "straddle":
            continue
        legs = [PayoffLeg.model_validate(l) for l in pos["legs"]]
        mids = [_mid(rt, pos["symbol"], l.right, l.strike, l.expiry) for l in legs]
        if any(m is None for m in mids):
            complete = False
        else:
            unrealized += (sum(l.qty * m for l, m in zip(legs, mids)) - net_value_per_share(legs)) * MULT
        if pos["open_premium"] < 0:
            open_premium += -pos["open_premium"]

    closed = ctx.db.closed_positions()
    realized += sum(c["realized_pnl"] or 0.0 for c in closed)
    realized_today += sum(c["realized_pnl"] or 0.0 for c in closed if (c["closed_at"] or "")[:10] == today.isoformat())

    book = build_book(ctx.db)
    deployed: Optional[float] = None
    if all(p.symbol in spots for p in book):
        deployed = compute_metrics(book, spots, today, s.risk).total_capital
    journal = ctx.db.journal(500)
    orders = ctx.db.list_orders(500)
    qualities = sorted({c.quality.value for c in rt.tracker.chains.values()})
    return _clean({
        "asof": today.isoformat(), "mode": s.mode.value, "data_source": s.data_source,
        "execution_backend": s.execution_backend, "kill_switch": ctx.db.kill_switch_engaged(),
        "starting_capital": s.starting_capital,
        "portfolio_value": s.starting_capital + realized + unrealized,
        "realized_pnl": realized, "unrealized_pnl": unrealized, "unrealized_complete": complete,
        "open_premium_collected": open_premium, "capital_deployed": deployed,
        "capital_utilization": (deployed / s.risk.max_capital_at_risk) if deployed is not None else None,
        "today": {
            "signals": sum(1 for j in journal if (j["ts"] or "")[:10] == today.isoformat()),
            "orders": sum(1 for o in orders if (o["created_at"] or "")[:10] == today.isoformat()),
            "realized_pnl": realized_today,
        },
        "data_quality": qualities,
    })


def wheel_view(rt: Runtime) -> dict:
    ctx, today = rt.ctx, rt.ctx.today()
    agent = rt.agents.get("wheel")
    rows = []
    by_symbol = {p.symbol: p for p in _wheel_positions(rt)}
    journal = ctx.db.journal(300)
    for sym in ctx.settings.tickers:
        p = by_symbol.get(sym) or WheelPosition(symbol=sym)
        snap = rt.tracker.chains.get(sym)
        leg = None
        if p.leg is not None:
            mid = _mid(rt, sym, p.leg.right, p.leg.strike, p.leg.expiry)
            spot = snap.underlying_price if snap else None
            leg = {"right": p.leg.right, "strike": p.leg.strike, "expiry": p.leg.expiry.isoformat(),
                   "dte": (p.leg.expiry - today).days, "contracts": p.leg.contracts, "premium": p.leg.premium,
                   "mid": mid, "pct_captured": (1 - mid / p.leg.premium) if mid is not None and p.leg.premium else None,
                   "spot": spot}
        action = None
        if isinstance(agent, WheelAgent) and snap is not None and sym not in agent.inflight:
            sig = agent.decide(agent.positions.get(sym, p), snap, today)
            if sig is not None:
                action = {"kind": sig.meta.get("kind"), "trigger": sig.meta.get("trigger"), "reason": sig.reason,
                          "action": sig.action.value}
        queue = [{"signal_id": j["signal_id"], "action": j["action"], "reason": j["reason"], "verdict": j["verdict"],
                  "order_status": j["order_status"]}
                 for j in journal if j["symbol"] == sym and j["strategy"] == "wheel"
                 and (j["order_status"] in (None, "pending_confirmation", "submitted"))][:5]
        rows.append({
            "symbol": sym, "state": p.state.value, "shares": p.shares, "cost_basis": p.cost_basis,
            "assignment_price": p.assignment_price, "premium_collected": p.lifetime_premium,
            "cycle_premium": p.cycle_premium, "cycles": p.cycles, "realized_pnl": p.realized_pnl,
            "annualized_return": p.total_annualized_return(), "cycle_annualized_return": p.cycle_annualized_return(today),
            "leg": leg, "suggested_action": action, "queue": queue,
            "history": p.history[-5:],
        })
    return _clean({"asof": today.isoformat(), "tickers": rows})


def straddle_view(rt: Runtime, mode: Optional[str] = None) -> dict:
    ctx, today = rt.ctx, rt.ctx.today()
    cfg = ctx.settings.straddle
    cfg = cfg.model_copy(update={"mode": mode}) if mode in ("long", "short") else cfg
    events = getattr(rt.agents.get("straddle"), "events", {})
    rows = []
    for sym in ctx.settings.tickers:
        iv, snap = rt.tracker.ivs.get(sym), rt.tracker.chains.get(sym)
        row: dict[str, Any] = {"symbol": sym, "iv_rank": None, "candidate": None}
        if iv:
            row.update(iv_rank=iv.iv_rank, iv_percentile=iv.iv_percentile, iv_current=iv.iv_current,
                       realized_vol=iv.realized_vol, history_days=iv.history_days,
                       expected_move_abs=iv.expected_move_abs, expected_move_pct=iv.expected_move_pct)
        ev = next_event(events, sym, today)
        row["days_to_event"] = (ev - today).days if ev else None
        if snap is not None:
            legs = build_legs(snap.quotes, snap.underlying_price, today, cfg.mode, cfg, event_date=ev)
            if legs:
                spot = snap.underlying_price
                dte = (legs[0].expiry - today).days
                implied = legs[0].price + legs[1].price
                rv = iv.realized_vol if iv else None
                realized = expected_abs_move(spot, rv, dte / 365.0) if rv else None
                dec = evaluate_entry(cfg.mode, cfg, iv.iv_rank if iv else None,
                                     (iv.iv_current / rv) if iv and rv else None, row["days_to_event"], dte, implied, realized)
                xs, ys = payoff_curve(legs, spot)
                mp, ml = max_profit_loss(legs)
                g = net_greeks(legs, spot, today)
                row["candidate"] = {
                    "mode": cfg.mode, "expiry": legs[0].expiry.isoformat(), "dte": dte, "strike": legs[0].strike,
                    "spot": spot, "implied_move": implied, "realized_move": realized, "breakevens": breakevens(legs),
                    "max_profit": mp, "max_loss": ml, "entry_ok": dec.ok, "reasons": dec.reasons, "flags": dec.flags,
                    "greeks": {"delta": g.delta, "gamma": g.gamma, "theta": g.theta, "vega": g.vega},
                    "payoff": {"spot": xs.tolist(), "pnl": ys.tolist()},
                    "legs": [l.model_dump(mode="json") for l in legs],
                }
        rows.append(row)

    open_rows = []
    for pos in ctx.db.open_positions():
        if pos["strategy"] != "straddle":
            continue
        legs = [PayoffLeg.model_validate(l) for l in pos["legs"]]
        spot = rt.tracker.spots().get(pos["symbol"]) or (legs[0].strike if legs else 100.0)
        xs, ys = payoff_curve(legs, spot)
        mp, ml = max_profit_loss(legs)
        g = net_greeks(legs, spot, today)
        open_rows.append({"id": pos["id"], "symbol": pos["symbol"], "mode": pos["meta"].get("mode"),
                          "opened_at": pos["opened_at"], "breakevens": breakevens(legs), "max_profit": mp, "max_loss": ml,
                          "greeks": {"delta": g.delta, "gamma": g.gamma, "theta": g.theta, "vega": g.vega},
                          "payoff": {"spot": xs.tolist(), "pnl": ys.tolist()},
                          "legs": [l.model_dump(mode="json") for l in legs]})
    return _clean({"asof": today.isoformat(), "mode": cfg.mode, "tickers": rows, "open_positions": open_rows,
                   "thresholds": {"long_max_iv_rank": cfg.long_max_iv_rank, "short_min_iv_rank": cfg.short_min_iv_rank}})


@dataclass
class ScannerFilters:
    strategy: str = "all"  # all | wheel_put | wheel_call | straddle
    min_delta: float = 0.0
    max_delta: float = 1.0
    min_dte: int = 0
    max_dte: int = 365
    min_iv_rank: float = 0.0
    max_spread_pct: float = 1.0
    min_open_interest: int = 0


def scanner(rt: Runtime, f: ScannerFilters) -> list[dict]:
    today = rt.ctx.today()
    rows: list[dict] = []
    for sym, snap in rt.tracker.chains.items():
        iv = rt.tracker.ivs.get(sym)
        rank = iv.iv_rank if iv else None
        if rank is not None and rank < f.min_iv_rank:
            continue
        spot = snap.underlying_price
        if f.strategy in ("all", "wheel_put", "wheel_call"):
            for q in snap.quotes:
                dte = (q.expiry - today).days
                if not f.min_dte <= dte <= f.max_dte:
                    continue
                kind = "wheel_put" if q.right == "P" else "wheel_call"
                if f.strategy not in ("all", kind):
                    continue
                if (q.right == "P" and q.strike >= spot) or (q.right == "C" and q.strike <= spot):
                    continue
                mid, sp = q.mid, q.spread_pct
                if mid is None or sp is None or sp > f.max_spread_pct:
                    continue
                if (q.open_interest or 0) < f.min_open_interest:
                    continue
                d = quote_abs_delta(q, spot, today)
                if d is None or not f.min_delta <= d[0] <= f.max_delta:
                    continue
                rows.append({"strategy": kind, "symbol": sym, "right": q.right, "strike": q.strike,
                             "expiry": q.expiry.isoformat(), "dte": dte, "delta": d[0], "delta_estimated": d[1],
                             "bid": q.bid, "ask": q.ask, "mid": mid, "spread_pct": sp, "open_interest": q.open_interest,
                             "iv": q.iv, "iv_rank": rank, "annualized_roc": mid / q.strike * 365.0 / max(dte, 1),
                             "score": mid / q.strike * 365.0 / max(dte, 1), "quality": q.quality.value})
        if f.strategy in ("all", "straddle"):
            rows.extend(_straddle_rows(sym, snap, rank, today, f))
    rows.sort(key=lambda r: r["score"] or 0.0, reverse=True)
    return _clean(rows[:500])


def _straddle_rows(sym: str, snap: ChainSnapshot, rank: Optional[float], today: date, f: ScannerFilters) -> list[dict]:
    spot = snap.underlying_price
    out = []
    for exp in sorted({q.expiry for q in snap.quotes}):
        dte = (exp - today).days
        if not f.min_dte <= dte <= f.max_dte or dte < 1:
            continue
        qs = [q for q in snap.quotes if q.expiry == exp]
        ks = sorted({q.strike for q in qs if q.right == "C"} & {q.strike for q in qs if q.right == "P"},
                    key=lambda k: abs(k - spot))
        if not ks:
            continue
        c = next(q for q in qs if q.right == "C" and q.strike == ks[0])
        p = next(q for q in qs if q.right == "P" and q.strike == ks[0])
        if c.mid is None or p.mid is None:
            continue
        spreads = [x for x in (c.spread_pct, p.spread_pct) if x is not None]
        sp = max(spreads) if len(spreads) == 2 else None
        oi = min((c.open_interest or 0), (p.open_interest or 0))
        if sp is None or sp > f.max_spread_pct or oi < f.min_open_interest:
            continue
        move = c.mid + p.mid
        out.append({"strategy": "straddle", "symbol": sym, "right": "C+P", "strike": ks[0], "expiry": exp.isoformat(),
                    "dte": dte, "delta": None, "delta_estimated": False, "bid": None, "ask": None, "mid": move,
                    "spread_pct": sp, "open_interest": oi, "iv": (c.iv + p.iv) / 2 if c.iv and p.iv else None,
                    "iv_rank": rank, "annualized_roc": None, "score": move / spot,  # implied move as % of spot
                    "quality": c.quality.value})
    return out


def risk_view(rt: Runtime) -> dict:
    ctx, today = rt.ctx, rt.ctx.today()
    r = ctx.settings.risk
    book = build_book(ctx.db)
    spots = spots_for(rt, [p.symbol for p in book])
    missing = [p.symbol for p in book if p.symbol not in spots]
    limits = {"max_capital_at_risk": r.max_capital_at_risk, "max_ticker_concentration": r.max_ticker_concentration,
              "max_net_delta": r.max_net_delta, "max_net_vega": r.max_net_vega, "max_net_theta_abs": r.max_net_theta_abs,
              "max_scenario_loss_pct": r.max_scenario_loss_pct, "warn_utilization": r.warn_utilization}
    if missing:
        return {"asof": today.isoformat(), "error": f"no price for: {', '.join(missing)}", "limits": limits}
    m = compute_metrics(book, spots, today, r)
    return _clean({
        "asof": today.isoformat(), "limits": limits, "utilization": utilization(m, r),
        "total_capital_at_risk": m.total_capital, "capital_by_ticker": m.capital_by_ticker,
        "concentration": m.concentration, "net_delta": m.net_delta, "net_vega": m.net_vega, "net_theta": m.net_theta,
        "scenario": {k: {str(mv): pnl for mv, pnl in v.items()} for k, v in m.scenario.items()},
        "worst_scenario_pnl": m.worst_scenario_pnl, "heatmap": m.heatmap, "unbounded": m.unbounded,
        "positions": [p.model_dump(mode="json") for p in book],
    })


def journal_view(rt: Runtime, limit: int = 200) -> list[dict]:
    rows = rt.ctx.db.journal(limit)
    for r in rows:
        r["risk_reasons"] = json.loads(r["risk_reasons"] or "[]")
    return rows
