"""
premia/paper.py — the daily paper-trading loop (phase 7). Run once a day, ~30 minutes before the close:

    python -m premia.paper --dry-run          # proposals, rule checks, marks and report; no orders (default)
    python -m premia.paper --execute          # same, and send limit orders to the IBKR PAPER account (port 7497)

Steps: refresh the research data for the traded names -> the strategy models propose (strike/expiry from the model;
the live bid/ask must confirm a positive edge net of costs) -> every proposal goes through PaperExecutor (paper account
check, kill switch, liquidity, hard limits) -> delta hedges sized by the band policy -> marks and the daily report.

Parameters are the walk-forward selection in results/premia/selected_params.json. The live account is never touched:
PaperExecutor refuses any port but 7497 and any account that is not a DU... paper account.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from . import market, pricing, risk, strategies as st
from .config import COSTS, LIMITS, RESULTS_DIR
from .execution import PAPER_DIR, PaperExecutor, Proposal
from .hedging import HedgePolicy, hedge_trade
from .report import append_marks, daily_report


def selected_config() -> dict:
    p = RESULTS_DIR / "selected_params.json"
    sel = json.loads(p.read_text()) if p.exists() else {"wheel": "wheel_l0.1_b1"}
    lab = sel["wheel"]
    lam = float(lab.split("_l")[1].split("_")[0]) if "_l" in lab else 0.1
    mult = float(lab.split("_b")[1]) if "_b" in lab else 1.0
    return {"lam": lam, "band_mult": mult, "labels": sel}


def wheel_proposals(underlyings: dict[str, market.Underlying], names: list[str], lam: float, nav: float,
                    live_quote, held: set[str]) -> list[Proposal]:
    """Model candidates at the latest close, confirmed against the live quote of the chosen contract."""
    out = []
    for sym in names:
        u = underlyings.get(sym)
        if u is None or sym in held:
            continue
        t = u.df.index[-1]
        c = st.best_candidate(st.wheel_candidates(u, t, "P", lam, hedged=True))
        if c is None:
            continue
        exp = c.expiry.strftime("%Y%m%d")
        q = live_quote(sym, exp, c.K, "P")
        bid, ask = q.get("bid"), q.get("ask")
        if not (bid and ask and ask >= bid > 0):
            continue
        fill = 0.5 * (bid + ask) - COSTS.slippage_frac_of_half_spread * 0.5 * (ask - bid)
        hedge_cost = c.fill - c.fair - COSTS.option_commission / 100 - c.edge        # the model's expected hedge cost
        edge_live = fill - c.fair - COSTS.option_commission / 100 - hedge_cost
        if edge_live <= 0:
            continue
        weight = 0.15 if sym == "SPY" else 0.025
        n = max(int(round(weight * nav / (c.K * 100))), 0)
        if n == 0:
            continue
        row = u.df.iloc[-1]
        T = market.year_frac(t, c.expiry)
        g = pricing.greeks(float(row["close"]), [c.K], T, float(row["r"]), float(row["q"]), [c.iv], "P")
        S = float(row["close"])
        impact = {"net_vega": float(g.vega[0]) * -n, "net_vanna": float(g.vanna[0]) * -n * S,
                  "net_volga": float(g.volga[0]) * -n / 100, "margin": c.K * 100 * n, "by_name": {sym: c.K * 100 * n}}
        out.append(Proposal(sym, "P", c.K, exp, -n, "csp", f"edge live {edge_live:.2f}/sh, model fair {c.fair:.2f}",
                            edge_live, c.fair, impact))
    return out


def hedge_orders(positions: list[dict], spots: dict[str, float], gammas: dict[str, float], stock: dict[str, float],
                 band_mult: float, nav: float) -> dict[str, int]:
    """Shares to trade per symbol under the band policy and the hard delta cap."""
    net: dict[str, float] = {}
    for p in positions:
        net[p["symbol"]] = net.get(p["symbol"], 0.0) + p["delta_shares"]
    out = {}
    for sym, d in net.items():
        d += stock.get(sym, 0.0)
        S = spots[sym]
        tr = hedge_trade(HedgePolicy("band", band_mult=band_mult), d, 0, S, gammas.get(sym, 0.0), 0.1, 0.04, nav,
                         LIMITS.max_net_delta_pct_nav * nav / S / 2)
        if tr:
            out[sym] = int(tr)
    return out


def book_from_portfolio(items, underlyings: dict[str, market.Underlying]) -> tuple[list[dict], dict, list[dict]]:
    """(option positions with model Greeks, stock shares, mark rows) from IBKR portfolio items."""
    positions, stock, marks = [], {}, []
    today = date.today().isoformat()
    for it in items:
        c = it.contract
        if c.secType == "STK":
            stock[c.symbol] = stock.get(c.symbol, 0.0) + float(it.position)
            continue
        if c.secType != "OPT" or c.symbol not in underlyings:
            continue
        u = underlyings[c.symbol]
        row = u.df.iloc[-1]
        S, r, q = float(row["close"]), float(row["r"]), float(row["q"])
        exp = pd.Timestamp(c.lastTradeDateOrContractMonth)
        T = max(market.year_frac(pd.Timestamp(today), exp), 1 / 365)
        px = float(it.marketPrice)
        iv = float(pricing.implied_vol(px, S, c.strike, T, r, q, c.right)) if px > 0 else float(row["iv"])
        iv = iv if np.isfinite(iv) else float(row["iv"])
        g = pricing.greeks(S, [c.strike], T, r, q, [iv], c.right)
        n = float(it.position) * 100
        positions.append({"symbol": c.symbol, "delta_shares": float(g.delta[0]) * n, "gamma_shares": float(g.gamma[0]) * n,
                          "vega": float(g.vega[0]) * n / 100, "vanna": float(g.vanna[0]) * n / 100 * S,
                          "volga": float(g.volga[0]) * n / 1e4, "S": S, "K": c.strike, "right": c.right,
                          "margin": c.strike * -n if (c.right == "P" and n < 0) else 0.0})
        marks.append({"date": today, "contract": f"{c.symbol} {c.lastTradeDateOrContractMonth} {c.strike:g}{c.right}",
                      "symbol": c.symbol, "right": c.right, "K": c.strike, "expiry": exp.date().isoformat(), "T": T,
                      "S": S, "iv": iv, "r": r, "q": q, "qty": float(it.position), "trade_type": "csp" if c.right == "P" else "cc",
                      "bid": None, "ask": None})
    return positions, stock, marks


def exposures_of(positions: list[dict], stock: dict, spots: dict) -> risk.Exposures:
    e = risk.Exposures()
    for p in positions:
        e.net_delta += p["delta_shares"] * p["S"]
        e.net_vega += p["vega"]
        e.net_vanna += p["vanna"]
        e.net_volga += p["volga"]
        e.margin += p["margin"]
        if p["margin"]:
            e.by_name[p["symbol"]] = e.by_name.get(p["symbol"], 0.0) + p["margin"]
    for sym, sh in stock.items():
        e.net_delta += sh * spots.get(sym, 0.0)
    return e


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m premia.paper")
    ap.add_argument("--execute", action="store_true", help="send limit orders to the IBKR paper account")
    ap.add_argument("--port", type=int, default=7497)
    ap.add_argument("--names", default="SPY", help="comma-separated underlyings (default SPY)")
    args = ap.parse_args(argv)
    from .execution import IBPaperBroker
    from .research.build import load_shapes, shape_for
    cfg = selected_config()
    broker = IBPaperBroker()
    ex = PaperExecutor(broker)
    ex.connect(port=args.port)                       # refuses anything but a verified paper account
    try:
        names = [s.strip().upper() for s in args.names.split(",") if s.strip()]
        shapes = load_shapes()
        U = {s: market.build_underlying(s, shape_for(s, shapes), use_cache=False) for s in names}
        U = {k: v for k, v in U.items() if v is not None}
        nav = float(sum(float(v.value) for v in broker.ib.accountValues() if v.tag == "NetLiquidation" and v.currency == "USD"))
        positions, stock, marks = book_from_portfolio(broker.ib.portfolio(), U)
        spots = {s: float(u.df["close"].iloc[-1]) for s, u in U.items()}
        exp = exposures_of(positions, stock, spots)
        held = {p["symbol"] for p in positions}
        for p in wheel_proposals(U, names, cfg["lam"], nav, broker.quote, held):
            if args.execute:
                print(ex.submit(p, exp, nav))
            else:
                ex.log.write("dry_run_proposal", **p.__dict__)
                print("DRY RUN proposal:", p)
        gam = {}
        for p in positions:
            gam[p["symbol"]] = gam.get(p["symbol"], 0.0) + p["gamma_shares"]
        for sym, sh in hedge_orders(positions, spots, gam, stock, cfg["band_mult"], nav).items():
            ex.log.write("hedge", symbol=sym, shares=sh, executed=False)
            print(f"hedge {sym}: {sh:+d} shares (stock orders are logged; place them in TWS paper or extend the broker)")
        if marks:
            allm = append_marks(marks)
            print(daily_report(date.today().isoformat(), allm, stock, nav))
    finally:
        broker.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
