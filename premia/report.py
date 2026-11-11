"""
premia/report.py — phase 7 daily log and report: positions, Greeks, hedges, P&L attribution.

Each day the marks of every open option (spot, implied vol at the strike, strike, time, rate, dividend yield, quantity,
trade type) are appended to `marks.csv`. The attribution between consecutive marks (premia.attribution) is how the
paper-trading period produces the *real-quote* residual P&L that phase 3B needs before any trade type can be called
alpha. The report is written as JSON and Markdown under results/premia/paper/reports/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import attribution, pricing
from .execution import PAPER_DIR

MARK_COLUMNS = ["date", "contract", "symbol", "right", "K", "expiry", "T", "S", "iv", "r", "q", "qty", "trade_type",
                "bid", "ask"]


def append_marks(rows: list[dict], path: Path = PAPER_DIR / "marks.csv") -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows, columns=MARK_COLUMNS)
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=MARK_COLUMNS)
    allm = pd.concat([old, new], ignore_index=True).drop_duplicates(["date", "contract"], keep="last")
    allm.to_csv(path, index=False)
    return allm


def position_greeks(marks: pd.DataFrame) -> pd.DataFrame:
    g = pricing.greeks(marks["S"].to_numpy(float), marks["K"].to_numpy(float), marks["T"].to_numpy(float),
                       marks["r"].to_numpy(float), marks["q"].to_numpy(float), marks["iv"].to_numpy(float),
                       marks["right"].to_numpy(str))
    n = marks["qty"].to_numpy(float) * 100
    S = marks["S"].to_numpy(float)
    return marks.assign(value=g.price * n, delta_shares=g.delta * n, dollar_delta=g.delta * n * S,
                        gamma_shares=g.gamma * n, vega_per_pt=g.vega * n / 100, theta_per_day=g.theta * n / 365,
                        vanna_usd=g.vanna * n / 100 * S, volga_usd=g.volga * n / 1e4)


def daily_report(date: str, marks: pd.DataFrame, stock: dict[str, float], nav: float,
                 out_dir: Path = PAPER_DIR / "reports") -> dict:
    """marks: all stored marks (any dates); stock: shares per symbol held now (assigned + hedges)."""
    today = marks[marks["date"] == date]
    pg = position_greeks(today) if len(today) else today
    by_sym = []
    for sym, g in (pg.groupby("symbol") if len(pg) else []):
        S = float(g["S"].iloc[0])
        net_sh = float(g["delta_shares"].sum()) + stock.get(sym, 0.0)
        by_sym.append({"symbol": sym, "option_delta_shares": float(g["delta_shares"].sum()), "stock_shares": stock.get(sym, 0.0),
                       "net_delta_shares": net_sh, "hedge_to_flatten_shares": -round(net_sh), "dollar_delta": net_sh * S,
                       "vega_per_pt": float(g["vega_per_pt"].sum()), "theta_per_day": float(g["theta_per_day"].sum()),
                       "vanna_usd": float(g["vanna_usd"].sum()), "volga_usd": float(g["volga_usd"].sum())})
    hist = marks[marks["date"] <= date]
    attr = attribution.attribute_frame(hist.rename(columns={})) if hist["date"].nunique() > 1 else pd.DataFrame()
    last = attr[attr["date"] == date] if len(attr) else attr
    comp = {c: float(last[c].sum()) for c in attribution.COMPONENTS + ["residual", "total"]} if len(last) else {}
    rep = {"date": date, "nav": nav, "positions": int(len(today)),
           "totals": {k: float(sum(r[k] for r in by_sym)) for k in ("dollar_delta", "vega_per_pt", "theta_per_day", "vanna_usd", "volga_usd")} if by_sym else {},
           "by_symbol": by_sym, "attribution_today": comp}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{date}.json").write_text(json.dumps(rep, indent=1, default=float))
    lines = [f"# premia paper report {date}", "", f"NAV {nav:,.0f}; {len(today)} option positions", "",
             "| symbol | net delta (sh) | hedge (sh) | vega/pt | theta/day | vanna $ | volga $ |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {r['symbol']} | {r['net_delta_shares']:,.0f} | {r['hedge_to_flatten_shares']:,.0f} | {r['vega_per_pt']:,.0f} | "
              f"{r['theta_per_day']:,.0f} | {r['vanna_usd']:,.0f} | {r['volga_usd']:,.0f} |" for r in by_sym]
    if comp:
        lines += ["", "P&L attribution since the previous mark: " + ", ".join(f"{k} {v:,.0f}" for k, v in comp.items())]
    (out_dir / f"{date}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rep
