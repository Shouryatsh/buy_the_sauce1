from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from .. import charts
from ..common import fetch, live, post

METRIC_ROWS = [("annualized_return", "Annualized return", "pct"), ("total_return", "Total return", "pct"),
               ("max_drawdown", "Max drawdown", "pct"), ("sharpe", "Sharpe", "num"), ("sortino", "Sortino", "num"),
               ("win_rate", "Win rate", "pct"), ("n_trades", "Trades", "int"), ("avg_trade_pnl", "Avg trade P&L", "usd"),
               ("annualized_return_on_deployed", "Ann. return on deployed capital", "pct")]


def _fmt(v, kind: str) -> str:
    if v is None:
        return "-"
    return {"pct": f"{v:.1%}", "num": f"{v:.2f}", "int": f"{int(v)}", "usd": f"${v:,.0f}"}[kind]


def _submit_form() -> None:
    with st.form("bt_form"):
        c = st.columns(3)
        strategy = c[0].selectbox("Strategy", ["wheel", "straddle_long", "straddle_short"])
        tickers = c[1].text_input("Tickers (comma separated)", "AAPL,MSFT")
        end = date.today()
        dr = c[2].date_input("Date range", (end - timedelta(days=730), end))
        with st.expander("Parameter overrides"):
            o = st.columns(4)
            pd_min = o[0].number_input("Put delta min", 0.05, 0.6, 0.25, 0.01)
            pd_max = o[1].number_input("Put delta max", 0.05, 0.6, 0.30, 0.01)
            dte_min = o[2].number_input("DTE min", 7, 90, 30)
            dte_max = o[3].number_input("DTE max", 7, 120, 45)
            o = st.columns(4)
            take = o[0].number_input("Profit take %", 0.1, 1.0, 0.5, 0.05)
            comm = o[1].number_input("Commission / contract ($)", 0.0, 5.0, 0.65, 0.05)
            slip = o[2].number_input("Slippage (fraction of half-spread)", 0.0, 2.0, 0.5, 0.1)
            wings = o[3].checkbox("Straddle wings", False)
        if st.form_submit_button("Run in background"):
            if not isinstance(dr, tuple) or len(dr) != 2:
                st.error("Pick a start and end date.")
                return
            overrides = {
                "wheel": {"put_delta_min": pd_min, "put_delta_max": pd_max, "dte_min": int(dte_min), "dte_max": int(dte_max),
                          "profit_take_pct": take},
                "straddle": {"use_wings": wings},
                "backtest": {"commission_per_contract": comm, "slippage_pct_of_spread": slip},
            }
            out = post("/backtests", {"strategy": strategy, "tickers": [t.strip().upper() for t in tickers.split(",") if t.strip()],
                                      "start": dr[0].isoformat(), "end": dr[1].isoformat(), "overrides": overrides},
                       success="Queued. Results appear below when finished; the UI stays responsive.")
            if out:
                st.session_state.pop("bt_compare", None)


def _results() -> None:
    runs = fetch("/backtests")
    if runs is None:
        return
    if not runs:
        st.info("No backtests yet.")
        return
    if any(r["status"] == "running" for r in runs):
        st.info("A backtest is running in the background...")
    table = pd.DataFrame([{
        "run": r["id"], "strategy": r["strategy"], "status": r["status"],
        "tickers": ",".join(r["params"]["tickers"]), "from": r["params"]["start"], "to": r["params"]["end"],
        "ann. return": (r["metrics"] or {}).get("annualized_return"), "max DD": (r["metrics"] or {}).get("max_drawdown"),
        "Sharpe": (r["metrics"] or {}).get("sharpe"), "error": r["error"] or ""} for r in runs])
    st.dataframe(table, hide_index=True, use_container_width=True, column_config={
        "ann. return": st.column_config.NumberColumn(format="percent"), "max DD": st.column_config.NumberColumn(format="percent"),
        "Sharpe": st.column_config.NumberColumn(format="%.2f")})

    done = [r for r in runs if r["status"] == "done"]
    if not done:
        return
    label = {r["id"]: f"{r['strategy']} {','.join(r['params']['tickers'])} [{r['id'][:6]}]" for r in done}
    chosen = st.multiselect("Compare runs side by side", list(label), default=list(label)[:2], format_func=label.get, key="bt_compare")
    if not chosen:
        return
    full = {}
    for rid in chosen:
        run = fetch(f"/backtests/{rid}")
        if run:
            full[label[rid]] = run
    if not full:
        return
    st.dataframe(pd.DataFrame({name: {lab: _fmt(run["metrics"].get(key), kind) for key, lab, kind in METRIC_ROWS}
                               for name, run in full.items()}), use_container_width=True)
    a, b = st.columns(2)
    a.plotly_chart(charts.equity_compare(full), use_container_width=True)
    b.plotly_chart(charts.drawdown_compare(full), use_container_width=True)
    st.caption("Option prices are modelled (Black-Scholes on a realized-vol IV proxy), fills happen the bar after the signal, "
               "with commissions and slippage. This tests mechanics and risk shape, not real-market edge.")


def render() -> None:
    st.header("Backtest lab")
    _submit_form()
    live(_results)
