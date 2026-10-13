from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .. import charts, theme
from ..common import fetch, live


def _body() -> None:
    w = fetch("/wheel")
    if w is None:
        return
    rows = w["tickers"]
    st.subheader("Per-ticker cycle")
    df = pd.DataFrame([{
        "symbol": r["symbol"], "state": r["state"],
        "position": (f"{r['leg']['right']} {r['leg']['strike']:g} exp {r['leg']['expiry']} ({r['leg']['dte']} DTE)"
                     if r["leg"] else (f"{r['shares']} sh" if r["shares"] else "-")),
        "cost basis": r["cost_basis"], "premium collected": r["premium_collected"], "cycles": r["cycles"],
        "realized P&L": r["realized_pnl"],
        "ann. return (done)": None if r["annualized_return"] is None else r["annualized_return"] * 100,
        "ann. return (current)": None if r["cycle_annualized_return"] is None else r["cycle_annualized_return"] * 100,
    } for r in rows])
    st.dataframe(df, hide_index=True, width="stretch", column_config={
        "cost basis": st.column_config.NumberColumn(format="$%.2f"),
        "premium collected": st.column_config.NumberColumn(format="$%.0f"),
        "realized P&L": st.column_config.NumberColumn(format="$%.0f"),
        "ann. return (done)": st.column_config.NumberColumn(format="%.1f%%"),
        "ann. return (current)": st.column_config.NumberColumn(format="%.1f%%"),
    })
    st.caption("Annualized return is on capital (strike x 100), counting assignment outcomes.")

    st.subheader("Action queue")
    any_action = False
    for r in rows:
        act, queue, leg = r["suggested_action"], r["queue"], r["leg"]
        if act:
            any_action = True
            kind = (act.get("trigger") or act.get("kind") or "").replace("_", " ")
            box = st.warning if (act.get("trigger") or "").startswith("roll") else st.info
            box(f"**{r['symbol']}** suggested: {act['action']} ({kind}). {act['reason']}")
        for q in queue:
            any_action = True
            st.caption(f"**{r['symbol']}** queued: {q['action']} - risk={q['verdict'] or 'pending'}, "
                       f"order={q['order_status'] or 'none'}: {q['reason']}")
        if leg and leg["pct_captured"] is not None:
            cap = leg["pct_captured"]
            st.progress(min(max(cap, 0.0), 1.0), text=f"{r['symbol']} {leg['right']}{leg['strike']:g}: {cap:.0%} of premium captured")
    if not any_action:
        st.caption("No actions pending.")

    held = [r for r in rows if r["premium_collected"]]
    if held:
        fig = go.Figure(go.Bar(x=[r["symbol"] for r in held], y=[r["premium_collected"] for r in held],
                               marker_color=[theme.pnl_color(r["premium_collected"]) for r in held]))
        st.plotly_chart(charts._layout(fig, "Premium collected (lifetime, $)", 260), width="stretch")

    for r in rows:
        if r["history"]:
            with st.expander(f"{r['symbol']}: completed cycles"):
                st.dataframe(pd.DataFrame(r["history"]), hide_index=True, width="stretch")


def render() -> None:
    st.header("Wheel")
    live(_body)
