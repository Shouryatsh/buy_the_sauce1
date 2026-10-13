from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import charts, theme
from ..common import fetch, live


def _body() -> None:
    r = fetch("/risk")
    if r is None:
        return
    if r.get("error"):
        st.warning(r["error"])
        return
    util, lim = r["utilization"], r["limits"]
    hot = [k for k, v in util.items() if v > 1.0]
    warn = [k for k, v in util.items() if lim["warn_utilization"] <= v <= 1.0]
    if hot:
        st.error("Over limit: " + ", ".join(hot))
    elif warn:
        st.warning("Approaching limit: " + ", ".join(warn))
    else:
        st.success("All limits within bounds.")

    a, b = st.columns([3, 2])
    a.plotly_chart(charts.utilization_bars(util, lim["warn_utilization"]), use_container_width=True)
    with b:
        k = st.columns(2)
        k[0].metric("Capital at risk", f"${r['total_capital_at_risk']:,.0f}")
        k[1].metric("Worst scenario P&L", f"${r['worst_scenario_pnl']:,.0f}")
        k = st.columns(3)
        k[0].metric("Net delta", f"{r['net_delta']:.1f}")
        k[1].metric("Net vega ($/pt)", f"{r['net_vega']:.1f}")
        k[2].metric("Net theta ($/day)", f"{r['net_theta']:.1f}")
        if r["unbounded"]:
            st.warning("Unbounded-loss positions: " + ", ".join(r["unbounded"]))

    if not r["heatmap"]:
        st.info("No open positions: exposure and scenario views appear once something is open.")
        return
    st.plotly_chart(charts.exposure_heatmap(r["heatmap"]), use_container_width=True)
    st.plotly_chart(charts.scenario_heatmap(r["scenario"]), use_container_width=True)
    conc = pd.DataFrame([{"ticker": t, "capital at risk": c, "concentration": r["concentration"][t] * 100}
                         for t, c in r["capital_by_ticker"].items()])
    st.dataframe(conc, hide_index=True, use_container_width=True, column_config={
        "capital at risk": st.column_config.NumberColumn(format="$%.0f"),
        "concentration": st.column_config.NumberColumn(format="%.1f%%", help="of max capital at risk")})
    with st.expander("Limits"):
        st.json(lim)


def render() -> None:
    st.header("Risk")
    live(_body)
