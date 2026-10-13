from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import theme
from ..common import badge, fetch, live


def _body() -> None:
    rows = fetch("/journal", limit=500)
    if rows is None:
        return
    if not rows:
        st.info("No signals yet.")
        return
    df = pd.DataFrame(rows)
    df = df[df["paper"].fillna(0) != 1]  # live only: hide anything tied to a paper order
    f = st.columns(4)
    strat = f[0].multiselect("Strategy", sorted(df["strategy"].unique()), key="j_strat")
    sym = f[1].multiselect("Symbol", sorted(df["symbol"].unique()), key="j_sym")
    ver = f[2].multiselect("Risk verdict", ["approve", "flag", "block"], key="j_ver")
    stat = f[3].multiselect("Order status", sorted(df["order_status"].dropna().unique()), key="j_stat")
    if strat:
        df = df[df["strategy"].isin(strat)]
    if sym:
        df = df[df["symbol"].isin(sym)]
    if ver:
        df = df[df["verdict"].isin(ver)]
    if stat:
        df = df[df["order_status"].isin(stat)]

    st.dataframe(df[["ts", "strategy", "symbol", "action", "verdict", "order_status", "fill_price", "reason", "correlation_id"]],
                 hide_index=True, width="stretch", column_config={
                     "ts": st.column_config.DatetimeColumn(format="MMM D, HH:mm:ss")})

    if df.empty:
        return
    st.subheader("Decision trace")
    pick = st.selectbox("Signal", df["signal_id"].tolist(),
                        format_func=lambda sid: f"{df.loc[df['signal_id'] == sid, 'symbol'].iloc[0]} "
                                                f"{df.loc[df['signal_id'] == sid, 'action'].iloc[0]} [{sid[:6]}]", key="j_pick")
    r = df[df["signal_id"] == pick].iloc[0]
    v = r["verdict"] or "pending"
    st.markdown(f"{badge(v, theme.STATUS_COLORS.get(v, theme.MUTED))} "
                f"{badge(r['order_status'] or 'no order', theme.STATUS_COLORS.get(r['order_status'] or '', theme.MUTED))} "
                f"correlation id `{r['correlation_id']}`", unsafe_allow_html=True)
    st.write("**Strategy reasoning**")
    st.write(r["reason"])
    if r["risk_reasons"]:
        st.write("**Risk reasoning**")
        for x in r["risk_reasons"]:
            st.write(f"- {x}")
    st.caption("Trace the same correlation id in the JSON logs to see every agent that touched this decision.")


def render() -> None:
    st.header("Trade journal")
    live(_body)
