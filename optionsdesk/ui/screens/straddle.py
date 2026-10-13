from __future__ import annotations

import streamlit as st

from .. import charts, theme
from ..common import fetch, live


def _body() -> None:
    mode_default = st.session_state.get("straddle_mode")
    data = fetch("/straddle", mode=mode_default)
    if data is None:
        return
    rows = data["tickers"]
    if not rows:
        st.info("No tickers configured.")
        return
    c1, c2 = st.columns([1, 2])
    mode = c1.radio("Mode", ["long", "short"], index=0 if data["mode"] == "long" else 1, horizontal=True, key="straddle_mode")
    symbol = c2.selectbox("Ticker", [r["symbol"] for r in rows], key="straddle_symbol")
    row = next(r for r in rows if r["symbol"] == symbol)
    th = data["thresholds"]

    g, m = st.columns([1, 2])
    with g:
        st.plotly_chart(charts.iv_gauge(row.get("iv_rank"), th["long_max_iv_rank"], th["short_min_iv_rank"],
                                        f"{symbol} IV rank"), width="stretch")
        if row.get("iv_rank") is None:
            st.warning("IV rank needs 30+ days of IV history.")
    with m:
        a = st.columns(4)
        a[0].metric("ATM IV", "-" if row.get("iv_current") is None else f"{row['iv_current']:.1%}")
        a[1].metric("Realized vol", "-" if row.get("realized_vol") is None else f"{row['realized_vol']:.1%}")
        a[2].metric("IV percentile", "-" if row.get("iv_percentile") is None else f"{row['iv_percentile']:.0f}")
        a[3].metric("Days to event", "-" if row.get("days_to_event") is None else row["days_to_event"])
        st.plotly_chart(charts.move_compare(rows), width="stretch")

    cand = row.get("candidate")
    if not cand:
        st.info("No straddle candidate yet (waiting for a chain with ATM quotes in the DTE window).")
    else:
        if cand["entry_ok"]:
            st.success(f"{cand['mode'].capitalize()} straddle passes the entry filters.")
        else:
            st.warning("Filters not met: " + "; ".join(cand["reasons"]))
        for f in cand["flags"]:
            st.caption(f"flag: {f}")
        st.plotly_chart(charts.payoff_fig(cand["payoff"]["spot"], cand["payoff"]["pnl"], cand["breakevens"], cand["spot"],
                                          f"{symbol} {cand['mode']} straddle {cand['strike']:g} exp {cand['expiry']}"),
                        width="stretch")
        k = st.columns(6)
        be = cand["breakevens"]
        k[0].metric("Breakevens", " / ".join(f"{b:.2f}" for b in be) if be else "-")
        k[1].metric("Max loss", "unbounded" if cand["max_loss"] is None else f"${cand['max_loss']:,.0f}")
        k[2].metric("Max profit", "unbounded" if cand["max_profit"] is None else f"${cand['max_profit']:,.0f}")
        gk = cand["greeks"]
        k[3].metric("Net delta", f"{gk['delta']:.1f}")
        k[4].metric("Net vega ($/pt)", f"{gk['vega']:.1f}")
        k[5].metric("Net theta ($/day)", f"{gk['theta']:.1f}")

    if data["open_positions"]:
        st.subheader("Open straddles")
        for p in data["open_positions"]:
            with st.expander(f"{p['symbol']} {p['mode']} - opened {p['opened_at']}", expanded=False):
                st.plotly_chart(charts.payoff_fig(p["payoff"]["spot"], p["payoff"]["pnl"], p["breakevens"], None),
                                width="stretch")
                st.caption(f"max loss {p['max_loss']}, max profit {p['max_profit']}, greeks {p['greeks']}")


def render() -> None:
    st.header("Straddle")
    live(_body)
