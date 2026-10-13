from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import theme
from ..common import DOT, badge, fetch, live, quality_badges
from ..live import EventFeed


def _feed() -> EventFeed | None:
    try:
        return st.session_state.get("_feed")
    except Exception:  # noqa: BLE001
        return None


def _body() -> None:
    o = fetch("/overview")
    if o is None:
        return
    mode_badge = badge("PAPER", theme.GREEN) if o["mode"] == "paper" else badge("LIVE", theme.RED)
    ks = badge("KILL SWITCH ENGAGED", theme.RED) if o["kill_switch"] else badge("trading enabled", theme.GREEN)
    st.markdown(f"{mode_badge} {ks} &nbsp; data: {quality_badges(o['data_quality'])} &nbsp; "
                f"source: `{o['data_source']}` / execution: `{o['execution_backend']}`", unsafe_allow_html=True)

    c = st.columns(4)
    c[0].metric("Portfolio value", f"${o['portfolio_value']:,.0f}", f"{o['portfolio_value'] - o['starting_capital']:+,.0f}")
    c[1].metric("Realized P&L", f"${o['realized_pnl']:,.0f}", f"{o['today']['realized_pnl']:+,.0f} today")
    c[2].metric("Unrealized P&L", f"${o['unrealized_pnl']:,.0f}", None if o["unrealized_complete"] else "partial (missing quotes)",
                delta_color="off")
    c[3].metric("Open premium collected", f"${o['open_premium_collected']:,.0f}")
    c = st.columns(4)
    dep = o["capital_deployed"]
    c[0].metric("Capital deployed", "-" if dep is None else f"${dep:,.0f}")
    util = o["capital_utilization"]
    c[1].metric("Capital utilization", "-" if util is None else f"{util:.0%}")
    c[2].metric("Signals today", o["today"]["signals"])
    c[3].metric("Orders today", o["today"]["orders"])
    if util is not None:
        st.progress(min(util, 1.0))
        if util >= 0.8:
            st.warning("Capital utilization is above 80% of the configured limit.")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Agent status")
        agents = fetch("/agents")
        if agents:
            df = pd.DataFrame([{
                "": DOT.get(a["status"], "⚪") + (" ⏱" if a["stale"] else ""), "agent": a["agent"], "status": a["status"],
                "last heartbeat": a["last_heartbeat"], "last run": a["last_run"], "restarts": a["restarts"],
                "last error": a["last_error"] or ""} for a in agents])
            st.dataframe(df, hide_index=True, use_container_width=True)
            if any(a["stale"] for a in agents):
                st.warning("⏱ = no heartbeat recently.")
    with right:
        st.subheader("Summary")
        n = fetch("/narrative")
        if n:
            st.text(n["text"])

    st.subheader("Live events")
    feed = _feed()
    events = feed.snapshot(15) if feed else []
    if not events:
        st.caption("No events yet (connect the API stream or wait for the next market-data poll).")
    for e in events:
        d, name = e["data"], e["event"]
        if name == "Heartbeat":
            continue
        txt = d.get("reason") or d.get("detail") or d.get("verdict") or ""
        st.caption(f"`{d.get('ts', '')[11:19]}` **{name}** {d.get('symbol', '')} {d.get('status', d.get('verdict', ''))} {txt}"[:240])


def render() -> None:
    st.header("Overview")
    live(_body)
