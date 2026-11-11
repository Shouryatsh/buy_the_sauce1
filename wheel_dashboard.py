"""
wheel_dashboard.py — entry point for the options wheel dashboard (read-only).

Run with:
    streamlit run wheel_dashboard.py

Layout: a vertical navigation rail on the left (st.navigation), a glass top
bar with the account summary, and one page at a time in the centre. Pages
live in app_pages/; shared session/data/style helpers in wheel_ui.py.

Requires TWS or IB Gateway running locally with API access enabled (see
README.md). The IBKR socket is opened with readonly=True: this dashboard
never places, modifies or cancels an order. The Order Ticket page only
previews an order for you to enter in TWS yourself.
"""

from __future__ import annotations

import logging

import streamlit as st

import wheel_config as wc
import wheel_ui as ui

logging.basicConfig(level=logging.WARNING)

st.set_page_config(page_title="Wheel Desk", page_icon=":material/donut_large:", layout="wide")
ui.init_state()
ui.inject_css()

page = st.navigation(
    {
        "": [
            st.Page("app_pages/overview.py", title="Overview", icon=":material/space_dashboard:", default=True),
            st.Page("app_pages/chart.py", title="Chart", icon=":material/candlestick_chart:"),
            st.Page("app_pages/volatility.py", title="Volatility & events", icon=":material/ssid_chart:"),
            st.Page("app_pages/analyst.py", title="Wheel analyst", icon=":material/target:"),
            st.Page("app_pages/order_ticket.py", title="Order ticket", icon=":material/receipt_long:"),
        ],
        "Portfolio": [
            st.Page("app_pages/pnl.py", title="Monthly P&L", icon=":material/payments:"),
            st.Page("app_pages/exposure.py", title="Exposure", icon=":material/shield:"),
            st.Page("app_pages/trade_log.py", title="Trade log", icon=":material/history:"),
        ],
        "Research": [
            st.Page("app_pages/screener.py", title="Fundamentals", icon=":material/filter_alt:"),
            st.Page("app_pages/iv_screener.py", title="IV screener", icon=":material/vertical_align_center:"),
            st.Page("app_pages/vol_surface.py", title="Vol surface", icon=":material/layers:"),
            st.Page("app_pages/radar.py", title="Options radar", icon=":material/radar:"),
            st.Page("app_pages/recommendations.py", title="Wheel recommendations", icon=":material/leaderboard:"),
            st.Page("app_pages/straddle_recommendations.py", title="Straddle recommendations", icon=":material/swap_vert:"),
            st.Page("app_pages/premia_research.py", title="Premia research", icon=":material/science:"),
        ],
    },
    position="sidebar",
    expanded=True,          # 15 pages: without this Streamlit hides all but ten behind "View 5 more"
)

# ---------------------------------------------------------------------------
# Left rail: connection controls under the navigation. The connection is
# explicit (not dialed on every rerun) so widget clicks don't each trigger
# a multi-second connect attempt when TWS is down.
# ---------------------------------------------------------------------------
with st.sidebar:
    st.caption(f"LIVE IBKR · port {wc.IBKR_LIVE_PORT} · read-only")
    with st.container(horizontal=True):
        st.button("Connect", on_click=ui.connect, icon=":material/link:", type="primary", width="stretch")
        st.button("Disconnect", on_click=ui.disconnect, icon=":material/link_off:", width="stretch")
    client = ui.get_client()
    if client is not None:
        st.success("Connected", icon=":material/check_circle:")
    elif st.session_state.ibkr_error:
        st.error(f"Connection failed: {st.session_state.ibkr_error}", icon=":material/error:")
    st.caption(
        "Real-time option greeks need an OPRA subscription; without it option "
        "quotes are delayed — verify before trading."
    )

ui.top_bar(client)
st.title(page.title)
page.run()
