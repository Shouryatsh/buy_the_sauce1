"""IV screener: IV percentile and RV percentile for every stock in the universe, each against its OWN history
(McMillan) — low-IV watchlist, high-IV watchlist, and IV-vs-RV side by side."""

import streamlit as st

import wheel_ui as ui
from ivscreen import ui as iui

client = ui.get_client()
holdings = sorted({p.symbol for p in client.get_positions() if p.sec_type == "STK"}) if client else []
cfg = iui.get_config()

st.caption("Where is each stock's implied volatility, and its realized volatility, compared with its own past? "
           "Percentiles only — raw IV and RV are never compared with each other. " + iui.DISCLAIMER)

df = iui.load_evaluated(cfg)
iui.render_coverage(df, cfg)

iui.render_scan_controls(client, holdings)

if df.empty:
    st.caption("Results appear here after the first scan.")
else:
    view = iui.render_filters(df, set(holdings))
    iui.render_summary(view)
    iui.render_low(view, cfg)
    iui.render_high(view, cfg)
    iui.render_comparison(view, cfg)
    iui.render_availability(view)

iui.render_settings()
