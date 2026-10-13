from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import theme
from ..common import badge, fetch, live, post


def _body() -> None:
    h = fetch("/health")
    if h is None:
        return
    live_mode = h["mode"] == "live"

    st.subheader("Kill switch")
    ks = fetch("/kill-switch")
    if ks and ks["engaged"]:
        st.markdown(badge("ENGAGED", theme.RED) + f" &nbsp; {ks['reason']}", unsafe_allow_html=True)
        st.error("No orders will be sent until the kill switch is released.")
        if st.button("Release kill switch", key="ks_release"):
            if post("/kill-switch", {"engaged": False, "reason": "released from dashboard"}, "Released."):
                st.rerun()
    else:
        st.markdown(badge("released", theme.GREEN), unsafe_allow_html=True)
        reason = st.text_input("Reason", "manual stop", key="ks_reason")
        if st.button("ENGAGE KILL SWITCH", type="primary", key="ks_engage"):
            if post("/kill-switch", {"engaged": True, "reason": reason}, "Engaged: pending orders cancelled."):
                st.rerun()

    st.subheader("Orders awaiting confirmation")
    orders = [o for o in (fetch("/orders", limit=200) or []) if not o["paper"]]
    pending = [o for o in orders if o["status"] == "pending_confirmation"]
    if not pending:
        st.caption("None. Live orders always wait here for your confirmation.")
    ack = False
    if pending and live_mode:
        st.error("LIVE trading: confirming sends a real order.")
        ack = st.checkbox("I understand this is a LIVE order", key="ack_live")
    for o in pending:
        with st.container(border=True):
            st.write(f"**{o['symbol']}** {o['strategy']} - {o['reason']}")
            c = st.columns([1, 1, 4])
            if c[0].button("Confirm", key=f"c_{o['order_id']}", disabled=live_mode and not ack):
                if post(f"/orders/{o['order_id']}/confirm", {"acknowledge_live": ack}, "Confirmed."):
                    st.rerun()
            if c[1].button("Reject", key=f"r_{o['order_id']}"):
                if post(f"/orders/{o['order_id']}/reject", None, "Rejected."):
                    st.rerun()

    st.subheader("Order log")
    if orders:
        df = pd.DataFrame(orders)[["created_at", "symbol", "strategy", "status", "fill_price", "detail", "reason", "correlation_id"]]
        st.dataframe(df, hide_index=True, width="stretch")
    else:
        st.caption("No orders yet.")


def render() -> None:
    st.header("Orders & controls")
    live(_body)
