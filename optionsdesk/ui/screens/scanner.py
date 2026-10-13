from __future__ import annotations

import pandas as pd
import streamlit as st

from ..common import fetch, live


def _body() -> None:
    with st.expander("Filters", expanded=True):
        a = st.columns(4)
        strategy = a[0].selectbox("Strategy", ["all", "wheel_put", "wheel_call", "straddle"], key="sc_strategy")
        dl, dh = a[1].slider("|Delta|", 0.0, 1.0, (0.20, 0.35), 0.01, key="sc_delta")
        tl, th = a[2].slider("DTE", 0, 120, (7, 60), key="sc_dte")
        iv = a[3].slider("Min IV rank", 0, 100, 0, key="sc_ivr")
        b = st.columns(2)
        spread = b[0].slider("Max bid/ask spread (% of mid)", 1, 100, 15, key="sc_spread") / 100
        oi = b[1].number_input("Min open interest", 0, 100000, 100, step=50, key="sc_oi")
    # delta does not apply to straddle rows (they carry no delta), so only filter them by the other fields
    rows = fetch("/scanner", strategy=strategy, min_delta=dl if strategy != "straddle" else None,
                 max_delta=dh if strategy != "straddle" else None, min_dte=tl, max_dte=th, min_iv_rank=iv,
                 max_spread_pct=spread, min_open_interest=oi)
    if rows is None:
        return
    if strategy == "all":
        rows = [r for r in rows if r["strategy"] == "straddle" or (dl <= (r["delta"] or 0) <= dh)]
    if not rows:
        st.info("No candidates match these filters.")
        return
    df = pd.DataFrame(rows)[["strategy", "symbol", "right", "strike", "expiry", "dte", "delta", "mid", "bid", "ask",
                             "spread_pct", "open_interest", "iv", "iv_rank", "annualized_roc", "score", "quality"]]
    for col in ("spread_pct", "iv", "annualized_roc"):
        df[col] = df[col] * 100.0  # shown with a % sign
    st.caption(f"{len(df)} candidates. Click a column header to sort. Wheel score = annualized return on strike; "
               "straddle score = implied move as % of spot. Quality 'estimated' = modelled, not broker data.")
    st.dataframe(df, hide_index=True, width="stretch", column_config={
        "delta": st.column_config.NumberColumn(format="%.2f"), "mid": st.column_config.NumberColumn(format="%.2f"),
        "spread_pct": st.column_config.NumberColumn("spread", format="%.1f%%", help="of mid"),
        "iv": st.column_config.NumberColumn(format="%.1f%%"),
        "iv_rank": st.column_config.NumberColumn(format="%.0f"),
        "annualized_roc": st.column_config.NumberColumn("ann. ROC", format="%.1f%%"),
        "score": st.column_config.NumberColumn(format="%.3f"),
    })


def render() -> None:
    st.header("Opportunity scanner")
    live(_body)
