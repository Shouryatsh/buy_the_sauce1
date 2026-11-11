import pandas as pd
import streamlit as st

import wheel_config as wc
import wheel_db
import wheel_ui as ui
from stockscreen import ui as sui
from wheel_watchlist import WHEEL_WATCHLIST, load_exclusions, save_exclusions

client = ui.get_client()
holdings = sorted({p.symbol for p in client.get_positions() if p.sec_type == "STK"}) if client else []
cfg = sui.get_config()
sui.maybe_auto_scan(holdings)

st.caption("Financially strong companies with consistent free cash flow, then where their price sits against its own "
           "history and what the trend says. " + sui.DISCLAIMER)

# --- results (top: coverage, summary) ----------------------------------------------
df, details = sui.load_evaluated(cfg)
sui.render_coverage(df)
sui.render_summary(df)

# --- scan controls ------------------------------------------------------------------
sui.render_scan_controls(holdings)

# --- filters, table, detail ---------------------------------------------------------
if df.empty:
    st.caption("Results appear here after the first scan.")
else:
    shown = sui.add_display_columns(df, details)
    filtered = sui.render_filters(shown)
    selected = sui.render_table(filtered)
    if selected:
        sui.render_detail(selected, details, cfg)

# --- alerts, thresholds ---------------------------------------------------------------
sui.render_alerts()
sui.render_profile()

# --- manage universe (manual exclusions) ------------------------------------------------
excluded = load_exclusions()
st.caption(f"Universe: {len(WHEEL_WATCHLIST)} tickers ({len(excluded)} excluded by you), "
           f"price <= \\${wc.MAX_ASSIGNMENT_PRICE} in the wheel quality screen below. Source: SEC EDGAR.")

full = ui._screen(tuple(WHEEL_WATCHLIST)) if st.session_state.get("screen_on") else None
by_sym = {p.symbol: p for p in full} if full else {}
names = ui.company_names()

with st.expander(f"Manage universe — {len(WHEEL_WATCHLIST)} tickers, {len(excluded)} excluded",
                 icon=":material/playlist_remove:", expanded=False):
    st.caption("Tick **Exclude** to drop a ticker from the screeners, Recommendations and symbol lists, then save. "
               "Exclusions are kept in wheel_exclusions.json, so they survive a re-scan of the universe. "
               + ("" if full else "Wheel quality-screen results appear here once that screen has run."))
    show = st.segmented_control("Show", ["All", "Passed", "Failed", "Excluded"], default="All", key="uni_show")
    rows = []
    for t in sorted(WHEEL_WATCHLIST):
        p = by_sym.get(t)
        rows.append({
            "Exclude": t.upper() in excluded, "Ticker": t,
            "Company": names.get(t.upper(), names.get(t.upper().replace(".", "-"), "")),
            "Screen": "—" if p is None else ("Pass" if p.passes else "Fail"),
            "Quality": None if p is None else p.quality_score,
            "Price": None if p is None else p.price,
            "Why it failed": "" if p is None or p.passes else "; ".join(p.fail_reasons),
        })
    uni = pd.DataFrame(rows)
    if show == "Passed":
        uni = uni[uni["Screen"] == "Pass"]
    elif show == "Failed":
        uni = uni[uni["Screen"] == "Fail"]
    elif show == "Excluded":
        uni = uni[uni["Exclude"]]
    edited = st.data_editor(
        uni, hide_index=True, key=f"uni_editor_{show}", height=420,
        disabled=["Ticker", "Company", "Screen", "Quality", "Price", "Why it failed"],
        column_config={
            "Exclude": st.column_config.CheckboxColumn(help="Remove this ticker from the screener universe"),
            "Quality": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
            "Price": st.column_config.NumberColumn(format="$%.2f"),
        })
    shown_syms = set(edited["Ticker"].str.upper())
    new_excluded = (excluded - shown_syms) | set(edited.loc[edited["Exclude"], "Ticker"].str.upper())
    added, restored = new_excluded - excluded, excluded - new_excluded
    if added or restored:
        st.info(f"Unsaved: exclude {', '.join(sorted(added)) or '—'}; restore {', '.join(sorted(restored)) or '—'}.",
                icon=":material/edit:")
    if st.button("Save exclusions", icon=":material/save:", type="primary", disabled=not (added or restored)):
        save_exclusions(new_excluded)
        st.toast(f"Saved — {len(new_excluded)} ticker(s) excluded.", icon=":material/check_circle:")
        st.rerun()

# --- the original wheel quality screen (kept: Recommendations and Options radar use it) -------
with st.expander("Wheel quality screen (used by Recommendations and Options radar)", icon=":material/filter_alt:"):
    profiles = ui.profiles_or_none()
    if profiles is None:
        ui.need_screen("need_screen_screener")
    else:
        unreliable = [p.symbol for p in profiles if not p.data_reliable]
        if unreliable:
            st.warning(f"Data looked incomplete for: {', '.join(unreliable)} — scored anyway but treat with caution.")
        wdf = pd.DataFrame([{
            "symbol": p.symbol, "passes": p.passes, "quality_score": p.quality_score, "price": p.price,
            "fcf_yield": p.fcf_yield, "net_margin": p.net_margin, "net_margin_trend": p.net_margin_trend,
            "fcf_trend_score": p.fcf_trend_score, "roe": p.return_on_equity, "revenue_growth": p.revenue_growth,
            "is_elite": p.is_elite, "data_reliable": p.data_reliable, "fail_reasons": "; ".join(p.fail_reasons),
        } for p in profiles])
        st.dataframe(wdf, hide_index=True, column_config={
            "quality_score": st.column_config.ProgressColumn("quality", min_value=0, max_value=100, format="%.0f"),
            "price": st.column_config.NumberColumn(format="$%.2f"),
            "fcf_yield": st.column_config.NumberColumn(format="percent"),
            "net_margin": st.column_config.NumberColumn(format="percent"),
            "roe": st.column_config.NumberColumn(format="percent"),
            "revenue_growth": st.column_config.NumberColumn(format="percent"),
        })
        if st.button("Save this snapshot to history", icon=":material/save:"):
            wheel_db.save_screen_snapshot(profiles)
            st.success("Saved.")
