import pandas as pd
import streamlit as st

import wheel_db
import wheel_ui as ui

client = ui.get_client()
if client is not None:
    try:
        n = client.archive_recent_fills()
        st.caption(f"Synced with IBKR: {n} recent fill(s) reported. New fills are added the moment they happen "
                   "while the dashboard is connected.")
    except Exception as exc:
        st.warning(f"Could not sync fills from IBKR: {exc}", icon=":material/sync_problem:")
else:
    st.caption("Not connected — showing the local archive only. Connect to sync today's fills.")

log = wheel_db.get_trade_log()
if not log:
    st.caption(
        "Nothing archived yet. Fills are archived automatically while connected; anything IBKR's API "
        "didn't show (e.g. an expiration) needs the manual-entry form on the Monthly P&L page."
    )
    st.stop()

df = pd.DataFrame(log)
df["when"] = pd.to_datetime(df["exec_time"].fillna(df["closed_date"]).fillna(df["logged_at"]), utc=True, errors="coerce")
df["contract"] = [
    f"{r.symbol} {r.strike:g}{r.opt_right} {r.expiry}" if r.sec_type == "OPT" and pd.notna(r.strike) else r.symbol
    for r in df.itertuples()
]
side = df["action"].map({"SLD": "Sell", "BOT": "Buy"}).fillna(df["action"])
mult = df["multiplier"].fillna(df["sec_type"].map({"OPT": 100.0}).fillna(1.0))
df["cash"] = [(1 if s == "Sell" else -1 if s == "Buy" else 0) * q * p * m
              for s, q, p, m in zip(side, df["quantity"], df["price"], mult)]
view = pd.DataFrame({
    "When (ET)": df["when"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d %H:%M"),
    "Contract": df["contract"], "Side": side, "Qty": df["quantity"], "Price": df["price"],
    "Cash flow": df["cash"], "Commission": df["commission"], "Source": df["source"], "Notes": df["notes"],
})
with st.container(key="glass_tradelog"):
    st.dataframe(view, hide_index=True, column_config={
        "Qty": st.column_config.NumberColumn(format="%d"),
        "Price": st.column_config.NumberColumn(format="$%.2f"),
        "Cash flow": st.column_config.NumberColumn(format="$%,.2f", help="Premium received (+) or paid (−), before commission"),
        "Commission": st.column_config.NumberColumn(format="$%.2f"),
    })
    st.download_button("Download CSV", df.drop(columns=["when"]).to_csv(index=False), file_name="trade_log.csv",
                       mime="text/csv", icon=":material/download:")
st.caption("`ibkr` = archived from IBKR's execution feed (one row per execution); `manual` = logged on Monthly P&L. "
           "IBKR's API only keeps about a week of executions, so the archive is the long-term record.")
