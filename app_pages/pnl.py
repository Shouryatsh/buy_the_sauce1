from datetime import datetime, timedelta

import streamlit as st

import wheel_config as wc
import wheel_db
import wheel_ui as ui

client = ui.get_client()
if client is None:
    st.info("Connect to IBKR in the left rail to see trade history.", icon=":material/link:")
    st.stop()

days = st.slider("Lookback window (days)", 7, 90, 30)
start_date = (datetime.now().date() - timedelta(days=days)).isoformat()

stats = client.get_trade_stats(days=days)
manual = wheel_db.get_manual_realized_pnl_since(start_date)
combined_realized_pnl = stats.realized_pnl + manual["realized_pnl"]

with st.container(horizontal=True):
    st.metric("Realized P&L", f"${combined_realized_pnl:,.2f}", border=True)
    st.metric("Win rate", f"{stats.win_rate:.0%}" if stats.win_rate is not None else "N/A", border=True)
    st.metric("Avg premium / contract",
              f"${stats.avg_premium_per_contract:,.2f}" if stats.avg_premium_per_contract else "N/A", border=True)
    st.metric("Avg days to close",
              f"{stats.avg_days_to_close:.1f}" if stats.avg_days_to_close is not None else "N/A", border=True)
if manual["trade_count"]:
    st.caption(
        f"\\${stats.realized_pnl:,.2f} from {stats.trade_count} IBKR-visible fill(s) "
        f"+ \\${manual['realized_pnl']:,.2f} from {manual['trade_count']} manually-logged trade(s)"
    )

progress = min(combined_realized_pnl / wc.TARGET_MONTHLY_INCOME, 1.0) if wc.TARGET_MONTHLY_INCOME else 0.0
st.progress(max(progress, 0.0),
            text=f"\\${combined_realized_pnl:,.0f} / \\${wc.TARGET_MONTHLY_INCOME:,.0f} monthly target")
st.caption(
    "IBKR's execution API has retained essentially nothing beyond the current session in testing against "
    "this account, and a worthless expiration never generates an execution at all. Use 'Log a closed "
    "trade' below for anything IBKR's API didn't surface."
)

with st.expander("Log a closed trade IBKR didn't show (expiration, assignment, etc.)", icon=":material/edit_note:"):
    with st.form("manual_trade_form", clear_on_submit=True):
        fc1, fc2, fc3 = st.columns(3)
        m_symbol = fc1.text_input("Symbol").upper().strip()
        m_sec_type = fc2.selectbox("Type", ["Put", "Call"])
        m_contracts = fc3.number_input("Contracts", min_value=1, value=1, step=1)

        fc4, fc5, fc6 = st.columns(3)
        m_strike = fc4.number_input("Strike", min_value=0.0, value=0.0, step=0.5)
        m_premium = fc5.number_input("Premium collected per contract ($/share)", min_value=0.0, value=0.0,
                                     step=0.01, format="%.2f")
        m_outcome = fc6.selectbox("Outcome", ["Expired worthless", "Assigned", "Bought to close"])

        m_close_price = st.number_input("Price paid to close ($/share, only for 'Bought to close')",
                                        min_value=0.0, value=0.0, step=0.01, format="%.2f")
        m_closed_date = st.date_input("Date closed", value=datetime.now().date())
        m_notes = st.text_input("Notes (optional)")

        if st.form_submit_button("Log trade"):
            if not m_symbol or m_premium <= 0:
                st.error("Symbol and premium collected are required.")
            else:
                if m_outcome == "Bought to close":
                    realized = (m_premium - m_close_price) * 100 * m_contracts
                else:
                    # Expired worthless or assigned: the premium is kept in full either way.
                    realized = m_premium * 100 * m_contracts
                action = {"Expired worthless": "EXPIRED_WORTHLESS", "Assigned": "ASSIGNED",
                          "Bought to close": "BOUGHT_TO_CLOSE"}[m_outcome]
                wheel_db.record_manual_trade(
                    symbol=m_symbol, sec_type="OPT", action=action, quantity=m_contracts,
                    premium_per_contract=m_premium, realized_pnl=realized,
                    closed_date=m_closed_date.isoformat(),
                    notes=f"{m_sec_type} ${m_strike:g}" + (f" — {m_notes}" if m_notes else ""),
                )
                st.success(f"Logged {m_symbol}: \\${realized:,.2f} realized.")
                st.rerun()
