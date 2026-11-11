import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import wheel_ui as ui

client = ui.get_client()
if client is None:
    st.info("Connect to IBKR in the left rail to see your live account.", icon=":material/link:")
    st.stop()

s = client.get_account_summary()
nlv = s.net_liquidation or 1.0
positions = client.get_positions()
stocks = [p for p in positions if p.sec_type == "STK"]
options = [p for p in positions if p.sec_type == "OPT"]
short_puts = [p for p in options if p.right == "P" and p.quantity < 0]
put_notional = sum((p.strike or 0) * 100 * abs(p.quantity) for p in short_puts)
unrealized = sum(p.unrealized_pnl or 0 for p in positions)

# --- KPI row ---------------------------------------------------------------
with st.container(horizontal=True):
    st.metric("Net liquidation", f"${s.net_liquidation:,.0f}", border=True)
    st.metric("Unrealized P&L", f"${unrealized:,.0f}", f"{unrealized / nlv:+.1%} of net liq", border=True)
    st.metric("Cash", f"${s.total_cash:,.0f}", border=True)
    st.metric("Buying power", f"${s.buying_power:,.0f}", border=True)
    st.metric("Short-put notional", f"${put_notional:,.0f}", f"{put_notional / nlv:.0%} of net liq",
              delta_color="inverse", delta_arrow="off", border=True)

left, right = st.columns([2, 1], gap="medium")

with left:
    # --- Exposure heatmap: tile size = value, colour = unrealized P&L % ------
    with st.container(key="glass_heatmap"):
        st.subheader("Position heatmap", anchor=False)
        rows = []
        for p in stocks:
            cost = p.avg_cost * p.quantity
            rows.append(dict(name=p.symbol, value=abs(p.market_value or 0),
                             pnl_pct=(p.unrealized_pnl or 0) / cost if cost else 0, kind="Shares"))
        for p in short_puts:
            notional = (p.strike or 0) * 100 * abs(p.quantity)
            rows.append(dict(name=f"{p.symbol} {p.strike:g}P {p.expiry[4:6]}/{p.expiry[6:]}", value=notional,
                             pnl_pct=(p.unrealized_pnl or 0) / notional if notional else 0, kind="Short puts (notional)"))
        if rows:
            kinds = sorted({r["kind"] for r in rows})
            # Leaf tiles, one parent per kind, and an explicit root so plotly doesn't
            # draw its grey implicit root ("remainder" = parents carry no own value).
            labels = [r["name"] for r in rows] + kinds + ["Portfolio"]
            parents = [r["kind"] for r in rows] + ["Portfolio"] * len(kinds) + [""]
            values = [r["value"] for r in rows] + [0] * (len(kinds) + 1)
            pnl = [r["pnl_pct"] for r in rows] + [0] * (len(kinds) + 1)
            fig = go.Figure(go.Treemap(
                labels=labels, parents=parents, values=values, branchvalues="remainder",
                root_color="rgba(0,0,0,0)", pathbar=dict(visible=False), tiling=dict(pad=3),
                customdata=[[v] for v in pnl],
                marker=dict(colors=pnl, colorscale=[[0, ui.RED], [0.5, "#1E2550"], [1, ui.TEAL]],
                            cmid=0, line=dict(width=1, color="rgba(255,255,255,0.08)")),
                texttemplate="<b>%{label}</b><br>$%{value:,.0f}<br>%{customdata[0]:+.1%}",
                hovertemplate="<b>%{label}</b><br>Value $%{value:,.0f}<br>P&L %{customdata[0]:+.2%}<extra></extra>",
            ))
            fig.update_layout(**ui.plotly_layout(height=360))
            st.plotly_chart(fig, config={"displayModeBar": False})
        else:
            st.caption("No positions.")

    # --- Positions with 3-month sparklines (slowest part: price history) ----
    with st.container(key="glass_positions"):
        st.subheader("Positions", anchor=False)
        with st.skeleton(height=320):
            table = []
            for p in stocks:
                hist, _src = ui.price_history(p.symbol)
                spark = hist["Close"].tail(63).round(2).tolist() if hist is not None and not hist.empty else []
                table.append({
                    "Symbol": p.symbol, "Qty": p.quantity, "Avg cost": p.avg_cost, "Last": p.market_price,
                    "Value": p.market_value, "Unrealized": p.unrealized_pnl,
                    "% of acct": (p.market_value or 0) / nlv, "3 months": spark,
                })
            st.dataframe(
                pd.DataFrame(table), hide_index=True,
                column_config={
                    "Qty": st.column_config.NumberColumn(format="%d"),
                    "Avg cost": st.column_config.NumberColumn(format="$%.2f"),
                    "Last": st.column_config.NumberColumn(format="$%.2f"),
                    "Value": st.column_config.NumberColumn(format="$%,.0f"),
                    "Unrealized": st.column_config.NumberColumn(format="$%,.0f"),
                    "% of acct": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=0.3),
                    "3 months": st.column_config.AreaChartColumn(width="medium"),
                },
            )
            if options:
                st.markdown("**Options**")
                st.dataframe(pd.DataFrame([{
                    "Contract": f"{p.symbol} {p.strike:g}{p.right} {p.expiry}", "Qty": p.quantity,
                    "Premium received": -p.quantity * p.avg_cost if p.quantity < 0 else None,
                    "Value now": p.market_value, "Unrealized": p.unrealized_pnl,
                } for p in options]), hide_index=True, column_config={
                    "Qty": st.column_config.NumberColumn(format="%d"),
                    "Premium received": st.column_config.NumberColumn(format="$%,.0f"),
                    "Value now": st.column_config.NumberColumn(format="$%,.0f"),
                    "Unrealized": st.column_config.NumberColumn(format="$%,.0f"),
                })

with right:
    # --- Margin gauge -------------------------------------------------------
    with st.container(key="glass_margin"):
        st.subheader("Margin usage", anchor=False)
        pct = 100 * s.margin_used / nlv
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=pct, number=dict(suffix="%", valueformat=".0f"),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor="rgba(255,255,255,0.3)"),
                bar=dict(color=ui.VIOLET, thickness=0.28),
                bgcolor="rgba(255,255,255,0.04)", borderwidth=0,
                steps=[dict(range=[0, 30], color="rgba(45,212,191,0.18)"),
                       dict(range=[30, 50], color="rgba(251,191,36,0.18)"),
                       dict(range=[50, 100], color="rgba(251,113,133,0.18)")],
            ),
        ))
        gauge.update_layout(**ui.plotly_layout(height=220))
        st.plotly_chart(gauge, config={"displayModeBar": False})
        st.caption(f"Maintenance margin \\${s.margin_used:,.0f} of \\${s.net_liquidation:,.0f} net liq.")

    # --- Exposure by asset class -------------------------------------------
    with st.container(key="glass_assetclass"):
        st.subheader("Exposure by type", anchor=False)
        stock_val = sum(p.market_value or 0 for p in stocks)
        st.bar_chart(
            pd.DataFrame({"Type": ["Shares", "Short-put notional", "Cash"],
                          "USD": [stock_val, put_notional, max(s.total_cash, 0)]}),
            x="Type", y="USD", horizontal=True, height=180,
        )

    # --- Alerts -------------------------------------------------------------
    with st.container(key="glass_alerts"):
        st.subheader("Alerts", anchor=False)
        ui.render_alerts(ui.compute_alerts(client))

    with st.container(key="glass_orders"):
        st.subheader("Open orders", anchor=False)
        orders = client.get_open_orders()
        if orders:
            st.dataframe(pd.DataFrame(orders), hide_index=True)
        else:
            st.caption("No open orders.")
