import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import wheel_analyst as wa
import wheel_ui as ui
from wheel_watchlist import active_watchlist

client = ui.get_client()
held = sorted({p.symbol for p in client.get_positions()}) if client else []
core = held + [s for s in wa.WATCHLIST if s not in held]
# Full screener universe too: only the selected symbol is ever loaded, so a long list costs nothing.
symbols = core + sorted(set(active_watchlist()) - set(core))

with st.container(horizontal=True, vertical_alignment="bottom", key="glass_chartbar"):
    symbol = st.selectbox("Symbol", symbols, accept_new_options=True, key="chart_symbol", width=180)
    rng = st.segmented_control("Range", ["3M", "6M", "1Y", "2Y"], default="6M", key="chart_range")
    overlays = st.pills("Overlays", ["EMA 20", "EMA 50", "VWAP", "Volume"],
                        selection_mode="multi", default=["EMA 20", "EMA 50", "Volume"], key="chart_overlays")

symbol = (symbol or "").upper().strip()
if not symbol:
    st.stop()

hist, src = ui.price_history(symbol, period_years=2)
if hist is None or hist.empty:
    st.error(f"No price history for {symbol}.", icon=":material/error:")
    st.stop()

days = {"3M": 63, "6M": 126, "1Y": 252, "2Y": 504}[rng or "6M"]
full = hist.copy()
full["EMA 20"] = full["Close"].ewm(span=20, adjust=False).mean()
full["EMA 50"] = full["Close"].ewm(span=50, adjust=False).mean()
df = full.tail(days).copy()
# VWAP anchored to the start of the visible range (daily bars, typical price).
tp = (df["High"] + df["Low"] + df["Close"]) / 3
df["VWAP"] = (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()
ret = full["Close"].pct_change()
df["Vol 20d"] = (ret.rolling(20).std() * (252 ** 0.5)).tail(days)

last, prev = float(df["Close"].iloc[-1]), float(df["Close"].iloc[-2])
with st.container(horizontal=True):
    st.metric(f"{symbol} last close", f"${last:,.2f}", f"{(last / prev - 1):+.2%}", border=True,
              chart_data=df["Close"].tail(30).round(2).tolist(), chart_type="area")
    st.metric("Range high / low", f"\\${df['High'].max():,.2f} / \\${df['Low'].min():,.2f}", border=True)
    st.metric("20d realized vol", f"{df['Vol 20d'].iloc[-1]:.1%}" if pd.notna(df["Vol 20d"].iloc[-1]) else "n/a",
              border=True, chart_data=df["Vol 20d"].dropna().tail(60).round(4).tolist(), chart_type="line")

show_vol = "Volume" in (overlays or [])
fig = make_subplots(rows=3 if show_vol else 2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                    row_heights=[0.62, 0.18, 0.20] if show_vol else [0.78, 0.22])
fig.add_trace(go.Candlestick(
    x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name=symbol,
    increasing=dict(line=dict(color=ui.TEAL), fillcolor="rgba(45,212,191,0.55)"),
    decreasing=dict(line=dict(color=ui.MAGENTA), fillcolor="rgba(244,114,182,0.55)"),
), row=1, col=1)
for name, color in (("EMA 20", ui.VIOLET), ("EMA 50", ui.AMBER), ("VWAP", "#38BDF8")):
    if name in (overlays or []):
        fig.add_trace(go.Scatter(x=df.index, y=df[name], name=name, mode="lines",
                                 line=dict(color=color, width=1.6)), row=1, col=1)
row = 2
if show_vol:
    colors = [ui.TEAL if c >= o else ui.MAGENTA for o, c in zip(df["Open"], df["Close"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker=dict(color=colors, opacity=0.45)),
                  row=2, col=1)
    row = 3
fig.add_trace(go.Scatter(x=df.index, y=df["Vol 20d"], name="20d vol", mode="lines", fill="tozeroy",
                         line=dict(color=ui.VIOLET, width=1.4), fillcolor="rgba(167,139,250,0.15)"),
              row=row, col=1)

layout = ui.plotly_layout(height=640, hovermode="x unified", showlegend=True,
                          legend=dict(orientation="h", y=1.04, x=0, bgcolor="rgba(0,0,0,0)"),
                          xaxis_rangeslider_visible=False)
for axis in ("xaxis", "yaxis"):  # per-subplot axes are styled below
    del layout[axis]
fig.update_layout(**layout)
grid = "rgba(255,255,255,0.06)" if st.context.theme.type != "light" else "rgba(20,26,51,0.08)"
fig.update_xaxes(gridcolor=grid, showspikes=True, spikemode="across", spikesnap="cursor",
                 spikecolor="rgba(167,139,250,0.6)", spikethickness=1,
                 rangebreaks=[dict(bounds=["sat", "mon"])])
fig.update_yaxes(gridcolor=grid, showspikes=True, spikecolor="rgba(167,139,250,0.6)", spikethickness=1)
fig.update_yaxes(tickformat=".0%", row=row, col=1)

with st.container(key="glass_chart"):
    st.plotly_chart(fig, config={"displayModeBar": True, "displaylogo": False,
                                 "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
    st.caption(f"Daily bars from {src}. VWAP is anchored to the start of the selected range. "
               "Hover any bar for exact values.")
