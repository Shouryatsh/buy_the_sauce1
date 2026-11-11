import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import wheel_config as wc
import wheel_ui as ui
from wheel_options import fetch_put_candidates, realized_volatility

profiles = ui.profiles_or_none()
client = ui.get_client()
if profiles is None:
    ui.need_screen("need_screen_radar")
    st.stop()
passing_symbols = [p.symbol for p in profiles if p.passes]
if not passing_symbols:
    st.info("No symbols currently pass the fundamentals screen.")
    st.stop()
if client is None:
    st.info("Connect to IBKR in the left rail to pull option chains.", icon=":material/link:")
    st.stop()

symbol = st.selectbox("Symbol", passing_symbols)
hist, _src = ui.price_history(symbol)
if hist is None or hist.empty:
    st.error(f"No price history for {symbol}.")
    st.stop()

underlying_price = float(hist["Close"].iloc[-1])
rv = realized_volatility(hist)
st.caption(f"Underlying price: \\${underlying_price:.2f}  |  Realized vol ({wc.RICHNESS_LOOKBACK_DAYS}d): {rv:.1%}"
           if rv else "Realized vol unavailable.")

candidates = fetch_put_candidates(client, symbol, underlying_price, rv)
if not candidates:
    st.warning("No candidates found within the DTE / delta filters.")
    st.stop()

with st.container(key="glass_radar"):
    st.dataframe(pd.DataFrame([c.__dict__ for c in candidates]), hide_index=True)
    fig = go.Figure(go.Scatter(
        x=[c.delta for c in candidates],
        y=[c.market_iv or c.realized_vol or 0 for c in candidates],
        mode="markers",
        marker=dict(size=12, color=[c.richness_ratio or 1.0 for c in candidates],
                    colorscale=[[0, ui.TEAL], [0.5, ui.VIOLET], [1, ui.MAGENTA]], showscale=True,
                    colorbar=dict(title="Richness"), line=dict(width=1, color="rgba(255,255,255,0.3)")),
        text=[f"{c.symbol} {c.strike} {c.expiry}" for c in candidates],
    ))
    fig.update_layout(**ui.plotly_layout(height=400, xaxis_title="Delta", yaxis_title="IV (or realized-vol proxy)"))
    st.plotly_chart(fig)

st.caption("Full IV smile/skew and 25-delta risk-reversal diagnostics: Phase 2 (wheel_skew.py). "
           "Merton jump-diffusion fair-value IV: Phase 3 (wheel_jump_diffusion.py).")
