import pandas as pd
import streamlit as st

import wheel_config as wc
import wheel_ui as ui
from wheel_options import realized_volatility
from wheel_risk import compute_exposure

client = ui.get_client()
if client is None:
    st.info("Connect to IBKR in the left rail to compute exposure.", icon=":material/link:")
    st.stop()

capital_targets = ui.capital_targets()
positions = client.get_positions()
short_put_symbols = {p.symbol for p in positions if p.sec_type == "OPT" and p.right == "P" and p.quantity < 0}
underlying_prices, implied_vols = {}, {}
for sym in short_put_symbols:
    hist, _src = ui.price_history(sym)
    if hist is not None and not hist.empty:
        underlying_prices[sym] = float(hist["Close"].iloc[-1])
        rv = realized_volatility(hist)
        if rv:
            implied_vols[sym] = rv   # proxy: no live IV pulled here

report = compute_exposure(positions, underlying_prices, implied_vols, total_book=capital_targets.total_book)
with st.container(horizontal=True):
    st.metric("Raw exposure", f"${report.raw_exposure:,.0f}", border=True)
    st.metric("Delta-adjusted exposure", f"${report.delta_adjusted_exposure:,.0f}", border=True)
    st.metric("Stress exposure", f"${report.stress_exposure:,.0f}", border=True,
              help=f"Underlying {wc.STRESS_UNDERLYING_SHOCK_PCT:+.0%}, IV {wc.STRESS_IV_SHOCK_PCT:+.0%}")
st.caption("Delta-adjusted and stress figures use the Black-Scholes proxy — see wheel_risk.py.")

if report.per_name_concentration:
    with st.container(key="glass_concentration"):
        st.subheader("Per-name concentration (short puts)", anchor=False)
        conc_df = pd.DataFrame(
            [{"Symbol": s, "% of book": p} for s, p in report.per_name_concentration.items()]
        ).sort_values("% of book", ascending=False)
        st.bar_chart(conc_df, x="Symbol", y="% of book", horizontal=True, height=220)
        if report.concentration_breaches:
            st.error(f"Over the {wc.MAX_NAME_CONCENTRATION:.0%} per-name cap: "
                     f"{', '.join(report.concentration_breaches)}", icon=":material/warning:")
else:
    st.caption("No open short puts.")
