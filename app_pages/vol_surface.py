"""Vol surface: Gatheral's volatility-surface ideas on real option quotes — smile, surface, total variance, a live SVI fit,
butterfly / calendar arbitrage, surface dynamics, and decisions for the account (analysis only)."""

import json

import streamlit as st

import wheel_ui as ui
from volsurface import ui as vui

client = ui.get_client()

st.caption("Following Jim Gatheral, *The Volatility Surface*: look at the smile, build the surface, switch to total variance, fit SVI, test for "
           "arbitrage, watch the surface move — then use it. SPX by default (S&P 500 index options). " + vui.DISCLAIMER)

qs = vui.render_controls(client)
if qs is None:
    st.stop()

opts = vui._opts_from_state()
fit_settings = vui._fit_from_state()
key = vui.qs_key(qs)
opts_json = json.dumps(opts.__dict__)

surf = vui._surface(key, opts_json, qs)
if not surf.slices:
    st.error("No expiry has enough usable quotes. If the market is closed, Yahoo and IBKR return no live bid/ask — load a saved snapshot "
             "or try again when the market is open.", icon=":material/error:")
    vui.banner(qs, surf)
    st.stop()

with st.spinner(f"Fitting an SVI smile to each of {len(surf.slices)} expiries…"):
    fs = vui._fit(key, opts_json, json.dumps(fit_settings.__dict__), surf)

vui.banner(qs, surf)
vui.render_smile(surf, fs)
vui.render_surface(surf, fs)
vui.render_total_variance(surf, fs)
vui.render_svi_lab(surf, fs, key)
vui.render_arbitrage(surf, fs, key, opts_json)
pcts = vui.render_dynamics(surf, fs, qs, st.session_state.get("vs_sid"))
vui.render_decisions(surf, fs, qs, pcts, client, opts_json)
