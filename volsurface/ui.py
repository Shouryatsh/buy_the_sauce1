"""
volsurface/ui.py — Streamlit rendering for the Vol surface page (app_pages/vol_surface.py).

The page follows Gatheral's route, one numbered section per idea:
  1 the smile of one expiry        2 the surface across expiries and strikes     3 total variance and moneyness
  4 an SVI fit you can tune live   5 butterfly and calendar arbitrage            6 surface dynamics
  7 decisions for your account (analysis only — this dashboard never places an order)
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

import wheel_config as wc
import wheel_ui as wui

from . import analysis as an
from . import arbitrage as ar
from . import data, model, store, surface
from .svi import PARAM_NAMES, SVI, error_stats, fit_slice

COMMON_SYMBOLS = ["SPX", "SPY", "XSP", "QQQ", "IWM", "NDX"]
SRC_YAHOO, SRC_IBKR, SRC_SAVED = "Yahoo (delayed)", "IBKR (live)", "Saved snapshot"
SOURCES = [SRC_YAHOO, SRC_IBKR, SRC_SAVED]
AWARE = dict(butterfly_strength=3.0, calendar_strength=3.0)
DISCLAIMER = ("Analysis only — this dashboard is read-only and never places, changes or cancels an order. Model values (SVI fits, "
              "risk-neutral probabilities) describe how the market prices options today; they are not forecasts, and a quote that "
              "sits off the smooth surface is not a guaranteed edge.")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def ramp(n: int) -> list[str]:
    """One colour per expiry, ordered by maturity (sequential: short -> long)."""
    return sample_colorscale("Viridis", list(np.linspace(0.12, 0.95, max(n, 1))))


def x_of(k, F: float, mode: str, w_atm: Optional[float] = None):
    k = np.asarray(k, dtype=float)
    if mode == "Strike":
        return F * np.exp(k)
    if mode == "K/F (%)":
        return 100.0 * np.exp(k)
    if mode.startswith("Standardized"):
        return k / np.sqrt(w_atm)
    return k


def x_title(mode: str) -> str:
    return {"Strike": "Strike", "K/F (%)": "Strike as % of forward (K/F)", "Log-moneyness k": "Log-moneyness k = ln(K/F)",
            "Standardized k/√w_ATM": "Standardized moneyness k/√w(ATM)"}.get(mode, mode)


def fmt_pct(x, d=1) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x * 100:.{d}f}%"


def _fig(height=420, **kw) -> go.Figure:
    f = go.Figure()
    f.update_layout(**wui.plotly_layout(height=height, **kw))
    return f


def _legend_h(y=-0.18):
    return dict(orientation="h", y=y, yanchor="top", x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=11))


# ---------------------------------------------------------------------------
# data: load, build, fit (cached)
# ---------------------------------------------------------------------------

def qs_key(qs: data.QuoteSet) -> str:
    return f"{qs.symbol}|{qs.source}|{qs.asof}|{len(qs.quotes)}"


@st.cache_data(ttl=900, show_spinner=False)
def listed_expiries(symbol: str) -> list[str]:
    try:
        return data.yahoo_listing(symbol)[0]
    except Exception:
        return []


@st.cache_data(show_spinner=False, max_entries=12)
def _surface(key: str, opts_json: str, _qs) -> surface.Surface:
    return surface.build_surface(_qs.quotes, _qs.spot, _qs.asof, _qs.symbol, _qs.source, surface.SurfaceOptions(**json.loads(opts_json)))


@st.cache_data(show_spinner=False, max_entries=24)
def _fit(key: str, opts_json: str, fit_json: str, _surf) -> model.FittedSurface:
    return model.fit_surface(_surf, model.FitSettings(**json.loads(fit_json)))


@st.cache_data(show_spinner=False, max_entries=64)
def _fit_one(key: str, expiry: str, style: str, _sl) -> Optional[tuple]:
    """Single-slice fit in a given style ('plain' | 'aware'), for the lab. Returns (params tuple, rmse, within_band) or None."""
    u = _sl.usable
    kw = AWARE if style == "aware" else {}
    try:
        f = fit_slice(u["k"].to_numpy(), u["iv"].to_numpy(), _sl.T, u["iv_bid"].to_numpy(), u["iv_ask"].to_numpy(),
                      u["vega"].to_numpy(), weights="spread", **kw)
    except Exception:
        return None
    return f.params.as_tuple(), f.rmse_iv, f.within_band


@st.cache_data(ttl=3600, show_spinner=False)
def vix_history() -> pd.DataFrame:
    import yfinance as yf
    tickers = ["^VIX", "^VIX9D", "^VIX3M", "^VIX6M", "^SKEW", "^VVIX", "^GSPC"]
    raw = yf.download(tickers, period="3y", interval="1d", auto_adjust=False, progress=False, group_by="ticker", threads=True)
    cols = {}
    for t in tickers:
        try:
            s = raw[t]["Close"].dropna()
        except KeyError:
            continue
        if len(s):
            s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
            cols[t] = s
    return pd.DataFrame(cols)


# ---------------------------------------------------------------------------
# controls
# ---------------------------------------------------------------------------

def _opts_from_state() -> surface.SurfaceOptions:
    s = st.session_state
    return surface.SurfaceOptions(min_bid=float(s.get("vs_min_bid", 0.05)), max_rel_spread=float(s.get("vs_max_spread", 0.5)),
                                  k_min=float(s.get("vs_kmin", -0.8)), k_max=float(s.get("vs_kmax", 0.5)),
                                  min_days=float(s.get("vs_min_days", 2.0)))


def _fit_from_state() -> model.FitSettings:
    s = st.session_state
    kw = AWARE if s.get("vs_aware", True) else {}
    return model.FitSettings(weights=s.get("vs_weights", "spread"), lee=True, nstarts=12, **kw)


def _load(symbol: str, source: str, expiries: list[str], client) -> Optional[str]:
    """Fetch quotes into session state; returns an error message or None."""
    try:
        if source == SRC_YAHOO:
            qs = data.fetch_yahoo(symbol, expiries or None)
        elif source == SRC_IBKR:
            qs = data.fetch_ibkr(client, symbol, expiries or None)
        else:
            return "choose a saved snapshot"
        sid = store.save_snapshot(qs)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    st.session_state.vs_qs, st.session_state.vs_sid = qs, sid
    return None


def render_controls(client) -> Optional[data.QuoteSet]:
    with st.container(key="glass_vs_controls"):
        with st.container(horizontal=True, vertical_alignment="bottom"):
            symbol = st.selectbox("Underlying", COMMON_SYMBOLS, accept_new_options=True, key="vs_symbol", width=170,
                                  help="SPX = S&P 500 index options (cash-settled, ~$770k notional per contract). SPY / XSP are the "
                                       "small-account analogues. Any optionable ticker works.")
            source = st.segmented_control("Quote source", SOURCES, default=SRC_YAHOO, key="vs_source",
                                          help="Yahoo: free, ~15 min delayed, no TWS needed. IBKR: live via TWS (read-only). "
                                               "Saved snapshot: any set you loaded before — works when the market is closed.")
        symbol = (symbol or "SPX").upper().strip()
        source = source or SRC_YAHOO
        qs: Optional[data.QuoteSet] = st.session_state.get("vs_qs")
        err = None

        if source == SRC_SAVED:
            snaps = store.list_snapshots(symbol)
            if snaps.empty:
                st.info(f"No saved {symbol} snapshots yet — load quotes from Yahoo or IBKR once and they are saved automatically.",
                        icon=":material/info:")
                return None
            labels = {int(r["id"]): f"{r['asof'][:16].replace('T', ' ')} · {r['source']} · {int(r['n_quotes'])} quotes" for _, r in snaps.iterrows()}
            sid = st.selectbox("Snapshot", list(labels), format_func=labels.get, key="vs_snap_pick")
            if st.session_state.get("vs_sid") != sid or qs is None or qs.symbol != symbol:
                qs = store.load_snapshot(int(sid))
                st.session_state.vs_qs, st.session_state.vs_sid = qs, int(sid)
        else:
            listing = listed_expiries(symbol)
            asof = data.now_et()
            default = data.default_expiries(listing, asof) if listing else []
            expiries: list[str] = []
            if listing:
                days = {e: (pd.Timestamp(e) - asof.tz_localize(None).normalize()).days for e in listing}
                ok = [e for e in listing if days[e] >= 2]
                expiries = st.multiselect("Expiries to load", ok, default=[e for e in default if e in ok], key=f"vs_exp_{symbol}",
                                          format_func=lambda e: f"{e} · {days[e]}d")
            first_time = qs is None and source == SRC_YAHOO and "vs_autoloaded" not in st.session_state
            go_ = st.button("Load quotes", icon=":material/download:", type="primary", key="vs_load")
            if source == SRC_IBKR and client is None:
                st.caption("Connect to IBKR in the left rail to load live quotes.")
            elif go_ or first_time:
                st.session_state["vs_autoloaded"] = True
                with st.spinner(f"Loading {symbol} option chains from {source}…"):
                    err = _load(symbol, source, expiries, client)
                qs = st.session_state.get("vs_qs")
            elif qs is not None and qs.symbol != symbol:
                st.caption(f"Showing {qs.symbol}; press **Load quotes** to switch to {symbol}.")
        if err:
            st.error(err, icon=":material/error:")
        with st.expander("Quote filters and fit settings", icon=":material/tune:"):
            a, b, c, d = st.columns(4)
            a.number_input("Minimum bid ($)", 0.0, 5.0, 0.05, 0.05, key="vs_min_bid", help="A bid below this is treated as no bid.")
            b.number_input("Maximum bid-ask spread (% of mid)", 5, 200, 50, 5, key="vs_max_spread_pct",
                           help="Quotes wider than this are too noisy to fit.")
            st.session_state["vs_max_spread"] = st.session_state.get("vs_max_spread_pct", 50) / 100.0
            c.number_input("Min log-moneyness k", -3.0, 0.0, -0.8, 0.05, key="vs_kmin")
            d.number_input("Max log-moneyness k", 0.0, 3.0, 0.5, 0.05, key="vs_kmax")
            e, f, g = st.columns(3)
            e.number_input("Skip expiries under (days)", 0.0, 30.0, 2.0, 1.0, key="vs_min_days")
            f.selectbox("Fit weighting", ["spread", "vega", "equal"], key="vs_weights",
                        help="spread: trust quotes in proportion to how tight their bid-ask is. vega: price-error weighting. equal.")
            g.toggle("Arbitrage-aware fit", value=True, key="vs_aware",
                     help="Adds a soft penalty for negative density and for crossing the previous expiry. Costs a few hundredths "
                          "of a vol point of fit error and keeps the extrapolated wings sensible.")
    return st.session_state.get("vs_qs")


def banner(qs: data.QuoteSet, surf: surface.Surface) -> None:
    n_ok = sum(int(s.usable.shape[0]) for s in surf.slices)
    with st.container(horizontal=True):
        st.metric("Underlying", qs.symbol, f"spot {qs.spot:,.2f}" if qs.spot else None, delta_color="off", delta_arrow="off", border=True)
        st.metric("Source", qs.source, f"as of {str(qs.asof)[:16].replace('T', ' ')} ET", delta_color="off", delta_arrow="off", border=True)
        st.metric("Expiries used", len(surf.slices), f"{len(surf.dropped)} dropped" if surf.dropped else None,
                  delta_color="off", delta_arrow="off", border=True)
        st.metric("Usable OTM quotes", n_ok, f"of {len(qs.quotes)} loaded", delta_color="off", delta_arrow="off", border=True)
    for n in qs.notes:
        st.warning(n, icon=":material/warning:")
    if surf.dropped:
        with st.expander(f"{len(surf.dropped)} expiries not used", icon=":material/info:"):
            st.dataframe(pd.DataFrame({"Expiry": list(surf.dropped), "Why": list(surf.dropped.values())}), hide_index=True)
    fallback = [s.expiry for s in surf.slices if s.forward_source != "parity"]
    if fallback:
        st.warning(f"Put-call parity could not set the forward for {', '.join(fallback)} — a rate-based fallback is used there.",
                   icon=":material/warning:")


# ---------------------------------------------------------------------------
# 1 · the smile
# ---------------------------------------------------------------------------

def _default_index(slices, days=30.0) -> int:
    return int(np.argmin([abs(s.days - days) for s in slices]))


def render_smile(surf: surface.Surface, fs: model.FittedSurface) -> None:
    with st.container(key="glass_vs_smile"):
        st.subheader("1 · The smile of one expiry", anchor=False)
        st.caption("Implied volatility is not flat across strikes: out-of-the-money puts cost more vol than calls. Each dot is an "
                   "out-of-the-money option (puts left of the forward, calls right of it) with its bid-ask range as an error bar.")
        labels = {s.expiry: f"{s.expiry} · {s.days:.0f} days" for s in surf.slices}
        order = [s.expiry for s in surf.slices]
        c1, c2 = st.columns([2, 3])
        exp = c1.selectbox("Expiry", order, index=_default_index(surf.slices), format_func=labels.get, key="vs_smile_exp")
        mode = c2.segmented_control("Horizontal axis", ["Strike", "K/F (%)", "Log-moneyness k"], default="Strike", key="vs_smile_x")
        mode = mode or "Strike"
        sl = surf.slice(exp)
        fsl = fs.by_expiry(exp)
        p = sl.points
        used, other_otm, itm = p[p["use"]], p[p["otm"] & ~p["use"] & p["iv"].notna()], p[~p["otm"] & p["iv"].notna()]
        fig = _fig(430, hovermode="closest", margin=dict(l=8, r=8, t=16, b=8), legend=_legend_h(),
                   xaxis_title=x_title(mode), yaxis_title="Implied volatility")
        if len(itm):
            fig.add_trace(go.Scatter(x=x_of(itm["k"], sl.F, mode), y=itm["iv"], mode="markers", name="ITM quotes (not used)",
                                     marker=dict(size=4, color="rgba(148,163,184,0.45)", symbol="circle-open"), hoverinfo="skip"))
        if len(other_otm):
            fig.add_trace(go.Scatter(x=x_of(other_otm["k"], sl.F, mode), y=other_otm["iv"], mode="markers",
                                     name="OTM filtered out (no bid / too wide)", marker=dict(size=5, color="rgba(148,163,184,0.7)"),
                                     hovertemplate="strike %{customdata[0]:.0f}<br>IV %{y:.1%}<extra>filtered</extra>", customdata=other_otm[["K"]]))
        for cp, name, color in ((-1.0, "OTM puts", wui.VIOLET), (1.0, "OTM calls", wui.TEAL)):
            u = used[used["cp"] == cp]
            if u.empty:
                continue
            up = (u["iv_ask"] - u["iv"]).fillna(0).clip(lower=0)
            dn = (u["iv"] - u["iv_bid"]).fillna(0).clip(lower=0)
            fig.add_trace(go.Scatter(
                x=x_of(u["k"], sl.F, mode), y=u["iv"], mode="markers", name=name, marker=dict(size=6, color=color),
                error_y=dict(type="data", symmetric=False, array=up, arrayminus=dn, thickness=1, width=0, color=color),
                customdata=np.stack([u["K"], u["bid"], u["ask"]], axis=1),
                hovertemplate="strike %{customdata[0]:.0f}<br>IV %{y:.2%}<br>bid %{customdata[1]:.2f} / ask %{customdata[2]:.2f}<extra></extra>"))
        if fsl.fit is not None and len(used):
            kk = np.linspace(used["k"].min() - 0.03, used["k"].max() + 0.03, 200)
            fig.add_trace(go.Scatter(x=x_of(kk, sl.F, mode), y=fsl.params.iv(kk, sl.T), mode="lines", name="SVI fit",
                                     line=dict(color=wui.AMBER, width=2, dash="dot"), hoverinfo="skip"))
        fx = float(x_of(0.0, sl.F, mode))
        fig.add_vline(x=fx, line=dict(color="rgba(230,234,245,0.4)", dash="dot", width=1), annotation_text="forward", annotation_position="top")
        if surf.spot and mode == "Strike":
            fig.add_vline(x=surf.spot, line=dict(color="rgba(244,114,182,0.5)", dash="dot", width=1), annotation_text="spot", annotation_position="bottom")
        if len(used):
            lo_iv, hi_iv = float(used["iv"].min()), float(used["iv"].max())
            pad = (hi_iv - lo_iv) * 0.08 + 0.005
            fig.update_yaxes(range=[max(lo_iv - pad, 0.0), hi_iv + pad])
            xs_used = x_of(used["k"], sl.F, mode, float(fsl.params.w(0.0)) if fsl.fit else None)
            xpad = (float(xs_used.max()) - float(xs_used.min())) * 0.05
            fig.update_xaxes(range=[float(xs_used.min()) - xpad, float(xs_used.max()) + xpad])
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, config={"displayModeBar": False})
        atm = float(fsl.params.iv(0.0, sl.T)) if fsl.fit else float("nan")
        sk = float(fsl.params.iv(np.log(0.9), sl.T)) - atm if fsl.fit else float("nan")
        with st.container(horizontal=True):
            st.metric("Days to expiry", f"{sl.days:.1f}", f"{sl.root} · {'AM' if sl.root.upper() in surface.AM_SETTLED_ROOTS else 'PM'} settled",
                      delta_color="off", delta_arrow="off", border=True)
            st.metric("Forward (parity)", f"{sl.F:,.2f}", f"carry r−q {fmt_pct(sl.carry, 2)}" if sl.carry is not None else None,
                      delta_color="off", delta_arrow="off", border=True,
                      help="Solved from put-call parity at this expiry: C − P = D (F − K). No rate or dividend assumption.")
            st.metric("ATM implied vol", fmt_pct(atm), "from the SVI fit", delta_color="off", delta_arrow="off", border=True)
            st.metric("Skew: IV(90% K/F) − ATM", f"{sk * 100:+.1f} vol pts" if np.isfinite(sk) else "n/a", delta_color="off", delta_arrow="off",
                      border=True, help="How much more vol the market charges for a put struck 10% below the forward than for the at-the-money option.")
            st.metric("Quotes used", int(sl.usable.shape[0]), f"{int(sl.points['otm'].sum())} OTM listed", delta_color="off", delta_arrow="off", border=True)


# ---------------------------------------------------------------------------
# 2 · the surface
# ---------------------------------------------------------------------------

def term_table(fs: model.FittedSurface) -> pd.DataFrame:
    rows = []
    for f in fs.slices:
        sl = f.slice
        rows.append({"Expiry": sl.expiry, "Days": sl.days, "Forward": sl.F, "Carry r−q": sl.carry,
                     "ATM IV (fit)": float(f.params.iv(0.0, sl.T)) if f.fit else None,
                     "ATM total variance": float(f.params.w(0.0)) if f.fit else None,
                     "Fit error (vol pts)": f.fit.rmse_iv * 100 if f.fit else None, "Quotes": int(sl.usable.shape[0]),
                     "Forward from": sl.forward_source, "Fit": "ok" if f.fit else f.error[:50]})
    return pd.DataFrame(rows)


def render_surface(surf: surface.Surface, fs: model.FittedSurface) -> None:
    with st.container(key="glass_vs_surface"):
        st.subheader("2 · The surface across expiries and strikes", anchor=False)
        st.caption("Fit one smile per expiry and join them: implied vol as a function of moneyness and time to expiry. The surface is the "
                   "fitted SVI slices, interpolated in total variance between expiries; the white dots are the market quotes it is built from.")
        fit = fs.fitted
        if len(fit) < 2:
            st.info("Need at least two fitted expiries to draw a surface.", icon=":material/info:")
            return
        c1, c2, c3 = st.columns([3, 3, 2])
        win = c1.slider("Strike window (% of forward)", 60, 130, (80, 108), key="vs_surf_win")
        d_lo, d_hi = int(np.floor(fit[0].slice.days)), int(np.ceil(fit[-1].slice.days))
        start = next((int(np.ceil(f.slice.days)) for f in fit if f.slice.days >= 14), d_lo)       # a one-week smile is so steep it hides the rest
        dwin = c2.slider("Maturity window (days)", d_lo, d_hi, (min(start, d_hi - 1), min(d_hi, max(start + 30, 400))), key=f"vs_surf_days_{d_lo}_{d_hi}")
        view = c3.segmented_control("View", ["3D surface", "Heatmap"], default="3D surface", key="vs_surf_view") or "3D surface"
        k_grid = np.linspace(np.log(win[0] / 100), np.log(win[1] / 100), 70)
        T_grid = np.linspace(max(dwin[0], fit[0].slice.days) / 365.0, dwin[1] / 365.0, 48)
        Z = fs.iv_grid(k_grid, T_grid) * 100
        days = T_grid * 365.0
        zlo, zhi = float(np.nanpercentile(Z, 1)), float(np.nanpercentile(Z, 99))
        xs = 100 * np.exp(k_grid)
        if view == "3D surface":
            fig = go.Figure(go.Surface(x=xs, y=days, z=Z, colorscale="Viridis", opacity=0.92, cmin=zlo, cmax=zhi, colorbar=dict(title="IV %", thickness=12),
                                       hovertemplate="K/F %{x:.1f}%<br>%{y:.0f} days<br>IV %{z:.1f}%<extra></extra>"))
            pts = surf.points
            pts = pts[pts["use"] & (pts["k"] >= k_grid[0]) & (pts["k"] <= k_grid[-1]) & (pts["T"] * 365.0 >= dwin[0] - 1) & (pts["T"] * 365.0 <= dwin[1] + 1)]
            if len(pts) > 1500:
                pts = pts.iloc[:: int(np.ceil(len(pts) / 1500))]
            fig.add_trace(go.Scatter3d(x=100 * np.exp(pts["k"]), y=pts["T"] * 365.0, z=pts["iv"] * 100, mode="markers",
                                       marker=dict(size=2, color="rgba(255,255,255,0.85)"), name="market quotes",
                                       hovertemplate="K/F %{x:.1f}%<br>%{y:.0f} days<br>IV %{z:.1f}%<extra>quote</extra>"))
            layout = wui.plotly_layout(height=640, margin=dict(l=0, r=0, t=10, b=0))
            layout.pop("xaxis", None)
            layout.pop("yaxis", None)
            fig.update_layout(**layout, showlegend=False, scene=dict(
                xaxis=dict(title="Strike / forward (%)"), yaxis=dict(title="Days to expiry"),
                zaxis=dict(title="Implied vol (%)", range=[max(float(np.nanmin(Z)) - 1, 0.0), float(np.nanmax(Z)) + 1]),
                camera=dict(eye=dict(x=-1.6, y=-1.7, z=0.9)), aspectmode="manual", aspectratio=dict(x=1.2, y=1.3, z=0.95)))
        else:
            fig = _fig(480, margin=dict(l=8, r=8, t=16, b=8), xaxis_title="Strike as % of forward", yaxis_title="Days to expiry")
            fig.add_trace(go.Heatmap(x=xs, y=days, z=Z, colorscale="Viridis", zmin=zlo, zmax=zhi, colorbar=dict(title="IV %", thickness=12),
                                     hovertemplate="K/F %{x:.1f}%<br>%{y:.0f} days<br>IV %{z:.1f}%<extra></extra>"))
            for f in fit:
                fig.add_hline(y=f.slice.days, line=dict(color="rgba(255,255,255,0.15)", width=1))
        st.plotly_chart(fig, config={"displayModeBar": view == "3D surface"})
        st.markdown("**Term structure**")
        t = term_table(fs)
        st.dataframe(t, hide_index=True, column_config={
            "Days": st.column_config.NumberColumn(format="%.0f"), "Forward": st.column_config.NumberColumn(format="%,.2f"),
            "Carry r−q": st.column_config.NumberColumn(format="percent", help="ln(F/spot)/T — only meaningful for longer expiries; "
                                                       "short ones are dominated by quote timing."),
            "ATM IV (fit)": st.column_config.NumberColumn(format="percent"),
            "ATM total variance": st.column_config.NumberColumn(format="%.5f"),
            "Fit error (vol pts)": st.column_config.NumberColumn(format="%.2f")})


# ---------------------------------------------------------------------------
# 3 · total variance and moneyness
# ---------------------------------------------------------------------------

def render_total_variance(surf: surface.Surface, fs: model.FittedSurface) -> None:
    with st.container(key="glass_vs_tv"):
        st.subheader("3 · Total variance and moneyness", anchor=False)
        st.caption("Gatheral works with total implied variance w = σ²·T instead of σ, against log-moneyness k = ln(K/F). In these units the "
                   "curves of successive expiries must nest — each one above the last, never crossing (no calendar arbitrage) — and "
                   "every smile has wings that grow only linearly in |k|.")
        fit = fs.fitted
        if not fit:
            st.info("No fitted expiries.", icon=":material/info:")
            return
        mode = st.segmented_control("Horizontal axis", ["Log-moneyness k", "K/F (%)", "Standardized k/√w_ATM"], default="Log-moneyness k",
                                    key="vs_tv_x") or "Log-moneyness k"
        colors = ramp(len(fit))
        fig_iv = _fig(430, hovermode="closest", margin=dict(l=8, r=8, t=16, b=8), xaxis_title=x_title(mode), yaxis_title="Implied volatility σ",
                      showlegend=False)
        fig_w = _fig(430, hovermode="closest", margin=dict(l=8, r=8, t=16, b=8), xaxis_title=x_title(mode), yaxis_title="Total variance w = σ²T",
                     legend=dict(orientation="v", x=1.01, y=1, bgcolor="rgba(0,0,0,0)", font=dict(size=10)))
        for f, col in zip(fit, colors):
            sl, p = f.slice, f.params
            u = sl.usable
            w_atm = float(p.w(0.0))
            kk = np.linspace(u["k"].min(), u["k"].max(), 160)
            xx = x_of(kk, sl.F, mode, w_atm)
            name = f"{sl.days:.0f}d"
            fig_iv.add_trace(go.Scatter(x=xx, y=p.iv(kk, sl.T), mode="lines", line=dict(color=col, width=2), name=name,
                                        hovertemplate=f"{name}<br>σ %{{y:.1%}}<extra></extra>"))
            fig_iv.add_trace(go.Scatter(x=x_of(u["k"], sl.F, mode, w_atm), y=u["iv"], mode="markers", marker=dict(size=3, color=col, opacity=0.55),
                                        hoverinfo="skip", showlegend=False))
            fig_w.add_trace(go.Scatter(x=xx, y=p.w(kk), mode="lines", line=dict(color=col, width=2), name=name,
                                       hovertemplate=f"{name}<br>w %{{y:.4f}}<extra></extra>"))
            fig_w.add_trace(go.Scatter(x=x_of(u["k"], sl.F, mode, w_atm), y=u["w"], mode="markers", marker=dict(size=3, color=col, opacity=0.55),
                                       hoverinfo="skip", showlegend=False))
        fig_iv.update_yaxes(tickformat=".0%")
        a, b = st.columns(2)
        a.markdown("**Implied volatility σ** — short expiries have the steepest, highest smiles")
        a.plotly_chart(fig_iv, config={"displayModeBar": False})
        b.markdown("**Total variance w = σ²T** — the curves stack in maturity order")
        b.plotly_chart(fig_w, config={"displayModeBar": False})
        if mode.startswith("Standardized"):
            st.caption("Dividing k by √w(ATM) rescales every expiry by its own width: the short-dated smiles that looked very different in σ "
                       "line up much more closely. That scaling is the idea behind Gatheral's SSVI surface.")
        ok, bad = ar.atm_variance_monotone(fs.labelled())
        T_atm = np.array([f.T for f in fit])
        w_atm = np.array([float(f.params.w(0.0)) for f in fit])
        fig = _fig(260, margin=dict(l=8, r=8, t=16, b=8), xaxis_title="Years to expiry T", yaxis_title="ATM total variance w(0,T)", showlegend=False)
        fig.add_trace(go.Scatter(x=T_atm, y=w_atm, mode="lines+markers", line=dict(color=wui.VIOLET, width=2), marker=dict(size=6),
                                 hovertemplate="T %{x:.3f}y<br>w(0,T) %{y:.5f}<extra></extra>"))
        st.markdown("**ATM total variance against maturity** — must be non-decreasing: " + (":green[it is]" if ok else f":red[it is not, between {bad}]"))
        st.plotly_chart(fig, config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# 4 · SVI lab (a fragment: moving a slider re-runs only this block)
# ---------------------------------------------------------------------------

PARAM_HELP = {
    "a": "Vertical level of the whole smile (total-variance units). Raising it lifts every strike equally.",
    "b": "Angle between the wings: larger b = steeper wings and a more pronounced smile.",
    "rho": "Rotation / skew. ρ < 0 tilts the smile so low strikes carry more variance (equity skew); ρ > 0 the reverse.",
    "m": "Horizontal translation: moves the vertex of the smile along k.",
    "sigma": "Smoothness of the vertex: small σ = a sharp V at the minimum, large σ = a rounded bowl.",
}


def _slider_spec(name: str, base: float) -> tuple:
    spec = {"a": (-0.10, 0.10, 0.0005), "b": (0.001, 1.0, 0.001), "rho": (-0.99, 0.99, 0.01), "m": (-0.5, 0.5, 0.005), "sigma": (0.005, 1.0, 0.005)}[name]
    lo, hi, step = spec
    return min(lo, base - 5 * step), max(hi, base + 5 * step), step


def _reset_lab(expiry: str, base: SVI) -> None:
    for n, v in zip(PARAM_NAMES, base.as_tuple()):
        st.session_state[f"vsp_{expiry}_{n}"] = float(v)


@st.fragment
def render_svi_lab(surf: surface.Surface, fs: model.FittedSurface, qkey: str) -> None:
    with st.container(key="glass_vs_lab"):
        st.subheader("4 · Fit an SVI smile and tune it live", anchor=False)
        st.caption("Raw SVI:  w(k) = a + b·( ρ·(k − m) + √((k − m)² + σ²) ).  Start from the automatic fit, then move the five parameters "
                   "and watch the curve, the error against the market and the risk-neutral density react.")
        fitted = fs.fitted
        if not fitted:
            st.info("No fitted expiries.", icon=":material/info:")
            return
        labels = {f.expiry: f"{f.expiry} · {f.slice.days:.0f} days" for f in fitted}
        c1, c2, c3 = st.columns([2, 2, 2])
        exp = c1.selectbox("Expiry", list(labels), index=_default_index([f.slice for f in fitted]), format_func=labels.get, key="vs_lab_exp")
        style = c2.segmented_control("Automatic fit", ["As configured", "Plain least squares", "Arbitrage-aware"], default="As configured", key="vs_lab_style")
        yaxis = c3.segmented_control("Show", ["Implied vol", "Total variance"], default="Implied vol", key="vs_lab_y") or "Implied vol"
        f = fs.by_expiry(exp)
        sl = f.slice
        u = sl.usable
        auto = f.params
        if style == "Plain least squares":
            got = _fit_one(qkey, exp, "plain", sl)
            auto = SVI(*got[0]) if got else auto
        elif style == "Arbitrage-aware":
            got = _fit_one(qkey, exp, "aware", sl)
            auto = SVI(*got[0]) if got else auto
        sig = (qkey, exp, style, tuple(round(x, 8) for x in auto.as_tuple()))
        if st.session_state.get("vs_lab_sig") != sig:                              # new expiry / data / fit style: reset the sliders
            _reset_lab(exp, auto)
            st.session_state["vs_lab_sig"] = sig
        cols = st.columns(5)
        vals = []
        for col, n, base in zip(cols, PARAM_NAMES, auto.as_tuple()):
            lo, hi, step = _slider_spec(n, base)
            v = col.slider({"a": "a", "b": "b", "rho": "ρ", "m": "m", "sigma": "σ"}[n], lo, hi, step=step, key=f"vsp_{exp}_{n}", help=PARAM_HELP[n],
                           format="%.4f" if n in ("a", "m") else "%.3f")
            vals.append(v)
        st.button("Reset to the automatic fit", icon=":material/restart_alt:", on_click=_reset_lab, args=(exp, auto), key="vs_lab_reset")
        manual = SVI(*vals)

        k, iv = u["k"].to_numpy(), u["iv"].to_numpy()
        ib, ia = u["iv_bid"].to_numpy(), u["iv_ask"].to_numpy()
        rm_m, in_m = error_stats(manual, k, iv, sl.T, ib, ia)
        rm_a, in_a = error_stats(auto, k, iv, sl.T, ib, ia)
        rng = (float(k.min()), float(k.max()))
        bf = ar.butterfly(manual, rng)
        with st.container(horizontal=True):
            st.metric("Fit error, your curve", f"{rm_m * 100:.2f} vol pts", f"auto-fit {rm_a * 100:.2f}", delta_color="off", delta_arrow="off", border=True,
                      help="Root-mean-square difference between the curve and the market mid implied vols.")
            st.metric("Inside the bid-ask band", "n/a" if in_m is None else f"{in_m * 100:.0f}%", None if in_a is None else f"auto-fit {in_a * 100:.0f}%",
                      delta_color="off", delta_arrow="off", border=True, help="Share of quotes where your curve lies between the bid and ask implied vols.")
            st.metric("Butterfly", "no arbitrage" if bf.ok else ("extrapolation only" if bf.ok_in_data else "ARBITRAGE in data"),
                      f"min g = {bf.min_g:.4f}", delta_color="off", delta_arrow="off", border=True,
                      help="Durrleman's g(k) must be ≥ 0 for the density to be non-negative. 'Extrapolation only' means it dips below zero "
                           "only at strikes nobody quotes.")
            st.metric("Wing slopes (Lee ≤ 2)", f"{manual.wing_left:.2f} / {manual.wing_right:.2f}", "left / right  ·  " + ("ok" if bf.lee_ok else "too steep"),
                      delta_color="off", delta_arrow="off", border=True)
        if not bf.ok:
            where = ", ".join(f"k {a:+.2f} to {b:+.2f}" for a, b in bf.bad_ranges[:3])
            where_k = "inside" if not bf.ok_in_data else "outside"
            st.warning(f"Negative risk-neutral density at {where} — {where_k} the quoted range ({rng[0]:+.2f} … {rng[1]:+.2f}). "
                       + ("A real butterfly arbitrage against these quotes." if not bf.ok_in_data else "An artefact of extrapolating the form into strikes that are not quoted."),
                       icon=":material/warning:")
        if manual.min_w <= 0:
            st.error("Minimum total variance is not positive: the curve implies negative variance somewhere.", icon=":material/error:")

        kk = np.linspace(rng[0] - 0.12, rng[1] + 0.25, 360)
        fig = make_subplots(rows=2, cols=1, row_heights=[0.68, 0.32], shared_xaxes=True, vertical_spacing=0.06)
        layout = wui.plotly_layout(height=560, hovermode="x unified", margin=dict(l=8, r=8, t=16, b=8), legend=_legend_h(-0.1))
        layout.pop("xaxis", None)
        layout.pop("yaxis", None)
        fig.update_layout(**layout)
        up = (ia - iv)
        dn = (iv - ib)
        up, dn = np.nan_to_num(np.clip(up, 0, None)), np.nan_to_num(np.clip(dn, 0, None))
        if yaxis == "Implied vol":
            yd = lambda s: s.iv(kk, sl.T)                                       # noqa: E731
            pts_y, e_up, e_dn = iv, up, dn
        else:
            yd = lambda s: s.w(kk)                                              # noqa: E731
            pts_y = iv ** 2 * sl.T
            e_up, e_dn = (np.nan_to_num(np.clip(ia ** 2 * sl.T - pts_y, 0, None)), np.nan_to_num(np.clip(pts_y - ib ** 2 * sl.T, 0, None)))
        fig.add_trace(go.Scatter(x=k, y=pts_y, mode="markers", name="market (mid, bid–ask range)", marker=dict(size=5, color="rgba(230,234,245,0.85)"),
                                 error_y=dict(type="data", symmetric=False, array=e_up, arrayminus=e_dn, thickness=1, width=0, color="rgba(230,234,245,0.35)")),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=kk, y=yd(auto), mode="lines", name="automatic fit", line=dict(color=wui.TEAL, width=2, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=kk, y=yd(manual), mode="lines", name="your curve", line=dict(color=wui.VIOLET, width=3)), row=1, col=1)
        for r_ in (1, 2):
            fig.add_vrect(x0=rng[0], x1=rng[1], fillcolor="rgba(255,255,255,0.035)", line_width=0, row=r_, col=1)
        d_manual = manual.density(kk)
        fig.add_trace(go.Scatter(x=kk, y=np.where(d_manual >= 0, d_manual, np.nan), mode="lines", name="density (your curve)",
                                 line=dict(color=wui.VIOLET, width=2), fill="tozeroy", fillcolor="rgba(167,139,250,0.18)"), row=2, col=1)
        neg = np.where(d_manual < 0, d_manual, np.nan)
        if np.isfinite(neg).any():
            fig.add_trace(go.Scatter(x=kk, y=neg, mode="lines", name="NEGATIVE density", line=dict(color=wui.RED, width=3), fill="tozeroy",
                                     fillcolor="rgba(251,113,133,0.35)"), row=2, col=1)
        fig.add_trace(go.Scatter(x=kk, y=auto.density(kk), mode="lines", name="density (automatic fit)", line=dict(color=wui.TEAL, width=1.5, dash="dot")),
                      row=2, col=1)
        fig.add_hline(y=0, line=dict(color="rgba(230,234,245,0.5)", width=1), row=2, col=1)
        fig.update_yaxes(tickformat=".0%" if yaxis == "Implied vol" else ".4f", title_text="σ" if yaxis == "Implied vol" else "w", row=1, col=1)
        fig.update_yaxes(title_text="density of k", row=2, col=1)
        fig.update_xaxes(title_text="Log-moneyness k = ln(K/F)", row=2, col=1)
        st.plotly_chart(fig, config={"displayModeBar": False})
        st.caption("The shaded band is where the market quotes. Below it: the risk-neutral density of the log-forward-price implied by your curve, "
                   "p(k) = g(k)/√(2πw) · exp(−d₂²/2). Where g(k) < 0 the density is negative — that is what a butterfly arbitrage is.")
        with st.expander("What each parameter does", icon=":material/help:"):
            for n in PARAM_NAMES:
                st.markdown(f"**{ {'a': 'a', 'b': 'b', 'rho': 'ρ', 'm': 'm', 'sigma': 'σ'}[n] }** — {PARAM_HELP[n]}")


# ---------------------------------------------------------------------------
# 5 · arbitrage
# ---------------------------------------------------------------------------

def arbitrage_tables(fs: model.FittedSurface, summ: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for f in fs.fitted:
        r = summ["butterfly"][f.expiry]
        rows.append({"Expiry": f.expiry, "Days": f.slice.days, "Butterfly": "clean" if r.ok else ("extrapolation only" if r.ok_in_data else "ARBITRAGE in data"),
                     "Min g(k)": r.min_g, "At k": r.k_at_min,
                     "Violating k": "; ".join(f"{a:+.2f}…{b:+.2f}" for a, b in r.bad_ranges[:3]), "Data range k": f"{r.data_range[0]:+.2f}…{r.data_range[1]:+.2f}",
                     "Lee wings L/R": f"{r.lee_left:.2f} / {r.lee_right:.2f}", "Lee ≤ 2": "ok" if r.lee_ok else "too steep"})
    cal = []
    for c in summ["calendar"]:
        cal.append({"Short expiry": c.label_short, "Long expiry": c.label_long,
                    "Calendar": "clean" if c.ok else ("extrapolation only" if c.ok_in_data else "ARBITRAGE in data"),
                    "Min Δw": c.min_dw, "At k": c.k_at_min, "Violating k": "; ".join(f"{a:+.2f}…{b:+.2f}" for a, b in c.bad_ranges[:3]),
                    "Common data k": "none" if c.common_range is None else f"{c.common_range[0]:+.2f}…{c.common_range[1]:+.2f}"})
    return pd.DataFrame(rows), pd.DataFrame(cal)


def render_arbitrage(surf: surface.Surface, fs: model.FittedSurface, qkey: str, opts_json: str) -> None:
    with st.container(key="glass_vs_arb"):
        st.subheader("5 · Butterfly and calendar arbitrage", anchor=False)
        st.caption("A surface is free of static arbitrage if (a) within each expiry the risk-neutral density is non-negative — Durrleman's "
                   "g(k) ≥ 0, the butterfly condition — and (b) total variance never falls as maturity grows at a fixed k — the calendar "
                   "condition. A violation where the market quotes is a real problem; one beyond the quoted strikes is only the fitted form "
                   "extrapolating.")
        if len(fs.fitted) < 1:
            st.info("No fitted expiries.", icon=":material/info:")
            return
        summ = model.arbitrage_summary(fs)
        bt, ct = arbitrage_tables(fs, summ)
        n_bf_data = int((bt["Butterfly"] == "ARBITRAGE in data").sum())
        n_cal_data = int((ct["Calendar"] == "ARBITRAGE in data").sum()) if len(ct) else 0
        n_bf_ext = int((bt["Butterfly"] == "extrapolation only").sum())
        n_cal_ext = int((ct["Calendar"] == "extrapolation only").sum()) if len(ct) else 0
        with st.container(horizontal=True):
            st.metric("Butterfly, in the quoted range", f"{len(bt) - n_bf_data} of {len(bt)} clean", f"{n_bf_ext} flagged only in extrapolated wings",
                      delta_color="off", delta_arrow="off", border=True)
            st.metric("Calendar, in the common range", f"{len(ct) - n_cal_data} of {len(ct)} clean" if len(ct) else "n/a",
                      f"{n_cal_ext} flagged only in extrapolated wings", delta_color="off", delta_arrow="off", border=True)
            st.metric("ATM variance monotone", "yes" if summ["atm_monotone"] else "NO", delta_color="off", delta_arrow="off", border=True)
        if n_bf_data or n_cal_data:
            st.warning("The fitted surface has arbitrage inside the quoted region — see the tables. Try the arbitrage-aware fit in the settings.", icon=":material/warning:")
        else:
            st.success("No static arbitrage where the market actually quotes. Anything flagged below lies beyond the quoted strikes.", icon=":material/check_circle:")

        st.markdown("**Butterfly — Durrleman's g(k) per expiry** (the risk-neutral density is proportional to it)")
        fit = fs.fitted
        colors = ramp(len(fit))
        fig = _fig(380, hovermode="x unified", margin=dict(l=8, r=8, t=16, b=8), xaxis_title="Log-moneyness k", yaxis_title="g(k)  (≥ 0 required)",
                   legend=dict(orientation="v", x=1.01, y=1, bgcolor="rgba(0,0,0,0)", font=dict(size=10)))
        for f, col in zip(fit, colors):
            r = summ["butterfly"][f.expiry]
            mask = (r.k >= -1.0) & (r.k <= 0.8)
            fig.add_trace(go.Scatter(x=r.k[mask], y=np.clip(r.g[mask], -1.0, 3.0), mode="lines", line=dict(color=col, width=1.8), name=f"{f.slice.days:.0f}d",
                                     hovertemplate=f"{f.slice.days:.0f}d<br>g %{{y:.3f}}<extra></extra>"))
        fig.add_hline(y=0, line=dict(color=wui.RED, width=1.5, dash="dash"))
        fig.update_yaxes(range=[-0.6, 2.0])
        st.plotly_chart(fig, config={"displayModeBar": False})
        st.dataframe(bt, hide_index=True, column_config={"Days": st.column_config.NumberColumn(format="%.0f"), "Min g(k)": st.column_config.NumberColumn(format="%.4f"),
                                                          "At k": st.column_config.NumberColumn(format="%+.2f")})

        if len(ct):
            st.markdown("**Calendar — change in total variance from one expiry to the next, w(k,T₂) − w(k,T₁)**")
            fig = _fig(340, hovermode="x unified", margin=dict(l=8, r=8, t=16, b=8), xaxis_title="Log-moneyness k", yaxis_title="Δw  (≥ 0 required)",
                       legend=dict(orientation="v", x=1.01, y=1, bgcolor="rgba(0,0,0,0)", font=dict(size=10)))
            for c, col in zip(summ["calendar"], ramp(len(summ["calendar"]))):
                lo, hi = (c.common_range or (c.k[0], c.k[-1]))
                m = (c.k >= -0.8) & (c.k <= 0.6)
                fig.add_trace(go.Scatter(x=c.k[m], y=c.dw[m], mode="lines", line=dict(color=col, width=1.8), name=f"{c.label_short[5:]}→{c.label_long[5:]}",
                                         hovertemplate=f"{c.label_short}→{c.label_long}<br>Δw %{{y:.5f}}<extra></extra>"))
            fig.add_hline(y=0, line=dict(color=wui.RED, width=1.5, dash="dash"))
            st.plotly_chart(fig, config={"displayModeBar": False})
            st.dataframe(ct, hide_index=True, column_config={"Min Δw": st.column_config.NumberColumn(format="%.5f"), "At k": st.column_config.NumberColumn(format="%+.2f")})

        st.markdown("**Model-free check on the quotes themselves**")
        rows = []
        for sl in surf.slices:
            qc = ar.quote_checks(sl.points[["strike", "right", "bid", "ask", "k"]], min_bid=surf.options.min_bid,
                                 k_window=(surf.options.k_min, surf.options.k_max))
            first = ""
            if qc.convexity_violations:
                v = qc.convexity_violations[0]
                first = f"{v[0]} {v[1]:.0f}/{v[2]:.0f}/{v[3]:.0f} (${v[4]:.2f})"
            elif qc.monotone_violations:
                v = qc.monotone_violations[0]
                first = f"{v[0]} {v[1]:.0f}→{v[2]:.0f} (${v[3]:.2f})"
            rows.append({"Expiry": sl.expiry, "Strikes": qc.n_strikes, "Monotonicity violations": len(qc.monotone_violations),
                         "Convexity (butterfly) violations": len(qc.convexity_violations), "Worst ($)": qc.worst, "First violation": first})
        st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={"Worst ($)": st.column_config.NumberColumn(format="%.2f")})
        st.caption("Calls must fall with strike and puts must rise with it, and each must be convex in strike. These tests use bid and ask, and only "
                   "count a violation larger than one price tick, so a flag is a vertical or butterfly spread you could lock in after paying the "
                   "spread. Delayed or stale quotes can still produce a few spurious flags.")

        with st.expander("Compare plain least squares with the arbitrage-aware fit", icon=":material/compare_arrows:"):
            if st.button("Run comparison", key="vs_cmp_run", icon=":material/play_arrow:"):
                st.session_state["vs_cmp"] = True
            if st.session_state.get("vs_cmp"):
                with st.spinner("Fitting both ways…"):
                    plain = _fit(qkey, opts_json, json.dumps(model.FitSettings(weights=fs.settings.weights, nstarts=12).__dict__), surf)
                    aware = _fit(qkey, opts_json, json.dumps(model.FitSettings(weights=fs.settings.weights, nstarts=12, **AWARE).__dict__), surf)
                rows = []
                for name, f_ in (("Plain least squares", plain), ("Arbitrage-aware", aware)):
                    s_ = model.arbitrage_summary(f_)
                    rows.append({"Fit": name, "Mean fit error (vol pts)": float(np.mean([x.fit.rmse_iv for x in f_.fitted])) * 100,
                                 "Butterfly: slices with any violation": sum(1 for r in s_["butterfly"].values() if not r.ok),
                                 "…of which inside the quoted range": sum(1 for r in s_["butterfly"].values() if not r.ok_in_data),
                                 "Calendar pairs with any violation": sum(1 for c in s_["calendar"] if not c.ok),
                                 "…of which inside the common range": sum(1 for c in s_["calendar"] if not c.ok_in_data)})
                st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={"Mean fit error (vol pts)": st.column_config.NumberColumn(format="%.3f")})
                st.caption("The penalty trades a sliver of in-sample accuracy for a density that stays non-negative beyond the quoted strikes.")


# ---------------------------------------------------------------------------
# 6 · dynamics
# ---------------------------------------------------------------------------

def vix_percentiles(hist: pd.DataFrame, lookback: int = 504) -> dict:
    """Own-history percentiles (McMillan: each series only against its own past) of the market-wide vol indicators."""
    out: dict = {}
    def put(name, s):                                                           # noqa: E306
        if s is not None and len(s.dropna()) > 80:
            out[name] = {"value": float(s.dropna().iloc[-1]), "pct": an.percentile_of_last(s, lookback)}
    if "^VIX" in hist:
        put("VIX", hist["^VIX"])
    if "^VIX" in hist and "^VIX3M" in hist:
        both = pd.concat([hist["^VIX"], hist["^VIX3M"]], axis=1).dropna()
        put("VIX / VIX3M", both.iloc[:, 0] / both.iloc[:, 1])
    if "^SKEW" in hist:
        put("CBOE SKEW", hist["^SKEW"])
    if "^VVIX" in hist:
        put("VVIX", hist["^VVIX"])
    if "^GSPC" in hist:
        import wheel_vol as wv
        put("SPX 20d realized vol", wv.realized_vol(hist["^GSPC"], 20))
    return out


def render_dynamics(surf: surface.Surface, fs: model.FittedSurface, qs: data.QuoteSet, sid: Optional[int]) -> dict:
    """Returns the market-wide percentiles for the decisions section."""
    pcts: dict = {}
    with st.container(key="glass_vs_dyn"):
        st.subheader("6 · Surface dynamics", anchor=False)
        st.caption("How the surface moves: every quote set you load is saved, so you can compare today's surface with an earlier one, "
                   "chart the level, skew and term structure over time, and see where the market-wide volatility indicators sit in their "
                   "own history. Schedule `python -m volsurface.record --symbol SPX` to build the history while the dashboard is closed.")
        snaps = store.list_snapshots(qs.symbol)
        others = snaps[snaps["id"] != sid] if sid is not None else snaps
        st.markdown("**Compare with an earlier surface**")
        if others.empty:
            st.caption("No earlier snapshot of this underlying yet — load again later (or schedule the recorder) and the comparison appears here.")
        else:
            labels = {int(r["id"]): f"{r['asof'][:16].replace('T', ' ')} · {r['source']}" for _, r in others.iterrows()}
            pick = st.selectbox("Earlier snapshot", list(labels), format_func=labels.get, key="vs_dyn_pick")
            old_qs = store.load_snapshot(int(pick))
            old_opts = _opts_from_state()
            old_surf = _surface(qs_key(old_qs), json.dumps(old_opts.__dict__), old_qs)
            old_fs = _fit(qs_key(old_qs), json.dumps(old_opts.__dict__), json.dumps(fs.settings.__dict__), old_surf)
            diff = an.surface_difference(old_fs, fs)
            if diff.empty:
                st.caption("The two surfaces share no maturities to compare.")
            else:
                fig = _fig(330, margin=dict(l=8, r=8, t=16, b=8), xaxis_title="Strike as % of forward", yaxis_title="Days to expiry")
                z = diff.to_numpy()
                lim = max(float(np.nanmax(np.abs(z))), 0.5)
                fig.add_trace(go.Heatmap(x=[str(c) for c in diff.columns], y=[f"{int(d)}d" for d in diff.index], z=z, zmin=-lim, zmax=lim,
                                         colorscale=[[0, "#38BDF8"], [0.5, "#1E293B"], [1, "#FB7185"]], colorbar=dict(title="Δ vol pts", thickness=12),
                                         hovertemplate="%{y} · %{x}% of forward<br>Δ %{z:+.2f} vol pts<extra></extra>"))
                st.plotly_chart(fig, config={"displayModeBar": False})
                st.caption(f"Implied-vol change, now minus {str(old_qs.asof)[:16].replace('T', ' ')}. Red = options got dearer, blue = cheaper. "
                           "A parallel red band is a level move; a tilt across moneyness is a change in skew; a tilt down the maturities is the term structure moving.")
                t_days = np.array([7, 14, 30, 60, 90, 180, 365])
                fig = _fig(280, margin=dict(l=8, r=8, t=16, b=8), xaxis_title="Days to expiry", yaxis_title="ATM implied vol", legend=_legend_h(-0.25))
                for label, f_, col, dash in (("now", fs, wui.VIOLET, "solid"), ("earlier", old_fs, wui.TEAL, "dot")):
                    rng = an.surface_range(f_)
                    td = [d for d in t_days if rng[0] * 0.999 <= d / 365 <= rng[1] * 1.001]
                    fig.add_trace(go.Scatter(x=td, y=[float(f_.iv(0.0, d / 365)) for d in td], mode="lines+markers", name=label, line=dict(color=col, width=2, dash=dash)))
                fig.update_yaxes(tickformat=".0%")
                st.plotly_chart(fig, config={"displayModeBar": False})

        st.markdown("**History of this surface's level, skew and term structure**")
        mkey = f"{sid}|{json.dumps(fs.settings.__dict__)}"
        if sid is not None and st.session_state.get("vs_metrics_saved") != mkey:        # cache this snapshot's summary once per fit setting
            store.update_metrics(sid, an.snapshot_metrics(fs))
            st.session_state["vs_metrics_saved"] = mkey
        hist = store.snapshot_metrics_df(qs.symbol)
        if len(hist) < 2:
            st.caption(f"{len(hist)} snapshot with metrics so far — the history charts need at least two.")
        else:
            a, b = st.columns(2)
            fig = _fig(280, margin=dict(l=8, r=8, t=16, b=8), yaxis_title="ATM implied vol", legend=_legend_h(-0.25))
            for col, name, c in (("atm30", "30-day", wui.VIOLET), ("atm90", "90-day", wui.TEAL)):
                if col in hist:
                    fig.add_trace(go.Scatter(x=hist["asof"], y=hist[col], mode="lines+markers", name=name, line=dict(color=c, width=2)))
            fig.update_yaxes(tickformat=".0%")
            a.plotly_chart(fig, config={"displayModeBar": False})
            fig = _fig(280, margin=dict(l=8, r=8, t=16, b=8), yaxis_title="Skew: IV(90% K/F) − ATM, vol pts", showlegend=False)
            if "skew90_30" in hist:
                fig.add_trace(go.Scatter(x=hist["asof"], y=hist["skew90_30"] * 100, mode="lines+markers", line=dict(color=wui.MAGENTA, width=2)))
            b.plotly_chart(fig, config={"displayModeBar": False})

        st.markdown("**Market-wide volatility indicators against their own history**")
        try:
            vh = vix_history()
            pcts = vix_percentiles(vh)
        except Exception as exc:
            vh, pcts = pd.DataFrame(), {}
            st.caption(f"Could not load VIX history: {type(exc).__name__}: {exc}")
        if pcts:
            from ivscreen import metrics as imx
            with st.container(horizontal=True):
                for name, v in pcts.items():
                    pct = v["pct"]
                    st.metric(name, f"{v['value']:.2f}" if name in ("VIX / VIX3M",) else f"{v['value']:.1f}" if name != "SPX 20d realized vol" else fmt_pct(v["value"]),
                              "n/a" if pct is None else f"{pct:.0f}th pct · {imx.classify(pct).lower()}", delta_color="off", delta_arrow="off", border=True,
                              help="Percentile of today's value among the prior two years of its own history (McMillan: each series only against itself).")
            if "^VIX" in vh and "^VIX3M" in vh:
                tail = vh[["^VIX", "^VIX3M"]].dropna().iloc[-252:]
                fig = _fig(280, margin=dict(l=8, r=8, t=16, b=8), yaxis_title="Index level", legend=_legend_h(-0.25))
                fig.add_trace(go.Scatter(x=tail.index, y=tail["^VIX"], name="VIX (30-day)", line=dict(color=wui.VIOLET, width=2)))
                fig.add_trace(go.Scatter(x=tail.index, y=tail["^VIX3M"], name="VIX3M (3-month)", line=dict(color=wui.TEAL, width=2)))
                st.plotly_chart(fig, config={"displayModeBar": False})
                st.caption("VIX above VIX3M (backwardation) means the market is paying more for near-term protection than for the next quarter — "
                           "typical of stress; the usual state is the reverse (contango).")
    return pcts


# ---------------------------------------------------------------------------
# 7 · decisions
# ---------------------------------------------------------------------------

def _account(client) -> Optional[object]:
    try:
        return client.get_account_summary() if client is not None else None
    except Exception:
        return None


def render_decisions(surf: surface.Surface, fs: model.FittedSurface, qs: data.QuoteSet, pcts: dict, client, opts_json: str) -> None:
    with st.container(key="glass_vs_dec"):
        st.subheader("7 · Decisions for your account", anchor=False)
        st.info(DISCLAIMER, icon=":material/shield:")
        cm = an.constant_maturity(fs)
        st.markdown("**What the surface and the indicators say today**")
        for item in an.scorecard(pcts, cm):
            icon = {"info": ":material/info:", "good": ":material/check_circle:", "warn": ":material/warning:"}[item["tone"]]
            st.markdown(f"{icon} **{item['title']}** — {item['text']}")

        fit = fs.fitted
        if not fit:
            return
        st.markdown("**Cash-secured put scan on the fitted smile**")
        st.caption("Every out-of-the-money put of one expiry against its fitted smile. RICH = even the bid implied vol is above the fit by at least a "
                   "quarter vol point (you could sell above the smooth surface at the bid); CHEAP = even the ask is that far below it; FAIR = the fit "
                   "sits inside (or within a quarter point of) the bid-ask band, so the difference is noise. P(below) is the risk-neutral probability of finishing under the strike, read off the smile.")
        labels = {f.expiry: f"{f.expiry} · {f.slice.days:.0f} days" for f in fit}
        c1, c2, c3, c4 = st.columns(4)
        exp = c1.selectbox("Expiry", list(labels), index=_default_index([f.slice for f in fit]), format_func=labels.get, key="vs_dec_exp")
        max_delta = c2.number_input("Max |delta|", 0.02, 0.5, float(wc.MAX_DELTA_ABS), 0.01, key="vs_dec_delta", help="Your wheel rule: 0.15.")
        max_spread = c3.number_input("Max spread (% of mid)", 5, 100, int(wc.MAX_SPREAD_PCT * 100), 5, key="vs_dec_spread", help="Your wheel rule: 40%.")
        min_bid = c4.number_input("Min bid ($)", 0.0, 50.0, 0.10, 0.05, key="vs_dec_minbid", help="Ignore dust premiums.")
        d1, d2 = st.columns(2)
        only = d1.segmented_control("Show", ["Passing my rules", "RICH only", "All"], default="Passing my rules", key="vs_dec_show") or "All"
        sort_label = d2.segmented_control("Sort by", ["Annualised yield", "Richest vs fit", "Strike"], default="Annualised yield", key="vs_dec_sort") or "Annualised yield"
        f = fs.by_expiry(exp)
        scan = an.put_scan(f, surf.spot, max_delta, max_spread / 100.0, min_bid, {"Annualised yield": "yield", "Richest vs fit": "rich", "Strike": "strike"}[sort_label])
        view = scan
        if only == "Passing my rules":
            view = scan[scan["passes"]]
        elif only == "RICH only":
            view = scan[scan["signal"] == "RICH"]
        acct = _account(client)
        sl = f.slice
        with st.container(horizontal=True):
            st.metric("Contracts shown", len(view), f"of {len(scan)} OTM puts", delta_color="off", delta_arrow="off", border=True)
            if len(view):
                cash = float(view["K"].median()) * 100
                st.metric("Cash to secure one contract", f"${cash:,.0f}", "median strike × 100", delta_color="off", delta_arrow="off", border=True)
            if acct is not None:
                st.metric("Your buying power", f"${acct.buying_power:,.0f}", f"cash ${acct.total_cash:,.0f}", delta_color="off", delta_arrow="off", border=True)
        if acct is not None and len(view):
            cash = float(view["K"].median()) * 100
            if cash > max(acct.buying_power, acct.total_cash):
                st.warning(f"One contract needs about ${cash:,.0f} of cash and your buying power is ${acct.buying_power:,.0f}: this underlying is too "
                           "large for the account. SPY or XSP are the practical analogues for index exposure.", icon=":material/account_balance_wallet:")
        if len(view):
            show = pd.DataFrame({
                "Strike": view["K"], "% below spot": (1 - view["K"] / (surf.spot or sl.F)) * 100, "Bid": view["bid"], "Ask": view["ask"], "Spread %": view["rel_spread"] * 100,
                "Market IV": view["iv"], "Fit IV": view["iv_fit"], "vs fit (vol pts)": view["diff_vp"], "Signal": view["signal"], "|Delta|": view["abs_delta"],
                "P(below) smile": view["p_below"] * 100, "P(below) flat": view["p_below_flat"] * 100, "Yield %": view["yield"] * 100,
                "Annualised %": view["annual_yield"] * 100, "Break-even cushion %": view["cushion"] * 100, "Passes my rules": view["passes"]})
            st.dataframe(show, hide_index=True, height=36 * (min(len(show), 14) + 1) + 4, column_config={
                "Strike": st.column_config.NumberColumn(format="%.0f"), "% below spot": st.column_config.NumberColumn(format="%.1f"),
                "Bid": st.column_config.NumberColumn(format="%.2f"), "Ask": st.column_config.NumberColumn(format="%.2f"),
                "Spread %": st.column_config.NumberColumn(format="%.0f"), "Market IV": st.column_config.NumberColumn(format="percent"),
                "Fit IV": st.column_config.NumberColumn(format="percent"), "vs fit (vol pts)": st.column_config.NumberColumn(format="%+.2f"),
                "|Delta|": st.column_config.NumberColumn(format="%.3f"), "P(below) smile": st.column_config.NumberColumn(format="%.1f"),
                "P(below) flat": st.column_config.NumberColumn(format="%.1f"), "Yield %": st.column_config.NumberColumn(format="%.2f"),
                "Annualised %": st.column_config.NumberColumn(format="%.1f"), "Break-even cushion %": st.column_config.NumberColumn(format="%.1f")})
            st.caption("Delta is Black-Scholes at the fitted vol (forward delta). Yield is bid ÷ strike; annualised on cash secured. Break-even cushion is how far "
                       "the underlying can fall from spot before the position loses money at expiry. A RICH signal means a favourable price versus the model — "
                       "it does not change the assignment risk or the gap risk.")
        else:
            st.caption("No put matches the current filters.")

        st.markdown("**Where your short options sit on their own smiles**")
        if client is None:
            st.caption("Connect to IBKR in the left rail to check your open short options.")
        elif st.button("Check my short options", icon=":material/fact_check:", key="vs_pos_run"):
            with st.spinner("Loading each position's option chain and fitting its smile…"):
                st.session_state["vs_pos"] = check_positions(client, fs.settings)
        pos = st.session_state.get("vs_pos")
        if pos is not None:
            if pos.empty:
                st.caption("No short option positions found in the account.")
            else:
                st.dataframe(pos, hide_index=True, column_config={
                    "Strike": st.column_config.NumberColumn(format="%.2f"), "Days": st.column_config.NumberColumn(format="%.0f"),
                    "Market IV": st.column_config.NumberColumn(format="percent"), "Fit IV": st.column_config.NumberColumn(format="percent"),
                    "vs fit (vol pts)": st.column_config.NumberColumn(format="%+.2f"), "P(ITM) smile %": st.column_config.NumberColumn(format="%.1f"),
                    "P(ITM) flat %": st.column_config.NumberColumn(format="%.1f"), "Model price": st.column_config.NumberColumn(format="%.2f"),
                    "Bid": st.column_config.NumberColumn(format="%.2f"), "Ask": st.column_config.NumberColumn(format="%.2f")})
                st.caption("Prices come from Yahoo (delayed). P(ITM) is the risk-neutral probability of finishing in the money at expiry, from the smile "
                           "versus a flat at-the-money vol. 'Model price' is the Black-Scholes value at the fitted vol.")


def check_positions(client, settings: model.FitSettings) -> pd.DataFrame:
    """Each open SHORT option position located on the fitted smile of its own expiry (Yahoo chains, read-only)."""
    rows = []
    positions = [p for p in client.get_positions() if p.sec_type == "OPT" and p.quantity < 0 and p.expiry and p.strike]
    groups: dict = {}
    for p in positions:
        groups.setdefault((p.symbol.replace(" ", "-"), f"{p.expiry[:4]}-{p.expiry[4:6]}-{p.expiry[6:8]}"), []).append(p)
    for (sym, exp), plist in groups.items():
        try:
            qs = data.fetch_yahoo(sym, [exp])
            surf = surface.build_surface(qs.quotes, qs.spot, qs.asof, sym, qs.source)
            fs = model.fit_surface(surf, settings)
            f = fs.by_expiry(exp) if surf.slices else None
        except Exception as exc:
            f = None
            note = f"{type(exc).__name__}: {exc}"
        else:
            note = "" if f is not None and f.fit else "no usable quotes for this expiry"
        for p in plist:
            base = {"Symbol": sym, "Expiry": exp, "Strike": p.strike, "Right": p.right, "Qty": p.quantity}
            if f is None or f.fit is None:
                rows.append({**base, "Note": note})
                continue
            r = an.check_position(f, float(p.strike), p.right or "P", surf.spot)
            rows.append({**base, "Days": r["days"], "Market IV": r["market_iv"], "Fit IV": r["fit_iv"], "vs fit (vol pts)": r["diff_vp"], "Signal": r["signal"],
                         "P(ITM) smile %": r["p_itm"] * 100, "P(ITM) flat %": r["p_itm_flat"] * 100, "Model price": r["model_price"], "Bid": r["bid"], "Ask": r["ask"],
                         "Note": "" if r["inside_data"] else "strike outside the quoted range — extrapolated"})
    return pd.DataFrame(rows)
