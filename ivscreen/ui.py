"""
ivscreen/ui.py — Streamlit rendering for the IV screener page. Helper functions only;
app_pages/iv_screener.py composes them top to bottom.

Everything shown is a PERCENTILE of a symbol's own history (IV against IV, RV against RV). Raw IV and RV
levels are deliberately not displayed here — see the Volatility & events page for the underlying charts.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import wheel_ui as wui
from stockscreen import prices as pr
from stockscreen import universe as su

from . import evaluate as ev
from . import metrics as mx
from . import scan as scan_mod
from . import store
from .config import INSTRUMENT_TYPES, IVConfig, load_config, save_config

DISCLAIMER = ("Research signals only, not investment advice. A high IV percentile means implied volatility is high compared with "
              "this stock's own past — it is not proof that options are overpriced or that selling them will be profitable "
              "(events such as earnings legitimately lift IV). A low percentile is not proof that options are cheap. "
              "Implied and realized volatility are each ranked against their own history and never compared as raw levels.")

BAND_CSS = {
    mx.VERY_LOW: "background-color: rgba(45,212,191,0.38); font-weight: 600",
    mx.LOW: "background-color: rgba(45,212,191,0.16)",
    mx.HIGH: "background-color: rgba(251,191,36,0.20)",
    mx.VERY_HIGH: "background-color: rgba(244,114,182,0.40); font-weight: 600",
}
REL_COLOR = {mx.REL_IV_ELEVATED: wui.VIOLET, mx.REL_RV_ELEVATED: wui.TEAL, mx.REL_BOTH_HIGH: wui.MAGENTA,
             mx.REL_BOTH_LOW: "#64748B", mx.REL_IN_LINE: "#94A3B8"}
DEFAULT_TYPES = ["Common stock", "ADR", "REIT / trust"]       # ETFs and "Other" are opt-in, never silently mixed in


@st.cache_resource
def scan_manager() -> scan_mod.IVScanManager:
    return scan_mod.IVScanManager()


def get_config() -> IVConfig:
    if "iv_cfg" not in st.session_state:
        st.session_state.iv_cfg = load_config()
    return st.session_state.iv_cfg


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def _price_stamp() -> str:
    try:
        with os.scandir(pr.PRICE_DIR) as it:
            files = [e.stat().st_mtime for e in it if e.name.endswith(".pkl")]
        return f"{len(files)}|{max(files) if files else 0:.0f}"
    except FileNotFoundError:
        return "0|0"


def _close(symbol: str) -> Optional[pd.Series]:
    df = pr.load_cached(symbol)
    return df["Close"] if df is not None and "Close" in df else None


@st.cache_data(show_spinner=False)
def _evaluate_cached(data_stamp: str, price_stamp: str, cfg_json: str, day: str, exclusions: str) -> pd.DataFrame:
    cfg = IVConfig.from_dict(json.loads(cfg_json))
    return ev.evaluate_all(store.load_symbols(), store.load_all_iv(), _close, cfg, date.fromisoformat(day), su.excluded())


def load_evaluated(cfg: IVConfig) -> pd.DataFrame:
    """Every stored symbol, classified. Cached on the stored data, price cache, settings, day and exclusions, so a
    settings change is instant and nothing is recomputed on an unchanged rerun."""
    if su.excluded():
        store.purge_symbols(su.excluded())                   # exclusions are absolute: nothing of theirs survives
    return _evaluate_cached(f"{store.stamp()}", _price_stamp(), json.dumps(cfg.to_dict(), sort_keys=True),
                            date.today().isoformat(), su.signature())


# ---------------------------------------------------------------------------
# coverage and summary
# ---------------------------------------------------------------------------

def render_coverage(df: pd.DataFrame, cfg: IVConfig) -> None:
    if not scan_manager().running:
        store.interrupt_orphans()
    run = store.latest_run()
    if run is None or df.empty:
        st.info("No scan has been run yet. Connect to IBKR in the left rail, choose a universe below and click **Run scan**.",
                icon=":material/info:")
        return
    n = len(df)
    ranked = int((df["status"] == ev.OK).sum())
    kind = {"complete": ("success", ":material/check_circle:"), "running": ("info", ":material/hourglass_top:")}.get(
        run["status"], ("warning", ":material/warning:"))
    asof = df["iv_asof"].dropna()
    checked = df["fetched_at"].dropna()
    msg = (f"**Last scan:** {run['universe_label']} · {n} symbols stored · **{ranked} ranked** · status **{run['status']}** · "
           f"started {run['started_at'][:16].replace('T', ' ')}"
           + (f" · latest implied-vol data point {asof.max()} (oldest ranked {df.loc[df['status'] == ev.OK, 'iv_asof'].min()})"
              if len(asof) and ranked else "")
           + (f" · last checked {max(checked)[:16].replace('T', ' ')}" if len(checked) else ""))
    getattr(st, kind[0])(msg, icon=kind[1])
    if run["status"] != "complete":
        st.warning("The last scan was **partial** (cancelled, failed or with errors) — do not read the counts as full coverage.",
                   icon=":material/warning:")
    counts = df["status"].value_counts()
    unavailable = int(sum(counts.get(s, 0) for s in ev.UNAVAILABLE))
    if unavailable or counts.get(ev.SHORT_HISTORY) or counts.get(ev.STALE):
        bits = [f"{int(counts.get(s, 0))} {ev.STATUS_TEXT[s].lower()}" for s in
                (ev.NO_OPTIONS_DATA, ev.FETCH_ERROR, ev.UNRESOLVED, ev.UNSUPPORTED, ev.DUPLICATE, ev.SHORT_HISTORY, ev.STALE, ev.NOT_SCANNED)
                if counts.get(s, 0)]
        st.caption("Not ranked, and kept out of every list below (their volatility is unknown, not low): " + "; ".join(bits) + ".")
    st.caption(f"The universe is a shortlist you control, not the whole market. Percentiles rank today's reading against the "
               f"prior {cfg.lookback_days} trading days of the same series (at least {cfg.min_history_days} needed). "
               "Implied vol is IBKR's 30-day figure; realized vol is computed from daily closes.")


def render_summary(view: pd.DataFrame) -> None:
    ok = view[view["status"] == ev.OK]
    c = lambda band: int((ok["iv_band"] == band).sum())                                   # noqa: E731
    with st.container(horizontal=True):
        st.metric("Ranked symbols", len(ok), border=True)
        st.metric("Very low IV", c(mx.VERY_LOW), border=True, help="IV percentile at or below the 'very low' threshold")
        st.metric("Low IV", c(mx.LOW), border=True)
        st.metric("High IV", c(mx.HIGH), border=True)
        st.metric("Very high IV", c(mx.VERY_HIGH), border=True)
        st.metric("Not ranked", int((view["status"] != ev.OK).sum()), border=True,
                  help="No options data, failed fetch, too little history, stale or unsupported — unknown, not low")


# ---------------------------------------------------------------------------
# scan controls
# ---------------------------------------------------------------------------

@st.fragment(run_every="2s")
def _progress_fragment() -> None:
    mgr = scan_manager()
    p = mgr.progress
    if mgr.running:
        eta = p.eta_seconds
        st.progress(min(p.fraction, 1.0), text=f"{p.phase}: {p.done}/{p.total} · {p.current} · {p.ok} ok, {p.no_options} no options "
                    f"data, {p.unsupported} unsupported, {p.errors} failed"
                    + (f" · ~{eta / 60:.0f} min left" if eta else ""))
        st.caption("The scan runs in the background — you can keep using other pages.")
    elif p.state in ("done", "cancelled", "error") and p.finished and not st.session_state.get("_ivscan_seen") == p.run_id:
        st.session_state["_ivscan_seen"] = p.run_id
        st.rerun()


def render_scan_controls(client, holdings: list[str]) -> None:
    cfg = get_config()
    mgr = scan_manager()
    with st.container(key="glass_ivscanctl"):
        st.subheader("Run a scan", anchor=False)
        c1, c2 = st.columns([3, 2])
        sources = c1.multiselect("Universe", list(scan_mod.ALL_SOURCES), default=[scan_mod.SOURCE_WHEEL, scan_mod.SOURCE_HOLDINGS],
                                 key="iv_sources", help="Wheel universe = your screener shortlist. ETFs are an opt-in category "
                                 "(edit the list in Settings). Manual exclusions always apply.")
        c2.text_input("Custom tickers", key="iv_custom", placeholder="e.g. NVDA, SAP, CHWY")
        uni = scan_mod.build_universe(sources, holdings, st.session_state.get("iv_custom", ""), cfg)
        st.caption(uni["note"] + ". Read-only: one IBKR historical-data request per symbol after the first scan, no orders.")
        with st.container(horizontal=True, vertical_alignment="center"):
            force = st.checkbox("Re-download full histories", key="iv_force",
                                help="Normally only the last ~2 months are fetched for symbols already stored.")
            if mgr.running:
                st.button("Cancel scan", icon=":material/cancel:", on_click=mgr.cancel, key="iv_cancel")
            elif client is None:
                st.button("Run scan", icon=":material/play_arrow:", type="primary", disabled=True, key="iv_run_off")
                st.caption("Connect to IBKR in the left rail to scan.")
            elif st.button("Run scan", icon=":material/play_arrow:", type="primary", disabled=not uni["symbols"], key="iv_run"):
                mgr.start(client, uni, " + ".join(sources) or "(none)", cfg, force=force)
                st.rerun()
        _progress_fragment()
        p = mgr.progress
        if not mgr.running and p.state in ("done", "cancelled", "error") and p.message:
            st.caption(f"Last run in this session: {p.message}")


# ---------------------------------------------------------------------------
# filters
# ---------------------------------------------------------------------------

def render_filters(df: pd.DataFrame, held: set[str]) -> pd.DataFrame:
    """Filters shared by every section below. Returns the rows that pass."""
    df = df.copy()
    df["_sources"] = df["sources"].fillna("").apply(lambda s: [x for x in s.split(",") if x])
    df["_sources"] = [src + (["Holdings"] if sym in held and "Holdings" not in src else []) for sym, src in zip(df["symbol"], df["_sources"])]
    present = [t for t in INSTRUMENT_TYPES if (df["instrument"] == t).any()]
    with st.container(key="glass_ivfilters"):
        st.subheader("Filters", anchor=False)
        a, b, c = st.columns([2, 2, 2])
        types = a.multiselect("Instrument type", present, default=[t for t in DEFAULT_TYPES if t in present], key="iv_f_types",
                              help="ETFs are a separate category and are not mixed in unless you add them here.")
        lists = b.multiselect("Watchlist", scan_mod.ALL_SOURCES, key="iv_f_lists", placeholder="Any",
                              help="Which universe list(s) the symbol came from. Holdings = your current stock positions.")
        text = c.text_input("Ticker contains", key="iv_f_text", placeholder="e.g. AXP, MSFT, N")
        d, e, f = st.columns([2, 2, 2])
        exch = d.multiselect("Exchange", sorted(x for x in df["exchange"].unique() if x), key="iv_f_exch", placeholder="Any")
        sect = e.multiselect("Sector (IBKR)", sorted(x for x in df["sector"].unique() if x), key="iv_f_sect", placeholder="Any")
        earn = f.selectbox("Earnings", ["Any", "Within the 30-day IV window", "Not within the window"], key="iv_f_earn",
                           help="Earnings inside the window lift 30-day IV for a known reason.")
    out = df
    out = out[out["instrument"].isin(types)] if types else out.iloc[0:0]
    if lists:
        out = out[out["_sources"].apply(lambda s: any(x in s for x in lists))]
    if text.strip():
        terms = [t.strip().upper() for t in text.replace(",", " ").split() if t.strip()]
        out = out[out["symbol"].apply(lambda s: any(t in s for t in terms))]
    if exch:
        out = out[out["exchange"].isin(exch)]
    if sect:
        out = out[out["sector"].isin(sect)]
    if earn == "Within the 30-day IV window":
        out = out[out["earnings_in_window"]]
    elif earn == "Not within the window":
        out = out[~out["earnings_in_window"]]
    hidden = len(df) - len(out)
    st.caption(f"{len(out)} of {len(df)} stored symbols match the filters" + (f" ({hidden} hidden)." if hidden else "."))
    return out.drop(columns=["_sources"])


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

_ARROW = {"Rising": "▲", "Falling": "▼", "Flat": "▬"}


def _trend_text(trend: str, chg) -> str:
    if chg is None or trend == mx.NA:
        return "n/a"
    return f"{_ARROW[trend]} {chg:+.0f}"


def _display(df: pd.DataFrame, with_event: bool = False) -> pd.DataFrame:
    out = pd.DataFrame({
        "Symbol": df["symbol"], "Name": df["name"], "Type": df["instrument"],
        "IV pct": df["iv_pct"], "IV band": df["iv_band"],
        "IV trend": [_trend_text(t, c) for t, c in zip(df["iv_trend"], df["iv_chg"])],
        "RV pct": df["rv_pct"], "RV band": df["rv_band"],
        "RV trend": [_trend_text(t, c) for t, c in zip(df["rv_trend"], df["rv_chg"])],
        "Relationship": df["relationship"],
        "Next earnings": df["next_earnings"], "In (days)": df["days_to_earnings"],
        "History": [f"{n}d" + ("*" if p else "") for n, p in zip(df["iv_obs"], df["partial_window"])],
    })
    if with_event:
        out["Event note"] = ["Earnings in the 30-day window — may explain elevated IV" if e else "" for e in df["earnings_in_window"]]
    return out.reset_index(drop=True)


def _config(cfg: IVConfig) -> dict:
    return {
        "IV pct": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f",
                                                  help="Share of the prior lookback window's daily IV readings below today's"),
        "RV pct": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f",
                                                  help=f"Same for {cfg.rv_window}-day realized volatility"),
        "IV trend": st.column_config.TextColumn(help=f"Change in IV percentile points over {cfg.trend_days} sessions "
                                                      f"(Flat inside ±{cfg.trend_tolerance:g})"),
        "RV trend": st.column_config.TextColumn(help=f"Change in RV percentile points over {cfg.trend_days} sessions"),
        "Next earnings": st.column_config.DateColumn(format="MMM DD"),
        "In (days)": st.column_config.NumberColumn(format="%d"),
        "History": st.column_config.TextColumn(help="Daily readings in the percentile window; * = shorter than the full lookback"),
    }


def _table(df: pd.DataFrame, key: str, cfg: IVConfig, with_event: bool = False, height_rows: int = 12, columns=None) -> None:
    disp = _display(df, with_event)
    if columns:
        disp = disp[columns]
    styled = disp.style.map(lambda v: BAND_CSS.get(v, ""), subset=[c for c in ("IV band", "RV band") if c in disp.columns])
    sel = st.dataframe(styled, hide_index=True, column_config=_config(cfg), on_select="rerun", selection_mode="single-row",
                       key=key, height=36 * (min(len(disp), height_rows) + 1) + 4)
    rows = sel.selection.rows if sel and sel.selection else []
    if rows:
        st.session_state["iv_pick"] = disp.iloc[rows[0]]["Symbol"]


def render_low(view: pd.DataFrame, cfg: IVConfig) -> None:
    ok = view[view["status"] == ev.OK]
    low = ok[ok["iv_pct"].apply(lambda p: mx.classify(p, cfg.very_low_max, cfg.low_max, cfg.high_min, cfg.very_high_min) in (mx.VERY_LOW, mx.LOW))]
    low = low.sort_values(["iv_pct", "symbol"])
    with st.container(key="glass_ivlow"):
        st.subheader("A · Low IV watchlist", anchor=False)
        st.caption(f"Implied volatility is low compared with the stock's own history: Very low ≤ {cfg.very_low_max:g}th percentile, "
                   f"Low > {cfg.very_low_max:g} and ≤ {cfg.low_max:g}. Lowest first. Click a row to open it in the comparison below.")
        if low.empty:
            st.caption("No ranked symbol is in the low zone with the current filters.")
            return
        vl = int((low["iv_band"] == mx.VERY_LOW).sum())
        st.caption(f"{len(low)} symbols — {vl} very low, {len(low) - vl} low.")
        _table(low, "iv_tbl_low", cfg)


def render_high(view: pd.DataFrame, cfg: IVConfig) -> None:
    ok = view[view["status"] == ev.OK]
    hi = ok[ok["iv_pct"].apply(lambda p: mx.classify(p, cfg.very_low_max, cfg.low_max, cfg.high_min, cfg.very_high_min) in (mx.HIGH, mx.VERY_HIGH))]
    hi = hi.sort_values(["iv_pct", "symbol"], ascending=[False, True])
    with st.container(key="glass_ivhigh"):
        st.subheader("B · High IV watchlist", anchor=False)
        st.caption(f"Options are pricing unusually high volatility compared with the stock's own history: High ≥ {cfg.high_min:g}th "
                   f"percentile, Very high ≥ {cfg.very_high_min:g}th (highlighted). Highest first. Check the event note before reading "
                   "anything into it — scheduled events lift IV for a reason.")
        if hi.empty:
            st.caption("No ranked symbol is in the high zone with the current filters.")
            return
        vh = int((hi["iv_band"] == mx.VERY_HIGH).sum())
        ev_n = int(hi["earnings_in_window"].sum())
        st.caption(f"{len(hi)} symbols — {vh} very high, {len(hi) - vh} high · {ev_n} have earnings inside the 30-day IV window.")
        _table(hi, "iv_tbl_high", cfg, with_event=True)


# ---------------------------------------------------------------------------
# IV vs RV comparison
# ---------------------------------------------------------------------------

def _scatter(both: pd.DataFrame, cfg: IVConfig) -> None:
    fig = go.Figure()
    fig.add_shape(type="rect", x0=cfg.high_min, x1=100, y0=0, y1=cfg.low_max, fillcolor="rgba(167,139,250,0.08)", line_width=0)
    fig.add_shape(type="rect", x0=0, x1=cfg.low_max, y0=cfg.high_min, y1=100, fillcolor="rgba(45,212,191,0.08)", line_width=0)
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", line=dict(color="rgba(230,234,245,0.3)", dash="dot", width=1),
                             hoverinfo="skip", showlegend=False))
    for rel in mx.RELATIONSHIPS[:-1]:
        d = both[both["relationship"] == rel]
        if d.empty:
            continue
        fig.add_trace(go.Scatter(
            x=d["iv_pct"], y=d["rv_pct"], mode="markers", name=f"{rel} ({len(d)})", customdata=d["symbol"],
            text=d["symbol"], marker=dict(size=9, color=REL_COLOR[rel], line=dict(width=1, color="rgba(0,0,0,0.35)"), opacity=0.9),
            hovertemplate="<b>%{text}</b><br>IV pct %{x:.0f} · RV pct %{y:.0f}<extra></extra>"))
    fig.update_layout(**wui.plotly_layout(height=430, hovermode="closest", showlegend=True, margin=dict(l=8, r=8, t=16, b=8),
                                          xaxis_title="IV percentile (vs its own history)", yaxis_title="RV percentile (vs its own history)",
                                          legend=dict(orientation="h", y=-0.2, yanchor="top", x=0, bgcolor="rgba(0,0,0,0)")))
    fig.update_xaxes(range=[-2, 102])
    fig.update_yaxes(range=[-2, 102])
    sel = st.plotly_chart(fig, config={"displayModeBar": False}, on_select="rerun", selection_mode="points", key="iv_scatter")
    pts = getattr(getattr(sel, "selection", None), "points", None) or []
    if pts and pts[0].get("customdata"):
        cd = pts[0]["customdata"]
        st.session_state["iv_pick"] = cd[0] if isinstance(cd, (list, tuple)) else cd


def render_comparison(view: pd.DataFrame, cfg: IVConfig) -> None:
    ok = view[view["status"] == ev.OK]
    both = ok[ok["rv_pct"].notna()]
    no_rv = ok[ok["rv_pct"].isna()]
    with st.container(key="glass_ivcompare"):
        st.subheader("C · IV percentile vs RV percentile", anchor=False)
        st.caption(f"Each stock's implied-vol percentile beside its {cfg.rv_window}-day realized-vol percentile, both against their own "
                   f"history. A gap of ≥ {cfg.gap_threshold:g} percentile points counts as one being elevated relative to the other. "
                   "This compares where each sits in its own range — not raw volatility levels, and not a verdict on whether options are overpriced.")
        if both.empty:
            st.caption("No symbol has both an IV and an RV percentile with the current filters.")
        else:
            counts = both["relationship"].value_counts()
            with st.container(horizontal=True):
                for rel in mx.RELATIONSHIPS[:-1]:
                    st.metric(rel, int(counts.get(rel, 0)), border=True)
            _scatter(both, cfg)
            pick_rel = st.pills("Show relationship", list(mx.RELATIONSHIPS[:-1]), selection_mode="multi",
                                default=list(mx.RELATIONSHIPS[:-1]), key="iv_rel_filter")
            tbl = both[both["relationship"].isin(pick_rel or [])].copy()
            tbl["abs_gap"] = tbl["gap"].abs()
            tbl = tbl.sort_values(["abs_gap", "symbol"], ascending=[False, True])
            disp = _display(tbl)
            disp["Gap (IV−RV)"] = tbl["gap"].to_numpy()
            disp["RV following IV?"] = tbl["trend_summary"].to_numpy()
            cols = ["Symbol", "Name", "IV pct", "IV band", "IV trend", "RV pct", "RV band", "RV trend", "Gap (IV−RV)",
                    "Relationship", "RV following IV?"]
            styled = disp[cols].style.map(lambda v: BAND_CSS.get(v, ""), subset=["IV band", "RV band"])
            sel = st.dataframe(styled, hide_index=True, column_config={**_config(cfg), "Gap (IV−RV)": st.column_config.NumberColumn(
                format="%+.0f", help="IV percentile minus RV percentile")}, on_select="rerun", selection_mode="single-row",
                key="iv_tbl_cmp", height=36 * (min(len(disp), 14) + 1) + 4)
            rows = sel.selection.rows if sel and sel.selection else []
            if rows:
                st.session_state["iv_pick"] = disp.iloc[rows[0]]["Symbol"]
        if len(no_rv):
            with st.expander(f"{len(no_rv)} ranked on IV but realized vol is unavailable", icon=":material/info:"):
                st.caption("Not enough price history (or stale prices) for a realized-vol percentile; their IV percentile is still valid. "
                           "Run a scan to download prices.")
                st.dataframe(no_rv[["symbol", "name", "instrument", "iv_pct", "iv_band", "rv_status"]].rename(columns={
                    "symbol": "Symbol", "name": "Name", "instrument": "Type", "iv_pct": "IV pct", "iv_band": "IV band",
                    "rv_status": "RV data"}), hide_index=True, column_config={"IV pct": st.column_config.ProgressColumn(
                        min_value=0, max_value=100, format="%.0f")})
    render_detail(ok, cfg)


def render_detail(ok: pd.DataFrame, cfg: IVConfig) -> None:
    if ok.empty:
        return
    syms = sorted(ok["symbol"])
    pick = st.session_state.get("iv_pick")
    with st.container(key="glass_ivdetail"):
        st.subheader("Symbol detail — percentile history", anchor=False)
        sym = st.selectbox("Symbol", syms, index=syms.index(pick) if pick in syms else 0, key=f"iv_detail_{pick}")
        row = ok[ok["symbol"] == sym].iloc[0]
        with st.container(horizontal=True):
            st.metric("IV percentile", f"{row['iv_pct']:.0f}", f"{row['iv_band']} · {_trend_text(row['iv_trend'], row['iv_chg'])}",
                      delta_color="off", delta_arrow="off", border=True)
            if row["rv_pct"] is not None and pd.notna(row["rv_pct"]):
                st.metric(f"RV percentile ({cfg.rv_window}d)", f"{row['rv_pct']:.0f}",
                          f"{row['rv_band']} · {_trend_text(row['rv_trend'], row['rv_chg'])}", delta_color="off", delta_arrow="off", border=True)
                st.metric("Gap (IV − RV)", f"{row['gap']:+.0f}", row["relationship"], delta_color="off", delta_arrow="off", border=True)
            else:
                st.metric(f"RV percentile ({cfg.rv_window}d)", "n/a", f"RV data: {row['rv_status']}", delta_color="off", delta_arrow="off", border=True)
        st.markdown(f"**{row['relationship']}.** {mx.REL_NOTE[row['relationship']]}")
        if row["trend_summary"] != "n/a":
            st.caption(f"Last {cfg.trend_days} sessions: {row['trend_summary']}.")
        if row["earnings_in_window"]:
            st.caption(f"Earnings on {row['next_earnings']} ({int(row['days_to_earnings'])} days) fall inside the 30-day IV window, "
                       "so part of any elevation is the known event.")
        if row["partial_window"]:
            st.caption(f"Only {int(row['iv_obs'])} daily readings are available (window is {cfg.lookback_days}): the percentile is "
                       "ranked against the shorter history.")
        hist = ev.percentile_history(store.load_iv(sym), _close(sym), cfg)
        if hist.empty:
            st.caption("Not enough history for a percentile chart.")
            return
        fig = go.Figure()
        fig.add_hrect(y0=cfg.high_min, y1=100, fillcolor="rgba(251,191,36,0.07)", line_width=0)
        fig.add_hrect(y0=0, y1=cfg.low_max, fillcolor="rgba(45,212,191,0.07)", line_width=0)
        if "iv_pct" in hist:
            fig.add_trace(go.Scatter(x=hist.index, y=hist["iv_pct"], name="IV percentile", mode="lines", line=dict(color=wui.VIOLET, width=2.4),
                                     hovertemplate="%{x|%b %d %Y}<br>IV pct %{y:.0f}<extra></extra>"))
        if "rv_pct" in hist:
            fig.add_trace(go.Scatter(x=hist.index, y=hist["rv_pct"], name=f"RV percentile ({cfg.rv_window}d)", mode="lines",
                                     line=dict(color=wui.TEAL, width=2), hovertemplate="%{x|%b %d %Y}<br>RV pct %{y:.0f}<extra></extra>"))
        fig.update_layout(**wui.plotly_layout(height=360, hovermode="x unified", showlegend=True, margin=dict(l=8, r=8, t=16, b=8),
                                              legend=dict(orientation="h", y=-0.14, yanchor="top", x=0, bgcolor="rgba(0,0,0,0)")))
        fig.update_yaxes(range=[0, 100], title="percentile of own history")
        st.plotly_chart(fig, config={"displayModeBar": False})
        st.caption(f"Each day's reading is ranked against the {cfg.lookback_days} trading days before it. Shaded: below "
                   f"{cfg.low_max:g} (low) and above {cfg.high_min:g} (high). Raw IV / RV levels and earnings overlays are on "
                   "the Volatility & events page.")
        if st.button("Open raw charts on Volatility & events", icon=":material/ssid_chart:", key=f"iv_open_vol_{sym}"):
            st.session_state["vol_symbol"] = sym
            st.switch_page("app_pages/volatility.py")


# ---------------------------------------------------------------------------
# data availability
# ---------------------------------------------------------------------------

def render_availability(view: pd.DataFrame) -> None:
    bad = view[view["status"] != ev.OK]
    with st.expander(f"Data availability — {len(bad)} of {len(view)} symbols not ranked", icon=":material/rule:"):
        st.caption("A symbol with no options data, a failed request or too little history is **unknown**, not low-IV. "
                   "None of these appear in the lists above.")
        by = view["status"].map(lambda s: ev.STATUS_TEXT.get(s, s)).value_counts().rename_axis("Status").reset_index(name="Symbols")
        st.dataframe(by, hide_index=True, width=420)
        if len(bad):
            st.dataframe(bad[["symbol", "name", "instrument", "status_text", "detail", "iv_asof", "iv_obs"]].rename(columns={
                "symbol": "Symbol", "name": "Name", "instrument": "Type", "status_text": "Why not ranked", "detail": "Detail",
                "iv_asof": "Latest IV point", "iv_obs": "Prior readings"}), hide_index=True,
                column_config={"Latest IV point": st.column_config.DateColumn(format="MMM DD YYYY")})


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------

def render_settings() -> None:
    cfg = get_config()
    with st.expander("Settings — bands, lookback, windows, trend, universe", icon=":material/tune:"):
        st.caption("Changing these re-classifies instantly (no rescan), except concurrency and the ETF list. Saved to ivscreen_config.json.")
        with st.form("iv_settings", border=False):
            st.markdown("**Percentile bands**")
            a, b, c, d = st.columns(4)
            vl = a.number_input("Very low ≤", 0.0, 100.0, cfg.very_low_max, 1.0)
            lo = b.number_input("Low ≤", 0.0, 100.0, cfg.low_max, 1.0)
            hi = c.number_input("High ≥", 0.0, 100.0, cfg.high_min, 1.0)
            vh = d.number_input("Very high ≥", 0.0, 100.0, cfg.very_high_min, 1.0)
            st.markdown("**Percentile definition**")
            e, f, g = st.columns(3)
            lookback = e.selectbox("Lookback", [252, 504, 756], [252, 504, 756].index(cfg.lookback_days) if cfg.lookback_days in (252, 504, 756) else 1,
                                   format_func=lambda n: {252: "1 year (252d)", 504: "2 years (504d)", 756: "3 years (756d)"}[n])
            rvw = f.selectbox("Realized-vol window (days)", [10, 20, 50, 100], [10, 20, 50, 100].index(cfg.rv_window) if cfg.rv_window in (10, 20, 50, 100) else 1)
            minh = g.number_input("Minimum prior readings", 20, 756, cfg.min_history_days, 10,
                                  help="Fewer prior readings than this => no percentile (never a guess)")
            st.markdown("**Trend and relationship**")
            h, i, j = st.columns(3)
            td = h.number_input("Trend window (sessions)", 2, 60, cfg.trend_days)
            tt = i.number_input("Flat if |change| below (pts)", 0.0, 50.0, cfg.trend_tolerance, 1.0)
            gap = j.number_input("IV/RV gap counts at (pts)", 5.0, 100.0, cfg.gap_threshold, 5.0)
            st.markdown("**Freshness and universe**")
            k, l, m = st.columns(3)
            iv_age = k.number_input("IV stale after (days)", 1, 60, cfg.max_iv_age_days)
            px_age = l.number_input("Prices stale after (days)", 1, 60, cfg.max_price_age_days)
            conc = m.number_input("Concurrent IBKR requests", 1, 8, cfg.concurrency)
            etfs = st.text_area("ETF list (opt-in universe category)", " ".join(cfg.etf_symbols), height=80)
            c1, c2 = st.columns(2)
            saved = c1.form_submit_button("Save settings", type="primary", icon=":material/save:")
            reset = c2.form_submit_button("Reset to defaults", icon=":material/restart_alt:")
        if reset:
            new = IVConfig()
            save_config(new)
            st.session_state.iv_cfg = new
            st.rerun()
        if saved:
            new = IVConfig.from_dict({**cfg.to_dict(), "very_low_max": vl, "low_max": lo, "high_min": hi, "very_high_min": vh,
                                      "lookback_days": lookback, "rv_window": rvw, "min_history_days": minh,
                                      "trend_days": td, "trend_tolerance": tt, "gap_threshold": gap, "max_iv_age_days": iv_age,
                                      "max_price_age_days": px_age, "concurrency": conc,
                                      "etf_symbols": [scan_mod.normalize_symbol(t) for t in etfs.replace(",", " ").split()]})
            problems = new.problems()
            if problems:
                st.error(" ".join(problems), icon=":material/error:")
            else:
                save_config(new)
                st.session_state.iv_cfg = new
                st.rerun()
