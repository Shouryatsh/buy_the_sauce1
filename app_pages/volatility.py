"""Volatility: implied vs ITS OWN history, realized vs ITS OWN history
(McMillan), with the events that move IV and an upcoming-events calendar."""

from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import wheel_analyst as wa
import wheel_events as ev
import wheel_ui as ui
import wheel_vol as wv
from wheel_watchlist import active_watchlist

CATEGORY_STYLE = {   # colour, marker symbol for event markers on the charts
    "Earnings": (ui.MAGENTA, "diamond"),
    "FOMC": (ui.AMBER, "triangle-up"),
    "CPI": ("#38BDF8", "square"),
    "Jobs": (ui.GREEN, "circle"),
}

client = ui.get_client()
held = sorted({p.symbol for p in client.get_positions()}) if client else []
core = held + [s for s in wa.WATCHLIST if s not in held]          # holdings + wheel watchlist
symbols = core + sorted(set(active_watchlist()) - set(core))         # + full screener universe (searchable)

st.caption("Following McMillan: implied volatility is compared only with its own history, and realized "
           "volatility only with its own history — each as a percentile of where it has been. "
           "IV and realized vol are not compared with each other (except at expiration).")

with st.container(horizontal=True, vertical_alignment="bottom", key="glass_volbar"):
    symbol = st.selectbox("Symbol", symbols, accept_new_options=True, key="vol_symbol", width=180)
    lb_name = st.segmented_control("Lookback", list(wv.LOOKBACKS), default="2Y", key="vol_lookback")
    hv_win = st.segmented_control("Realized-vol window (days)", list(wv.HV_WINDOWS), default=20, key="vol_hvwin")
    shown = st.pills("Events on charts", list(CATEGORY_STYLE), selection_mode="multi",
                     default=list(CATEGORY_STYLE), key="vol_events")

symbol = (symbol or "").upper().strip()
if not symbol:
    st.stop()
lookback = wv.LOOKBACKS[lb_name or "2Y"]
hv_win = hv_win or 20

# --- data ---------------------------------------------------------------
hist, src = ui.price_history(symbol, period_years=6)   # long enough for ~5 years of earnings moves
hv = wv.realized_vol(hist["Close"], hv_win).dropna() if hist is not None and not hist.empty else pd.Series(dtype=float)

iv, iv_error = pd.Series(dtype=float), None
if client is None:
    iv_error = "Connect to IBKR in the left rail to load implied-vol history (realized vol works without it)."
else:
    try:
        iv = ui.iv_history(symbol, "5 Y", _client=client)   # 5Y: ~20 past earnings to compare against
    except Exception as exc:
        iv_error = f"Implied-vol history unavailable: {exc}"

iv_read = wv.read(iv, lookback) if not iv.empty else wv.VolRead(None, None, None, None, None)
hv_read = wv.read(hv, lookback) if not hv.empty else wv.VolRead(None, None, None, None, None)

start_hist = date.today() - timedelta(days=int(lookback * 365 / 252) + 10)
events = ev.events_between([symbol], start_hist, date.today() + timedelta(days=60))
upcoming = events[events["date"] >= date.today()]


# --- KPI cards -----------------------------------------------------------
def pct_text(r: wv.VolRead) -> str:
    return f"{r.pct:.0f}th pct · {r.label}" if r.pct is not None else "n/a"


with st.container(horizontal=True):
    st.metric("Implied vol (30-day)", f"{iv_read.value:.1%}" if iv_read.value else "n/a", pct_text(iv_read),
              delta_color="off", delta_arrow="off", border=True,
              chart_data=(iv.tail(60) * 100).round(2).tolist() if not iv.empty else None, chart_type="area",
              help=f"Percentile of today's IV within its own {lb_name} history.")
    st.metric(f"Realized vol ({hv_win}d)", f"{hv_read.value:.1%}" if hv_read.value else "n/a", pct_text(hv_read),
              delta_color="off", delta_arrow="off", border=True,
              chart_data=(hv.tail(60) * 100).round(2).tolist() if not hv.empty else None, chart_type="area",
              help=f"Percentile of today's {hv_win}-day realized vol within its own {lb_name} history.")
    st.metric("IV percentile", f"{iv_read.pct:.0f}" if iv_read.pct is not None else "n/a",
              f"normal band {iv_read.lo:.0%}–{iv_read.hi:.0%}" if iv_read.lo else None,
              delta_color="off", delta_arrow="off", border=True)

if iv_error:
    st.info(iv_error, icon=":material/info:")
else:
    has_event = any(ev.days_until(d) <= 30 for d in upcoming["date"])
    st.markdown(f"**Read:** {wv.interpretation(iv_read, has_event)}")


# --- charts ---------------------------------------------------------------
def vol_chart(series: pd.Series, name: str, color: str, fill: str) -> go.Figure:
    s = series.dropna()
    bands = wv.rolling_percentile_bands(s, lookback)
    view = s.iloc[-lookback:]
    b = bands.loc[view.index]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=b.index, y=b["p90"], line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=b.index, y=b["p10"], fill="tonexty", fillcolor=fill, line=dict(width=0),
                             name="10th–90th percentile (trailing)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=b.index, y=b["p50"], name="Median (trailing)", mode="lines",
                             line=dict(color="rgba(230,234,245,0.45)", width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=view.index, y=view, name=name, mode="lines", line=dict(color=color, width=2),
                             hovertemplate="%{x|%b %d %Y}<br>" + name + " %{y:.1%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[view.index[-1]], y=[view.iloc[-1]], mode="markers", showlegend=False,
                             marker=dict(size=11, color=color, line=dict(width=2, color="white")),
                             hovertemplate="Today %{y:.1%}<extra></extra>"))
    # Event markers below the series, one row per event type (earnings on top,
    # drawn last so nothing covers it); hover names the event.
    lo, span = float(view.min()), float(view.max() - view.min()) or 0.01
    rows = {"Jobs": 4, "CPI": 3, "FOMC": 2, "Earnings": 1}
    for cat in ("Jobs", "CPI", "FOMC", "Earnings"):
        ccol, sym = CATEGORY_STYLE[cat]
        if cat not in (shown or []):
            continue
        y0 = lo - span * 0.06 * rows[cat]
        e = events[(events["category"] == cat) & (events["date"] >= view.index[0].date())
                   & (events["date"] <= date.today())]
        if e.empty:
            continue
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(e["date"]), y=[y0] * len(e), mode="markers", name=cat,
            marker=dict(symbol=sym, size=9, color=ccol, line=dict(width=1, color="rgba(0,0,0,0.4)")),
            text=e["event"], hovertemplate="%{x|%b %d %Y}<br>%{text}<extra></extra>"))
        if cat == "Earnings":   # earnings move single-stock IV most: draw full-height guides
            for d in e["date"]:
                fig.add_vline(x=pd.Timestamp(d), line=dict(color="rgba(244,114,182,0.6)", width=1.3, dash="dot"))
    fig.update_layout(**ui.plotly_layout(height=420, hovermode="x unified", showlegend=True,
                                         margin=dict(l=8, r=8, t=16, b=8),
                                         legend=dict(orientation="h", y=-0.12, yanchor="top", x=0,
                                                     bgcolor="rgba(0,0,0,0)", font=dict(size=11))))
    fig.update_yaxes(tickformat=".0%")
    return fig


left, right = st.columns(2, gap="medium")
with left:
    with st.container(key="glass_ivchart"):
        st.subheader("Implied vol vs its own history", anchor=False)
        if iv.empty:
            st.caption(iv_error or "No implied-vol history.")
        else:
            st.plotly_chart(vol_chart(iv, "Implied vol", ui.VIOLET, "rgba(167,139,250,0.14)"),
                            config={"displayModeBar": False})
            st.caption("IBKR 30-day implied volatility. Shaded band = where IV has normally been "
                       f"(trailing {lb_name} 10th–90th percentile).")
with right:
    with st.container(key="glass_hvchart"):
        st.subheader("Realized vol vs its own history", anchor=False)
        if hv.empty:
            st.caption("No price history.")
        else:
            st.plotly_chart(vol_chart(hv, f"Realized vol {hv_win}d", ui.TEAL, "rgba(45,212,191,0.12)"),
                            config={"displayModeBar": False})
            st.caption(f"{hv_win}-day close-to-close realized volatility from {src} daily prices.")


# --- where today sits in the distribution ----------------------------------
def distribution(series: pd.Series, name: str, color: str) -> go.Figure:
    s = series.dropna().iloc[-(lookback + 1):]
    fig = go.Figure(go.Histogram(x=s.iloc[:-1], nbinsx=40, marker=dict(color=color, opacity=0.55), name=name,
                                 hovertemplate="%{x:.1%}: %{y} days<extra></extra>"))
    fig.add_vline(x=s.iloc[-1], line=dict(color="white", width=2),
                  annotation_text=f"today {s.iloc[-1]:.1%}", annotation_position="top")
    fig.update_layout(**ui.plotly_layout(height=200, showlegend=False, bargap=0.05))
    fig.update_xaxes(tickformat=".0%")
    return fig


with st.container(key="glass_dists"):
    st.subheader(f"Where today sits ({lb_name} of daily readings)", anchor=False)
    d1, d2 = st.columns(2)
    if not iv.empty:
        d1.plotly_chart(distribution(iv, "IV", ui.VIOLET), config={"displayModeBar": False})
        d1.caption(f"Implied vol: lower than today on {iv_read.pct:.0f}% of days.")
    if not hv.empty:
        d2.plotly_chart(distribution(hv, "HV", ui.TEAL), config={"displayModeBar": False})
        d2.caption(f"Realized vol: lower than today on {hv_read.pct:.0f}% of days.")


# --- earnings: IV run-up and crush, today vs the same point in past cycles ---
E_K_MIN, E_K_MAX = -30, 5   # -30: the report enters IBKR's 30-day IV window ~21 sessions out
earn_events = ev.earnings_events(symbol)
past_events = [(d, t) for d, t in earn_events if d < date.today()]
next_event = next(((d, t) for d, t in earn_events if d >= date.today()), None)

with st.container(key="glass_earnings"):
    st.subheader("Earnings: implied-vol run-up and crush", anchor=False)
    st.caption("Each past earnings is aligned on the last close before the announcement (day 0). "
               "Day −5 is one trading week before; day +1 is the first close after the news. "
               "Comparing today's IV with IV at the same point in past earnings cycles shows whether it is "
               "high even allowing for the coming event.")
    if iv.empty:
        st.caption(iv_error or "No implied-vol history.")
    elif not earn_events:
        st.caption(f"No earnings dates found for {symbol} (ETFs have none).")
    else:
        table = wv.earnings_table(iv, hist["Close"] if hist is not None else None, past_events)
        cmp_ = wv.compare_to_past_earnings(iv, past_events, next_event) if next_event else None

        with st.container(horizontal=True):
            if next_event:
                d, t = next_event
                st.metric("Next earnings", f"{d:%b %d}",
                          f"{'before open' if t == 'BMO' else 'after close' if t == 'AMC' else 'time n/a'}"
                          + (f" · {-cmp_.k} sessions to day 0" if cmp_ else ""),
                          delta_color="off", delta_arrow="off", border=True)
            if cmp_ and cmp_.pct is not None:
                st.metric("IV vs same point, past cycles", f"{cmp_.pct:.0f}th pct",
                          f"today {cmp_.today:.1%} · past median {cmp_.same_point.median():.1%}",
                          delta_color="off", delta_arrow="off", border=True,
                          help=f"Share of the last {len(cmp_.same_point)} earnings where IV {-cmp_.k} sessions "
                               "before day 0 was lower than today's IV.")
            if not table.empty:
                st.metric("Typical run-up (week before → day 0)", f"{table['Run-up'].median():+.0%}",
                          f"median of {len(table)} earnings", delta_color="off", delta_arrow="off", border=True)
                st.metric("Typical crush (day 0 → day +1)", f"{table['Crush'].median():+.0%}",
                          f"median stock move ±{table['Stock move'].abs().median():.1%}"
                          if table["Stock move"].notna().any() else None,
                          delta_color="off", delta_arrow="off", border=True)

        if cmp_ and cmp_.pct is not None:
            if cmp_.pct >= 75:
                verdict = (f"IV is **high even for this point before earnings** — above {cmp_.pct:.0f}% of past cycles "
                           "at the same distance from the report. Premium is richer than the event alone usually explains.")
            elif cmp_.pct <= 25:
                verdict = (f"IV is **low for this point before earnings** — below most past cycles "
                           f"({cmp_.pct:.0f}th percentile). The usual run-up may still be ahead, or the market expects a quiet report.")
            else:
                verdict = (f"IV is **in its usual range for this point before earnings** ({cmp_.pct:.0f}th percentile "
                           "of past cycles).")
            if cmp_.typical_runup_from_here is not None and cmp_.typical_peak is not None:
                verdict += (f" In past cycles IV rose a median **{cmp_.typical_runup_from_here:+.0%}** from here to day 0 "
                            f"(typical day-0 IV {cmp_.typical_peak:.1%}).")
            st.markdown(verdict)
        elif next_event:
            st.caption("Next earnings is too far out (or too few past events) for a same-point comparison.")

        paths = wv.event_paths(iv, past_events, E_K_MIN, E_K_MAX)
        if not paths.empty:
            fig = go.Figure()
            for col in paths.columns:
                fig.add_trace(go.Scatter(x=paths.index, y=paths[col], mode="lines", name=col, showlegend=False,
                                         line=dict(color="rgba(167,139,250,0.28)", width=1),
                                         hovertemplate=f"{col}<br>day %{{x}}: %{{y:.1%}}<extra></extra>"))
            fig.add_trace(go.Scatter(x=paths.index, y=paths.median(axis=1), mode="lines", name="Median of past earnings",
                                     line=dict(color=ui.VIOLET, width=3),
                                     hovertemplate="median day %{x}: %{y:.1%}<extra></extra>"))
            if cmp_ and cmp_.k >= E_K_MIN:
                n = cmp_.k - E_K_MIN + 1
                today_path = iv.dropna().iloc[-n:]
                fig.add_trace(go.Scatter(x=list(range(E_K_MIN, cmp_.k + 1)), y=today_path.values, mode="lines+markers",
                                         name=f"Now (next: {next_event[0]:%b %d})", line=dict(color=ui.TEAL, width=3),
                                         marker=dict(size=5), hovertemplate="now, day %{x}: %{y:.1%}<extra></extra>"))
            fig.add_vline(x=0, line=dict(color=ui.MAGENTA, width=1.5, dash="dot"),
                          annotation_text="day 0 · last close before report", annotation_position="top left")
            fig.add_vline(x=-5, line=dict(color="rgba(230,234,245,0.35)", width=1, dash="dot"),
                          annotation_text="1 week before", annotation_position="bottom right")
            fig.update_layout(**ui.plotly_layout(height=400, hovermode="closest", showlegend=True,
                                                 legend=dict(orientation="h", y=-0.15, yanchor="top", x=0,
                                                             bgcolor="rgba(0,0,0,0)"),
                                                 xaxis_title="Trading days relative to the last close before earnings"))
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, config={"displayModeBar": False})

        if not table.empty:
            st.dataframe(table.sort_values("Earnings", ascending=False), hide_index=True, column_config={
                "Earnings": st.column_config.DateColumn(format="MMM DD YYYY"),
                "Timing": st.column_config.TextColumn(help="BMO = before the open, AMC = after the close"),
                "IV 1 week before": st.column_config.NumberColumn(format="percent"),
                "IV last close before": st.column_config.NumberColumn(format="percent"),
                "Run-up": st.column_config.NumberColumn(format="percent", help="IV at day 0 vs one week before"),
                "IV after": st.column_config.NumberColumn(format="percent"),
                "Crush": st.column_config.NumberColumn(format="percent", help="IV at day +1 vs day 0"),
                "Stock move": st.column_config.NumberColumn(format="percent",
                                                            help="Close at day 0 to close at day +1"),
            })
        st.caption("IBKR's 30-day implied volatility for the stock (not one specific expiry). Because it is a "
                   "30-day measure, the coming report is already priced in once it falls inside that window "
                   "(~21 sessions out), so much of the earnings premium appears before the final week and the "
                   "week-before run-up can look small; the same-point comparison and the crush are the clearest "
                   "signals. Earnings dates and before/after-market timing from Yahoo Finance. Day counts skip "
                   "weekends, not exchange holidays.")


# --- watchlist scan ---------------------------------------------------------
SCAN_SCOPES = ["Holdings & watchlist", "Fundamentals passes", f"Full universe ({len(symbols)})"]

with st.container(key="glass_volscan"):
    st.subheader("Volatility scan", anchor=False)
    st.caption("IV and realized-vol percentiles for every name, each against its own history, plus IV versus "
               "the same point in past earnings cycles when earnings is within 30 days. "
               "One IBKR history request per symbol.")
    st.page_link("app_pages/iv_screener.py", icon=":material/vertical_align_center:",
                 label="Want a saved scan with low/high-IV watchlists, filters and IV-vs-RV percentiles? Open the IV screener.")
    scope = st.segmented_control("Scope", SCAN_SCOPES, default="Holdings & watchlist", key="vol_scope")
    profiles = ui.profiles_or_none()
    if scope == "Fundamentals passes":
        scan_syms = [p.symbol for p in profiles if p.passes] if profiles else []
        if profiles is None:
            ui.need_screen("need_screen_volscan")
    elif scope and scope.startswith("Full universe"):
        scan_syms = symbols
        st.caption(f"{len(scan_syms)} symbols — this takes several minutes (price history + IBKR request each).")
    else:
        scan_syms = core
    if client is None:
        st.caption("Connect to IBKR to scan.")
    elif scan_syms and st.button(f"Scan {len(scan_syms)} symbols", icon=":material/radar:", key="vol_scan"):
        rows = []
        prog = st.progress(0.0)
        for i, sym in enumerate(scan_syms):
            prog.progress(i / len(scan_syms), text=f"{sym} ({i + 1}/{len(scan_syms)})")
            try:
                s_iv = ui.iv_history(sym, "5 Y", _client=client)
            except Exception:
                s_iv = pd.Series(dtype=float)
            h, _ = ui.price_history(sym, period_years=6)
            s_hv = wv.realized_vol(h["Close"], hv_win).dropna() if h is not None and not h.empty else pd.Series(dtype=float)
            r_iv, r_hv = wv.read(s_iv, lookback), wv.read(s_hv, lookback)
            evs = ev.earnings_events(sym)
            nxt = next(((d, t) for d, t in evs if d >= date.today()), None)
            cyc = None
            if nxt and not s_iv.empty and ev.days_until(nxt[0]) <= 30:
                c = wv.compare_to_past_earnings(s_iv, [(d, t) for d, t in evs if d < date.today()], nxt)
                cyc = c.pct if c else None
            rows.append({"Symbol": sym, "Held": sym in held, "IV": r_iv.value, "IV pct": r_iv.pct,
                         "IV read": r_iv.label, f"HV {hv_win}d": r_hv.value, "HV pct": r_hv.pct,
                         "HV read": r_hv.label, "Next earnings": nxt[0] if nxt else None,
                         "Days to earnings": ev.days_until(nxt[0]) if nxt else None,
                         "IV vs past earnings cycles": cyc})
        prog.empty()
        st.session_state.vol_scan_rows = rows
    if st.session_state.get("vol_scan_rows"):
        scan_df = pd.DataFrame(st.session_state.vol_scan_rows).sort_values("IV pct", ascending=False)
        st.dataframe(scan_df, hide_index=True, height=36 * (len(scan_df) + 1) + 4, column_config={
                         "IV": st.column_config.NumberColumn(format="percent"),
                         f"HV {hv_win}d": st.column_config.NumberColumn(format="percent"),
                         "IV pct": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
                         "HV pct": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
                         "IV vs past earnings cycles": st.column_config.ProgressColumn(
                             min_value=0, max_value=100, format="%.0f",
                             help="Percentile of today's IV among past earnings at the same distance from the report "
                                  "(only when earnings is within 30 days)"),
                         "Next earnings": st.column_config.DateColumn(format="MMM DD"),
                     })


# --- events calendar --------------------------------------------------------
with st.container(key="glass_calendar"):
    st.subheader("Events calendar — next 60 days", anchor=False)
    st.caption("Market-wide releases (FOMC, CPI, jobs report) tend to lift IV across all names into the "
               "release; earnings lift a single stock's IV. IV usually falls right after the event.")
    cal_syms = st.multiselect("Earnings for", symbols, default=held or symbols[:5], key="cal_syms")
    cats = st.pills("Show", ["FOMC", "CPI", "Jobs", "Earnings"], selection_mode="multi",
                    default=["FOMC", "CPI", "Jobs", "Earnings"], key="cal_cats")
    cal = ev.events_between(cal_syms, date.today(), date.today() + timedelta(days=60))
    cal = cal[cal["category"].isin(cats or [])]
    if cal.empty:
        st.caption("No events in range.")
    else:
        cal = cal.assign(**{"In (days)": [ev.days_until(d) for d in cal["date"]]})
        st.dataframe(cal[["date", "In (days)", "event", "category", "scope", "time_et"]].rename(columns={
            "date": "Date", "event": "Event", "category": "Type", "scope": "Affects", "time_et": "Time (ET)"}),
            hide_index=True, column_config={"Date": st.column_config.DateColumn(format="ddd MMM DD")})
    st.caption("Macro dates: federalreserve.gov and bls.gov schedules (macro_calendar.csv — add rows for other "
               "releases). Earnings: Yahoo Finance; confirm with the company before trading around them.")
