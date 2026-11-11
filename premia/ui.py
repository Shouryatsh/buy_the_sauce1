"""
premia/ui.py — the dashboard's "Premia research" page: verdicts, walk-forward results, direction independence,
P&L attribution, hedging, sensitivity and the vol-model evidence.

Reads the saved research outputs in results/premia/ only (written by `python -m premia.research.*`). Opens no broker
connection and builds no order.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import wheel_ui as wui

from .config import RESULTS_DIR

PHASES = [("phase1", "1 Data"), ("phase2", "2 Screener"), ("phase3", "3 Vol models"), ("phase3b", "3B Attribution"),
          ("phase4", "4 Strategy models"), ("phase5", "5 Direction independence"), ("phase6", "6 Backtester"),
          ("phase7", "7 Risk & execution")]
SLEEVES = {"combined": wui.MAGENTA, "wheel": wui.TEAL, "straddle": wui.VIOLET}     # fixed order, fixed colour
NOTES = {
    "phase3": "Judgment call: work continued under SPEC §6's drop rule. FHS probabilities are dropped; FHS paths are "
              "used only for hedged P&L, which depends on realized variance, not drift. With a point-in-time drift the "
              "calibration error falls to 1.9 pts.",
    "phase3b": "The only failed criterion (detect Sharpe 0.5 at 80 % power) needs ~31 years of daily data; the test "
               "that can be run (size ≤ 10 %, power ≥ 80 % at the minimum detectable effect) passes.",
    "phase4": "'Beats the baseline' here means less negative: both models and both baselines lose money out of sample.",
    "phase5": "Direction independence is achieved; the net return after costs is negative.",
    "phase7": "Tests only. The 2–3 month paper-trading period has not been run.",
}
DISCLAIMER = ("Research results, read from saved files. Backtests price options from real 30-day ATM IV but a modelled "
              "smile and spread; they are not predictions. Nothing on this page connects to a broker or places an order.")


# ---------------------------------------------------------------------------
# loading (cached on file modification time, so a re-run of the research shows up on refresh)
# ---------------------------------------------------------------------------

def _path(name: str) -> Path:
    return RESULTS_DIR / name


@st.cache_data(ttl=3600, max_entries=64, show_spinner=False)
def _read(path: str, mtime: float):
    p = Path(path)
    if p.suffix == ".json":
        return json.loads(p.read_text())
    return pd.read_csv(p)


def load(name: str):
    p = _path(name)
    return _read(str(p), p.stat().st_mtime) if p.exists() else None


def _fig(height: int = 360, **kw) -> go.Figure:
    f = go.Figure()
    f.update_layout(**wui.plotly_layout(height=height, **kw))
    return f


def _legend():
    return dict(orientation="h", y=1.02, yanchor="bottom", x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=11))


def _pct(x, d=2) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x * 100:.{d}f}%"


def _missing(what: str, cmd: str) -> None:
    st.info(f"No {what} yet. Run `{cmd}` to produce it.", icon=":material/hourglass_empty:")


# ---------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------

def verdict_table() -> pd.DataFrame:
    rows = []
    for key, label in PHASES:
        v = load(f"verdict_{key}.json")
        if v is None:
            rows.append({"Phase": label, "Verdict": "not run", "Evidence": "", "At": ""})
            continue
        failed = [c for c in v["checks"] if not c["pass"]]
        shown = failed or v["checks"][-1:]
        ev = "; ".join(f"{c['criterion']}: {c['evidence']}" for c in shown)
        rows.append({"Phase": label, "Verdict": v["verdict"], "Evidence": ev, "At": v.get("at", "")[:16].replace("T", " ")})
    return pd.DataFrame(rows)


def render_verdicts() -> None:
    vt = verdict_table()
    p5 = vt.loc[vt["Phase"].str.startswith("5"), "Verdict"]
    if len(p5) and p5.iloc[0] == "FAIL":
        st.error("**Phase 5 fails: no positive return after costs while direction-neutral.** Per the working rules the "
                 "system is not ready for paper trading as a profit strategy. The deciding next step is real historical "
                 "option chains (see the sensitivity tab: the wheel's result hinges on the skew).", icon=":material/block:")
    with st.container(horizontal=True):
        for r in vt.itertuples():
            icon = {"PASS": ":material/check_circle:", "FAIL": ":material/cancel:"}.get(r.Verdict, ":material/schedule:")
            color = {"PASS": "green", "FAIL": "red"}.get(r.Verdict, "gray")
            st.badge(f"{r.Phase}: {r.Verdict}", icon=icon, color=color)
    st.dataframe(vt, hide_index=True, width="stretch",
                 column_config={"Evidence": st.column_config.TextColumn(width="large"),
                                "Verdict": st.column_config.TextColumn(width="small")})
    with st.expander("Every criterion, as pre-registered in SPEC.md §6", icon=":material/checklist:"):
        for key, label in PHASES:
            v = load(f"verdict_{key}.json")
            if v is None:
                continue
            st.markdown(f"**{label}: {v['verdict']}**")
            st.dataframe(pd.DataFrame(v["checks"]).rename(columns=str.capitalize), hide_index=True, width="stretch",
                         column_config={"Pass": st.column_config.CheckboxColumn(width="small")})
            if key in NOTES:
                st.caption(NOTES[key])


# ---------------------------------------------------------------------------
# performance
# ---------------------------------------------------------------------------

def render_performance() -> None:
    nav = load("wf_oos_nav.csv")
    a = load("analysis.json")
    if nav is None or a is None:
        _missing("walk-forward results", "python -m premia.research.analysis")
        return
    nav = nav.set_index(pd.to_datetime(nav["date"])).drop(columns="date")
    perf = a["performance_oos"]
    beta = a["phase5"]
    st.caption("Walk-forward, out of sample from 2013-01: every 6 months the configuration with the best trailing "
               "3-year Sharpe is chosen and run for the next 6 months. $1 M book.")
    with st.container(horizontal=True):
        for s in SLEEVES:
            p = perf[f"{s}_wf"]
            with st.container(border=True):
                st.markdown(f"**{s.capitalize()}**")
                with st.container(horizontal=True):
                    st.metric("CAGR", _pct(p["cagr"]))
                    st.metric("Sharpe", f"{p['sharpe']:.2f}")
                    st.metric("Max drawdown", _pct(p["max_drawdown"], 1))
                    st.metric("β to SPY", f"{beta[s]['beta']['beta']:.3f}",
                              help=f"Newey-West t {beta[s]['beta']['t_hac']:.2f}, correlation {beta[s]['beta']['corr']:.2f}")

    f = _fig(380, legend=_legend(), yaxis=dict(title="Return since 2013", tickformat=".0%",
                                               gridcolor=wui.plotly_layout()["yaxis"]["gridcolor"], zeroline=False))
    for s, c in SLEEVES.items():
        f.add_scatter(x=nav.index, y=nav[s] / 1e6 - 1, name=s.capitalize(), line=dict(color=c, width=2),
                      hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2%}<extra>" + s + "</extra>")
    f.update_layout(hovermode="x unified")
    st.plotly_chart(f, width="stretch", config={"displayModeBar": False})

    y = nav.resample("YE").last()
    y.loc[pd.Timestamp(nav.index[0].year - 1, 12, 31)] = 1e6
    y = y.sort_index().pct_change().dropna()
    y.index = [str(i.year) + (" YTD" if i.year == nav.index[-1].year else "") for i in y.index]
    st.markdown("**Calendar-year returns**")
    st.dataframe(y.T.rename(index=str.capitalize).style.format("{:.2%}"), width="stretch")

    others = {k: v for k, v in perf.items() if not k.endswith("_wf")}
    if others:
        with st.expander("All 24 backtest configurations (OOS period, before walk-forward selection)",
                         icon=":material/table_rows:"):
            t = pd.DataFrame(others).T[["cagr", "ann_vol", "sharpe", "max_drawdown", "mean_t_hac"]]
            t.columns = ["CAGR", "Annual vol", "Sharpe", "Max drawdown", "Mean-return HAC t"]
            st.dataframe(t.sort_index().style.format({"CAGR": "{:.2%}", "Annual vol": "{:.2%}", "Sharpe": "{:.2f}",
                                                      "Max drawdown": "{:.1%}", "Mean-return HAC t": "{:.2f}"}),
                         width="stretch")
            st.caption(f"Deflated Sharpe (9 trials each): wheel {a['dsr_wheel']['dsr']:.3f}, "
                       f"straddle {a['dsr_straddle']['dsr']:.4f}. A value near 1 would mean the result survives "
                       "the number of configurations tried.")
    ch = load("wf_wheel_choices.csv")
    if ch is not None:
        with st.expander("Walk-forward choices per 6-month period", icon=":material/history:"):
            cs = load("wf_straddle_choices.csv")
            both = ch.rename(columns={"chosen": "wheel", "train_sharpe": "wheel train Sharpe"})
            if cs is not None:
                both = both.merge(cs.rename(columns={"chosen": "straddle", "train_sharpe": "straddle train Sharpe"}),
                                  on="boundary", how="outer")
            st.dataframe(both, hide_index=True, width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="%.2f") for c in both.columns if "Sharpe" in c})


# ---------------------------------------------------------------------------
# direction independence
# ---------------------------------------------------------------------------

def render_direction() -> None:
    a = load("analysis.json")
    if a is None:
        _missing("phase 5 results", "python -m premia.research.analysis")
        return
    p5 = a["phase5"]
    st.caption("Pass needs |β| < 0.10 and not significant, |correlation| < 0.20 to SPY and to the underlyings, "
               "**and** a positive net return. Market months: SPY down < −2 %, flat ±2 %, up > +2 %.")
    f = _fig(320, legend=_legend(), barmode="group",
             yaxis=dict(title="Mean monthly return", tickformat=".2%", zeroline=True, zerolinewidth=1,
                        gridcolor=wui.plotly_layout()["yaxis"]["gridcolor"]))
    cats = [("market_down", "SPY down"), ("market_flat", "SPY flat"), ("market_up", "SPY up")]
    for s, c in SLEEVES.items():
        rg = p5[s]["regimes"]
        f.add_bar(x=[l for _, l in cats], y=[rg[k]["mean_monthly_return"] for k, _ in cats], name=s.capitalize(),
                  marker=dict(color=c, cornerradius=4), customdata=[[rg[k]["months"], rg[k]["share_positive"]] for k, _ in cats],
                  hovertemplate="%{x}: %{y:.3%} per month<br>%{customdata[0]} months, %{customdata[1]:.0%} positive"
                                "<extra>" + s + "</extra>")
    f.update_layout(bargap=0.25, bargroupgap=0.08)
    st.plotly_chart(f, width="stretch", config={"displayModeBar": False})

    rows = []
    for s in SLEEVES:
        b, rg = p5[s]["beta"], p5[s]["regimes"]
        rows.append({"Sleeve": s.capitalize(), "β to SPY": b["beta"], "NW t": b["t_hac"], "Corr to SPY": b["corr"],
                     "High VIX (ann.)": rg["vix_high"]["ann_return"], "Low VIX (ann.)": rg["vix_low"]["ann_return"],
                     "Feb–Apr 2020": rg["crash_2020"]["return"], "2022": rg["year_2022"]["return"],
                     "Feb 2018": rg["volmageddon_2018"]["return"]})
    spy = p5["combined"]["regimes"]
    rows.append({"Sleeve": "SPY (reference)", "Feb–Apr 2020": spy["crash_2020"]["spy_return"],
                 "2022": spy["year_2022"]["spy_return"], "Feb 2018": spy["volmageddon_2018"]["spy_return"]})
    t = pd.DataFrame(rows).set_index("Sleeve")
    st.dataframe(t.style.format({"β to SPY": "{:.3f}", "NW t": "{:.2f}", "Corr to SPY": "{:.2f}",
                                 **{c: "{:.2%}" for c in t.columns[4:]}}, na_rep=""), width="stretch")

    st.markdown("**Hedged vs unhedged wheel**: what removing the direction costs")
    h = []
    for lab, name in (("wheel_base_none", "0.30-delta wheel, unhedged"), ("wheel_base_b1", "0.30-delta wheel, hedged"),
                      ("wheel_l0.1_none", "Model wheel, unhedged")):
        if lab in p5 and lab in a["performance_oos"]:
            p, b = a["performance_oos"][lab], p5[lab]["beta"]
            h.append({"Run": name, "CAGR": p["cagr"], "Sharpe": p["sharpe"], "β": b["beta"], "β t": b["t_hac"],
                      "Corr to SPY": b["corr"], "Alpha t": b["alpha_t_hac"]})
    if h:
        st.dataframe(pd.DataFrame(h).set_index("Run").style.format(
            {"CAGR": "{:.2%}", "Sharpe": "{:.2f}", "β": "{:.3f}", "β t": "{:.1f}", "Corr to SPY": "{:.2f}",
             "Alpha t": "{:.2f}"}), width="stretch")
        st.caption("The unhedged wheel's return comes with high correlation to SPY and an alpha t-stat near zero: it is "
                   "the equity risk premium collected through short puts. Hedged, the same rules lose money after costs.")
    hc = p5.get("wheel_hedge_cost")
    if hc:
        st.caption(f"Wheel hedge cost: stock trading {_pct(hc['stock_trading_per_year_pct_nav'], 3)} of NAV/yr, borrow "
                   f"{_pct(hc['borrow_per_year_pct_nav'], 3)}, about {hc['hedge_shares_per_day']:.0f} shares/day.")


# ---------------------------------------------------------------------------
# attribution
# ---------------------------------------------------------------------------

KIND_NAMES = {"csp": "Short put (wheel)", "cc": "Covered call", "straddle": "Long straddle", "tail": "SPY tail put"}
COMPONENTS = ["delta", "gamma", "vega", "theta", "vanna", "volga", "residual"]


def render_attribution() -> None:
    a = load("analysis.json")
    if a is None:
        _missing("attribution", "python -m premia.research.analysis")
        return
    att = pd.DataFrame(a["attribution_by_kind"]).set_index("kind")
    cost_by_kind = {k: sum(v for c, v in a["costs_total"].items() if c.endswith(f"cost_{k}")) for k in att.index}
    att["costs"] = -pd.Series(cost_by_kind)
    st.caption("Daily Taylor attribution of every option held, summed over the walk-forward OOS period ($ on a $1 M "
               "book). Delta is offset by the stock hedges, which are booked separately.")
    kind = st.segmented_control("Trade type", list(att.index), default="straddle" if "straddle" in att.index else att.index[0],
                                format_func=lambda k: KIND_NAMES.get(k, k), key="premia_att_kind")
    if kind:
        r = att.loc[kind]
        comps = COMPONENTS + ["costs"]
        vals = [float(r[c]) for c in comps]
        f = _fig(340, yaxis=dict(title="$ over the OOS period", tickformat=",.0f", zeroline=True,
                                 gridcolor=wui.plotly_layout()["yaxis"]["gridcolor"]), showlegend=False)
        f.add_waterfall(x=[c.capitalize() for c in comps] + ["Net"], y=vals + [sum(vals)],
                        measure=["relative"] * len(comps) + ["total"],
                        increasing=dict(marker=dict(color=wui.GREEN)), decreasing=dict(marker=dict(color=wui.RED)),
                        totals=dict(marker=dict(color=wui.VIOLET)), connector=dict(line=dict(width=0)),
                        hovertemplate="%{x}: $%{y:,.0f}<extra></extra>")
        st.plotly_chart(f, width="stretch", config={"displayModeBar": False})

    t = att[COMPONENTS + ["total", "costs"]].rename(index=lambda k: KIND_NAMES.get(k, k), columns=str.capitalize)
    t = t.rename(columns={"Total": "Option total", "Costs": "Option costs"})
    st.dataframe(t.style.format("{:,.0f}"), width="stretch")

    st.markdown("**Alpha or risk premium?**")
    cls = pd.DataFrame(a["classification"])
    if not cls.empty:
        cls["trade_type"] = cls["trade_type"].map(lambda k: KIND_NAMES.get(k, k))
        cols = ["trade_type", "label", "mean_residual", "t_hac", "p", "n_days", "note"]
        if "raw_label" in cls:
            cols.insert(2, "raw_label")
        st.dataframe(cls[cols], hide_index=True, width="stretch",
                     column_config={"trade_type": "Trade type", "label": "Label", "raw_label": "Raw statistical label",
                                    "mean_residual": st.column_config.NumberColumn("Mean residual $/day", format="%.2f"),
                                    "t_hac": st.column_config.NumberColumn("HAC t", format="%.2f"),
                                    "p": st.column_config.NumberColumn("p", format="%.3f"), "n_days": "Days",
                                    "note": st.column_config.TextColumn("Note", width="large")})
        st.caption("On model-priced history the residual is the pricing model's own higher-order terms, so no trade type "
                   "can be labelled alpha until real quotes (paper-trading marks) exist (SPEC §3).")

    exp = load("phase3b_premium_by_strike_dte.csv")
    if exp is not None:
        with st.expander("Delta-hedged P&L per $ of vega, by moneyness and days to expiry", icon=":material/grid_on:"):
            k = st.selectbox("Trade type", sorted(exp["kind"].unique()), format_func=lambda x: KIND_NAMES.get(x, x),
                             key="premia_heat_kind")
            sub = exp[exp["kind"] == k]
            piv = sub.pivot(index="mny_bucket", columns="dte_bucket", values="pnl_per_abs_vega")
            order = lambda idx: sorted(idx, key=lambda s: float(s.strip("(]").split(",")[0]))
            piv = piv.reindex(index=order(piv.index), columns=order(piv.columns))
            days = sub.pivot(index="mny_bucket", columns="dte_bucket", values="position_days").reindex_like(piv)
            lim = float(np.nanpercentile(np.abs(piv.values), 90)) or 1.0
            f = _fig(360, xaxis=dict(title="Days to expiry"), yaxis=dict(title="Strike / spot"))
            f.add_heatmap(z=piv.values, x=list(piv.columns), y=list(piv.index), zmid=0, zmin=-lim, zmax=lim,
                          colorscale=[[0, wui.RED], [0.5, "rgba(128,128,128,0.25)"], [1, wui.GREEN]],
                          customdata=days.values, xgap=2, ygap=2,
                          hovertemplate="K/S %{y}, DTE %{x}<br>%{z:.2f} $ per $ vega<br>%{customdata:.0f} position-days"
                                        "<extra></extra>", colorbar=dict(title="$/vega", thickness=10))
            st.plotly_chart(f, width="stretch", config={"displayModeBar": False})
            st.caption("Cells with few position-days are noise; hover shows the count.")


# ---------------------------------------------------------------------------
# hedging and sensitivity
# ---------------------------------------------------------------------------

def render_hedging() -> None:
    a = load("analysis.json")
    if a is None:
        _missing("hedging results", "python -m premia.research.analysis")
        return
    fr = pd.DataFrame(a["phase4"]["frontier"])
    if not fr.empty:
        st.markdown("**Hedging frontier**: wheel (λ = 0.1), Whalley-Wilmott band vs fixed-interval hedging")
        names = {"wheel_l0.1_b0.5": "Band ×0.5", "wheel_l0.1_b1": "Band ×1", "wheel_l0.1_b2": "Band ×2",
                 "wheel_l0.1_time1": "Daily", "wheel_l0.1_time5": "Every 5 days", "wheel_l0.1_none": "None"}
        fr["Policy"] = fr["run"].map(names).fillna(fr["run"])
        fr["kind"] = np.where(fr["run"].str.contains("_b"), "band", np.where(fr["run"].str.contains("time"), "time", "none"))
        f = _fig(340, legend=_legend(), xaxis=dict(title="Hedge cost, % of NAV per year", tickformat=".2%"),
                 yaxis=dict(title="Daily P&L vol, % of NAV", tickformat=".3%",
                            gridcolor=wui.plotly_layout()["yaxis"]["gridcolor"]))
        for kind, color, sym in (("band", wui.TEAL, "circle"), ("time", wui.AMBER, "diamond"), ("none", wui.RED, "x")):
            s = fr[fr["kind"] == kind]
            f.add_scatter(x=s["hedge_cost_pct_nav_per_year"], y=s["daily_pnl_vol_pct_nav"], mode="markers+text",
                          text=s["Policy"], textposition="top center", name=kind.capitalize(),
                          marker=dict(color=color, size=11, symbol=sym, line=dict(width=2, color="rgba(0,0,0,0.4)")),
                          customdata=s[["sharpe", "mean_abs_delta_pct_nav"]].values,
                          hovertemplate="%{text}<br>cost %{x:.3%}/yr, P&L vol %{y:.3%}/day<br>Sharpe %{customdata[0]:.2f}, "
                                        "mean |delta| %{customdata[1]:.2%} NAV<extra></extra>")
        st.plotly_chart(f, width="stretch", config={"displayModeBar": False})
        dom = a["phase4"].get("band_points_dominated_by_time_hedging") or []
        st.caption(f"Band points dominated by time hedging: {', '.join(dom) if dom else 'none'}, so the band is kept. "
                   "Unhedged sits far above the chart's useful range in P&L vol and carries market beta.")

    sens = a.get("sensitivity", {})
    if sens:
        st.markdown("**Sensitivity of the selected parameters**")
        rows = []
        for k, p in sens.items():
            base, _, tag = k.partition("__")
            rows.append({"Sleeve": "Wheel" if base.startswith("wheel") else "Straddle", "Config": base,
                         "Variant": {"": "as selected", "skew0.5": "smile skew ×0.5", "skew1.5": "smile skew ×1.5",
                                     "slip1.0": "fills at full half-spread"}.get(tag, tag),
                         "CAGR": p["cagr"], "Sharpe": p["sharpe"], "HAC t": p["mean_t_hac"], "Max drawdown": p["max_drawdown"]})
        t = pd.DataFrame(rows).sort_values(["Sleeve", "Variant"])
        f = _fig(300, legend=_legend(), barmode="group", yaxis=dict(title="OOS Sharpe", zeroline=True,
                                                                     gridcolor=wui.plotly_layout()["yaxis"]["gridcolor"]))
        for sl, color in (("Wheel", wui.TEAL), ("Straddle", wui.VIOLET)):
            s = t[t["Sleeve"] == sl]
            f.add_bar(x=s["Variant"], y=s["Sharpe"], name=sl, marker=dict(color=color, cornerradius=4),
                      customdata=s["HAC t"], hovertemplate="%{x}: Sharpe %{y:.2f}, t %{customdata:.2f}<extra>" + sl + "</extra>")
        st.plotly_chart(f, width="stretch", config={"displayModeBar": False})
        st.dataframe(t, hide_index=True, width="stretch",
                     column_config={"CAGR": st.column_config.NumberColumn(format="percent"),
                                    "Sharpe": st.column_config.NumberColumn(format="%.2f"),
                                    "HAC t": st.column_config.NumberColumn(format="%.2f"),
                                    "Max drawdown": st.column_config.NumberColumn(format="percent")})
        st.caption("The straddle loses in every variant. The wheel's result depends on how much skew the modelled smile "
                   "carries, which the backtest assumes instead of observing; real historical chains would settle it. "
                   "The ×1.5 run is a sensitivity check, not a result to select.")


# ---------------------------------------------------------------------------
# vol models and today's signals
# ---------------------------------------------------------------------------

def render_vol() -> None:
    s = load("phase3_summary.json")
    if s is None:
        _missing("phase 3 results", "python -m premia.research.phase3")
        return
    with st.container(horizontal=True):
        st.metric("SVI median RMSE", f"{s['svi_median_rmse_volpts']:.2f} vol pt", help=f"{s['svi_slices']} real slices")
        st.metric("Arbitrage-clean slices", _pct(s["svi_share_clean"], 1))
        st.metric("Local-vol PDE max error", f"{s['local_vol_max_err_volpts']:.2f} vol pt",
                  help=f"{s['local_vol_points']} quotes, 80–120 % strikes")
        hi = s["rv_pooled"].get(s["rv_best_model"], {})
        st.metric("Best RV forecast", s["rv_best_model"].upper().replace("_", "-"),
                  f"DM t {hi.get('t_hac', float('nan')):.1f}", delta_color="off",
                  help=f"Beats trailing RV for {_pct(hi.get('share_names_better'), 1)} of names (QLIKE)")
        cal = s.get("calibration", {}).get("p_fhs", {})
        if cal:
            st.metric("FHS calibration error", f"{cal['ece_pts']:.2f} pts", "limit 3.0", delta_color="off")

    vrp = pd.DataFrame(s["vrp_spx_tenor"])
    vrp["Tenor"] = vrp["tenor"].str.replace("^", "", regex=False) + " (" + vrp["days"].astype(str) + "d)"
    names = s.get("vrp_names_mean_volpts")
    f = _fig(300, yaxis=dict(title="IV − later realized vol (vol pts)", gridcolor=wui.plotly_layout()["yaxis"]["gridcolor"],
                             zeroline=True), showlegend=False)
    x = list(vrp["Tenor"]) + (["Single names 30d"] if names is not None else [])
    y = list(vrp["vrp_volpts"]) + ([names] if names is not None else [])
    share = list(vrp["share_iv_above_rv"]) + ([s.get("vrp_names_share_positive_mean")] if names is not None else [])
    f.add_bar(x=x, y=y, marker=dict(color=[wui.TEAL] * len(vrp) + [wui.VIOLET] * (names is not None), cornerradius=4),
              customdata=share, text=[f"{v:.1f}" for v in y], textposition="outside",
              hovertemplate="%{x}: %{y:.2f} vol pts<br>IV above RV %{customdata:.0%} of the time<extra></extra>")
    st.markdown("**Variance risk premium**: implied vol minus the realized vol that followed")
    st.plotly_chart(f, width="stretch", config={"displayModeBar": False})

    sig = load("phase3_option_signals_real_spx.csv")
    if sig is not None and not sig.empty:
        asof, exp = sig["asof"].iloc[0], sig["expiry"].iloc[0]
        st.markdown(f"**SPX puts: market vs the model's physical fair value** (snapshot {asof}, expiry {exp})")
        sig = sig.sort_values("K/F")
        f = _fig(340, legend=_legend(), xaxis=dict(title="Strike / forward", tickformat=".0%"),
                 yaxis=dict(title="Implied vol", tickformat=".0%", gridcolor=wui.plotly_layout()["yaxis"]["gridcolor"]))
        f.add_scatter(x=sig["K/F"], y=sig["iv_bid"], name="Market bid IV", mode="lines+markers",
                      line=dict(color=wui.MAGENTA, width=2), marker=dict(size=8),
                      hovertemplate="K/F %{x:.1%}: bid IV %{y:.1%}<extra></extra>")
        f.add_scatter(x=sig["K/F"], y=sig["fair_iv_P"], name="Physical fair IV (FHS)", mode="lines+markers",
                      line=dict(color=wui.TEAL, width=2, dash="dot"), marker=dict(size=8),
                      hovertemplate="K/F %{x:.1%}: fair IV %{y:.1%}<extra></extra>")
        f.update_layout(hovermode="x unified")
        st.plotly_chart(f, width="stretch", config={"displayModeBar": False})
        st.dataframe(sig[["K", "K/F", "bid", "ask", "iv_bid", "iv_ask", "fair_iv_P", "edge_sell_per_share", "signal"]],
                     hide_index=True, width="stretch",
                     column_config={"K/F": st.column_config.NumberColumn(format="percent"),
                                    "iv_bid": st.column_config.NumberColumn("IV bid", format="percent"),
                                    "iv_ask": st.column_config.NumberColumn("IV ask", format="percent"),
                                    "fair_iv_P": st.column_config.NumberColumn("Fair IV", format="percent"),
                                    "edge_sell_per_share": st.column_config.NumberColumn("Edge selling, $/share", format="%.2f"),
                                    "signal": "Signal"})
        st.caption("RICH means the bid is above the physical fair value plus a spread buffer. The physical model failed "
                   "its calibration test (drift-neutral), so read this as the variance premium on display, not as an "
                   "unhedged trade signal.")


MCM_RUNS = {"mcm_base": "As in the book", "mcm_hist80": "Criterion 3 at 80 %", "mcm_3month": "Three-month options",
            "mcm_fullspread": "Fills at the full half-spread", "mcm_no_followup": "No follow-up (stop + time only)",
            "mcm_flat_term": "Flat term structure (biased)"}


def render_mcmillan() -> None:
    rep, v = load("mcmillan_report.json"), load("verdict_mcmillan_straddle.json")
    nav = load("mcmillan_nav.csv")
    if rep is None or "mcm_base" not in rep or nav is None:
        _missing("McMillan straddle backtest", "python -m premia.research.mcmillan_bt run")
        return
    b = rep["mcm_base"]
    st.caption("Lawrence McMillan's straddle buying (*McMillan on Options* ch. 7): IV at or below its 10th percentile, "
               "≥ 80 % chance of ever touching a breakeven, the move seen often in the past, no takeover; risk 60 %, "
               "exit with a month left, ride breakouts with the 20-day average. Every rule comes from the book, so "
               "nothing was fitted: 2013 onward is out of sample. Universe: the wheel watchlist minus exclusions. "
               "Spec and pass criteria: `premia/MCMILLAN.md`.")
    if v is not None:
        icon = ":material/check_circle:" if v["verdict"] == "PASS" else ":material/cancel:"
        (st.success if v["verdict"] == "PASS" else st.error)(
            f"**Pre-registered verdict: {v['verdict']}**  \n" + "  \n".join(
                f"{'✓' if c['pass'] else '✗'} {c['criterion']}: {c['evidence']}" for c in v["checks"]), icon=icon)
    o, t = b["oos"], b.get("trades", {})
    with st.container(horizontal=True):
        st.metric("CAGR (2013→)", _pct(o["cagr"]))
        st.metric("Sharpe", f"{o['sharpe']:.2f}", help=f"Newey-West t {o['mean_t_hac']:.2f}; deflated Sharpe "
                  f"{b.get('dsr', float('nan')):.2f} over {len([k for k in rep if k.startswith('mcm_')])} runs")
        st.metric("Max drawdown", _pct(o["max_drawdown"], 1))
        st.metric("β to SPY", f"{b['beta']['beta']:.3f}", help=f"t {b['beta']['t_hac']:.2f}, correlation {b['beta']['corr']:.2f}")
        if t:
            st.metric("Straddles", f"{t['n']:,}", help=f"{t['names']} names")
            st.metric("Winners", _pct(t["win_rate"], 0))
            st.metric("Avg return on premium", _pct(t["mean_ret_on_premium"], 1))
    nav = nav.set_index(pd.to_datetime(nav.iloc[:, 0])).iloc[:, -1]
    f = _fig(320, showlegend=False, yaxis=dict(title="Return since 2009", tickformat=".0%",
                                              gridcolor=wui.plotly_layout()["yaxis"]["gridcolor"], zeroline=False))
    f.add_scatter(x=nav.index, y=nav / nav.iloc[0] - 1, line=dict(color=wui.AMBER, width=2), name="McMillan straddles",
                  hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2%}<extra></extra>")
    f.add_vline(x=pd.Timestamp("2013-01-01"), line=dict(color="rgba(128,128,128,0.6)", dash="dot"))
    st.plotly_chart(f, width="stretch", config={"displayModeBar": False})
    st.caption("$1 M book, 0.5 % of NAV of premium per straddle, at most 10 % in all; dotted line: start of the "
               "reported period (2009-2012 is warm-up for the IV percentile history).")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**How the straddles ended** (2013→)")
        ex = pd.DataFrame(t.get("by_first_exit", []))
        if not ex.empty:
            ex.columns = ["First exit", "Straddles", "Avg return on premium"]
            st.dataframe(ex.sort_values("Straddles", ascending=False), hide_index=True, width="stretch",
                         column_config={"Avg return on premium": st.column_config.NumberColumn(format="percent")})
    with c2:
        st.markdown("**Criterion 2 against what happened**")
        cal = pd.DataFrame(b.get("touch_calibration", []))
        if not cal.empty:
            cal.columns = ["Predicted P(ever)", "Mean predicted", "Actually touched", "Name-days"]
            st.dataframe(cal, hide_index=True, width="stretch",
                         column_config={"Mean predicted": st.column_config.NumberColumn(format="percent"),
                                        "Actually touched": st.column_config.NumberColumn(format="percent")})
            st.caption("For every name-day with IV at or below the 10th percentile: did the price really reach a "
                       "breakeven before expiry? McMillan argues real markets touch more often than the lognormal model "
                       "says (p. 227).")

    rg = b.get("regimes", {})
    if rg:
        cats = [("market_down", "SPY down"), ("market_flat", "SPY flat"), ("market_up", "SPY up")]
        f = _fig(260, showlegend=False, yaxis=dict(title="Mean monthly return", tickformat=".2%", zeroline=True,
                                                   gridcolor=wui.plotly_layout()["yaxis"]["gridcolor"]))
        f.add_bar(x=[l for _, l in cats], y=[rg[k]["mean_monthly_return"] for k, _ in cats],
                  marker=dict(color=wui.AMBER, cornerradius=4),
                  customdata=[[rg[k]["months"], rg[k]["share_positive"]] for k, _ in cats],
                  hovertemplate="%{x}: %{y:.3%}/month<br>%{customdata[0]} months, %{customdata[1]:.0%} positive<extra></extra>")
        st.markdown("**By market direction** (no delta hedging, as the book prescribes)")
        st.plotly_chart(f, width="stretch", config={"displayModeBar": False})

    years = b.get("calendar_years")
    if years:
        st.markdown("**Calendar years**")
        st.dataframe(pd.DataFrame([years]).style.format("{:.2%}"), hide_index=True, width="stretch")

    st.markdown("**Sensitivity runs** (none is a selection: the book fixes the rules)")
    rows = []
    for k, name in MCM_RUNS.items():
        if k in rep:
            r = rep[k]
            rows.append({"Run": name, "CAGR": r["oos"]["cagr"], "Sharpe": r["oos"]["sharpe"], "HAC t": r["oos"]["mean_t_hac"],
                         "Max drawdown": r["oos"]["max_drawdown"], "Straddles": r.get("trades", {}).get("n"),
                         "Win rate": r.get("trades", {}).get("win_rate")})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={"CAGR": st.column_config.NumberColumn(format="percent"),
                                    "Sharpe": st.column_config.NumberColumn(format="%.2f"),
                                    "HAC t": st.column_config.NumberColumn(format="%.2f"),
                                    "Max drawdown": st.column_config.NumberColumn(format="percent"),
                                    "Win rate": st.column_config.NumberColumn(format="percent")})
    trades = load("mcmillan_trades.csv")
    if trades is not None and not trades.empty:
        with st.expander("Every straddle the backtest bought", icon=":material/receipt_long:"):
            st.dataframe(trades[["symbol", "entry", "expiry", "K", "contracts", "premium", "pnl", "ret_on_premium",
                                 "path", "ever_prob", "hist_freq", "iv_pct"]], hide_index=True, width="stretch",
                         column_config={"premium": st.column_config.NumberColumn(format="$%.0f"),
                                        "pnl": st.column_config.NumberColumn(format="$%.0f"),
                                        "ret_on_premium": st.column_config.NumberColumn("return", format="percent"),
                                        "ever_prob": st.column_config.NumberColumn("P(ever)", format="percent"),
                                        "hist_freq": st.column_config.NumberColumn("hist. freq", format="percent")})


def render_page() -> None:
    st.caption("Model-driven wheel and long-straddle research (`premia/`): phases 0–7 with pre-registered pass/fail "
               "criteria, walk-forward out-of-sample results, and the evidence behind the recommendation. " + DISCLAIMER)
    if not RESULTS_DIR.exists():
        _missing("research results", "python -m premia.research.analysis")
        return
    render_verdicts()
    tabs = st.tabs([":material/show_chart: Performance", ":material/balance: Direction independence",
                    ":material/stacked_bar_chart: Attribution", ":material/tune: Hedging & sensitivity",
                    ":material/ssid_chart: Vol models & signals", ":material/menu_book: McMillan straddle"])
    with tabs[0]:
        render_performance()
    with tabs[1]:
        render_direction()
    with tabs[2]:
        render_attribution()
    with tabs[3]:
        render_hedging()
    with tabs[4]:
        render_vol()
    with tabs[5]:
        render_mcmillan()
    st.caption(f"Source files: `{RESULTS_DIR}`. Full write-up: `premia/RESULTS.md`.")
