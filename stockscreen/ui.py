"""
stockscreen/ui.py — Streamlit rendering for the Fundamentals (stock screener) page.
Helper functions only; app_pages/screener.py composes them top to bottom.
"""

from __future__ import annotations

import time
from datetime import date, datetime
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import wheel_ui as wui

from . import alerts as alerts_mod
from . import evaluate as ev
from . import prices as pr
from . import scan as scan_mod
from . import scoring as sc
from . import store
from . import universe
from . import technicals as tc
from .config import ScreenConfig, load_config, save_config

DISCLAIMER = ("Research signals only, not investment advice or return predictions. Scores and opportunity classes are "
              "heuristic rankings that have not been backtested. A low price is not evidence of undervaluation.")


@st.cache_resource
def scan_manager() -> scan_mod.ScanManager:
    return scan_mod.ScanManager()


def get_config() -> ScreenConfig:
    if "screen_cfg" not in st.session_state:
        st.session_state.screen_cfg = load_config()
    return st.session_state.screen_cfg


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _evaluate_cached(results_stamp: str, cfg_json: str, day: str, exclusions: str):
    """Re-evaluate every stored record. Keyed on the results timestamp, the
    config and the calendar day (data ages advance daily), so threshold edits
    are instant and unchanged state is never recomputed."""
    import json
    cfg = ScreenConfig.from_dict(json.loads(cfg_json))
    ex = universe.excluded()
    records = [r for r in store.load_results() if r["symbol"].upper() not in ex]     # excluded tickers are never evaluated
    df, details = ev.evaluate_all(records, cfg, date.fromisoformat(day))
    return df, details


def load_evaluated(cfg: ScreenConfig):
    import json
    universe.apply_exclusions()          # purge anything an older scan left behind for tickers you have since excluded
    return _evaluate_cached(f"{store.db_path()}|{store.results_stamp()}", json.dumps(cfg.to_dict(), sort_keys=True),
                            date.today().isoformat(), universe.signature())


# ---------------------------------------------------------------------------
# header: coverage + summary
# ---------------------------------------------------------------------------

def render_coverage(df: pd.DataFrame) -> None:
    if not scan_manager().running:
        store.interrupt_orphans()                                    # e.g. a scan whose process was killed
    run = store.latest_run()
    if run is None:
        st.info("No scan has been run yet. Choose a universe below and click **Run scan**.", icon=":material/info:")
        return
    total = int(run["universe_size"] or 0)
    status = run["status"]
    n = len(df)
    kind = {"complete": ("success", ":material/check_circle:"), "running": ("info", ":material/hourglass_top:")}.get(
        status, ("warning", ":material/warning:"))
    msg = (f"**Last scan:** {run['universe_label']} — {n} of {total} tickers have results · "
           f"{run['evaluated']} with full financial statements, {run['no_statements']} without SEC statements "
           f"(ETFs, funds, foreign filers), {run['errors']} errors · status **{status}**"
           f" · started {run['started_at'][:16].replace('T', ' ')}")
    getattr(st, kind[0])(msg, icon=kind[1])
    if status != "complete":
        st.warning("This was a **partial** scan (cancelled, failed, or with errors) — do not read the counts below as "
                   "full coverage of the configured universe.", icon=":material/warning:")
    if run.get("universe_note"):
        st.caption(run["universe_note"] + " It is not a market-wide scan.")


def render_summary(df: pd.DataFrame) -> None:
    if df.empty:
        return
    ok = df[df["status"] != "ERROR"]
    cnt = lambda mask: int(mask.sum())                                                  # noqa: E731
    low = df["price_zone"].isin(["Deep value-zone candidate", "Low-price candidate"])
    passed = df["status"] == "PASSED"
    issues = cnt(df["status"].isin(["ERROR", "STALE_DATA"])) + cnt((df["status"] == "INSUFFICIENT_DATA"))
    with st.container(horizontal=True):
        st.metric("Stocks scanned", len(df), border=True)
        st.metric("Pass all fundamentals", cnt(passed), border=True)
        st.metric("In the low-price zone", cnt(passed & low), border=True, help="Passing fundamentals AND in the low-price zone")
        st.metric("Healthy pullbacks", cnt(df["opp_code"] == "C"), border=True)
        st.metric("Reversal confirmation", cnt(df["opp_code"] == "D"), border=True)
        st.metric("Declining fundamentals", cnt(df["fund_trend"] == "Declining"), border=True)
        st.metric("Data / coverage issues", issues, border=True,
                  help="Insufficient data (incl. no SEC statements), stale data, or errors")
    with st.expander("Breakdown by status", icon=":material/bar_chart:"):
        by = df["status"].value_counts().rename_axis("Status").reset_index(name="Stocks")
        st.dataframe(by, hide_index=True, width=300)


# ---------------------------------------------------------------------------
# scan controls
# ---------------------------------------------------------------------------

@st.fragment(run_every="2s")
def _progress_fragment() -> None:
    mgr = scan_manager()
    p = mgr.progress
    if mgr.running:
        st.progress(min(p.fraction, 1.0), text=f"{p.phase}: {p.done}/{p.total} · {p.current} · "
                                               f"{p.evaluated} evaluated, {p.no_statements} no statements, {p.errors} errors")
        st.caption("The scan runs in the background — you can keep using other pages.")
    elif p.state in ("done", "cancelled", "error") and p.finished and not st.session_state.get("_scan_seen") == p.run_id:
        st.session_state["_scan_seen"] = p.run_id
        st.rerun()


def render_scan_controls(holdings: list[str]) -> None:
    cfg = get_config()
    mgr = scan_manager()
    with st.container(key="glass_scanctl"):
        st.subheader("Run a scan", anchor=False)
        opts = scan_mod.universe_options(holdings, custom=st.session_state.get("custom_tickers", ""))
        c1, c2 = st.columns([3, 2])
        label = c1.selectbox("Universe", list(opts), key="scan_universe",
                             help="Which tickers to scan. Manual exclusions on the Manage universe table always apply.")
        c2.text_input("Custom tickers (optional)", key="custom_tickers", placeholder="e.g. NVDA, SAP, CHWY",
                      help="Adds a 'Custom list' option above.")
        u = opts.get(label) or next(iter(opts.values()))
        n_ex = len(universe.excluded())
        st.caption(f"{len(u['symbols'])} tickers to scan. {u['note']}"
                   + (f" {n_ex} manually excluded tickers are never scanned or shown." if n_ex else ""))
        with st.container(horizontal=True, vertical_alignment="center"):
            force = st.checkbox("Force refresh (ignore caches)", key="scan_force",
                                help="Re-download SEC statements and prices even if the cache is fresh.")
            if mgr.running:
                st.button("Cancel scan", icon=":material/cancel:", on_click=mgr.cancel)
            elif st.button("Run scan", icon=":material/play_arrow:", type="primary", disabled=not u["symbols"]):
                mgr.start(u["symbols"], label, u["note"], cfg, force=force)
                st.rerun()
            st.caption(f"Fundamentals cached {cfg.fundamentals_ttl_days}d · prices {cfg.price_ttl_hours:g}h · "
                       f"{cfg.scan_workers} workers · auto-scan {'on, after ' + cfg.auto_scan_time if cfg.auto_scan_enabled else 'off'}")
        _progress_fragment()
        p = mgr.progress
        if not mgr.running and p.state in ("done", "cancelled", "error") and p.message:
            st.caption(f"Last run in this session: {p.message}. Alerts: {p.alert_stats or 'none'}")


def maybe_auto_scan(holdings: list[str]) -> None:
    cfg = get_config()
    mgr = scan_manager()
    if mgr.running or not scan_mod.should_auto_scan(cfg):
        return
    u = scan_mod.universe_options(holdings)["Wheel universe (shortlist)"]
    mgr.start(u["symbols"], "Wheel universe (shortlist)", u["note"] + " (auto-scan)", cfg)


# ---------------------------------------------------------------------------
# profile (thresholds) editor
# ---------------------------------------------------------------------------

def render_profile() -> None:
    cfg = get_config()
    with st.expander("Screening profile — thresholds, weights and alert preferences", icon=":material/tune:"):
        st.caption("Changing a threshold re-classifies every stock instantly (no rescan). Only the slope window and "
                   "history grace period need a rescan. Saved to stockscreen_config.json.")
        with st.form("profile_form", border=False):
            a, b, c = st.columns(3)
            min_margin = a.number_input("Net margin must exceed (%)", 0.0, 100.0, cfg.min_net_margin * 100, 0.5)
            min_growth = b.number_input("Revenue growth must exceed (%)", -50.0, 200.0, cfg.min_revenue_growth * 100, 0.5)
            min_yield = c.number_input("FCF yield at least (%)", 0.0, 100.0, cfg.min_fcf_yield * 100, 0.25)
            d, e, f = st.columns(3)
            min_cap = d.number_input("Market cap at least ($B)", 0.0, 5000.0, cfg.min_market_cap / 1e9, 0.5)
            basis = e.selectbox("Growth basis", ["auto", "ttm", "fy"], ["auto", "ttm", "fy"].index(cfg.growth_basis),
                                help="auto = TTM vs prior TTM, else last fiscal year vs prior (labelled).")
            age = f.number_input("Stale after (days since latest quarter)", 30, 730, cfg.max_data_age_days, 10)
            g, h, i = st.columns(3)
            tol = g.number_input("Trend tolerance — level metrics (%)", 0.0, 50.0, cfg.trend_tol_level * 100, 0.5)
            tol_m = h.number_input("Trend tolerance — margin (pp)", 0.0, 20.0, cfg.trend_tol_margin_pp * 100, 0.25)
            tol_g = i.number_input("Trend tolerance — growth (pp)", 0.0, 30.0, cfg.trend_tol_growth_pp * 100, 0.5)
            crit = st.multiselect("Critical trend metrics (Declining on any makes a stock ineligible)",
                                  ["fcf", "net_margin", "revenue", "revenue_growth", "fcf_yield"],
                                  list(cfg.critical_trend_metrics))
            st.markdown("**Low-price thresholds** (0–100)")
            j, k, l, m_ = st.columns(4)
            r52 = j.number_input("52w range position ≤", 0.0, 100.0, cfg.low_range_pos_52w, 1.0)
            r2 = k.number_input("2y range position ≤", 0.0, 100.0, cfg.low_range_pos_2y, 1.0)
            r3 = l.number_input("3y range position ≤", 0.0, 100.0, cfg.low_range_pos_3y, 1.0)
            pct = m_.number_input("Closing-price percentile ≤", 0.0, 100.0, cfg.low_percentile, 1.0)
            n_, o_, p_ = st.columns(3)
            deep = n_.number_input("Lookbacks for 'Deep value-zone'", 1, 3, cfg.deep_zone_min_lookbacks)
            confirm = o_.number_input("Recovery signals to confirm a reversal", 1, 4, cfg.reversal_confirm_signals)
            near_hi = p_.number_input("'Near highs' at 52w range position ≥", 50.0, 100.0, cfg.near_high_range_pos, 1.0)
            st.markdown("**Rank weights** (fundamental / price / technical — renormalised)")
            w1, w2, w3 = st.columns(3)
            wf = w1.number_input("Fundamental", 0.0, 1.0, cfg.weights_rank["fundamental"], 0.05)
            wp = w2.number_input("Price opportunity", 0.0, 1.0, cfg.weights_rank["price"], 0.05)
            wt = w3.number_input("Technical", 0.0, 1.0, cfg.weights_rank["technical"], 0.05)
            st.markdown("**Alerts**")
            kinds = st.multiselect("Alert kinds", list(alerts_mod.KINDS), list(cfg.alert_kinds),
                                   format_func=lambda k: alerts_mod.KINDS[k])
            q, r, s = st.columns(3)
            cool = q.number_input("Cooldown (days) before the same alert can repeat", 0, 90, cfg.alert_cooldown_days)
            auto = r.checkbox("Auto-scan on weekdays while the dashboard is open", cfg.auto_scan_enabled)
            atime = s.text_input("Auto-scan time (HH:MM, local)", cfg.auto_scan_time)
            col1, col2 = st.columns(2)
            saved = col1.form_submit_button("Save profile", type="primary", icon=":material/save:")
            reset = col2.form_submit_button("Reset to defaults", icon=":material/restart_alt:")
        if reset:
            new = ScreenConfig()
            save_config(new)
            st.session_state.screen_cfg = new
            st.rerun()
        if saved:
            new = ScreenConfig.from_dict({**cfg.to_dict(),
                "min_net_margin": min_margin / 100, "min_revenue_growth": min_growth / 100, "min_fcf_yield": min_yield / 100,
                "min_market_cap": min_cap * 1e9, "growth_basis": basis, "max_data_age_days": int(age),
                "trend_tol_level": tol / 100, "trend_tol_margin_pp": tol_m / 100, "trend_tol_growth_pp": tol_g / 100,
                "critical_trend_metrics": crit, "low_range_pos_52w": r52, "low_range_pos_2y": r2, "low_range_pos_3y": r3,
                "low_percentile": pct, "deep_zone_min_lookbacks": int(deep), "reversal_confirm_signals": int(confirm),
                "near_high_range_pos": near_hi, "weights_rank": {"fundamental": wf, "price": wp, "technical": wt},
                "alert_kinds": kinds, "alert_cooldown_days": int(cool), "auto_scan_enabled": auto, "auto_scan_time": atime})
            save_config(new)
            st.session_state.screen_cfg = new
            st.toast("Profile saved — results re-classified.", icon=":material/check_circle:")
            st.rerun()


# ---------------------------------------------------------------------------
# filters + presets
# ---------------------------------------------------------------------------

PRESETS = {
    "Quality stocks": dict(status=["PASSED"]),
    "Quality stocks near lows": dict(status=["PASSED"], zone=["Deep value-zone candidate", "Low-price candidate"]),
    "Healthy pullbacks": dict(opp=["C"]),
    "Potential reversals": dict(opp=["D"]),
    "Strong uptrends": dict(status=["PASSED"], tech=["Strong uptrend"]),
    "Fundamental deterioration": dict(trend=["Declining"]),
}


def render_filters(df: pd.DataFrame) -> pd.DataFrame:
    with st.container(key="glass_filters"):
        st.subheader("Filters", anchor=False)
        preset = st.segmented_control("Preset", list(PRESETS), default="Quality stocks", key="preset",
                                      help="Presets set the status / zone / trend filters below; you can then refine them.")
        preset = preset or "Quality stocks"
        base = PRESETS[preset]
        key = f"flt_{preset}"                                   # new widget identities per preset so defaults reapply
        c1, c2, c3 = st.columns(3)
        status = c1.multiselect("Fundamental status", ["PASSED", "FAILED", "INSUFFICIENT_DATA", "STALE_DATA", "ERROR"],
                                base.get("status", []), key=key + "_status",
                                help="Empty = every status. Add FAILED to explore near-misses with the filters below.")
        zones = c2.multiselect("Price zone", tc.PRICE_ZONES, base.get("zone", []), key=key + "_zone")
        opps = c3.multiselect("Opportunity class", list(sc.OPPORTUNITY), base.get("opp", []), key=key + "_opp",
                              format_func=lambda c: sc.OPPORTUNITY[c])
        c4, c5, c6 = st.columns(3)
        ftrend = c4.multiselect("Fundamental trend", ["Improving", "Stable", "Declining", "Insufficient data"],
                                base.get("trend", []), key=key + "_trend")
        ttrend = c5.multiselect("Technical trend", tc.TREND_LABELS, base.get("tech", []), key=key + "_tech")
        rsi_lo, rsi_hi = c6.slider("RSI range", 0, 100, (0, 100), key=key + "_rsi")
        with st.expander("More filters (fundamentals, price position, scores)"):
            d1, d2, d3, d4 = st.columns(4)
            f_cap = d1.number_input("Min market cap ($B)", 0.0, 5000.0, 0.0, 0.5, key=key + "_cap")
            f_yield = d2.number_input("Min FCF yield (%)", -100.0, 100.0, -100.0, 0.5, key=key + "_yld")
            f_margin = d3.number_input("Min net margin (%)", -500.0, 100.0, -500.0, 1.0, key=key + "_mrg")
            f_growth = d4.number_input("Min revenue growth (%)", -100.0, 500.0, -100.0, 1.0, key=key + "_grw")
            e1, e2, e3 = st.columns(3)
            m52 = e1.slider("Max 52w range position", 0, 100, 100, key=key + "_m52")
            m2 = e2.slider("Max 2y range position", 0, 100, 100, key=key + "_m2")
            m3 = e3.slider("Max 3y range position", 0, 100, 100, key=key + "_m3")
            g1, g2, g3 = st.columns(3)
            s_f = g1.slider("Min fundamental score", 0, 100, 0, key=key + "_sf")
            s_p = g2.slider("Min price opportunity score", 0, 100, 0, key=key + "_sp")
            s_t = g3.slider("Min technical score", 0, 100, 0, key=key + "_st")
            need_fcf = st.checkbox("Require positive FCF history (4 quarters + 2 fiscal years)", False, key=key + "_fcf")

    out = df
    if status:
        out = out[out["status"].isin(status)]
    if zones:
        out = out[out["price_zone"].isin(zones)]
    if opps:
        out = out[out["opp_code"].isin(opps)]
    if ftrend:
        out = out[out["fund_trend"].isin(ftrend)]
    if ttrend:
        out = out[out["tech_trend"].isin(ttrend)]
    if (rsi_lo, rsi_hi) != (0, 100):
        out = out[out["rsi"].between(rsi_lo, rsi_hi)]

    def at_least(col, thr, scale=1.0):
        return out[col].fillna(float("-inf")) >= thr / scale

    if f_cap > 0:
        out = out[at_least("market_cap", f_cap, 1 / 1e9)]
    if f_yield > -100:
        out = out[at_least("fcf_yield", f_yield, 100)]
    if f_margin > -500:
        out = out[at_least("net_margin", f_margin, 100)]
    if f_growth > -100:
        out = out[at_least("revenue_growth", f_growth, 100)]
    for col, lim in (("pos_52w", m52), ("pos_2y", m2), ("pos_3y", m3)):
        if lim < 100:
            out = out[out[col].fillna(1e9) <= lim]
    for col, lim in (("fund_score", s_f), ("price_score", s_p), ("tech_score", s_t)):
        if lim > 0:
            out = out[out[col].fillna(-1) >= lim]
    if need_fcf:
        out = out[out["fcf_history_ok"]]
    return out


# ---------------------------------------------------------------------------
# main table
# ---------------------------------------------------------------------------

COLUMN_GROUPS = {
    "Company": ["symbol", "name", "sector"],
    "Size": ["market_cap_b"],
    "Cash flow": ["q_fcf_m", "ttm_fcf_m", "fy_fcf_latest_m", "fy_fcf_prior_m"],
    "Profitability": ["net_margin"],
    "Growth": ["revenue_growth"],
    "Yield": ["fcf_yield"],
    "Price": ["price", "pos_52w", "pos_2y", "pos_3y", "pct_52w"],
    "Technicals": ["rsi", "macd_status", "sma50", "sma200"],
    "Classification": ["status", "fund_trend", "tech_trend", "price_zone", "opportunity"],
    "Ranking": ["fund_score", "price_score", "tech_score", "composite"],
}

CONFIG = {
    "symbol": st.column_config.TextColumn("Ticker", width="small"),
    "name": st.column_config.TextColumn("Company"),
    "sector": st.column_config.TextColumn("Industry (SIC)"),
    "market_cap_b": st.column_config.NumberColumn("Mkt cap ($B)", format="$%.1f"),
    "q_fcf_m": st.column_config.NumberColumn("Latest Q FCF ($M)", format="$%,.0f"),
    "ttm_fcf_m": st.column_config.NumberColumn("TTM FCF ($M)", format="$%,.0f"),
    "fy_fcf_latest_m": st.column_config.NumberColumn("FY FCF ($M)", format="$%,.0f"),
    "fy_fcf_prior_m": st.column_config.NumberColumn("Prior FY FCF ($M)", format="$%,.0f"),
    "net_margin": st.column_config.NumberColumn("TTM net margin", format="percent"),
    "revenue_growth": st.column_config.NumberColumn("Revenue growth", format="percent",
                                                    help="TTM vs prior TTM (or fiscal year, labelled in the detail view)"),
    "fcf_yield": st.column_config.NumberColumn("FCF yield", format="percent"),
    "price": st.column_config.NumberColumn("Price", format="$%.2f"),
    "pos_52w": st.column_config.NumberColumn("52w pos %", format="%.0f", help="0 = at the 52-week low, 100 = at the high"),
    "pos_2y": st.column_config.NumberColumn("2y pos %", format="%.0f"),
    "pos_3y": st.column_config.NumberColumn("3y pos %", format="%.0f"),
    "pct_52w": st.column_config.NumberColumn("52w pctile", format="%.0f", help="Percentile of today's close among the prior year's closes"),
    "rsi": st.column_config.NumberColumn("RSI", format="%.0f"),
    "macd_status": st.column_config.TextColumn("MACD"),
    "sma50": st.column_config.NumberColumn("SMA50", format="$%.2f"),
    "sma200": st.column_config.NumberColumn("SMA200", format="$%.2f"),
    "status": st.column_config.TextColumn("Fundamentals"),
    "fund_trend": st.column_config.TextColumn("Fund. trend"),
    "tech_trend": st.column_config.TextColumn("Tech. trend"),
    "price_zone": st.column_config.TextColumn("Price zone"),
    "opportunity": st.column_config.TextColumn("Opportunity"),
    "fund_score": st.column_config.ProgressColumn("Fund. score", min_value=0, max_value=100, format="%.0f"),
    "price_score": st.column_config.ProgressColumn("Price score", min_value=0, max_value=100, format="%.0f"),
    "tech_score": st.column_config.ProgressColumn("Tech. score", min_value=0, max_value=100, format="%.0f"),
    "composite": st.column_config.ProgressColumn("Rank score", min_value=0, max_value=100, format="%.0f"),
}


def add_display_columns(df: pd.DataFrame, details: dict) -> pd.DataFrame:
    d = df.copy()
    d["market_cap_b"] = d["market_cap"] / 1e9
    for src, dst in (("q_fcf", "q_fcf_m"), ("ttm_fcf", "ttm_fcf_m"), ("fy_fcf_latest", "fy_fcf_latest_m"), ("fy_fcf_prior", "fy_fcf_prior_m")):
        d[dst] = d[src] / 1e6
    d["fcf_history_ok"] = [
        any(c["key"] == "fcf_consistency" and c["status"] == "pass"
            for c in details.get(s, {}).get("fund_eval", {}).get("criteria", []))
        for s in d["symbol"]]
    return d


def render_table(df: pd.DataFrame) -> Optional[str]:
    with st.container(key="glass_results"):
        st.subheader(f"Results — {len(df)} stocks", anchor=False)
        st.caption(DISCLAIMER)
        if df.empty:
            st.info("No stocks match the current filters. Loosen a filter or choose another preset.", icon=":material/filter_alt_off:")
            return None
        groups = st.pills("Columns", list(COLUMN_GROUPS), selection_mode="multi",
                          default=list(COLUMN_GROUPS), key="col_groups")
        cols = [c for g in (groups or ["Company"]) for c in COLUMN_GROUPS[g]]
        sort_by = st.segmented_control("Default sort", ["Rank score", "Fund. score", "Price score", "Tech. score", "Mkt cap"],
                                       default="Rank score", key="sort_by")
        key = {"Rank score": "composite", "Fund. score": "fund_score", "Price score": "price_score",
               "Tech. score": "tech_score", "Mkt cap": "market_cap"}[sort_by or "Rank score"]
        view = df.sort_values(key, ascending=False, na_position="last").reset_index(drop=True)
        event = st.dataframe(view[cols], hide_index=True, column_config=CONFIG, on_select="rerun",
                             selection_mode="single-row", key="results_table", height=min(620, 38 + 35 * len(view)))
        st.caption("Click a row to open the stock detail below. Click a column header to sort. Scores are heuristic and "
                   "not backtested.")
        rows = event.selection.rows if event and event.selection else []
        if rows:
            return view.iloc[rows[0]]["symbol"]
        # keyboard-friendly alternative to clicking a row
        pick = st.selectbox("Or open a stock by ticker", ["—"] + list(view["symbol"]), key="detail_pick",
                            help="Opens the same detail view as clicking a row.")
        return None if pick == "—" else pick
    return None


# ---------------------------------------------------------------------------
# detail view
# ---------------------------------------------------------------------------

def _money(x) -> str:
    return "n/a" if x is None or pd.isna(x) else f"${x / 1e6:,.1f}M"


def render_detail(symbol: str, details: dict, cfg: ScreenConfig) -> None:
    d = details.get(symbol)
    rec = next((r for r in store.load_results([symbol])), None)
    if not d or not rec:
        return
    row, fe = d["row"], d["fund_eval"]
    m = fe.get("metrics", {})
    with st.container(key="glass_detail"):
        st.subheader(f"{symbol} — {row['name'] or ''}", anchor=False)
        with st.container(horizontal=True):
            st.badge(row["status"], color={"PASSED": "green", "FAILED": "red"}.get(row["status"], "orange"))
            st.badge(d["opportunity"]["label"], color="violet")
            st.badge(row["price_zone"], color="blue")
            st.badge(row["tech_trend"], color="gray")
        st.write(d["opportunity"]["why"])
        scored = d["scores"]
        with st.container(horizontal=True):
            for label, key in (("Fundamental score", "fundamental"), ("Price opportunity", "price"), ("Technical", "technical")):
                s = scored[key]
                st.metric(label, "n/a" if s["score"] is None else f"{s['score']:.0f}", f"coverage {s['coverage']:.0%}" if s.get("coverage") is not None else None,
                          delta_color="off", delta_arrow="off", border=True)
            st.metric("Rank score (heuristic)", "n/a" if scored["composite"] is None else f"{scored['composite']:.0f}", border=True)
        st.caption("Scores are weighted averages over the components that could be computed (shown below); missing components "
                   "are excluded and the weights renormalised. Not backtested.")

        st.markdown("**Screening criteria**")
        crit = pd.DataFrame([{"Criterion": c["label"], "Result": c["status"].upper(), "Value": _fmt_crit(c),
                              "Rule": c["threshold"], "Detail": c["detail"]} for c in fe.get("criteria", [])])
        if not crit.empty:
            st.dataframe(crit, hide_index=True)
        if m.get("fcf_not_meaningful"):
            st.error(m["fcf_note"], icon=":material/report:")
        if fe.get("reasons"):
            st.warning("**Why it was excluded:** " + " | ".join(fe["reasons"]), icon=":material/rule:")
        if fe.get("flags"):
            for f in fe["flags"]:
                st.caption(f":material/flag: {f}")

        tabs_needed = bool(m.get("fcf_quarters"))
        if tabs_needed:
            st.markdown("**Free cash flow observations** (FCF = operating cash flow − capital expenditures)")
            obs = []
            for x in m["fcf_quarters"][-cfg.require_fcf_quarters:]:
                obs.append({"Period": f"Quarter ended {x['end']}", "OCF": _money(x["ocf"]), "Capex": _money(x["capex"]),
                            "FCF": _money(x["fcf"]), "Basis": x["basis"], "First reported": x["filed"] or ""})
            for x in m["fcf_years"][-cfg.require_fcf_years:]:
                obs.append({"Period": f"Fiscal year ended {x['end']}", "OCF": _money(x["ocf"]), "Capex": _money(x["capex"]),
                            "FCF": _money(x["fcf"]), "Basis": "reported FY", "First reported": x["filed"] or ""})
            st.dataframe(pd.DataFrame(obs), hide_index=True)
            with st.expander("Earlier quarters, revenue and net income history"):
                hist = []
                Q = rec["fund"]["quarters"]
                revs = {o["end"]: o for o in Q.get("revenue", [])}
                nis = {o["end"]: o for o in Q.get("net_income", [])}
                for x in m["fcf_quarters"][::-1]:
                    r, n = revs.get(x["end"]), nis.get(x["end"])
                    hist.append({"Quarter end": x["end"], "FCF": _money(x["fcf"]), "Revenue": _money(r["value"]) if r else "n/a",
                                 "Net income": _money(n["value"]) if n else "n/a",
                                 "Net margin": f"{n['value'] / r['value']:.1%}" if r and n and r["value"] else "n/a",
                                 "Basis": x["basis"]})
                st.dataframe(pd.DataFrame(hist), hide_index=True)
                ann = []
                Y = rec["fund"]["years"]
                yr = {o["end"]: o for o in Y.get("revenue", [])}
                yn = {o["end"]: o for o in Y.get("net_income", [])}
                for x in m["fcf_years"][::-1]:
                    r, n = yr.get(x["end"]), yn.get(x["end"])
                    ann.append({"Fiscal year end": x["end"], "FCF": _money(x["fcf"]), "Revenue": _money(r["value"]) if r else "n/a",
                                "Net income": _money(n["value"]) if n else "n/a",
                                "Net margin": f"{n['value'] / r['value']:.1%}" if r and n and r["value"] else "n/a"})
                st.dataframe(pd.DataFrame(ann), hide_index=True)

            st.markdown("**FCF yield calculation**")
            sh = m.get("shares")
            yc = pd.DataFrame([
                {"Item": "TTM free cash flow", "Value": _money(m.get("ttm_fcf")), "As of": f"4 quarters ending {m.get('ttm_period_end')}"},
                {"Item": "Share count", "Value": f"{sh['value']:,.0f}" if sh else "n/a",
                 "As of": f"{sh['end']} — {sh['source']}" if sh else ""},
                {"Item": "Price", "Value": f"${m['price']:,.2f}" if m.get("price") else "n/a", "As of": m.get("price_asof") or ""},
                {"Item": "Market capitalization", "Value": _money(m.get("market_cap")), "As of": "shares × price"},
                {"Item": "FCF yield", "Value": f"{m['fcf_yield']:.2%}" if m.get("fcf_yield") is not None else "n/a",
                 "As of": "TTM FCF / market cap"},
            ])
            st.dataframe(yc, hide_index=True)

            st.markdown("**Fundamental trends** (TTM series, last 5 quarter ends)")
            tr = fe.get("trends", {})
            trows = []
            for k, label in (("fcf", "Free cash flow"), ("net_margin", "Net margin"), ("revenue", "Revenue"),
                             ("revenue_growth", "Revenue growth"), ("fcf_yield", "FCF yield")):
                t = tr.get(k)
                if not t:
                    continue
                ch = t.get("change")
                kind_pp = k in ("net_margin", "revenue_growth")
                trows.append({"Metric": label + (" (critical)" if k in cfg.critical_trend_metrics else ""), "Trend": t["label"],
                              "Change over window": "n/a" if ch is None else (f"{ch * 100:+.1f} pp" if kind_pp else f"{ch:+.1%}"),
                              "Moves up / down": "n/a" if t.get("ups") is None else f"{t['ups']} / {t['downs']}",
                              "Note": t.get("note", "")})
            st.dataframe(pd.DataFrame(trows), hide_index=True)

        render_price_section(symbol, rec, d, cfg)
        st.caption(f"Sources: SEC EDGAR XBRL filings (latest filing {rec['fund'].get('latest_filed', 'n/a')}), Yahoo Finance prices "
                   f"(through {row['price_asof']}). Scan time {rec['scanned_at']}. Quarterly Q4 and cash-flow quarters are derived "
                   f"from year-to-date filings where a standalone quarter isn't reported.")


def _fmt_crit(c: dict) -> str:
    v = c["value"]
    if v is None:
        return "n/a"
    if c["key"] in ("net_margin", "revenue_growth", "fcf_yield"):
        return f"{v:.2%}"
    if c["key"] == "market_cap":
        return f"${v / 1e9:,.2f}B"
    if c["key"] == "fcf_consistency":
        return f"min {_money(v)}"
    return str(v)


def render_price_section(symbol: str, rec: dict, d: dict, cfg: ScreenConfig) -> None:
    pos = rec["price"].get("positions") or {}
    st.markdown("**Price position**")
    rows = []
    for label, name in (("52w", "52 weeks"), ("2y", "2 years"), ("3y", "3 years")):
        p = pos.get(label, {})
        if p.get("available"):
            rows.append({"Lookback": name, "High": p["high"], "Low": p["low"], "Below high %": p["pct_below_high"],
                         "Above low %": p["pct_above_low"], "Range position %": p["range_pos"], "Closing-price percentile": p["percentile"]})
        else:
            rows.append({"Lookback": name, "High": None, "Low": None, "Below high %": None, "Above low %": None,
                         "Range position %": None, "Closing-price percentile": None})
    st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={
        c: st.column_config.NumberColumn(format="%.1f") for c in ("High", "Low", "Below high %", "Above low %", "Range position %", "Closing-price percentile")})
    unavailable = [k for k, p in pos.items() if not p.get("available")]
    if unavailable:
        st.caption("Not enough trading history for: " + ", ".join(f"{k} ({pos[k].get('reason')})" for k in unavailable))
    st.caption(f"Range position = (price − low)/(high − low) of closing prices; percentile = share of prior closes below today's close. "
               f"Price zone: **{d['zone']['label']}**" + (f" (low on: {', '.join(d['zone']['in_zone'])})" if d["zone"]["in_zone"] else "") + ".")

    t = rec.get("tech") or {}
    if t:
        tr = d["trend"]
        st.markdown(f"**Technicals — {tr['label']}**")
        st.write(" · ".join(tr.get("reasons", [])))
        if tr.get("signals"):
            st.write("Recovery signals: " + ", ".join(tr["signals"]))
        st.dataframe(pd.DataFrame([{
            "SMA20": t.get("sma20"), "SMA50": t.get("sma50"), "SMA200": t.get("sma200"),
            "vs SMA50 %": t.get("dev_sma50_pct"), "vs SMA200 %": t.get("dev_sma200_pct"),
            "RSI": t.get("rsi"), "RSI band": tc.rsi_band(t.get("rsi"), cfg), "MACD": tc.macd_status(t),
            "ATR % of price": t.get("atr_pct"), "ADX": t.get("adx"), "Rel. volume": t.get("rel_volume")}]), hide_index=True)
        comps = d["scores"]["technical"]["components"]
        with st.expander("Score components"):
            for name, key in (("Fundamental", "fundamental"), ("Price opportunity", "price"), ("Technical", "technical")):
                s = d["scores"][key]
                st.write(f"**{name}**: " + ", ".join(f"{k} {v:.0f}" if v is not None else f"{k} n/a" for k, v in s["components"].items()))

    df = pr.load_cached(symbol)
    if df is not None and len(df) > 60:
        close = df["Close"]
        last = df.index[-1]
        view = df[df.index >= tc.window_start(last, 3)]
        s20, s50, s200 = (tc.sma(close, n).loc[view.index] for n in (20, 50, 200))
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])
        fig.add_trace(go.Scatter(x=view.index, y=view["Close"], name="Close", line=dict(color=wui.TEAL, width=1.8)), row=1, col=1)
        for s, name, color in ((s20, "SMA20", wui.VIOLET), (s50, "SMA50", wui.AMBER), (s200, "SMA200", wui.MAGENTA)):
            fig.add_trace(go.Scatter(x=view.index, y=s, name=name, line=dict(color=color, width=1.2)), row=1, col=1)
        for label, yrs, dash in (("52w", 1, "dot"), ("2y", 2, "dash")):
            p = pos.get(label, {})
            if p.get("available"):
                fig.add_hline(y=p["low"], line=dict(color="rgba(251,113,133,0.55)", dash=dash, width=1), row=1, col=1,
                              annotation_text=f"{label} low", annotation_position="bottom right")
                fig.add_hline(y=p["high"], line=dict(color="rgba(52,211,153,0.5)", dash=dash, width=1), row=1, col=1,
                              annotation_text=f"{label} high", annotation_position="top right")
        r = tc.rsi(close).loc[view.index]
        fig.add_trace(go.Scatter(x=view.index, y=r, name="RSI", line=dict(color=wui.VIOLET, width=1.3)), row=2, col=1)
        fig.add_hline(y=30, line=dict(color="rgba(251,113,133,0.5)", dash="dot", width=1), row=2, col=1)
        fig.add_hline(y=70, line=dict(color="rgba(52,211,153,0.5)", dash="dot", width=1), row=2, col=1)
        mac = tc.macd(close).loc[view.index]
        fig.add_trace(go.Bar(x=view.index, y=mac["hist"], name="MACD hist",
                             marker=dict(color=[wui.TEAL if v >= 0 else wui.MAGENTA for v in mac["hist"]], opacity=0.55)), row=3, col=1)
        fig.add_trace(go.Scatter(x=view.index, y=mac["macd"], name="MACD", line=dict(color=wui.AMBER, width=1)), row=3, col=1)
        fig.add_trace(go.Scatter(x=view.index, y=mac["signal"], name="Signal", line=dict(color=wui.VIOLET, width=1)), row=3, col=1)
        layout = wui.plotly_layout(height=620, hovermode="x unified", showlegend=True,
                                   legend=dict(orientation="h", y=-0.06, bgcolor="rgba(0,0,0,0)"))
        for ax in ("xaxis", "yaxis"):
            layout.pop(ax, None)
        fig.update_layout(**layout)
        grid = "rgba(255,255,255,0.06)" if st.context.theme.type != "light" else "rgba(20,26,51,0.08)"
        fig.update_xaxes(gridcolor=grid)
        fig.update_yaxes(gridcolor=grid)
        st.plotly_chart(fig, config={"displayModeBar": False})

        # rolling closing-price percentile over a 1-year window
        roll = close.rolling(252, min_periods=252).apply(
            lambda w: ((w[:-1] < w[-1]).sum() + 0.5 * (w[:-1] == w[-1]).sum()) / (len(w) - 1) * 100, raw=True).dropna()
        roll = roll[roll.index >= view.index[0]]
        if not roll.empty:
            fig2 = go.Figure(go.Scatter(x=roll.index, y=roll, name="1-year closing-price percentile",
                                        line=dict(color=wui.VIOLET, width=1.6), fill="tozeroy", fillcolor="rgba(167,139,250,0.12)"))
            fig2.add_hline(y=cfg.low_percentile, line=dict(color="rgba(251,113,133,0.7)", dash="dot"),
                           annotation_text=f"low zone ≤ {cfg.low_percentile:.0f}")
            fig2.update_layout(**wui.plotly_layout(height=240, yaxis_title="percentile", yaxis_range=[0, 100]))
            st.plotly_chart(fig2, config={"displayModeBar": False})
            st.caption("Where each day's close ranked among the prior 251 closes (a rolling 1-year percentile).")


# ---------------------------------------------------------------------------
# alerts
# ---------------------------------------------------------------------------

def render_alerts() -> None:
    ex = universe.excluded()
    items = [a for a in store.list_alerts(limit=300) if a["symbol"].upper() not in ex][:100]
    unread = [a for a in items if not a["is_read"]]
    with st.container(key="glass_screen_alerts"):
        st.subheader(f"Screener alerts ({len(unread)} unread)", anchor=False)
        if not items:
            st.caption("No alerts yet. They fire after a scan when a stock that passes the fundamental screen enters the "
                       "low-price zone (see alert kinds in the screening profile). The same unchanged condition never repeats; "
                       "it can fire again after the stock leaves and re-enters the zone, subject to the cooldown. Alerts are "
                       "never sent on stale or unverifiable data.")
            return
        if unread and st.button("Mark all as read", icon=":material/done_all:"):
            store.mark_alerts_read()
            st.rerun()
        for a in items[:25]:
            icon = ":material/notifications_active:" if not a["is_read"] else ":material/notifications:"
            with st.container(border=True):
                st.markdown(f"{icon} **{a['symbol']}** — {alerts_mod.KINDS.get(a['kind'], a['kind'])} · "
                            f"{a['created_at'][:16].replace('T', ' ')}")
                st.write(f"{a['name'] or ''} · price ${a['price']:,.2f}" if a["price"] else (a["name"] or ""))
                st.caption(f"{a['trigger_text']} · lowest percentile {a['percentile']:.0f} · FCF yield "
                           f"{a['fcf_yield']:.1%} · fundamentals {a['fund_status']} · technicals {a['tech_status']}"
                           if a["percentile"] is not None and a["fcf_yield"] is not None else
                           f"{a['trigger_text']} · fundamentals {a['fund_status']} · technicals {a['tech_status']}")
                st.caption(f"Reason: {a['reason']} · financial data through {a['data_asof']} · prices through {a['price_asof']}")
