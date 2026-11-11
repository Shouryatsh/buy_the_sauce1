import json
from datetime import date, datetime

import pandas as pd
import streamlit as st

import wheel_config as wc
import wheel_events
import wheel_ui as ui
from premia import mcmillan as mm
from premia.config import RESULTS_DIR
from straddle_options import fetch_straddle_quotes_batch, iv_history_batch
from straddle_recommend import (combine_iv, evaluate_quote, follow_up_for, held_straddles, pick_expiry, rank,
                                recent_duration, stored_iv, wheel_universe)

client = ui.get_client()
if client is None:
    st.info("Connect to IBKR in the left rail to scan for straddles (needs implied-vol history and option chains).",
            icon=":material/link:")
    st.stop()

st.caption("Lawrence McMillan's straddle buying (*McMillan on Options*, ch. 7, \"My Favorite Strategy\"): buy cheap "
           "volatility, then leave it alone until it either moves or loses 60 %. Read-only: nothing here places an order.")

verdict_file = RESULTS_DIR / "verdict_mcmillan_straddle.json"
if verdict_file.exists():
    v = json.loads(verdict_file.read_text())
    lines = "; ".join(f"{'✓' if c['pass'] else '✗'} {c['criterion']} ({c['evidence']})" for c in v["checks"])
    if v["verdict"] == "PASS":
        st.success(f"**Backtest verdict: PASS.** {lines}. Details on the Premia research page.", icon=":material/check_circle:")
    else:
        st.warning(f"**Backtest verdict: FAIL.** {lines}. The rules below are the book's, as you asked; the backtest "
                   "says they have not made money net of costs. Details on the Premia research page.", icon=":material/warning:")

with st.expander("The rules (all four must hold) and the follow-up", icon=":material/rule:"):
    st.markdown(
        f"1. **Cheap options**: implied volatility at or below the **{mm.IV_PCT_MAX:.0f}th percentile** of all its past "
        f"readings; options with **at least {mm.MIN_LIFE_DAYS} days** of life (the expiry nearest six months is used).\n"
        f"2. **Probability**: at least **{mm.EVER_PROB_MIN:.0%}** chance the stock *ever* trades at a breakeven "
        f"(strike ± straddle price) before expiry, lognormal, using the **lowest** of the 10/20/50/100-day historical vols.\n"
        f"3. **History**: the stock made the required move in at least **{mm.HIST_FREQ_MIN:.0%}** of same-length windows "
        f"over the past {mm.HIST_YEARS} years.\n"
        f"4. **Fundamentals**: no takeover bid or other reason for cheap options. The page flags a price pinned after a "
        f"jump; **check the news yourself** before buying.\n\n"
        f"**Follow-up**: risk **{mm.RISK_FRAC:.0%}** of the price paid; sell with about **{mm.TIME_EXIT_DAYS} days** of "
        f"life left. When the stock closes beyond a breakeven: sell all of the losing side and about a third of the "
        f"winning side, then sell the rest when the stock closes back through its **{mm.TRAIL_MA}-day moving average**. "
        f"No delta hedging.\n\n"
        f"Liquidity and size (premia limits): straddle spread ≤ {wc.STRADDLE_MAX_SPREAD_PCT:.0%} of mid, open interest ≥ "
        f"{wc.STRADDLE_MIN_OPEN_INTEREST:,} per leg, {wc.STRADDLE_BUDGET_PCT_NLV:.1%} of net liquidation per straddle, "
        f"{wc.STRADDLE_MAX_TOTAL_PCT_NLV:.0%} for all of them.")

TOP, ALL = f"Top {wc.STRADDLE_SCREEN_TOP} by fundamentals (fast)", "Every wheel name (slow)"
scope = st.segmented_control("Names to scan", [TOP, ALL], default=TOP, key="mcm_scope",
                             help="Top: the fundamentals screen's passing names with the best quality score, as on the "
                                  "Wheel recommendations page. Every: your whole watchlist minus exclusions "
                                  "(about 270 IBKR requests, several minutes).")
if scope == ALL:
    universe = wheel_universe()
    st.caption(f"Universe: all {len(universe)} names the wheel evaluates (your watchlist minus your exclusions).")
else:
    profiles = ui.profiles_or_none()
    if profiles is None:
        ui.need_screen("need_screen_straddle")
        st.stop()
    passing = sorted((p for p in profiles if p.passes), key=lambda p: p.quality_score or 0, reverse=True)
    universe = [p.symbol for p in passing][: wc.STRADDLE_SCREEN_TOP]
    if not universe:
        st.warning("No name passes the fundamentals screen.", icon=":material/filter_alt_off:")
        st.stop()
    st.caption(f"Universe: the {len(universe)} fundamentals-passing names with the best quality score "
               f"(of {len(passing)} passing): {', '.join(universe)}")


@st.cache_data(ttl=3600, show_spinner=False)
def _stored(symbol: str) -> pd.Series:
    return stored_iv(symbol)


if st.button("Scan the universe", icon=":material/radar:", type="primary"):
    with st.spinner(f"Step 1 of 3: implied-vol percentile for {len(universe)} names (stored history + recent IBKR)..."):
        stored = {s: _stored(s) for s in universe}
        recent = iv_history_batch(client, [(s, recent_duration(stored[s])) for s in universe])
        rows = []
        for s in universe:
            iv = combine_iv(stored[s], recent.get(s))
            pct = mm.latest_iv_percentile(iv)
            rows.append({"Symbol": s, "IV": float(iv.iloc[-1]) if len(iv) else None, "IV percentile": pct,
                         "Readings": len(iv), "As of": iv.index[-1].date() if len(iv) else None})
        scan = pd.DataFrame(rows)
    cheap = scan[scan["IV percentile"].notna() & (scan["IV percentile"] <= mm.IV_PCT_MAX)]["Symbol"].tolist()
    ideas, hist = [], {}
    if cheap:
        with st.spinner(f"Step 2 of 3: price history for {len(cheap)} name(s) with cheap options..."):
            for s in cheap:
                df, _src = ui.price_history(s, wc.STRADDLE_HISTORY_YEARS)
                if df is not None and not df.empty:
                    hist[s] = df
        with st.spinner(f"Step 3 of 3: six-month straddle quotes for {len(hist)} name(s)..."):
            quotes = fetch_straddle_quotes_batch(client, [(s, float(df["Close"].iloc[-1])) for s, df in hist.items()],
                                                 dte_min=wc.STRADDLE_DTE_MIN, dte_max=wc.STRADDLE_DTE_MAX, max_expiries=6)
        nlv = client.get_account_summary().net_liquidation
        positions = client.get_positions()
        held = held_straddles(positions)
        committed = sum((c.market_value or 0) + (p.market_value or 0) for _s, _k, _e, c, p in held)
        pct_by = dict(zip(scan["Symbol"], scan["IV percentile"]))
        for s, qs in quotes.items():
            q = pick_expiry(qs)
            if q is not None:
                ideas.append(evaluate_quote(q, hist[s], pct_by.get(s), nlv, committed, s in {h[0] for h in held}))
        missing = [s for s in hist if not quotes.get(s)]
    else:
        missing, nlv, committed = [], None, 0.0
    st.session_state.mcm_scan = {"scan": scan, "ideas": ideas, "missing": missing, "at": datetime.now().strftime("%H:%M"),
                                 "nlv": nlv, "committed": committed}

res = st.session_state.get("mcm_scan")

buy_tab, scan_tab = st.tabs([":material/shopping_cart: Straddles to buy", ":material/list: Universe scan"])

with buy_tab:
    if res is None:
        st.info("Press **Scan the universe** to look for straddles that meet all four criteria.", icon=":material/radar:")
    else:
        buys, rest = rank(res["ideas"])
        n_cheap = int((res["scan"]["IV percentile"] <= mm.IV_PCT_MAX).sum())
        st.caption(f"Scanned at {res['at']}: {n_cheap} of {len(res['scan'])} names have IV at or below the "
                   f"{mm.IV_PCT_MAX:.0f}th percentile; {len(res['ideas'])} had a quoted straddle 3–8 months out; "
                   f"{len(buys)} meet every criterion.")
        if res["missing"]:
            st.caption(f"No two-sided straddle quote 3–8 months out for: {', '.join(res['missing'])}")
        if not buys:
            st.info("**No straddle meets all four criteria right now.** McMillan notes that cheap straddles come and "
                    "go; the table below shows how close each candidate got.", icon=":material/do_not_disturb_on:")
        for i, r in enumerate(buys):
            c = r.criteria
            exp = datetime.strptime(r.expiry, "%Y%m%d")
            nxt = wheel_events.next_earnings(r.symbol)
            with st.container(key=f"glass_mcm_{i}"):
                c1, c2, c3 = st.columns([2, 2, 1])
                c1.markdown(f"**{r.symbol}** \\${r.strike:g} straddle (buy call + put), exp {exp:%b %d %Y} ({r.dte}d)")
                c1.caption(f"Bid \\${r.bid:.2f} / Ask \\${r.ask:.2f} · expected fill \\${r.cost:.2f} · stock "
                           f"\\${r.underlying_price:,.2f} · spread {r.spread_pct:.0%} · min open interest "
                           f"{r.min_open_interest or 0:,} per leg")
                with c1.container(horizontal=True):
                    if r.quote_type != "live":
                        st.badge(f"{r.quote_type} quote — verify before trading", icon=":material/schedule:", color="orange")
                    st.badge("check the news for a takeover (criterion 4)", icon=":material/newspaper:", color="blue")
                    if nxt is not None:
                        st.badge(f"earnings {nxt:%b %d}", icon=":material/event:", color="gray")
                c2.markdown(f"1 · IV percentile **{c.iv_percentile:.0f}** · 2 · P(ever touch ±\\${r.cost:.2f}) "
                            f"**{c.ever_prob:.0%}** at {c.min_hv:.0%} vol · 3 · moved that far in **{c.hist_freq:.0%}** "
                            f"of past windows")
                c2.caption(f"Breakevens \\${c.breakeven_down:,.2f} / \\${c.breakeven_up:,.2f} "
                           f"(±{r.cost / r.underlying_price:.1%})"
                           + (f" · straddle IV {r.straddle_iv:.1%}" if r.straddle_iv else ""))
                c3.metric("Cost", f"${r.cost_total:,.0f}", f"{r.contracts_suggested} straddle(s)", delta_color="off")
                with st.expander("Follow-up plan", icon=":material/logout:"):
                    st.markdown(
                        f"- **Stop**: sell both sides if the straddle is worth **\\${r.stop_value:.2f}** or less "
                        f"(60 % of \\${r.cost:.2f} lost).\n"
                        f"- **Time**: sell by **{r.time_exit:%b %d %Y}** (about a month of life left).\n"
                        f"- **Breakout up** (close above \\${c.breakeven_up:,.2f}): sell all the puts and a third of the "
                        f"calls; sell the other calls when the stock closes below its 20-day moving average.\n"
                        f"- **Breakout down** (close below \\${c.breakeven_down:,.2f}): sell all the calls and a third of "
                        f"the puts; sell the other puts when the stock closes above its 20-day moving average.")
                st.caption("Enter it in TWS yourself as a combo order (BUY call + BUY put, limit near the mid).")
        if rest:
            st.markdown("**Cheap-volatility candidates that miss a criterion**")
            t = pd.DataFrame([{
                "Symbol": r.symbol, "Expiry": datetime.strptime(r.expiry, "%Y%m%d").date(), "DTE": r.dte,
                "Strike": r.strike, "Cost": r.cost, "IV pct": r.criteria.iv_percentile, "P(ever)": r.criteria.ever_prob,
                "Hist. freq": r.criteria.hist_freq, "Spread": r.spread_pct, "Min OI": r.min_open_interest,
                "Why not": "; ".join(r.reasons)} for r in rest])
            st.dataframe(t, hide_index=True, width="stretch",
                         column_config={"Cost": st.column_config.NumberColumn(format="$%.2f"),
                                        "IV pct": st.column_config.NumberColumn(format="%.0f"),
                                        "P(ever)": st.column_config.NumberColumn(format="percent"),
                                        "Hist. freq": st.column_config.NumberColumn(format="percent"),
                                        "Spread": st.column_config.NumberColumn(format="percent"),
                                        "Why not": st.column_config.TextColumn(width="large")})

    held = held_straddles(client.get_positions())
    if held:
        st.markdown("**Straddles you hold: follow-up**")
        rows = []
        for sym, k, e, c, p in held:
            df, _src = ui.price_history(sym, 1)
            if df is None or df.empty:
                continue
            close, sma = float(df["Close"].iloc[-1]), float(df["Close"].rolling(mm.TRAIL_MA).mean().iloc[-1])
            h = follow_up_for(sym, k, e, c, p, close, sma)
            act = h.action
            todo = ("hold" if act.mode == "both" and not (act.sell_calls or act.sell_puts) else
                    f"sell {act.sell_calls} call(s), {act.sell_puts} put(s): {act.reason}")
            rows.append({"Symbol": sym, "Strike": k, "Expiry": datetime.strptime(e, "%Y%m%d").date(), "Straddles": h.contracts,
                         "Cost/sh": h.cost, "Value/sh": h.value, "Stop at": (1 - mm.RISK_FRAC) * h.cost,
                         "Breakevens": f"{h.breakeven_down:,.2f} / {h.breakeven_up:,.2f}", "Close": close,
                         f"{mm.TRAIL_MA}d MA": sma, "Action (on today's close)": todo})
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="$%.2f")
                                        for c in ("Cost/sh", "Value/sh", "Stop at", "Close", f"{mm.TRAIL_MA}d MA")})
            st.caption("Straddles where one side was already sold are not shown: the page cannot tell which follow-up "
                       "stage they are in. Apply the 20-day moving-average stop to the remaining side yourself.")

with scan_tab:
    if res is None:
        st.info("Press **Scan the universe** first.", icon=":material/radar:")
    else:
        s = res["scan"].sort_values("IV percentile", na_position="last")
        st.dataframe(s, hide_index=True, width="stretch",
                     column_config={"IV": st.column_config.NumberColumn(format="percent"),
                                    "IV percentile": st.column_config.ProgressColumn(
                                        "IV percentile (vs all past readings)", min_value=0, max_value=100, format="%.0f"),
                                    "Readings": st.column_config.NumberColumn("Days of IV history")})
        st.caption("Percentiles need at least two years of readings. History: the research store's IBKR implied-vol "
                   "download, topped up with recent IBKR history on every scan.")
