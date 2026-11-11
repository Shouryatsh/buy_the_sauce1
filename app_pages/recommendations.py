import streamlit as st

import wheel_config as wc
import wheel_ui as ui
import wheel_vol as wv
from wheel_options import fetch_put_candidates_batch, realized_volatility
from wheel_recommend import rank_recommendations

client = ui.get_client()
if client is None:
    st.info("Connect to IBKR in the left rail to build recommendations (needs option chains).",
            icon=":material/link:")
    st.stop()
profiles = ui.profiles_or_none()
if profiles is None:
    ui.need_screen("need_screen_recs")
    st.stop()

capital_targets = ui.capital_targets()
profiles_by_symbol = {p.symbol: p for p in profiles if p.passes}
positions = client.get_positions()
short_puts = [p for p in positions if p.sec_type == "OPT" and p.right == "P" and p.quantity < 0]
capital_committed = sum((p.strike or 0) * 100 * abs(p.quantity) for p in short_puts)
spare_capital_remaining = max(capital_targets.spare_capital - capital_committed, 0.0)
existing_exposure_by_symbol: dict = {}
for p in short_puts:
    existing_exposure_by_symbol[p.symbol] = (existing_exposure_by_symbol.get(p.symbol, 0.0)
                                             + (p.strike or 0) * 100 * abs(p.quantity))

st.caption(f"Spare capital remaining: \\${spare_capital_remaining:,.0f} of "
           f"\\${capital_targets.spare_capital:,.0f} ({capital_targets.source})")
st.caption(f"Expiries: {'monthly (third-Friday) only' if wc.MONTHLY_EXPIRIES_ONLY else 'all'}, "
           f"{wc.DEFAULT_DTE_MIN}–{wc.DEFAULT_DTE_MAX} days out · |delta| ≤ {wc.MAX_DELTA_ABS} · "
           f"spread ≤ {wc.MAX_SPREAD_PCT:.0%} of mid · premium ≥ \\${wc.MIN_PREMIUM_PER_CONTRACT:.0f}/contract · "
           f"open interest ≥ {wc.MIN_OPEN_INTEREST}")
w = wc.RECOMMENDATION_WEIGHTS
st.caption(f"Ranking: quality {w['quality']:.0%} · IV percentile vs its own "
           f"{wc.IV_PERCENTILE_LOOKBACK_DAYS // 252}-year history {w['iv_percentile']:.0%} · "
           f"capital efficiency {w['capital_efficiency']:.0%} (weighted geometric mean).")

# Each symbol costs at least one ~11s IBKR snapshot per expiry (8 symbols are
# pulled concurrently); default to the top names by quality score and let the
# user widen or narrow it.
ranked_symbols = sorted(profiles_by_symbol, key=lambda s: profiles_by_symbol[s].quality_score or 0, reverse=True)
selected_symbols = st.multiselect(
    "Symbols to pull option chains for (fewer = faster; each one is a live IBKR request)",
    options=ranked_symbols, default=ranked_symbols[:min(wc.RECOMMEND_DEFAULT_SYMBOLS, len(ranked_symbols))],
)

if st.button("Build recommendations", icon=":material/bolt:", type="primary"):
    if not selected_symbols:
        st.warning("Select at least one symbol above.")
        st.stop()

    requests, iv_pct = [], {}
    with st.spinner(f"Fetching prices and implied-vol history for {len(selected_symbols)} symbol(s)..."):
        for symbol in selected_symbols:
            hist, _src = ui.price_history(symbol)
            if hist is None or hist.empty:
                continue
            requests.append((symbol, float(hist["Close"].iloc[-1]), realized_volatility(hist)))
            try:   # IV vs its OWN history (McMillan), same data as the Volatility page
                iv_pct[symbol] = wv.percentile_rank(ui.iv_history(symbol, "5 Y", _client=client),
                                                    wc.IV_PERCENTILE_LOOKBACK_DAYS)
            except Exception:
                iv_pct[symbol] = None

    with st.spinner(f"Pulling option chains for {len(requests)} symbol(s)..."):
        candidates_by_symbol = fetch_put_candidates_batch(client, requests)

    recs = rank_recommendations(profiles_by_symbol, candidates_by_symbol, spare_capital_remaining,
                                existing_exposure_by_symbol, total_book=capital_targets.total_book,
                                iv_percentile_by_symbol=iv_pct)
    with_chains = sum(1 for s in candidates_by_symbol.values() if s)
    st.caption(f"{with_chains} of {len(requests)} symbols had at least one put clearing every filter.")
    if not recs:
        st.warning("No recommendations cleared the filters. Outside market hours option spreads widen "
                   "far past the 10% limit — run this between 9:45 AM and 4 PM ET.")
    for i, r in enumerate(recs):
        with st.container(key=f"glass_rec_{i}"):
            c1, c2, c3 = st.columns([2, 2, 1])
            c1.markdown(f"**{r.symbol}** \\${r.strike} put, exp {r.expiry} ({r.dte}d)")
            quote = (f"Bid \\${r.bid:.2f} / Ask \\${r.ask:.2f} · mid \\${r.premium:.2f}"
                     if r.bid is not None and r.ask is not None else f"Premium \\${r.premium:.2f}")
            bid_yield = (r.bid / r.strike * 365 / r.dte) if r.bid and r.strike and r.dte > 0 else None
            c1.caption(f"{quote} | Delta {r.delta:.2f}{' (est.)' if r.delta_is_estimated else ''} | "
                       f"Ann. yield {r.annualized_yield:.1%} at mid"
                       + (f", {bid_yield:.1%} at bid" if bid_yield is not None else "")
                       + " | OI " + (f"{r.open_interest:,}" if r.open_interest is not None else "n/a"))
            if r.quote_type != "live":
                c1.badge(f"{r.quote_type} quote — verify before trading", icon=":material/schedule:",
                         color="orange")
            if r.bid is not None and r.ask is not None and r.premium > 0:
                spread_pct = (r.ask - r.bid) / r.premium
                if spread_pct > wc.WIDE_SPREAD_WARN_PCT:
                    c1.badge(f"wide spread {spread_pct:.0%} of mid — use a limit order near the mid",
                             icon=":material/swap_horiz:", color="orange")
            iv_txt = (f"IV percentile **{r.iv_percentile:.0f}** ({wv.label(r.iv_percentile)})"
                      if r.iv_percentile is not None else "IV percentile n/a (scored neutral)")
            c2.markdown(f"Quality **{r.quality_score:.0f}** | {iv_txt}")
            c2.caption(f"Capital efficiency {r.capital_efficiency_score:.0f} | Composite rank **{r.composite_rank:.0f}**")
            c3.metric("Capital req.", f"${r.capital_required:,.0f}")
            if not r.capital_ok:
                st.warning("; ".join(r.capital_notes))
            st.caption("Preview the order on the Order ticket page; place it in TWS yourself.")
