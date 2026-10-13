"""
wheel_dashboard.py — Streamlit dashboard for the options wheel decision
support tool (Phase 1: read-only).

Run with:
    streamlit run wheel_dashboard.py

Requires TWS or IB Gateway running locally with API access enabled (see
README.md). Live/delayed OPRA options data is needed for broker-reported
greeks/IV -- without it, deltas and IV shown here are the Black-Scholes /
realized-vol proxy from wheel_options.py, and are labeled "(est.)".

This dashboard never places an order on its own. The "Execute" button in the
Ranked Recommendations panel is disabled until Phase 5 (wheel_execution.py).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import wheel_config as wc
from wheel_watchlist import WHEEL_WATCHLIST
from wheel_universe import scan_universe
from wheel_options import fetch_put_candidates, realized_volatility
from wheel_recommend import rank_recommendations
from wheel_risk import compute_exposure
from wheel_ibkr import WheelIBKRClient
from wheel_capital import resolve_capital_targets
import wheel_db

logging.basicConfig(level=logging.WARNING)

st.set_page_config(page_title="Options Wheel Dashboard", layout="wide")

# Note: WheelIBKRClient (wheel_ibkr.py) routes every ib_insync call through
# one dedicated background thread/event loop, so it stays safe to keep in
# st.session_state and call from whichever thread Streamlit uses for each
# script rerun -- no per-rerun setup needed here.


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner="Screening fundamentals (SEC EDGAR + bulk price)...")
def _screen(symbols: tuple[str, ...]):
    # scan_universe() concurrently fetches EDGAR fundamentals (rate-limited
    # in aggregate, see edgar.py) and bulk-prefetches prices via yfinance --
    # the same fast path used for the one-time universe scan, so this stays
    # workable even once WHEEL_WATCHLIST is a few hundred shortlisted names
    # rather than the original ~44. Cached for an hour since fundamentals
    # don't move intraday anyway.
    profiles = scan_universe(list(symbols))
    profiles.sort(key=lambda p: (p.passes, p.quality_score or 0), reverse=True)
    return profiles


@st.cache_data(ttl=900, show_spinner=False)
def _price_history(symbol: str, period_years: int = 1):
    """Cached so the same symbol's price history (needed for both the
    latest price and realized vol) is fetched once per 15 minutes, not
    once per tab per click -- the Fundamentals Screener, Options Radar, and
    Ranked Recommendations tabs all want it for the same symbols."""
    from edgar import get_price_history
    return get_price_history(symbol, period_years=period_years)


# ---------------------------------------------------------------------------
# Sidebar — explicit Connect/Disconnect.
#
# The connection lives in st.session_state (one IB() socket per browser
# session, kept across reruns) rather than being dialed automatically on
# every script rerun: an automatic attempt on every rerun means every widget
# click anywhere on the page re-attempts a multi-second TCP connect whenever
# TWS isn't reachable, and gives no visible feedback distinguishing "haven't
# tried yet" from "tried and failed".
# ---------------------------------------------------------------------------

st.sidebar.title("Options Wheel Dashboard")
use_paper = False  # live-only: no paper option
st.sidebar.warning(f"LIVE account (port {wc.IBKR_LIVE_PORT}). Read-only: this dashboard never places orders.")

if "ibkr_client" not in st.session_state:
    st.session_state.ibkr_client = None
if "ibkr_error" not in st.session_state:
    st.session_state.ibkr_error = None

col_connect, col_disconnect = st.sidebar.columns(2)
if col_connect.button("Connect", use_container_width=True):
    try:
        new_client = WheelIBKRClient(use_paper=use_paper)
        new_client.connect()
        st.session_state.ibkr_client = new_client
        st.session_state.ibkr_error = None
    except Exception as exc:
        st.session_state.ibkr_client = None
        st.session_state.ibkr_error = str(exc)
        logging.warning("IBKR connection failed: %s", exc)

if col_disconnect.button("Disconnect", use_container_width=True):
    if st.session_state.ibkr_client is not None:
        try:
            st.session_state.ibkr_client.disconnect()
        except Exception:
            pass
    st.session_state.ibkr_client = None
    st.session_state.ibkr_error = None

client = st.session_state.ibkr_client
if client is not None and client.is_connected():
    st.sidebar.success(f"Connected ({'Paper' if use_paper else 'Live'}, port {client._port})")
elif st.session_state.ibkr_error:
    st.sidebar.error(f"Connection failed: {st.session_state.ibkr_error}")
    client = None
else:
    st.sidebar.warning("Not connected — click Connect")
    client = None

st.sidebar.caption(
    "Fundamentals: SEC EDGAR (free, no key). Options data needs TWS/IB "
    "Gateway running with API access enabled; real greeks/IV need an OPRA "
    "subscription, else deltas shown are a Black-Scholes estimate."
)

capital_targets = resolve_capital_targets(client.get_account_summary() if client else None)
st.sidebar.divider()
st.sidebar.caption(
    f"Capital targets ({capital_targets.source}): "
    f"invested ${capital_targets.invested_capital:,.0f} | "
    f"spare ${capital_targets.spare_capital:,.0f} | "
    f"book ${capital_targets.total_book:,.0f}"
    + ("" if client else " — connect to use live account figures instead of wheel_config.py")
)

tabs = st.tabs([
    "Account Overview", "Monthly P&L", "Exposure", "Fundamentals Screener",
    "Options Radar", "Ranked Recommendations", "Trade Log",
])


# ---------------------------------------------------------------------------
# Account Overview
# ---------------------------------------------------------------------------

with tabs[0]:
    st.subheader("Account Overview")
    if client is None:
        st.info("Connect to IBKR to see live account data.")
    else:
        summary = client.get_account_summary()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Net Liquidation", f"${summary.net_liquidation:,.0f}")
        c2.metric("Total Cash", f"${summary.total_cash:,.0f}")
        c3.metric("Buying Power", f"${summary.buying_power:,.0f}")
        c4.metric("Margin Used", f"${summary.margin_used:,.0f}")

        st.markdown("**Open Positions**")
        positions = client.get_positions()
        if positions:
            df = pd.DataFrame([p.__dict__ for p in positions])
            st.dataframe(df, use_container_width=True)
        else:
            st.caption("No open positions.")

        st.markdown("**Open Orders**")
        orders = client.get_open_orders()
        st.dataframe(pd.DataFrame(orders), use_container_width=True) if orders else st.caption("No open orders.")


# ---------------------------------------------------------------------------
# Monthly P&L
# ---------------------------------------------------------------------------

with tabs[1]:
    st.subheader("Monthly P&L")
    if client is None:
        st.info("Connect to IBKR to see trade history.")
    else:
        days = st.slider("Lookback window (days)", 7, 90, 30)
        start_date = (datetime.now().date() - timedelta(days=days)).isoformat()

        stats = client.get_trade_stats(days=days)
        manual = wheel_db.get_manual_realized_pnl_since(start_date)
        combined_realized_pnl = stats.realized_pnl + manual["realized_pnl"]
        combined_trade_count = stats.trade_count + manual["trade_count"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Realized P&L", f"${combined_realized_pnl:,.2f}")
        c2.metric("Win Rate", f"{stats.win_rate:.0%}" if stats.win_rate is not None else "N/A")
        c3.metric("Avg Premium / Contract", f"${stats.avg_premium_per_contract:,.2f}" if stats.avg_premium_per_contract else "N/A")
        c4.metric("Avg Days to Close", f"{stats.avg_days_to_close:.1f}" if stats.avg_days_to_close is not None else "N/A")
        if manual["trade_count"]:
            st.caption(
                f"${stats.realized_pnl:,.2f} from {stats.trade_count} IBKR-visible fill(s) "
                f"+ ${manual['realized_pnl']:,.2f} from {manual['trade_count']} manually-logged trade(s)"
            )

        progress = min(combined_realized_pnl / wc.TARGET_MONTHLY_INCOME, 1.0) if wc.TARGET_MONTHLY_INCOME else 0.0
        st.progress(max(progress, 0.0), text=f"${combined_realized_pnl:,.0f} / ${wc.TARGET_MONTHLY_INCOME:,.0f} monthly target")
        st.caption(
            "IBKR's execution API has retained essentially nothing beyond the "
            "current session in testing against this account -- confirmed empty "
            "even on a 90-day request. A worthless expiration also never "
            "generates an execution at all (the position just disappears), so "
            "it's invisible to IBKR's fill history no matter the window. Use "
            "'Log a closed trade' below for anything IBKR's API didn't surface."
        )

        with st.expander("Log a closed trade IBKR didn't show (expiration, assignment, etc.)"):
            with st.form("manual_trade_form", clear_on_submit=True):
                fc1, fc2, fc3 = st.columns(3)
                m_symbol = fc1.text_input("Symbol").upper().strip()
                m_sec_type = fc2.selectbox("Type", ["Put", "Call"])
                m_contracts = fc3.number_input("Contracts", min_value=1, value=1, step=1)

                fc4, fc5, fc6 = st.columns(3)
                m_strike = fc4.number_input("Strike", min_value=0.0, value=0.0, step=0.5)
                m_premium = fc5.number_input("Premium collected per contract ($/share)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
                m_outcome = fc6.selectbox("Outcome", ["Expired worthless", "Assigned", "Bought to close"])

                m_close_price = 0.0
                if m_outcome == "Bought to close":
                    m_close_price = st.number_input("Price paid to close ($/share)", min_value=0.0, value=0.0, step=0.01, format="%.2f")

                m_closed_date = st.date_input("Date closed", value=datetime.now().date())
                m_notes = st.text_input("Notes (optional)")

                if st.form_submit_button("Log trade"):
                    if not m_symbol or m_premium <= 0:
                        st.error("Symbol and premium collected are required.")
                    else:
                        if m_outcome == "Bought to close":
                            realized = (m_premium - m_close_price) * 100 * m_contracts
                        else:
                            # Expired worthless or assigned: the option premium
                            # is kept in full either way -- assignment's
                            # resulting stock position isn't tracked here.
                            realized = m_premium * 100 * m_contracts
                        action = {
                            "Expired worthless": "EXPIRED_WORTHLESS",
                            "Assigned": "ASSIGNED",
                            "Bought to close": "BOUGHT_TO_CLOSE",
                        }[m_outcome]
                        wheel_db.record_manual_trade(
                            symbol=m_symbol,
                            sec_type="OPT",
                            action=action,
                            quantity=m_contracts,
                            premium_per_contract=m_premium,
                            realized_pnl=realized,
                            closed_date=m_closed_date.isoformat(),
                            notes=f"{m_sec_type} ${m_strike:g}" + (f" — {m_notes}" if m_notes else ""),
                        )
                        st.success(f"Logged {m_symbol}: ${realized:,.2f} realized.")
                        st.rerun()


# ---------------------------------------------------------------------------
# Exposure
# ---------------------------------------------------------------------------

with tabs[2]:
    st.subheader("Exposure")
    if client is None:
        st.info("Connect to IBKR to compute exposure.")
    else:
        positions = client.get_positions()
        short_put_symbols = {p.symbol for p in positions if p.sec_type == "OPT" and p.right == "P" and p.quantity < 0}
        underlying_prices, implied_vols = {}, {}
        for sym in short_put_symbols:
            hist, _src = _price_history(sym)
            if hist is not None and not hist.empty:
                underlying_prices[sym] = float(hist["Close"].iloc[-1])
                rv = realized_volatility(hist)
                if rv:
                    implied_vols[sym] = rv   # proxy: no live IV pulled here in Phase 1 overview

        report = compute_exposure(positions, underlying_prices, implied_vols, total_book=capital_targets.total_book)
        c1, c2, c3 = st.columns(3)
        c1.metric("Raw Exposure", f"${report.raw_exposure:,.0f}")
        c2.metric("Delta-Adjusted Exposure", f"${report.delta_adjusted_exposure:,.0f}")
        c3.metric(
            "Stress Exposure",
            f"${report.stress_exposure:,.0f}",
            help=f"Underlying {wc.STRESS_UNDERLYING_SHOCK_PCT:+.0%}, IV {wc.STRESS_IV_SHOCK_PCT:+.0%}",
        )
        st.caption("Delta-adjusted and stress figures use the Phase 1 Black-Scholes proxy — see wheel_risk.py.")

        if report.per_name_concentration:
            st.markdown("**Per-Name Concentration**")
            conc_df = pd.DataFrame(
                [{"symbol": s, "pct_of_book": p} for s, p in report.per_name_concentration.items()]
            ).sort_values("pct_of_book", ascending=False)
            st.dataframe(conc_df.style.format({"pct_of_book": "{:.1%}"}), use_container_width=True)
            if report.concentration_breaches:
                st.error(f"Over MAX_NAME_CONCENTRATION ({wc.MAX_NAME_CONCENTRATION:.0%}): {', '.join(report.concentration_breaches)}")
        else:
            st.caption("No open short puts.")


# ---------------------------------------------------------------------------
# Fundamentals Screener
# ---------------------------------------------------------------------------

with tabs[3]:
    st.subheader("Fundamentals Screener")
    st.caption(f"Universe: {len(WHEEL_WATCHLIST)} tickers, price <= ${wc.MAX_ASSIGNMENT_PRICE}. Source: SEC EDGAR.")
    profiles = _screen(tuple(WHEEL_WATCHLIST))
    unreliable = [p.symbol for p in profiles if not p.data_reliable]
    if unreliable:
        st.warning(f"Data looked incomplete for: {', '.join(unreliable)} — scored anyway but treat with caution.")

    rows = []
    for p in profiles:
        rows.append({
            "symbol": p.symbol,
            "passes": p.passes,
            "quality_score": p.quality_score,
            "price": p.price,
            "fcf_yield": p.fcf_yield,
            "net_margin": p.net_margin,
            "net_margin_trend": p.net_margin_trend,
            "fcf_trend_score": p.fcf_trend_score,
            "roe": p.return_on_equity,
            "revenue_growth": p.revenue_growth,
            "is_elite": p.is_elite,
            "data_reliable": p.data_reliable,
            "fail_reasons": "; ".join(p.fail_reasons),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    if st.button("Save this snapshot to history"):
        wheel_db.save_screen_snapshot(profiles)
        st.success("Saved.")


# ---------------------------------------------------------------------------
# Options Radar
# ---------------------------------------------------------------------------

with tabs[4]:
    st.subheader("Options Radar")
    passing_symbols = [p.symbol for p in _screen(tuple(WHEEL_WATCHLIST)) if p.passes]
    if not passing_symbols:
        st.info("No symbols currently pass the fundamentals screen.")
    elif client is None:
        st.info("Connect to IBKR to pull option chains.")
    else:
        symbol = st.selectbox("Symbol", passing_symbols)
        hist, _src = _price_history(symbol)
        if hist is None or hist.empty:
            st.error(f"No price history for {symbol}.")
        else:
            underlying_price = float(hist["Close"].iloc[-1])
            rv = realized_volatility(hist)
            st.caption(f"Underlying price: ${underlying_price:.2f}  |  Realized vol ({wc.RICHNESS_LOOKBACK_DAYS}d): {rv:.1%}" if rv else "Realized vol unavailable.")

            candidates = fetch_put_candidates(client, symbol, underlying_price, rv)
            if not candidates:
                st.warning("No candidates found within the DTE / delta filters.")
            else:
                cdf = pd.DataFrame([c.__dict__ for c in candidates])
                st.dataframe(cdf, use_container_width=True)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=[c.delta for c in candidates],
                    y=[c.market_iv or c.realized_vol or 0 for c in candidates],
                    mode="markers",
                    marker=dict(
                        size=10,
                        color=[c.richness_ratio or 1.0 for c in candidates],
                        colorscale="RdYlGn_r",
                        showscale=True,
                        colorbar=dict(title="Richness"),
                    ),
                    text=[f"{c.symbol} {c.strike} {c.expiry}" for c in candidates],
                ))
                fig.update_layout(xaxis_title="Delta", yaxis_title="IV (or realized-vol proxy)", height=400)
                st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.caption("Full IV smile/skew and 25-delta risk-reversal diagnostics: coming in Phase 2 (wheel_skew.py). "
               "Merton jump-diffusion fair-value IV: coming in Phase 3 (wheel_jump_diffusion.py).")


# ---------------------------------------------------------------------------
# Ranked Recommendations
# ---------------------------------------------------------------------------

with tabs[5]:
    st.subheader("Ranked Recommendations")
    if client is None:
        st.info("Connect to IBKR to build recommendations (needs live option chains).")
    else:
        profiles = _screen(tuple(WHEEL_WATCHLIST))
        profiles_by_symbol = {p.symbol: p for p in profiles if p.passes}

        positions = client.get_positions()
        short_puts = [p for p in positions if p.sec_type == "OPT" and p.right == "P" and p.quantity < 0]
        capital_committed = sum((p.strike or 0) * 100 * abs(p.quantity) for p in short_puts)
        spare_capital_remaining = max(capital_targets.spare_capital - capital_committed, 0.0)
        existing_exposure_by_symbol: dict = {}
        for p in short_puts:
            existing_exposure_by_symbol[p.symbol] = existing_exposure_by_symbol.get(p.symbol, 0.0) + (p.strike or 0) * 100 * abs(p.quantity)

        st.caption(
            f"Spare capital remaining: ${spare_capital_remaining:,.0f} of "
            f"${capital_targets.spare_capital:,.0f} ({capital_targets.source})"
        )

        # Bound the batch size: each symbol costs at least one ~11s IBKR
        # snapshot request per expiry, and while fetch_put_candidates_batch
        # runs them concurrently (not serially, see wheel_options.py), an
        # unbounded "all 20-30 passing symbols" selection can still hit
        # IBKR's own pacing limits and take a while. Default to the
        # top-ranked names by quality score; let the user widen it.
        ranked_symbols = sorted(
            profiles_by_symbol, key=lambda s: profiles_by_symbol[s].quality_score or 0, reverse=True
        )
        default_n = min(8, len(ranked_symbols))
        selected_symbols = st.multiselect(
            "Symbols to pull option chains for (fewer = faster; each one is a live IBKR request)",
            options=ranked_symbols,
            default=ranked_symbols[:default_n],
        )

        if st.button("Build recommendations (pulls live option chains)"):
            if not selected_symbols:
                st.warning("Select at least one symbol above.")
                st.stop()

            requests = []
            with st.spinner(f"Fetching prices for {len(selected_symbols)} symbol(s)..."):
                for symbol in selected_symbols:
                    hist, _src = _price_history(symbol)
                    if hist is None or hist.empty:
                        continue
                    underlying_price = float(hist["Close"].iloc[-1])
                    rv = realized_volatility(hist)
                    requests.append((symbol, underlying_price, rv))

            with st.spinner(f"Pulling option chains for {len(requests)} symbol(s) concurrently..."):
                from wheel_options import fetch_put_candidates_batch
                candidates_by_symbol = fetch_put_candidates_batch(client, requests)

            recs = rank_recommendations(
                profiles_by_symbol, candidates_by_symbol, spare_capital_remaining, existing_exposure_by_symbol,
                total_book=capital_targets.total_book,
            )
            if not recs:
                st.warning("No recommendations cleared the filters.")
            else:
                for r in recs:
                    with st.container(border=True):
                        c1, c2, c3 = st.columns([2, 2, 1])
                        c1.markdown(f"**{r.symbol}** ${r.strike} put, exp {r.expiry} ({r.dte}d)")
                        c1.caption(f"Delta {r.delta:.2f} | Premium ${r.premium:.2f} | Ann. yield {r.annualized_yield:.1%}")
                        c2.markdown(
                            f"Quality **{r.quality_score:.0f}** | Richness ratio "
                            f"**{r.richness_ratio:.2f}**" if r.richness_ratio else f"Quality **{r.quality_score:.0f}**"
                        )
                        c2.caption(f"Capital efficiency {r.capital_efficiency_score:.0f} | Composite rank **{r.composite_rank:.0f}**")
                        c3.metric("Capital req.", f"${r.capital_required:,.0f}")
                        if not r.capital_ok:
                            st.warning("; ".join(r.capital_notes))
                        st.button(
                            "Execute (Phase 5 — not yet available)",
                            key=f"exec_{r.symbol}_{r.strike}_{r.expiry}",
                            disabled=True,
                        )


# ---------------------------------------------------------------------------
# Trade Log
# ---------------------------------------------------------------------------

with tabs[6]:
    st.subheader("Trade Log")
    log = wheel_db.get_trade_log()
    if log:
        st.dataframe(pd.DataFrame(log), use_container_width=True)
        st.caption("`source`: 'ibkr' = archived from a live fill this dashboard saw; 'manual' = logged via Monthly P&L's form.")
    else:
        st.caption(
            "Nothing archived yet. Fills seen live under Monthly P&L are "
            "archived here automatically going forward; anything IBKR's API "
            "didn't show (e.g. an expiration) needs the manual-entry form on "
            "the Monthly P&L tab."
        )
