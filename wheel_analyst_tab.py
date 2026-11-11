"""
wheel_analyst_tab.py — Streamlit rendering for the "Wheel Analyst" tab.

Read-only: shows the account, a strict cash-secured-put screen, a budget fit
and a roll view. Nothing here places, modifies or cancels an order.
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import streamlit as st

import wheel_analyst as wa
import wheel_config as wc


def render(client) -> None:
    st.caption("Read-only analysis. Nothing here places, modifies or cancels an order; you decide and trade in TWS.")
    if client is None:
        st.info("Connect to IBKR in the left rail to run the analyst.", icon=":material/link:")
        return

    acct_ids = client.ib.managedAccounts()
    acct_id = acct_ids[0] if acct_ids else "?"
    if acct_id.startswith("DU"):
        st.error(f"Account {acct_id} is a PAPER account. Stopping — this tab is for the live account.")
        return

    summary = client.get_account_summary()
    nlv = summary.net_liquidation or 1.0
    positions = client.get_positions()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Net Liquidation", f"${summary.net_liquidation:,.0f}")
    c2.metric("Cash", f"${summary.total_cash:,.0f}")
    c3.metric("Buying Power", f"${summary.buying_power:,.0f}")
    c4.metric("Maint. Margin", f"${summary.margin_used:,.0f}")

    stocks = [p for p in positions if p.sec_type == "STK"]
    if stocks:
        st.markdown("**Stock positions**")
        st.dataframe(pd.DataFrame([{
            "symbol": p.symbol, "qty": p.quantity, "avg_cost": p.avg_cost, "last": p.market_price,
            "unrealized_pnl": p.unrealized_pnl, "pct_of_account": (p.market_value or 0) / nlv,
        } for p in stocks]).style.format({
            "qty": "{:,.0f}", "avg_cost": "{:,.2f}", "last": "{:,.2f}", "unrealized_pnl": "{:,.0f}",
            "pct_of_account": "{:.1%}"}),
            )

    short_puts = [p for p in positions if p.sec_type == "OPT" and p.right == "P" and p.quantity < 0]
    opt_rows = [p for p in positions if p.sec_type == "OPT"]
    if opt_rows:
        st.markdown("**Option positions** (premium and value per contract; delta in the roll view below)")
        st.dataframe(pd.DataFrame([{
            "symbol": p.symbol, "right": p.right, "strike": p.strike, "expiry": p.expiry, "qty": p.quantity,
            "premium_received": -p.quantity * p.avg_cost if p.quantity < 0 else None,
            "current_value": p.market_value,
            "dte": (datetime.strptime(p.expiry, "%Y%m%d").date() - date.today()).days,
        } for p in opt_rows]))

    put_capital = sum(p.strike * 100 * abs(p.quantity) for p in short_puts)
    if short_puts:
        st.warning(
            f"Short puts tie up \\${put_capital:,.0f} of strike exposure ({put_capital / nlv:.0%} of account) "
            f"against \\${summary.total_cash:,.0f} cash — "
            + ("these are margin-secured, not cash-secured." if put_capital > summary.total_cash else "covered by cash.")
        )
    st.caption("Assignments in the last 30 days: IBKR's API only returns recent executions — "
               "check Trade Confirmations in Client Portal to be sure.")

    st.divider()
    exp_date = wa.third_friday_next_month()
    expiry = exp_date.strftime("%Y%m%d")
    dte = (exp_date - date.today()).days
    budget = st.number_input("Capital budget for new puts ($)", min_value=0, value=30_000, step=1_000)
    st.caption(f"Expiry: {exp_date} ({dte} DTE, monthly). Filters: |delta| <= {wa.MAX_ABS_DELTA}, "
               f">= {wa.MIN_CUSHION:.0%} below spot, spread <= {wc.MAX_SPREAD_PCT:.0%} of mid, OI >= {wa.MIN_OPEN_INTEREST}.")

    if st.button("Run analysis (pulls option chains, ~1–2 min)"):
        st.session_state.analyst = _run(client, expiry, exp_date, dte, budget, short_puts)

    res = st.session_state.get("analyst")
    if res:
        _show(res, stocks, short_puts, nlv)


def _run(client, expiry, exp_date, dte, budget, short_puts) -> dict:
    scan, skipped, spots, any_delayed = [], {}, {}, False
    prog = st.progress(0.0)
    n = len(wa.WATCHLIST)
    error = None
    for i, sym in enumerate(wa.WATCHLIST):
        if not client.is_connected():
            error = "TWS/Gateway disconnected during the scan. Restart it, click Connect, and run again."
            break
        try:
            data = wa.fetch_symbol(client, sym, expiry)
        except Exception as exc:
            skipped[sym] = {f"fetch failed: {exc}": 1}
            prog.progress((i + 1) / n)
            continue
        spot = data["spot"]
        spots[sym] = spot
        if not spot:
            skipped[sym] = {"no spot price": 1}
            prog.progress((i + 1) / n)
            continue
        earn = wa.earnings_status(sym, exp_date)
        reasons: dict = {}
        for q in data["quotes"]:
            delayed = q["mdt"] != 1
            any_delayed |= delayed
            cand, why = wa.screen_put(sym, spot, q["strike"], expiry, dte, q, delayed, earn)
            if cand:
                scan.append(cand)
            else:
                reasons[why] = reasons.get(why, 0) + 1
        skipped[sym] = reasons
        prog.progress((i + 1) / n)

    roll = []
    for p in short_puts:
        spot = spots.get(p.symbol)
        if not spot:
            continue
        try:
            q = wa.fetch_held(client, p.symbol, p.expiry, p.strike)
        except Exception as exc:
            error = error or f"Could not quote held put {p.symbol} {p.strike:g}P ({exc}). Roll view is incomplete."
            continue
        d = abs(q["delta"]) if q["delta"] is not None else None
        status = wa.put_status(spot, p.strike, d)
        nxt = None
        if status != "safe":
            opts_ = sorted([c for c in scan if c.symbol == p.symbol and not c.earnings_blocks],
                           key=lambda c: c.strike, reverse=True)
            nxt = opts_[0] if opts_ else None
        roll.append((p, spot, q, d, status, nxt))
    return dict(scan=scan, skipped=skipped, roll=roll, budget=budget, any_delayed=any_delayed, spots=spots,
                error=error)


def _show(res: dict, stocks, short_puts, nlv: float) -> None:
    if res.get("error"):
        st.error(res["error"] + " Results below are partial.")
    if res["any_delayed"]:
        st.warning("Some option quotes are not live (delayed, or the last quote from before the close). "
                   "Every option figure below: verify before trading.")
    scan = sorted(res["scan"], key=lambda c: c.ann_mid, reverse=True)
    st.markdown("**Qualifying puts** — ranked by annualized ROI on mid")
    if not scan:
        st.info("No put qualifies right now. If the market is closed, options have no bid/ask — rerun during market hours.")
    else:
        st.dataframe(pd.DataFrame([{
            "symbol": c.symbol, "spot": c.spot, "strike": c.strike, "pct_below": c.pct_below_spot,
            "delta": c.delta, "dte": c.dte, "bid": c.bid, "ask": c.ask, "mid": c.mid,
            "prem/contract": c.premium_contract_mid, "capital": c.capital,
            "roi_mid": c.roi_mid, "roi_bid": c.roi_bid, "ann_mid": c.ann_mid, "ann_bid": c.ann_bid,
            "breakeven": c.breakeven, "be_pct_below": c.breakeven_pct_below,
            "earnings": c.earnings, "verify": "verify before trading" if c.delayed else "",
        } for c in scan]).style.format({
            "spot": "{:,.2f}", "strike": "{:,.1f}", "pct_below": "{:.1%}", "delta": "{:.3f}",
            "bid": "{:.2f}", "ask": "{:.2f}", "mid": "{:.2f}", "prem/contract": "{:,.0f}", "capital": "{:,.0f}",
            "roi_mid": "{:.2%}", "roi_bid": "{:.2%}", "ann_mid": "{:.1%}", "ann_bid": "{:.1%}",
            "breakeven": "{:,.2f}", "be_pct_below": "{:.1%}"}))
        if any(c.earnings == "unverified" for c in scan):
            st.caption("Earnings 'unverified' = calendar lookup failed; check before trading. "
                       "Names with a dated earnings event inside the window are excluded from the portfolio fit.")

        chosen = wa.select_portfolio(scan, res["budget"])
        st.markdown(f"**Portfolio fit** — budget \\${res['budget']:,.0f}, one put per underlying")
        if chosen:
            deployed = sum(c.capital for c in chosen)
            cc1, cc2, cc3, cc4 = st.columns(4)
            cc1.metric("Capital deployed", f"${deployed:,.0f}")
            cc2.metric("Total premium (mid)", f"${sum(c.premium_contract_mid for c in chosen):,.0f}")
            cc3.metric("Blended annualized ROI", f"{wa.blended_annualized(chosen):.1%}")
            cc4.metric("Budget remaining", f"${res['budget'] - deployed:,.0f}")
            held_shares = {p.symbol: p.market_value or 0 for p in stocks}
            held_puts: dict = {}
            for p in short_puts:
                held_puts[p.symbol] = held_puts.get(p.symbol, 0) + p.strike * 100 * abs(p.quantity)
            st.dataframe(pd.DataFrame([{
                "symbol": c.symbol, "strike": c.strike, "capital": c.capital, "ann_mid": c.ann_mid,
                "existing_shares_$": held_shares.get(c.symbol, 0), "existing_puts_$": held_puts.get(c.symbol, 0),
                "combined_pct_of_account": (c.capital + held_shares.get(c.symbol, 0) + held_puts.get(c.symbol, 0)) / nlv,
            } for c in chosen]).style.format({
                "capital": "{:,.0f}", "ann_mid": "{:.1%}", "existing_shares_$": "{:,.0f}",
                "existing_puts_$": "{:,.0f}", "combined_pct_of_account": "{:.0%}"}))
        else:
            st.caption("Nothing fits the budget.")

    if res["roll"]:
        st.markdown("**Roll and assignment view**")
        for p, spot, q, d, status, nxt in res["roll"]:
            dist = (spot - p.strike) / spot
            left = (datetime.strptime(p.expiry, "%Y%m%d").date() - date.today()).days
            delta_txt = "n/a (no model data)" if d is None else f"{d:.3f}"
            st.markdown(f"**{p.symbol} {p.strike:g}P {p.expiry}** x{abs(p.quantity):g} — status **{status.upper()}** | "
                        f"spot {spot:,.2f}, {dist:.1%} below | delta {delta_txt} | {left} days left")
            if status != "safe":
                if nxt and q["ask"]:
                    kind = "credit" if nxt.bid >= q["ask"] else "debit"
                    st.caption(f"Roll idea: sell {nxt.expiry} {nxt.strike:g}P at bid {nxt.bid:.2f} "
                               f"({nxt.pct_below_spot:.1%} below spot), buy back at ask {q['ask']:.2f} "
                               f"→ net {kind} \\${abs(nxt.bid - q['ask']) * 100:,.0f}/contract. verify before trading.")
                else:
                    st.caption("No qualifying roll target or no buy-back quote right now.")

    with st.expander("Contracts skipped, by reason"):
        for sym, reasons in res["skipped"].items():
            if reasons:
                st.write(f"{sym}: " + ", ".join(f"{k} ({v})" for k, v in reasons.items()))
