"""Order ticket — PREVIEW ONLY. Builds an order description and its risk
numbers; it never creates an ib_insync Order or calls placeOrder. The final
step is a disabled button: you enter the order in TWS yourself."""

from datetime import date

import streamlit as st

import wheel_analyst as wa
import wheel_config as wc
import wheel_ui as ui

client = ui.get_client()
summary = client.get_account_summary() if client else None
positions = client.get_positions() if client else []
nlv = summary.net_liquidation if summary else None

st.caption("Preview only. Nothing here is sent to IBKR — copy the ticket into TWS yourself.")

with st.container(key="glass_ticket"):
    instrument = st.segmented_control(
        "Instrument", ["Stock", "Put option", "Call option"], default="Put option", key="ot_instrument")
    with st.form("order_ticket", border=False):
        c1, c2, c3 = st.columns(3)
        symbol = c1.text_input("Symbol", value="WFC").upper().strip()
        action = c2.segmented_control("Action", ["Buy", "Sell"],
                                      default="Sell" if instrument != "Stock" else "Buy")
        qty = c3.number_input("Contracts" if instrument != "Stock" else "Shares", min_value=1, value=1, step=1)

        strike = expiry = None
        if instrument != "Stock":
            o1, o2 = st.columns(2)
            strike = o1.number_input("Strike", min_value=0.0, value=75.0, step=0.5)
            expiry = o2.date_input("Expiry", value=wa.third_friday_next_month(), min_value=date.today())

        order_type = st.segmented_control("Order type", ["Market", "Limit", "Stop", "Bracket"], default="Limit")
        p1, p2, p3, p4 = st.columns(4)
        price_label = "Limit price (premium/share)" if instrument != "Stock" else "Limit / entry price"
        limit = p1.number_input(price_label, min_value=0.0, value=0.60 if instrument != "Stock" else 80.0,
                                step=0.01, format="%.2f")
        stop = p2.number_input("Stop price", min_value=0.0, value=0.0, step=0.01, format="%.2f",
                               help="Stop and Bracket orders.")
        take = p3.number_input("Take-profit price", min_value=0.0, value=0.0, step=0.01, format="%.2f",
                               help="Bracket orders.")
        tif = p4.segmented_control("Time in force", ["DAY", "GTC"], default="DAY")
        submitted = st.form_submit_button("Preview order", icon=":material/preview:", type="primary")

if submitted:
    # Segmented controls return None when deselected; fall back to defaults.
    st.session_state.ot_preview = dict(instrument=instrument or "Put option", symbol=symbol,
                                       action=action or "Sell", qty=qty, strike=strike, expiry=expiry,
                                       order_type=order_type or "Limit", limit=limit, stop=stop, take=take,
                                       tif=tif or "DAY")

pv = st.session_state.get("ot_preview")
if not pv:
    st.stop()

# ---------------------------------------------------------------------------
# Step 1: preview with risk metrics
# ---------------------------------------------------------------------------
problems, warnings = [], []
is_opt = pv["instrument"] != "Stock"
mult = 100 if is_opt else 1
px = pv["limit"] if pv["order_type"] in ("Limit", "Bracket") else (pv["stop"] or pv["limit"])
if not pv["symbol"]:
    problems.append("Symbol is required.")
if pv["order_type"] in ("Limit", "Bracket") and pv["limit"] <= 0:
    problems.append("Limit price must be above zero.")
if pv["order_type"] in ("Stop", "Bracket") and pv["stop"] <= 0:
    problems.append("Stop price is required for Stop and Bracket orders.")
if pv["order_type"] == "Bracket" and pv["take"] <= 0:
    problems.append("Take-profit price is required for a Bracket order.")
if is_opt and pv["order_type"] == "Market":
    warnings.append("Market orders on options can fill far from mid — a limit is safer.")

notional = px * pv["qty"] * mult
metrics = {"Order value": f"${notional:,.2f}"}
cash_needed = 0.0
max_loss = None

if pv["instrument"] == "Put option" and pv["action"] == "Sell":
    cash_needed = pv["strike"] * 100 * pv["qty"]
    max_loss = (pv["strike"] - px) * 100 * pv["qty"]
    metrics.update({
        "Premium collected": f"${notional:,.2f}",
        "Cash to secure": f"${cash_needed:,.0f}",
        "Breakeven": f"${pv['strike'] - px:,.2f}",
        "Max loss (stock → 0)": f"${max_loss:,.0f}",
    })
    dte = (pv["expiry"] - date.today()).days
    if dte > 0 and pv["strike"]:
        metrics["Annualized ROI"] = f"{px / pv['strike'] * 365 / dte:.1%}"
    if summary and cash_needed > summary.total_cash:
        warnings.append(f"Cash to secure \\${cash_needed:,.0f} exceeds cash \\${summary.total_cash:,.0f} — "
                        "this would be margin-secured, not cash-secured.")
elif pv["instrument"] == "Call option" and pv["action"] == "Sell":
    held = sum(p.quantity for p in positions if p.sec_type == "STK" and p.symbol == pv["symbol"])
    metrics["Shares needed to cover"] = f"{100 * pv['qty']:,} (you hold {held:,.0f})"
    if held < 100 * pv["qty"]:
        warnings.append("Not enough shares to cover — this would be a naked call.")
elif pv["action"] == "Buy":
    cash_needed = notional
    max_loss = notional if is_opt else None
    if pv["order_type"] in ("Stop", "Bracket") and not is_opt and pv["stop"] > 0:
        max_loss = (px - pv["stop"]) * pv["qty"]
    if max_loss is not None:
        metrics["Max loss"] = f"${max_loss:,.0f}"
if pv["order_type"] == "Bracket" and pv["stop"] > 0 and pv["take"] > 0 and px > 0:
    risk, reward = abs(px - pv["stop"]), abs(pv["take"] - px)
    metrics["Reward : risk"] = f"{reward / risk:.2f} : 1" if risk else "n/a"

if nlv:
    exposure = cash_needed or notional
    metrics["% of net liq"] = f"{exposure / nlv:.1%}"
    if exposure / nlv > wc.MAX_NAME_CONCENTRATION:
        warnings.append(f"This order alone is {exposure / nlv:.0%} of net liq "
                        f"(cap {wc.MAX_NAME_CONCENTRATION:.0%} per name).")
    if summary and cash_needed > summary.buying_power:
        problems.append("Exceeds buying power.")
else:
    warnings.append("Not connected — account-based checks (cash, buying power, concentration) skipped.")

contract = pv["symbol"] + (f" {pv['expiry']:%Y-%m-%d} {pv['strike']:g} {'P' if 'Put' in pv['instrument'] else 'C'}"
                           if is_opt else "")
lines = [f"{pv['action'].upper()} {pv['qty']} {contract}",
         f"{pv['order_type'].upper()}" + (f" @ {pv['limit']:.2f}" if pv["order_type"] in ("Limit", "Bracket") else "")
         + (f"  STOP {pv['stop']:.2f}" if pv["order_type"] in ("Stop", "Bracket") else "")
         + (f"  TAKE-PROFIT {pv['take']:.2f}" if pv["order_type"] == "Bracket" else "")
         + f"  {pv['tif']}"]

with st.container(key="glass_preview"):
    st.subheader("Step 1 · Preview", anchor=False)
    st.code("\n".join(lines), language=None)
    items = list(metrics.items())
    for i in range(0, len(items), 4):   # fixed 4-wide grid so a lone card doesn't stretch
        for col, (k, v) in zip(st.columns(4), items[i:i + 4]):
            col.metric(k, v, border=True)
    for p in problems:
        st.error(p, icon=":material/block:")
    for w in warnings:
        st.warning(w, icon=":material/warning:")
    if is_opt:
        st.caption("Premium is your limit price, not a live quote. Option data is delayed without OPRA — "
                   "verify before trading.")

with st.container(key="glass_confirm"):
    st.subheader("Step 2 · Confirm", anchor=False)
    ok = st.checkbox("I have reviewed this preview", disabled=bool(problems), key="ot_reviewed")
    st.button("Send to IBKR", icon=":material/send:", disabled=True,
              help="Disabled by design: this dashboard is read-only. Enter the order in TWS.")
    if ok:
        st.info("Enter the order above in TWS yourself. This dashboard's IBKR connection is read-only "
                "and cannot place orders.", icon=":material/lock:")
