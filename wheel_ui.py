"""
wheel_ui.py — shared pieces for the multipage wheel dashboard (wheel_dashboard.py
+ app_pages/): IBKR session handling, cached data loaders, the aurora-glass
stylesheet, the top account bar, and read-only alert rules.

Nothing in here builds or sends an order.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional

import streamlit as st

import wheel_config as wc
import wheel_events
from wheel_capital import resolve_capital_targets
from wheel_ibkr import WheelIBKRClient
from wheel_watchlist import WHEEL_WATCHLIST, load_exclusions
from wheel_universe import scan_universe

# ---------------------------------------------------------------------------
# Palette (also used by plotly figures, which don't read config.toml)
# ---------------------------------------------------------------------------
TEAL, VIOLET, MAGENTA = "#2DD4BF", "#A78BFA", "#F472B6"
GREEN, RED, AMBER = "#34D399", "#FB7185", "#FBBF24"


def plotly_layout(height: int = 420, **extra) -> dict:
    """Transparent, low-contrast-grid plotly layout that sits on the glass panels."""
    dark = st.context.theme.type != "light"
    grid = "rgba(255,255,255,0.06)" if dark else "rgba(20,26,51,0.08)"
    text = "rgba(230,234,245,0.9)" if dark else "#141A33"
    layout = dict(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=text, family="Outfit, sans-serif"),
        margin=dict(l=8, r=8, t=24, b=8),
        xaxis=dict(gridcolor=grid, zeroline=False),
        yaxis=dict(gridcolor=grid, zeroline=False),
        hoverlabel=dict(bgcolor="rgba(17,24,58,0.92)" if dark else "rgba(255,255,255,0.95)",
                        bordercolor=VIOLET, font=dict(family="JetBrains Mono, monospace")),
        transition=dict(duration=250, easing="cubic-in-out"),
    )
    layout.update(extra)
    return layout


# ---------------------------------------------------------------------------
# Aurora glass stylesheet (explicitly requested). Glass panels are targeted
# via container keys starting with "glass" plus bordered st.metric cards, so
# nothing depends on Streamlit's internal test-ids for layout containers.
# ---------------------------------------------------------------------------
_CSS_DARK = """
.stApp {
  background:
    radial-gradient(1200px 600px at 10% -10%, rgba(45,212,191,0.16), transparent 60%),
    radial-gradient(900px 500px at 90% 0%, rgba(167,139,250,0.18), transparent 60%),
    radial-gradient(800px 600px at 50% 110%, rgba(244,114,182,0.12), transparent 60%),
    #060A1A;
  background-attachment: fixed;
}
[class*="st-key-glass"], [data-testid="stMetric"] {
  background: rgba(255,255,255,0.035) !important;
  backdrop-filter: blur(14px) saturate(140%);
  -webkit-backdrop-filter: blur(14px) saturate(140%);
  border: 1px solid rgba(167,139,250,0.22) !important;
  border-radius: 16px !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.06), 0 10px 30px rgba(0,0,0,0.35);
  transition: border-color 220ms ease, box-shadow 220ms ease, transform 220ms ease;
}
[class*="st-key-glass"]:hover, [data-testid="stMetric"]:hover {
  border-color: rgba(45,212,191,0.55) !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.08), 0 0 0 1px rgba(45,212,191,0.18),
              0 0 24px rgba(45,212,191,0.18), 0 10px 30px rgba(0,0,0,0.35);
}
[data-testid="stSidebar"] {
  background: rgba(8,12,32,0.72) !important;
  backdrop-filter: blur(18px);
  border-right: 1px solid rgba(167,139,250,0.18);
}
"""

_CSS_LIGHT = """
.stApp {
  background:
    radial-gradient(1200px 600px at 10% -10%, rgba(45,212,191,0.18), transparent 60%),
    radial-gradient(900px 500px at 90% 0%, rgba(167,139,250,0.20), transparent 60%),
    radial-gradient(800px 600px at 50% 110%, rgba(244,114,182,0.14), transparent 60%),
    #F4F6FB;
  background-attachment: fixed;
}
[class*="st-key-glass"], [data-testid="stMetric"] {
  background: rgba(255,255,255,0.55) !important;
  backdrop-filter: blur(14px) saturate(140%);
  -webkit-backdrop-filter: blur(14px) saturate(140%);
  border: 1px solid rgba(109,40,217,0.16) !important;
  border-radius: 16px !important;
  box-shadow: 0 8px 24px rgba(20,26,51,0.08);
  transition: border-color 220ms ease, box-shadow 220ms ease;
}
[class*="st-key-glass"]:hover, [data-testid="stMetric"]:hover {
  border-color: rgba(13,148,136,0.45) !important;
  box-shadow: 0 0 18px rgba(13,148,136,0.18), 0 8px 24px rgba(20,26,51,0.08);
}
[data-testid="stSidebar"] { background: rgba(255,255,255,0.6) !important; backdrop-filter: blur(18px); }
"""

_CSS_COMMON = """
[class*="st-key-glass"] { padding: 1rem 1.1rem; }
[data-testid="stMetric"] { padding: 0.85rem 1rem; }
[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; letter-spacing: -0.01em; }
.stButton button, .stFormSubmitButton button {
  transition: box-shadow 200ms ease, border-color 200ms ease, transform 120ms ease;
}
.stButton button:hover, .stFormSubmitButton button:hover {
  box-shadow: 0 0 16px rgba(167,139,250,0.45);
  border-color: rgba(167,139,250,0.8);
}
.stButton button:active, .stFormSubmitButton button:active { transform: scale(0.98); }
@media (prefers-reduced-motion: reduce) {
  * { transition: none !important; }
}
"""


def inject_css() -> None:
    css = (_CSS_LIGHT if st.context.theme.type == "light" else _CSS_DARK) + _CSS_COMMON
    st.html(f"<style>{css}</style>")


# ---------------------------------------------------------------------------
# IBKR connection: ONE read-only socket per server process, shared by every
# browser tab. A per-session socket outlived closed/reloaded tabs and kept
# the client id busy, so the next tab's Connect timed out. WheelIBKRClient
# routes all calls through its own loop thread, so sharing it is safe.
# ---------------------------------------------------------------------------

@st.cache_resource
def _shared() -> dict:
    return {"client": None}


def init_state() -> None:
    st.session_state.setdefault("ibkr_error", None)
    st.session_state.setdefault("screen_on", False)


def connect() -> None:
    holder = _shared()
    if holder["client"] is not None and holder["client"].is_connected():
        st.session_state.ibkr_error = None
        return
    # WHEEL_IBKR_CLIENT_ID lets a second dashboard/test run alongside without
    # colliding on TWS's one-connection-per-client-id rule.
    client_id = int(os.environ.get("WHEEL_IBKR_CLIENT_ID", wc.IBKR_CLIENT_ID))
    try:
        new_client = WheelIBKRClient(use_paper=False, client_id=client_id)
        new_client.connect()
        new_client.enable_fill_archiving()   # trade log updates the moment a fill happens
        new_client.archive_recent_fills()    # ...and catches anything since the last session
        holder["client"] = new_client
        st.session_state.ibkr_error = None
    except Exception as exc:
        holder["client"] = None
        if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
            msg = (f"TWS did not answer. Client ID {client_id} may already be in use by another browser "
                   "tab — disconnect there or restart the dashboard. Also check TWS is logged in.")
        elif isinstance(exc, ConnectionRefusedError):
            msg = f"Nothing listening on port {wc.IBKR_LIVE_PORT} — start TWS and enable the API."
        else:
            msg = f"{type(exc).__name__}: {exc}"
        st.session_state.ibkr_error = msg
        logging.warning("IBKR connection failed: %r", exc)


def disconnect() -> None:
    """Closes the shared socket — affects every open tab on this server."""
    holder = _shared()
    if holder["client"] is not None:
        try:
            holder["client"].disconnect()
        except Exception:
            pass
    holder["client"] = None
    st.session_state.ibkr_error = None


def get_client() -> Optional[WheelIBKRClient]:
    client = _shared()["client"]
    return client if client is not None and client.is_connected() else None


def account_id(client) -> str:
    ids = client.ib.managedAccounts() if client else []
    return ids[0] if ids else "?"


def capital_targets():
    client = get_client()
    return resolve_capital_targets(client.get_account_summary() if client else None)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner="Screening fundamentals (SEC EDGAR + bulk price)...")
def _screen(symbols: tuple[str, ...]):
    # Concurrent EDGAR fetch + bulk price prefetch; cached an hour since
    # fundamentals don't move intraday.
    profiles = scan_universe(list(symbols))
    profiles.sort(key=lambda p: (p.passes, p.quality_score or 0), reverse=True)
    return profiles


def profiles_or_none():
    """The full-watchlist screen takes minutes on a cold cache: only run it
    once the user asks for it this session."""
    if st.session_state.get("screen_on"):
        # Screen the full universe once (cached); manual exclusions are just
        # filtered out, so removing a ticker never triggers a re-screen.
        excluded = load_exclusions()
        return [p for p in _screen(tuple(WHEEL_WATCHLIST)) if p.symbol.upper() not in excluded]
    return None


def need_screen(key: str) -> None:
    st.info("This page needs the fundamentals screen (SEC EDGAR, takes a few minutes).", icon=":material/hourglass_top:")
    if st.button("Run fundamentals screen now", key=key, icon=":material/play_arrow:"):
        st.session_state.screen_on = True
        st.rerun()


@st.cache_data(ttl=86400, show_spinner=False)
def company_names() -> dict:
    """Ticker -> company name from SEC's company_tickers.json (cached a day)."""
    import requests
    from edgar import EDGAR_HEADERS, EDGAR_TICKERS_URL
    try:
        data = requests.get(EDGAR_TICKERS_URL, headers=EDGAR_HEADERS, timeout=15).json()
        return {v["ticker"].upper(): v["title"] for v in data.values()}
    except Exception:
        return {}


@st.cache_data(ttl=900, show_spinner=False)
def price_history(symbol: str, period_years: int = 1):
    """Daily OHLCV (IBKR → yfinance → Stooq), cached 15 minutes."""
    from edgar import get_price_history
    return get_price_history(symbol, period_years=period_years)


@st.cache_data(ttl=3600, show_spinner=False)
def iv_history(symbol: str, duration: str = "3 Y", _client=None):
    """IBKR implied-vol history, cached an hour. Raises (so nothing is
    cached) when IBKR returns no data."""
    import wheel_vol
    s = wheel_vol.iv_history(_client, symbol, duration)
    if s.empty:
        raise ValueError(f"IBKR returned no implied-vol history for {symbol}")
    return s


# ---------------------------------------------------------------------------
# Top account bar
# ---------------------------------------------------------------------------

def top_bar(client) -> None:
    with st.container(horizontal=True, vertical_alignment="center", key="glass_topbar"):
        if client is None:
            st.badge("Not connected", icon=":material/link_off:", color="gray")
            st.caption("Click **Connect** in the left rail (TWS must be running on port "
                       f"{wc.IBKR_LIVE_PORT}).")
            return
        s = client.get_account_summary()
        acct = account_id(client)
        if acct.startswith("DU"):
            st.badge(f"PAPER {acct}", icon=":material/warning:", color="orange")
        else:
            st.badge(f"LIVE {acct}", icon=":material/verified_user:", color="green")
        st.badge("Read-only", icon=":material/lock:", color="violet")
        margin_pct = s.margin_used / s.net_liquidation if s.net_liquidation else 0
        st.markdown(f"Net liq **\\${s.net_liquidation:,.0f}**")
        st.markdown(f"Cash **\\${s.total_cash:,.0f}**")
        st.markdown(f"Buying power **\\${s.buying_power:,.0f}**")
        color = "red" if margin_pct > 0.5 else "orange" if margin_pct > 0.3 else "green"
        st.markdown(f"Margin used :{color}[**{margin_pct:.0%}**]")


# ---------------------------------------------------------------------------
# Read-only alert rules
# ---------------------------------------------------------------------------

@dataclass
class Alert:
    level: str      # "error" | "warning" | "info"
    title: str
    detail: str


def compute_alerts(client) -> list[Alert]:
    s = client.get_account_summary()
    nlv = s.net_liquidation or 1.0
    positions = client.get_positions()
    stocks = {p.symbol: p for p in positions if p.sec_type == "STK"}
    short_puts = [p for p in positions if p.sec_type == "OPT" and p.right == "P" and p.quantity < 0]
    alerts: list[Alert] = []

    put_notional = sum((p.strike or 0) * 100 * abs(p.quantity) for p in short_puts)
    if short_puts and put_notional > s.total_cash:
        alerts.append(Alert("warning", "Short puts are margin-secured",
                            f"\\${put_notional:,.0f} strike exposure ({put_notional / nlv:.0%} of account) "
                            f"vs \\${s.total_cash:,.0f} cash."))
    margin_pct = s.margin_used / nlv
    if margin_pct > 0.5:
        alerts.append(Alert("error", "High margin usage", f"Maintenance margin is {margin_pct:.0%} of net liq."))
    elif margin_pct > 0.3:
        alerts.append(Alert("warning", "Margin usage rising", f"Maintenance margin is {margin_pct:.0%} of net liq."))

    # Concentration: shares + short-put notional per underlying.
    exposure: dict[str, float] = {}
    for p in stocks.values():
        exposure[p.symbol] = exposure.get(p.symbol, 0) + (p.market_value or 0)
    for p in short_puts:
        exposure[p.symbol] = exposure.get(p.symbol, 0) + (p.strike or 0) * 100 * abs(p.quantity)
    over = [(sym, val / nlv) for sym, val in sorted(exposure.items(), key=lambda kv: -kv[1])
            if val / nlv > wc.MAX_NAME_CONCENTRATION]
    if over:
        top_sym, top_pct = over[0]
        alerts.append(Alert("error" if top_pct > 0.5 else "warning",
                            f"{len(over)} name(s) over the {wc.MAX_NAME_CONCENTRATION:.0%} cap",
                            "Shares + short-put notional: " + ", ".join(f"{s} {p:.0%}" for s, p in over) + "."))

    today = date.today()
    for p in short_puts:
        dte = (datetime.strptime(p.expiry, "%Y%m%d").date() - today).days
        spot = stocks[p.symbol].market_price if p.symbol in stocks else None
        dist = (spot - p.strike) / spot if spot else None
        label = f"{p.symbol} {p.strike:g}P {p.expiry}"
        if dist is not None and dist < 0.02:
            alerts.append(Alert("error", f"{label} near/in the money",
                                f"Spot {spot:,.2f} is {dist:.1%} above strike — roll or assignment decision."))
        elif dist is not None and dist < 0.05:
            alerts.append(Alert("warning", f"{label} cushion under 5%", f"Spot {spot:,.2f}, {dist:.1%} above strike."))
        if 0 <= dte <= 7:
            alerts.append(Alert("info", f"{label} expires in {dte}d",
                                f"Current value \\${abs(p.market_value or 0):,.0f} — consider closing or letting expire."))
        earn = wheel_events.earnings_before(p.symbol, datetime.strptime(p.expiry, "%Y%m%d").date())
        if earn:
            alerts.append(Alert("warning", f"{p.symbol} earnings {earn:%b %d} before {label} expires",
                                "Earnings can gap the stock through your strike; IV is usually elevated into it."))

    # unread alerts from the quality-stock screener (in-app notifications)
    try:
        from stockscreen import store as _screen_store
        from stockscreen.alerts import KINDS as _KINDS
        from stockscreen.universe import excluded as _excluded
        _ex = _excluded()
        for a in [x for x in _screen_store.list_alerts(unread_only=True, limit=50) if x["symbol"].upper() not in _ex][:5]:
            alerts.append(Alert("info", f"Screener: {a['symbol']} — {_KINDS.get(a['kind'], a['kind'])}",
                                f"{a['trigger_text']}. Fundamentals {a['fund_status']}; technicals {a['tech_status']}. "
                                f"Open Fundamentals to review."))
    except Exception:
        pass   # the alert feed must never break the Overview page

    soon = wheel_events.events_between([], today, today + timedelta(days=3))
    if not soon.empty:
        alerts.append(Alert("info", "Market events in the next 3 days",
                            "; ".join(f"{r.event} {r.date:%a %b %d}" for r in soon.itertuples())
                            + ". Index-wide IV tends to rise into these releases."))
    return alerts


def render_alerts(alerts: list[Alert]) -> None:
    if not alerts:
        st.success("No alerts. Positions are inside your rules.", icon=":material/check_circle:")
        return
    icons = {"error": ":material/error:", "warning": ":material/warning:", "info": ":material/info:"}
    for a in alerts:
        getattr(st, a.level)(f"**{a.title}**  \n{a.detail}", icon=icons[a.level])
