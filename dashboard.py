"""
dashboard.py — Live dynamic dashboard for the buy-the-sauce trading system.

Tabs
----
  1. 📡 Screener        Live dip detector + ML signal for every watchlist ticker
  2. 💰 Risk & Capital  Position sizing, capital allocation, risk heat-map ($80k budget)
  3. 📈 Back-test       Equity curve + trade log from results/backtest_trades.csv
  4. 🗂 Screen Log      Historical screener runs from results/screen_log.csv
  5. 📒 My Trades       Manual transaction journal — amounts, quantities, P&L

Usage
-----
  python3 dashboard.py              # opens on http://127.0.0.1:8050
  python3 dashboard.py --port 8080  # custom port
  python3 dashboard.py --no-browser # don't auto-open browser tab

The screener tab has a "🔄 Refresh" button that re-runs the full data
fetch on demand (same pipeline as run_screen.py).  A countdown timer
shows the time since the last refresh.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import os
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc

import config
from risk_manager import calculate_order
from run_screen import WATCHLIST

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACCOUNT_EQUITY  = 80_000.0   # user's stated budget
RESULTS_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SCREEN_CSV      = os.path.join(RESULTS_DIR, "screen_log.csv")
BACKTEST_CSV    = os.path.join(RESULTS_DIR, "backtest_trades.csv")
TRADES_CSV      = os.path.join(RESULTS_DIR, "my_trades.csv")   # manual journal

BRAND_BG    = "#0d1117"
CARD_BG     = "#161b22"
ACCENT      = "#58a6ff"
GREEN       = "#3fb950"
RED         = "#f85149"
YELLOW      = "#d29922"
TEXT        = "#e6edf3"
MUTED       = "#8b949e"
BORDER      = "#30363d"

_CELL_STYLE = {
    "backgroundColor": CARD_BG,
    "color": TEXT,
    "border": f"1px solid {BORDER}",
    "fontFamily": "monospace",
    "fontSize": "13px",
}
_HDR_STYLE = {
    "backgroundColor": "#21262d",
    "color": ACCENT,
    "fontWeight": "bold",
    "fontFamily": "monospace",
    "fontSize": "13px",
    "border": f"1px solid {BORDER}",
}

# ── My Trades CSV schema ────────────────────────────────────────────────────
_TRADES_COLS = [
    "date", "symbol", "action", "quantity", "entry_price",
    "exit_price", "amount_invested", "realized_pnl", "notes",
]

def _ensure_trades_csv():
    """Create the trades CSV with headers if it doesn't exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not os.path.exists(TRADES_CSV):
        with open(TRADES_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_TRADES_COLS)
            writer.writeheader()

def _load_trades() -> pd.DataFrame:
    _ensure_trades_csv()
    try:
        df = pd.read_csv(TRADES_CSV)
        for col in _TRADES_COLS:
            if col not in df.columns:
                df[col] = None
        # coerce numeric cols
        for col in ["quantity", "entry_price", "exit_price", "amount_invested", "realized_pnl"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame(columns=_TRADES_COLS)

def _append_trade(row: dict):
    """Append a single trade dict to the CSV."""
    _ensure_trades_csv()
    with open(TRADES_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_TRADES_COLS)
        writer.writerow({k: row.get(k, "") for k in _TRADES_COLS})

def _delete_trade(idx: int):
    """Delete trade at zero-based index from CSV."""
    df = _load_trades()
    df = df.drop(index=idx).reset_index(drop=True)
    df.to_csv(TRADES_CSV, index=False)


# ---------------------------------------------------------------------------
# IBKR connectivity check
# ---------------------------------------------------------------------------

def _check_ibkr_status() -> tuple[bool, str]:
    """Return (connected: bool, message: str) for IBKR TWS/Gateway."""
    try:
        from ib_insync import IB
        ib = IB()
        ib.connect(config.IBKR_HOST, config.IBKR_PORT,
                   clientId=config.IBKR_CLIENT_ID + 98, readonly=True, timeout=3)
        connected = ib.isConnected()
        ib.disconnect()
        if connected:
            return True, f"✅ IBKR Connected  ({config.IBKR_HOST}:{config.IBKR_PORT})"
        return False, f"❌ IBKR Not Connected  ({config.IBKR_HOST}:{config.IBKR_PORT})"
    except Exception as exc:
        return False, f"❌ IBKR Unavailable — {exc}  →  Stooq fallback active"


# ---------------------------------------------------------------------------
# Data fetching (runs in-process — same as run_screen.py)
# ---------------------------------------------------------------------------

_last_fetch: dict = {}   # cached results to avoid re-fetching on every callback


def fetch_screen_data() -> list[dict]:
    """Run the full screener pipeline and return a list of record dicts."""
    import edgar
    from screener import screen_fundamental
    from dip_detector import score_dip

    records = []
    for symbol in WATCHLIST:
        try:
            info    = edgar.get_fundamentals(symbol)
            profile = screen_fundamental(symbol, info=info)
            hist, price_source = edgar.get_price_history(symbol, period_years=2)
            signal = score_dip(symbol, hist) if hist is not None and not hist.empty else None
        except Exception:
            profile = type("P", (), {"passes": False, "fail_reasons": ["fetch error"],
                                      "free_cash_flow": None, "fcf_yield": None,
                                      "profit_margin": None, "debt_to_equity": None,
                                      "return_on_equity": None})()
            signal = None
            price_source = "error"
            info = {}

        records.append({
            "symbol":       symbol,
            "profile":      profile,
            "signal":       signal,
            "info":         info,
            "price_source": price_source,
        })
    return records


def _records_to_df(records: list[dict]) -> pd.DataFrame:
    """Flatten screener records into a display DataFrame."""
    rows = []
    for r in records:
        s   = r["signal"]
        p   = r["profile"]
        sym = r["symbol"]

        price     = round(s.price, 2)       if s else None
        score     = s.score                 if s else None
        rsi       = round(s.rsi, 1)         if s else None
        ma50      = round(s.ma50, 2)        if s else None
        ma200     = round(s.ma200, 2)       if s else None
        rng_pct   = round((s.price - s.week52_low) / max(s.week52_high - s.week52_low, 1e-6) * 100, 0) if s else None
        is_dip    = s.is_dip                if s else False
        ml_dir    = s.ml_direction          if s else None
        ml_prob   = round(s.ml_probability * 100, 0) if s and s.ml_probability else None
        ml_conf   = s.ml_confidence         if s else None
        ml_ok     = s.ml_buy_confirmed      if s else False

        fund_pass = p.passes
        margin    = round(p.profit_margin * 100, 1)  if p.profit_margin  is not None else None
        de        = round(p.debt_to_equity, 2)        if p.debt_to_equity is not None else None
        roe       = round(p.return_on_equity * 100, 1) if p.return_on_equity is not None else None
        fcf_b     = round(p.free_cash_flow / 1e9, 2) if p.free_cash_flow is not None else None
        pe        = round(r["info"].get("trailingPE") or r["info"].get("forwardPE") or 0, 1) or None

        # Signal label
        if is_dip and fund_pass and (not config.ML_GATE_BUY_SIGNAL or not config.ML_ENABLED or ml_ok):
            signal_label = "🟢 BUY"
        elif is_dip and fund_pass and config.ML_GATE_BUY_SIGNAL and ml_dir and not ml_ok:
            signal_label = "🟡 DIP+FUND (ML↓)"
        elif is_dip and not fund_pass:
            signal_label = "🟠 DIP ONLY"
        else:
            signal_label = "⚪ WATCH"

        rows.append({
            "Symbol":     sym,
            "Signal":     signal_label,
            "Score":      f"{score}/4" if score is not None else "—",
            "Price":      price,
            "RSI":        rsi,
            "vs MA50%":   round((price - ma50) / ma50 * 100, 1) if price and ma50 else None,
            "vs MA200%":  round((price - ma200) / ma200 * 100, 1) if price and ma200 else None,
            "52wk%":      rng_pct,
            "Fund":       "✅" if fund_pass else "❌",
            "ML 5d":      f"{'↑' if ml_dir=='UP' else '↓' if ml_dir else '—'}{int(ml_prob) if ml_prob else ''}% {ml_conf or ''}".strip() if ml_dir else "—",
            "Margin%":    margin,
            "D/E":        de,
            "ROE%":       roe,
            "FCF $B":     fcf_b,
            "P/E":        pe,
            "Src":        r["price_source"],
            # internal — used for risk tab
            "_is_dip":    is_dip,
            "_fund_pass": fund_pass,
            "_ml_ok":     ml_ok,
            "_price":     price,
        })
    return pd.DataFrame(rows)


def _risk_df(screen_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate position sizing for all dip candidates."""
    rows = []
    candidates = screen_df[screen_df["_is_dip"] == True].copy()
    open_pos = 0
    for _, row in candidates.iterrows():
        price = row["_price"]
        if not price:
            continue
        spec = calculate_order(
            symbol         = row["Symbol"],
            entry_price    = price,
            account_equity = ACCOUNT_EQUITY,
            open_positions = open_pos,
        )
        if spec is None:
            rows.append({
                "Symbol": row["Symbol"], "Signal": row["Signal"],
                "Entry $": price, "Qty": "—", "Notional $": "—",
                "Stop $": "—", "Target $": "—",
                "Risk $": "—", "% Budget": "—", "Status": "Skipped",
            })
            continue
        rows.append({
            "Symbol":    spec.symbol,
            "Signal":    row["Signal"],
            "Entry $":   spec.entry_price,
            "Qty":       spec.quantity,
            "Notional $": round(spec.position_value, 0),
            "Stop $":    spec.stop_loss_price,
            "Target $":  spec.take_profit_price,
            "Risk $":    round(spec.risk_amount, 0),
            "% Budget":  round(spec.position_value / ACCOUNT_EQUITY * 100, 1),
            "Status":    row["Signal"],
        })
        open_pos += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _card(children, **kwargs):
    return dbc.Card(
        dbc.CardBody(children),
        style={"backgroundColor": CARD_BG, "border": f"1px solid {BORDER}", "borderRadius": "8px", **kwargs},
        className="mb-3",
    )


def _stat_card(label, value, colour=TEXT):
    return dbc.Col(_card([
        html.Div(label, style={"color": MUTED, "fontSize": "12px", "marginBottom": "4px"}),
        html.Div(value, style={"color": colour, "fontSize": "22px", "fontWeight": "bold", "fontFamily": "monospace"}),
    ], padding="12px"), width="auto")


def _make_table(df: pd.DataFrame, tid: str, hide_cols: list[str] | None = None,
                colour_col: str | None = None) -> dash_table.DataTable:
    hide_cols = hide_cols or []
    display_df = df[[c for c in df.columns if c not in hide_cols]]
    columns = [{"name": c, "id": c} for c in display_df.columns]

    style_data_conditional = []
    if "Signal" in display_df.columns:
        style_data_conditional += [
            {"if": {"filter_query": '{Signal} contains "BUY"',  "column_id": "Signal"}, "color": GREEN,  "fontWeight": "bold"},
            {"if": {"filter_query": '{Signal} contains "DIP+FUND"', "column_id": "Signal"}, "color": YELLOW},
            {"if": {"filter_query": '{Signal} contains "DIP ONLY"', "column_id": "Signal"}, "color": "#f0883e"},
            {"if": {"filter_query": '{Signal} contains "WATCH"', "column_id": "Signal"}, "color": MUTED},
        ]
    if "outcome" in display_df.columns or "Outcome" in display_df.columns:
        col = "outcome" if "outcome" in display_df.columns else "Outcome"
        style_data_conditional += [
            {"if": {"filter_query": f'{{{col}}} = "TP"', "column_id": col}, "color": GREEN},
            {"if": {"filter_query": f'{{{col}}} = "SL"', "column_id": col}, "color": RED},
            {"if": {"filter_query": f'{{{col}}} = "OPEN"', "column_id": col}, "color": YELLOW},
        ]
    if "pct_gain" in display_df.columns:
        style_data_conditional += [
            {"if": {"filter_query": "{pct_gain} > 0", "column_id": "pct_gain"}, "color": GREEN},
            {"if": {"filter_query": "{pct_gain} < 0", "column_id": "pct_gain"}, "color": RED},
        ]
    if "realized_pnl" in display_df.columns:
        style_data_conditional += [
            {"if": {"filter_query": "{realized_pnl} > 0", "column_id": "realized_pnl"}, "color": GREEN, "fontWeight": "bold"},
            {"if": {"filter_query": "{realized_pnl} < 0", "column_id": "realized_pnl"}, "color": RED,   "fontWeight": "bold"},
        ]
    if "Action" in display_df.columns:
        style_data_conditional += [
            {"if": {"filter_query": '{Action} = "BUY"',  "column_id": "Action"}, "color": GREEN},
            {"if": {"filter_query": '{Action} = "SELL"', "column_id": "Action"}, "color": RED},
        ]

    return dash_table.DataTable(
        id=tid,
        columns=columns,
        data=display_df.to_dict("records"),
        style_cell=_CELL_STYLE,
        style_header=_HDR_STYLE,
        style_data_conditional=style_data_conditional,
        style_table={"overflowX": "auto", "borderRadius": "6px"},
        sort_action="native",
        filter_action="native",
        page_size=40,
    )


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="Buy The Sauce — Dashboard",
)
app.layout = html.Div(
    style={"backgroundColor": BRAND_BG, "minHeight": "100vh", "padding": "0"},
    children=[
        # ── top nav ───────────────────────────────────────────────────────────
        dbc.Navbar(
            dbc.Container([
                html.Span("🍅 Buy The Sauce", style={
                    "color": ACCENT, "fontWeight": "bold", "fontSize": "20px", "fontFamily": "monospace",
                }),
                dbc.NavbarToggler(id="navbar-toggler"),
                html.Span(id="ibkr-status-label",
                          style={"color": MUTED, "fontSize": "12px", "fontFamily": "monospace", "marginLeft": "16px"}),
                html.Span(id="last-refresh-label",
                          style={"color": MUTED, "fontSize": "12px", "fontFamily": "monospace", "marginLeft": "auto"}),
            ], fluid=True),
            color=CARD_BG, dark=True,
            style={"borderBottom": f"1px solid {BORDER}", "padding": "8px 24px"},
        ),

        dbc.Container(fluid=True, style={"padding": "24px"}, children=[

            # ── store (holds fetched data as JSON) ────────────────────────────
            dcc.Store(id="screen-store"),
            dcc.Store(id="trades-store"),           # triggers trades table refresh
            dcc.Interval(id="clock-tick", interval=60_000, n_intervals=0),  # 1-min clock

            # ── tabs ──────────────────────────────────────────────────────────
            dbc.Tabs(id="tabs", active_tab="tab-screener", children=[
                dbc.Tab(label="📡  Screener",      tab_id="tab-screener"),
                dbc.Tab(label="💰  Risk & Capital", tab_id="tab-risk"),
                dbc.Tab(label="📈  Back-test",      tab_id="tab-backtest"),
                dbc.Tab(label="🗂  Screen Log",     tab_id="tab-log"),
                dbc.Tab(label="📒  My Trades",      tab_id="tab-trades"),
            ], style={"marginBottom": "20px"}),

            html.Div(id="tab-content"),
        ]),
    ],
)


# ---------------------------------------------------------------------------
# Tab 1 — Screener layout (static skeleton; data injected by callback)
# ---------------------------------------------------------------------------

def _screener_layout():
    return html.Div([
        dbc.Row([
            dbc.Col(
                dbc.Button("🔄 Refresh Data", id="refresh-btn", color="primary", size="sm",
                           style={"fontFamily": "monospace"}),
                width="auto",
            ),
            dbc.Col(
                dcc.Loading(html.Div(id="refresh-status",
                                     style={"color": MUTED, "fontFamily": "monospace", "fontSize": "13px",
                                            "paddingTop": "6px"})),
                width=True,
            ),
        ], className="mb-3"),

        # stat cards
        dbc.Row(id="stat-cards", className="mb-3"),

        # main table
        _card(html.Div(id="screener-table-container",
                        children=html.Div("Click 🔄 Refresh to load live data.",
                                          style={"color": MUTED, "fontFamily": "monospace"}))),
    ])


# ---------------------------------------------------------------------------
# Tab 2 — Risk & Capital layout
# ---------------------------------------------------------------------------

def _risk_layout(screen_df: pd.DataFrame | None):
    if screen_df is None or screen_df.empty:
        return _card(html.Div("Run the screener first (📡 Screener → 🔄 Refresh).",
                               style={"color": MUTED, "fontFamily": "monospace"}))

    risk_df = _risk_df(screen_df)
    n_candidates = len(risk_df)

    if risk_df.empty:
        return _card(html.Div("No dip candidates found — nothing to size.",
                               style={"color": MUTED, "fontFamily": "monospace"}))

    # ── summary stats ─────────────────────────────────────────────────────────
    total_deployed = sum(
        r for r in risk_df["Notional $"] if isinstance(r, (int, float))
    )
    total_risk     = sum(
        r for r in risk_df["Risk $"] if isinstance(r, (int, float))
    )
    cash_remaining = ACCOUNT_EQUITY - total_deployed
    pct_deployed   = total_deployed / ACCOUNT_EQUITY * 100

    stat_cards = dbc.Row([
        _stat_card("Budget",           f"${ACCOUNT_EQUITY:,.0f}", ACCENT),
        _stat_card("Deployed",         f"${total_deployed:,.0f}  ({pct_deployed:.1f}%)",
                   GREEN if pct_deployed <= 60 else YELLOW),
        _stat_card("Cash Remaining",   f"${cash_remaining:,.0f}", TEXT),
        _stat_card("Total $ at Risk",  f"${total_risk:,.0f}  ({total_risk/ACCOUNT_EQUITY*100:.1f}%)", RED),
        _stat_card("Candidates",       str(n_candidates), ACCENT),
        _stat_card("Max Positions",    str(config.MAX_POSITIONS), MUTED),
    ], className="mb-3")

    # ── position sizing table ─────────────────────────────────────────────────
    pos_table = _make_table(risk_df, "risk-table")

    # ── capital allocation pie ────────────────────────────────────────────────
    pie_labels = list(risk_df["Symbol"]) + ["Cash"]
    pie_values = [
        v if isinstance(v, (int, float)) else 0
        for v in risk_df["Notional $"]
    ] + [max(cash_remaining, 0)]

    pie_fig = go.Figure(go.Pie(
        labels=pie_labels,
        values=pie_values,
        hole=0.55,
        marker=dict(colors=px.colors.qualitative.Plotly),
        textinfo="label+percent",
        textfont=dict(color=TEXT, size=12),
    ))
    pie_fig.update_layout(
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT),
        margin=dict(t=30, b=10, l=10, r=10),
        showlegend=False,
        annotations=[dict(text=f"${ACCOUNT_EQUITY/1e3:.0f}k", x=0.5, y=0.5,
                          font_size=18, showarrow=False, font_color=ACCENT)],
    )

    # ── risk heat-map (symbol vs risk metrics) ────────────────────────────────
    hm_syms  = list(risk_df["Symbol"])
    hm_pct   = [v if isinstance(v, (int, float)) else 0 for v in risk_df["% Budget"]]
    hm_risk  = [v if isinstance(v, (int, float)) else 0 for v in risk_df["Risk $"]]

    # Stop loss distance %
    sl_pct = config.STOP_LOSS_PCT * 100
    tp_pct = config.TAKE_PROFIT_PCT * 100
    rr     = tp_pct / sl_pct  # reward:risk ratio

    heat_fig = go.Figure()
    heat_fig.add_trace(go.Bar(
        name="% of Budget", x=hm_syms, y=hm_pct,
        marker_color=ACCENT, text=[f"{v:.1f}%" for v in hm_pct],
        textposition="outside",
    ))
    heat_fig.add_trace(go.Bar(
        name="$ at Risk (hundreds)", x=hm_syms,
        y=[r / 100 for r in hm_risk],
        marker_color=RED,
        text=[f"${r:,.0f}" for r in hm_risk],
        textposition="outside",
    ))
    heat_fig.add_hline(
        y=config.MAX_POSITION_PCT * 100, line_dash="dash",
        line_color=YELLOW, annotation_text=f"Max position {config.MAX_POSITION_PCT:.0%}",
        annotation_font_color=YELLOW,
    )
    heat_fig.update_layout(
        barmode="group",
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT), legend=dict(font=dict(color=TEXT)),
        xaxis=dict(gridcolor=BORDER, color=TEXT),
        yaxis=dict(gridcolor=BORDER, color=TEXT, title="% of Budget / $Risk÷100"),
        margin=dict(t=20, b=10, l=10, r=10),
    )

    # ── risk parameters card ──────────────────────────────────────────────────
    params_card = _card([
        html.H6("Risk Parameters", style={"color": ACCENT, "fontFamily": "monospace"}),
        html.Hr(style={"borderColor": BORDER}),
        dbc.Row([
            dbc.Col([
                html.Div(f"Stop Loss:       {config.STOP_LOSS_PCT:.0%}  per trade", style={"fontFamily": "monospace", "fontSize": "13px"}),
                html.Div(f"Take Profit:     {config.TAKE_PROFIT_PCT:.0%}  per trade", style={"fontFamily": "monospace", "fontSize": "13px"}),
                html.Div(f"Reward : Risk:   {rr:.1f} : 1", style={"fontFamily": "monospace", "fontSize": "13px", "color": GREEN}),
            ], width=4),
            dbc.Col([
                html.Div(f"Risk per trade:  {config.RISK_PER_TRADE_PCT:.1%}  of equity  =  ${ACCOUNT_EQUITY * config.RISK_PER_TRADE_PCT:,.0f}", style={"fontFamily": "monospace", "fontSize": "13px"}),
                html.Div(f"Max position:    {config.MAX_POSITION_PCT:.0%}  of equity  =  ${ACCOUNT_EQUITY * config.MAX_POSITION_PCT:,.0f}", style={"fontFamily": "monospace", "fontSize": "13px"}),
                html.Div(f"Max positions:   {config.MAX_POSITIONS}  concurrent", style={"fontFamily": "monospace", "fontSize": "13px"}),
            ], width=4),
            dbc.Col([
                html.Div(f"Max total risk:  ${ACCOUNT_EQUITY * config.RISK_PER_TRADE_PCT * config.MAX_POSITIONS:,.0f}  ({config.RISK_PER_TRADE_PCT * config.MAX_POSITIONS:.0%} of budget)", style={"fontFamily": "monospace", "fontSize": "13px", "color": RED}),
                html.Div(f"Max deployed:    ${ACCOUNT_EQUITY * config.MAX_POSITION_PCT * config.MAX_POSITIONS:,.0f}  ({config.MAX_POSITION_PCT * config.MAX_POSITIONS:.0%} of budget)", style={"fontFamily": "monospace", "fontSize": "13px"}),
                html.Div(f"Budget:          ${ACCOUNT_EQUITY:,.0f}", style={"fontFamily": "monospace", "fontSize": "13px", "color": ACCENT}),
            ], width=4),
        ]),
    ])

    return html.Div([
        stat_cards,
        params_card,
        _card(html.H6("Capital Allocation", style={"color": ACCENT, "fontFamily": "monospace", "marginBottom": "12px"})),
        dbc.Row([
            dbc.Col(_card(dcc.Graph(figure=pie_fig, config={"displayModeBar": False})), width=4),
            dbc.Col(_card(dcc.Graph(figure=heat_fig, config={"displayModeBar": False})), width=8),
        ], className="mb-3"),
        _card([
            html.H6("Position Sizing — All Dip Candidates", style={"color": ACCENT, "fontFamily": "monospace", "marginBottom": "12px"}),
            pos_table,
        ]),
    ])


# ---------------------------------------------------------------------------
# Tab 3 — Back-test layout
# ---------------------------------------------------------------------------

def _backtest_layout():
    if not os.path.exists(BACKTEST_CSV):
        return _card(html.Div([
            html.Div("No back-test data found.", style={"color": MUTED, "fontFamily": "monospace"}),
            html.Div("Run:  python3 backtest.py  (or  python3 run_screen.py --backtest)",
                     style={"color": MUTED, "fontFamily": "monospace", "fontSize": "12px", "marginTop": "8px"}),
        ]))

    df = pd.read_csv(BACKTEST_CSV)
    df = df[df["outcome"].notna()]

    # ── equity curve (cumulative return over entry date, per run) ─────────────
    latest_run = df["backtest_date"].max()
    run_df     = df[df["backtest_date"] == latest_run].copy()
    run_df["pct_gain"] = pd.to_numeric(run_df["pct_gain"], errors="coerce")
    run_df = run_df.dropna(subset=["pct_gain"]).sort_values("entry_date")
    run_df["cum_return"] = (1 + run_df["pct_gain"] / 100).cumprod() - 1

    eq_fig = go.Figure()
    eq_fig.add_trace(go.Scatter(
        x=run_df["entry_date"], y=run_df["cum_return"] * 100,
        mode="lines+markers",
        line=dict(color=ACCENT, width=2),
        marker=dict(color=[GREEN if v >= 0 else RED for v in run_df["cum_return"]], size=7),
        name="Cumulative Return %",
        hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
    ))
    eq_fig.add_hline(y=0, line_color=BORDER, line_dash="dash")
    eq_fig.update_layout(
        title="Equity Curve (latest back-test run)",
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT),
        xaxis=dict(gridcolor=BORDER, color=TEXT, title="Entry Date"),
        yaxis=dict(gridcolor=BORDER, color=TEXT, title="Cumulative Return %"),
        margin=dict(t=40, b=20, l=20, r=20),
    )

    # ── outcome breakdown bar ─────────────────────────────────────────────────
    outcome_counts = run_df["outcome"].value_counts().reset_index()
    outcome_counts.columns = ["outcome", "count"]
    colour_map = {"TP": GREEN, "SL": RED, "OPEN": YELLOW}

    bar_fig = go.Figure(go.Bar(
        x=outcome_counts["outcome"],
        y=outcome_counts["count"],
        marker_color=[colour_map.get(o, MUTED) for o in outcome_counts["outcome"]],
        text=outcome_counts["count"],
        textposition="outside",
    ))
    bar_fig.update_layout(
        title="Trade Outcomes",
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT),
        xaxis=dict(color=TEXT), yaxis=dict(color=TEXT, gridcolor=BORDER),
        margin=dict(t=40, b=20, l=20, r=20),
        showlegend=False,
    )

    # ── gain distribution histogram ───────────────────────────────────────────
    hist_fig = go.Figure(go.Histogram(
        x=run_df["pct_gain"], nbinsx=30,
        marker_color=ACCENT, opacity=0.8,
        hovertemplate="Gain: %{x:.1f}%  Count: %{y}<extra></extra>",
    ))
    hist_fig.add_vline(x=0, line_color=BORDER, line_dash="dash")
    hist_fig.update_layout(
        title="Trade Gain Distribution",
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT),
        xaxis=dict(color=TEXT, title="% Gain"), yaxis=dict(color=TEXT, gridcolor=BORDER, title="Count"),
        margin=dict(t=40, b=20, l=20, r=20),
    )

    # ── per-symbol stats ──────────────────────────────────────────────────────
    sym_stats = (
        run_df.groupby("symbol")["pct_gain"]
        .agg(trades="count", avg_gain="mean", win_rate=lambda x: (x > 0).mean())
        .reset_index()
        .round(2)
    )
    sym_stats.columns = ["Symbol", "Trades", "Avg Gain %", "Win Rate"]
    sym_stats["Win Rate"] = (sym_stats["Win Rate"] * 100).round(1).astype(str) + "%"

    sym_bar = go.Figure(go.Bar(
        x=sym_stats["Symbol"], y=sym_stats["Avg Gain %"],
        marker_color=[GREEN if v >= 0 else RED for v in sym_stats["Avg Gain %"]],
        text=[f"{v:.1f}%" for v in sym_stats["Avg Gain %"]],
        textposition="outside",
    ))
    sym_bar.add_hline(y=0, line_color=BORDER, line_dash="dash")
    sym_bar.update_layout(
        title="Avg Gain % by Symbol",
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT),
        xaxis=dict(color=TEXT), yaxis=dict(color=TEXT, gridcolor=BORDER),
        margin=dict(t=40, b=20, l=20, r=20),
    )

    # ── summary stats row ─────────────────────────────────────────────────────
    n_closed = len(run_df[run_df["outcome"] != "OPEN"])
    n_wins   = len(run_df[run_df["pct_gain"] > 0])
    wr       = n_wins / n_closed if n_closed else 0
    avg_g    = run_df["pct_gain"].mean()
    cum_ret  = run_df["cum_return"].iloc[-1] * 100 if not run_df.empty else 0

    stat_row = dbc.Row([
        _stat_card("Back-test Date",    latest_run,                    MUTED),
        _stat_card("Total Trades",      str(len(run_df)),              TEXT),
        _stat_card("Win Rate",          f"{wr:.1%}",                   GREEN if wr >= 0.5 else RED),
        _stat_card("Avg Gain/Trade",    f"{avg_g:+.2f}%",              GREEN if avg_g >= 0 else RED),
        _stat_card("Cumulative Return", f"{cum_ret:+.1f}%",            GREEN if cum_ret >= 0 else RED),
    ], className="mb-3")

    return html.Div([
        stat_row,
        dbc.Row([
            dbc.Col(_card(dcc.Graph(figure=eq_fig,   config={"displayModeBar": False})), width=12),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(_card(dcc.Graph(figure=bar_fig,  config={"displayModeBar": False})), width=4),
            dbc.Col(_card(dcc.Graph(figure=hist_fig, config={"displayModeBar": False})), width=4),
            dbc.Col(_card(dcc.Graph(figure=sym_bar,  config={"displayModeBar": False})), width=4),
        ], className="mb-3"),
        _card([
            html.H6("Trade Log (latest run)", style={"color": ACCENT, "fontFamily": "monospace", "marginBottom": "12px"}),
            _make_table(
                run_df[["symbol","entry_date","entry_price","exit_date","exit_price","pct_gain","outcome","hold_days","score"]],
                "backtest-table",
            ),
        ]),
        _card([
            html.H6("Per-Symbol Summary", style={"color": ACCENT, "fontFamily": "monospace", "marginBottom": "12px"}),
            _make_table(sym_stats, "sym-stats-table"),
        ]),
    ])


# ---------------------------------------------------------------------------
# Tab 4 — Screen Log layout
# ---------------------------------------------------------------------------

def _log_layout():
    if not os.path.exists(SCREEN_CSV):
        return _card(html.Div("No screen log found. Run the screener first.",
                               style={"color": MUTED, "fontFamily": "monospace"}))

    df = pd.read_csv(SCREEN_CSV)
    df["dip_score"]   = pd.to_numeric(df["dip_score"],   errors="coerce")
    df["rsi"]         = pd.to_numeric(df["rsi"],         errors="coerce")
    df["price"]       = pd.to_numeric(df["price"],       errors="coerce")

    # ── dip score over time per symbol ───────────────────────────────────────
    score_fig = px.line(
        df.sort_values("date"), x="date", y="dip_score", color="symbol",
        title="Dip Score Over Time by Symbol",
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Plotly,
    )
    score_fig.update_layout(
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT), legend=dict(font=dict(color=TEXT)),
        xaxis=dict(gridcolor=BORDER), yaxis=dict(gridcolor=BORDER, title="Dip Score"),
        margin=dict(t=40, b=20, l=20, r=20),
    )

    # ── RSI heat-map (symbol × date) ─────────────────────────────────────────
    rsi_pivot = df.pivot_table(index="symbol", columns="date", values="rsi", aggfunc="mean")
    rsi_fig = go.Figure(go.Heatmap(
        z=rsi_pivot.values,
        x=rsi_pivot.columns.tolist(),
        y=rsi_pivot.index.tolist(),
        colorscale=[[0, RED], [0.35, YELLOW], [0.65, ACCENT], [1, GREEN]],
        zmin=0, zmax=100,
        colorbar=dict(title="RSI", tickfont=dict(color=TEXT), titlefont=dict(color=TEXT)),
        hovertemplate="Date: %{x}<br>Symbol: %{y}<br>RSI: %{z:.1f}<extra></extra>",
    ))
    rsi_fig.add_hline(y=-0.5, annotation_text="Oversold < 35",
                      line_color=RED, line_dash="dot", annotation_font_color=RED)
    rsi_fig.update_layout(
        title="RSI Heat-map (symbol × date)",
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT),
        xaxis=dict(color=TEXT), yaxis=dict(color=TEXT),
        margin=dict(t=40, b=20, l=20, r=20),
    )

    return html.Div([
        dbc.Row([
            dbc.Col(_card(dcc.Graph(figure=score_fig, config={"displayModeBar": False})), width=12),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(_card(dcc.Graph(figure=rsi_fig,   config={"displayModeBar": False})), width=12),
        ], className="mb-3"),
        _card([
            html.H6("Full Screen Log", style={"color": ACCENT, "fontFamily": "monospace", "marginBottom": "12px"}),
            _make_table(df, "log-table"),
        ]),
    ])


# ---------------------------------------------------------------------------
# Tab 5 — My Trades (manual transaction journal)
# ---------------------------------------------------------------------------

_INPUT_STYLE = {
    "backgroundColor": "#21262d",
    "color": TEXT,
    "border": f"1px solid {BORDER}",
    "borderRadius": "4px",
    "fontFamily": "monospace",
    "fontSize": "13px",
    "width": "100%",
    "padding": "6px 10px",
}

def _trades_layout():
    """Build the full My Trades tab layout (form + summary + chart + table)."""
    df = _load_trades()

    # ── summary stat cards ────────────────────────────────────────────────────
    total_invested  = df["amount_invested"].sum() if not df.empty else 0
    realized_pnl    = df["realized_pnl"].dropna().sum() if not df.empty else 0
    n_trades        = len(df)
    n_wins          = int((df["realized_pnl"] > 0).sum()) if not df.empty else 0
    win_rate        = n_wins / n_trades if n_trades > 0 else 0
    roi_pct         = realized_pnl / total_invested * 100 if total_invested else 0

    stat_row = dbc.Row([
        _stat_card("Total Trades",      str(n_trades),                    ACCENT),
        _stat_card("Total Invested",    f"${total_invested:,.0f}",        TEXT),
        _stat_card("Realized P&L",      f"${realized_pnl:+,.2f}",        GREEN if realized_pnl >= 0 else RED),
        _stat_card("ROI %",             f"{roi_pct:+.2f}%",               GREEN if roi_pct >= 0 else RED),
        _stat_card("Win Rate",          f"{win_rate:.0%}  ({n_wins}/{n_trades})", GREEN if win_rate >= 0.5 else (YELLOW if n_trades else MUTED)),
    ], className="mb-3")

    # ── cumulative P&L chart ──────────────────────────────────────────────────
    if not df.empty and df["realized_pnl"].notna().any():
        pnl_df = df[df["realized_pnl"].notna()].copy()
        pnl_df = pnl_df.sort_values("date")
        pnl_df["cum_pnl"] = pnl_df["realized_pnl"].cumsum()

        pnl_fig = go.Figure()
        pnl_fig.add_trace(go.Scatter(
            x=pnl_df["date"], y=pnl_df["cum_pnl"],
            mode="lines+markers",
            line=dict(color=ACCENT, width=2),
            marker=dict(
                color=[GREEN if v >= 0 else RED for v in pnl_df["cum_pnl"]],
                size=8,
            ),
            fill="tozeroy",
            fillcolor="rgba(88,166,255,0.08)",
            hovertemplate="%{x}<br>Cum P&L: $%{y:+,.2f}<extra></extra>",
        ))
        pnl_fig.add_hline(y=0, line_color=BORDER, line_dash="dash")
        pnl_fig.update_layout(
            title="Cumulative Realized P&L",
            paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
            font=dict(color=TEXT),
            xaxis=dict(gridcolor=BORDER, color=TEXT, title="Date"),
            yaxis=dict(gridcolor=BORDER, color=TEXT, title="Cumulative P&L ($)"),
            margin=dict(t=40, b=20, l=20, r=20),
        )

        # Per-symbol P&L bar
        sym_pnl = pnl_df.groupby("symbol")["realized_pnl"].sum().reset_index()
        sym_pnl.columns = ["Symbol", "P&L"]
        sym_fig = go.Figure(go.Bar(
            x=sym_pnl["Symbol"], y=sym_pnl["P&L"],
            marker_color=[GREEN if v >= 0 else RED for v in sym_pnl["P&L"]],
            text=[f"${v:+,.2f}" for v in sym_pnl["P&L"]],
            textposition="outside",
        ))
        sym_fig.add_hline(y=0, line_color=BORDER, line_dash="dash")
        sym_fig.update_layout(
            title="Realized P&L by Symbol",
            paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
            font=dict(color=TEXT),
            xaxis=dict(color=TEXT), yaxis=dict(color=TEXT, gridcolor=BORDER, title="P&L ($)"),
            margin=dict(t=40, b=20, l=20, r=20),
        )
        charts = dbc.Row([
            dbc.Col(_card(dcc.Graph(figure=pnl_fig, config={"displayModeBar": False})), width=8),
            dbc.Col(_card(dcc.Graph(figure=sym_fig, config={"displayModeBar": False})), width=4),
        ], className="mb-3")
    else:
        charts = html.Div()

    # ── trade entry form ──────────────────────────────────────────────────────
    form_card = _card([
        html.H6("➕ Log a New Trade", style={"color": ACCENT, "fontFamily": "monospace", "marginBottom": "12px"}),
        dbc.Row([
            dbc.Col([
                html.Label("Date", style={"color": MUTED, "fontSize": "12px"}),
                dcc.Input(
                    id="trade-date", type="text",
                    placeholder=datetime.date.today().isoformat(),
                    value=datetime.date.today().isoformat(),
                    debounce=True, style=_INPUT_STYLE,
                ),
            ], width=2),
            dbc.Col([
                html.Label("Symbol", style={"color": MUTED, "fontSize": "12px"}),
                dcc.Input(id="trade-symbol", type="text", placeholder="AAPL",
                          debounce=True, style=_INPUT_STYLE),
            ], width=2),
            dbc.Col([
                html.Label("Action", style={"color": MUTED, "fontSize": "12px"}),
                dcc.Dropdown(
                    id="trade-action",
                    options=[{"label": "BUY", "value": "BUY"}, {"label": "SELL", "value": "SELL"}],
                    value="BUY",
                    style={"backgroundColor": "#21262d", "color": BRAND_BG, "fontFamily": "monospace"},
                    clearable=False,
                ),
            ], width=1),
            dbc.Col([
                html.Label("Quantity", style={"color": MUTED, "fontSize": "12px"}),
                dcc.Input(id="trade-qty", type="number", placeholder="10",
                          debounce=True, style=_INPUT_STYLE),
            ], width=1),
            dbc.Col([
                html.Label("Entry Price $", style={"color": MUTED, "fontSize": "12px"}),
                dcc.Input(id="trade-entry", type="number", placeholder="150.00",
                          debounce=True, style=_INPUT_STYLE),
            ], width=2),
            dbc.Col([
                html.Label("Exit Price $ (opt)", style={"color": MUTED, "fontSize": "12px"}),
                dcc.Input(id="trade-exit", type="number", placeholder="165.00",
                          debounce=True, style=_INPUT_STYLE),
            ], width=2),
            dbc.Col([
                html.Label("Notes (opt)", style={"color": MUTED, "fontSize": "12px"}),
                dcc.Input(id="trade-notes", type="text", placeholder="Screener signal",
                          debounce=True, style=_INPUT_STYLE),
            ], width=2),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(
                dbc.Button("💾 Save Trade", id="save-trade-btn", color="success", size="sm",
                           style={"fontFamily": "monospace"}),
                width="auto",
            ),
            dbc.Col(
                html.Div(id="trade-save-status",
                         style={"color": MUTED, "fontFamily": "monospace", "fontSize": "13px",
                                "paddingTop": "6px"}),
                width=True,
            ),
        ]),
        html.Div([
            html.Hr(style={"borderColor": BORDER, "marginTop": "16px"}),
            html.Div([
                html.Span("💡 Realized P&L is auto-calculated from Entry/Exit when you save a SELL trade. "
                          "You can also enter it manually via CSV at ",
                          style={"color": MUTED, "fontSize": "11px", "fontFamily": "monospace"}),
                html.Code("results/my_trades.csv",
                          style={"color": ACCENT, "fontSize": "11px"}),
            ]),
        ]),
    ])

    # ── trade log table ───────────────────────────────────────────────────────
    if not df.empty:
        display_df = df.copy()
        display_df.columns = ["Date", "Symbol", "Action", "Qty", "Entry $",
                               "Exit $", "Invested $", "Realized P&L $", "Notes"]
        table_section = _card([
            html.H6(f"Trade Journal  ({n_trades} entries)",
                    style={"color": ACCENT, "fontFamily": "monospace", "marginBottom": "12px"}),
            _make_table(display_df, "trades-table"),
            html.Div([
                html.Hr(style={"borderColor": BORDER}),
                html.Span("✏️  Edit directly in ",
                          style={"color": MUTED, "fontSize": "11px", "fontFamily": "monospace"}),
                html.Code("results/my_trades.csv",
                          style={"color": ACCENT, "fontSize": "11px"}),
                html.Span(" and refresh the page to reload.",
                          style={"color": MUTED, "fontSize": "11px", "fontFamily": "monospace"}),
            ]),
        ])
    else:
        table_section = _card(
            html.Div("No trades logged yet. Use the form above to add your first trade.",
                     style={"color": MUTED, "fontFamily": "monospace"})
        )

    return html.Div([stat_row, charts, form_card, table_section])


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab"),
    State("screen-store", "data"),
    State("trades-store", "data"),
)
def render_tab(active_tab, store_data, trades_store):
    if active_tab == "tab-screener":
        return _screener_layout()
    if active_tab == "tab-risk":
        if store_data:
            screen_df = pd.DataFrame(store_data)
        else:
            screen_df = None
        return _risk_layout(screen_df)
    if active_tab == "tab-backtest":
        return _backtest_layout()
    if active_tab == "tab-log":
        return _log_layout()
    if active_tab == "tab-trades":
        return _trades_layout()
    return html.Div("Unknown tab")


@app.callback(
    Output("screen-store",           "data"),
    Output("screener-table-container","children"),
    Output("stat-cards",             "children"),
    Output("refresh-status",         "children"),
    Output("last-refresh-label",     "children"),
    Output("ibkr-status-label",      "children"),
    Input("refresh-btn",             "n_clicks"),
    prevent_initial_call=True,
)
def refresh_screener(n_clicks):
    """Re-run the screener pipeline and update the table + store."""
    global _last_fetch

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Check IBKR status first (non-blocking)
    ibkr_ok, ibkr_msg = _check_ibkr_status()

    try:
        records  = fetch_screen_data()
        df       = _records_to_df(records)
        _last_fetch = {"records": records, "df": df}
    except Exception as exc:
        return (
            dash.no_update, dash.no_update, dash.no_update,
            html.Span(f"❌ Error: {exc}", style={"color": RED}),
            f"Last refresh: {ts}",
            html.Span(ibkr_msg, style={"color": GREEN if ibkr_ok else YELLOW}),
        )

    # Count price sources to display in status
    src_counts = df["Src"].value_counts().to_dict() if "Src" in df.columns else {}
    src_summary = "  ".join(f"{src}:{cnt}" for src, cnt in src_counts.items())

    # stat cards
    n_buy   = df["Signal"].str.contains("BUY").sum()
    n_dip   = df["Signal"].str.contains("DIP").sum()
    n_watch = df["Signal"].str.contains("WATCH").sum()

    stat_cards = dbc.Row([
        _stat_card("🟢 Buy Signals",  str(n_buy),               GREEN),
        _stat_card("🟡 Dip Alerts",   str(n_dip),               YELLOW),
        _stat_card("⚪ Watch",         str(n_watch),             MUTED),
        _stat_card("Tickers Scanned", str(len(df)),              ACCENT),
        _stat_card("Price Sources",   src_summary or "—",        MUTED),
        _stat_card("Last Refresh",    ts,                        MUTED),
    ])

    hide = ["_is_dip", "_fund_pass", "_ml_ok", "_price"]
    table = _make_table(df, "screener-table", hide_cols=hide)

    return (
        df.to_dict("records"),
        table,
        stat_cards,
        html.Span(f"✅ Updated {ts}", style={"color": GREEN}),
        f"Last refresh: {ts}",
        html.Span(ibkr_msg, style={"color": GREEN if ibkr_ok else YELLOW}),
    )


@app.callback(
    Output("trades-store",      "data"),
    Output("trade-save-status", "children"),
    Output("trade-symbol",      "value"),
    Output("trade-qty",         "value"),
    Output("trade-entry",       "value"),
    Output("trade-exit",        "value"),
    Output("trade-notes",       "value"),
    Input("save-trade-btn",     "n_clicks"),
    State("trade-date",         "value"),
    State("trade-symbol",       "value"),
    State("trade-action",       "value"),
    State("trade-qty",          "value"),
    State("trade-entry",        "value"),
    State("trade-exit",         "value"),
    State("trade-notes",        "value"),
    prevent_initial_call=True,
)
def save_trade(n_clicks, date, symbol, action, qty, entry, exit_p, notes):
    """Validate inputs, auto-calculate P&L, append to CSV, refresh display."""
    errors = []
    if not symbol or not str(symbol).strip():
        errors.append("Symbol is required.")
    if not qty or float(qty) <= 0:
        errors.append("Quantity must be > 0.")
    if not entry or float(entry) <= 0:
        errors.append("Entry Price must be > 0.")
    if errors:
        return (
            dash.no_update,
            html.Span("⚠️ " + "  ".join(errors), style={"color": YELLOW}),
            dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
        )

    qty_f    = float(qty)
    entry_f  = float(entry)
    exit_f   = float(exit_p) if exit_p else None

    # Auto-compute realized P&L for SELL trades if exit is provided
    realized = None
    if action == "SELL" and exit_f is not None:
        realized = (exit_f - entry_f) * qty_f
    amount_invested = round(qty_f * entry_f, 2)

    row = {
        "date":            date or datetime.date.today().isoformat(),
        "symbol":          str(symbol).strip().upper(),
        "action":          action or "BUY",
        "quantity":        qty_f,
        "entry_price":     entry_f,
        "exit_price":      exit_f if exit_f is not None else "",
        "amount_invested": amount_invested,
        "realized_pnl":    round(realized, 2) if realized is not None else "",
        "notes":           (notes or "").strip(),
    }
    _append_trade(row)

    pnl_msg = f"  →  P&L: ${realized:+,.2f}" if realized is not None else ""
    status_msg = html.Span(
        f"✅ Saved {row['action']} {qty_f:.0f}x {row['symbol']} @ ${entry_f:.2f}{pnl_msg}",
        style={"color": GREEN},
    )
    # Return a timestamp as store data to trigger re-render of trades tab
    return (
        {"saved_at": datetime.datetime.now().isoformat()},
        status_msg,
        "",    # clear symbol
        None,  # clear qty
        None,  # clear entry
        None,  # clear exit
        "",    # clear notes
    )


@app.callback(
    Output("last-refresh-label", "children", allow_duplicate=True),
    Input("clock-tick", "n_intervals"),
    State("last-refresh-label", "children"),
    prevent_initial_call=True,
)
def tick_clock(_, current_label):
    """Keep the navbar label fresh without re-fetching data."""
    return current_label or "Not refreshed yet"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="Buy-the-sauce live dashboard")
    parser.add_argument("--port", type=int, default=8051)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    _ensure_trades_csv()   # make sure journal exists on startup
    args = _parse_args()
    print(f"\n  🍅 Buy The Sauce Dashboard")
    print(f"  Open → http://127.0.0.1:{args.port}\n")
    print(f"  IBKR config: {config.IBKR_HOST}:{config.IBKR_PORT}  (clientId {config.IBKR_CLIENT_ID})")
    print(f"  Trade journal: {TRADES_CSV}\n")
    ibkr_ok, ibkr_msg = _check_ibkr_status()
    print(f"  {ibkr_msg}\n")
    app.run(
        debug=False,
        port=args.port,
        use_reloader=False,
    )
