"""Plotly figure builders (pure functions of API data)."""

from __future__ import annotations

from typing import Optional, Sequence

import plotly.graph_objects as go

from . import theme


def _layout(fig: go.Figure, title: str = "", height: int = 340) -> go.Figure:
    fig.update_layout(template=theme.PLOTLY_TEMPLATE, title=title, height=height, margin=dict(l=10, r=10, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=-0.2))
    return fig


def iv_gauge(rank: Optional[float], long_max: float, short_min: float, title: str = "IV rank") -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=rank if rank is not None else 0, title={"text": title},
        number={"suffix": "" if rank is not None else " (n/a)"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": theme.BLUE},
               "steps": [{"range": [0, long_max], "color": "rgba(46,204,113,0.35)"},   # cheap vol: long candidates
                         {"range": [long_max, short_min], "color": "rgba(138,147,166,0.25)"},
                         {"range": [short_min, 100], "color": "rgba(245,166,35,0.35)"}]}))  # rich vol: short candidates
    return _layout(fig, height=230)


def payoff_fig(spots: Sequence[float], pnl: Sequence[float], breakevens: Sequence[float], spot: Optional[float],
               title: str = "Payoff at expiry") -> go.Figure:
    pos = [max(v, 0.0) for v in pnl]
    neg = [min(v, 0.0) for v in pnl]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spots, y=pos, fill="tozeroy", mode="lines", line=dict(color=theme.GREEN, width=0),
                             fillcolor="rgba(46,204,113,0.35)", name="Profit", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=spots, y=neg, fill="tozeroy", mode="lines", line=dict(color=theme.RED, width=0),
                             fillcolor="rgba(231,76,60,0.35)", name="Loss", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=spots, y=pnl, mode="lines", line=dict(color="white", width=2), name="P&L ($)",
                             hovertemplate="spot %{x:.2f}<br>P&L %{y:,.0f}<extra></extra>"))
    for be in breakevens:
        fig.add_vline(x=be, line=dict(color=theme.AMBER, dash="dash"), annotation_text=f"BE {be:.2f}")
    if spot:
        fig.add_vline(x=spot, line=dict(color=theme.BLUE, dash="dot"), annotation_text="spot")
    fig.add_hline(y=0, line=dict(color=theme.MUTED, width=1))
    fig.update_xaxes(title="Underlying price")
    fig.update_yaxes(title="P&L ($)")
    return _layout(fig, title)


def move_compare(rows: Sequence[dict]) -> go.Figure:
    """Implied move (ATM straddle) vs the move implied by realized vol, per symbol."""
    rows = [r for r in rows if r.get("candidate")]
    syms = [r["symbol"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=syms, y=[r["candidate"]["implied_move"] for r in rows], name="Implied move ($)",
                         marker_color=theme.BLUE))
    fig.add_trace(go.Bar(x=syms, y=[r["candidate"].get("realized_move") or 0 for r in rows], name="Realized-vol move ($)",
                         marker_color=theme.AMBER))
    fig.update_layout(barmode="group")
    return _layout(fig, "Expected move: implied vs realized-vol", 300)


def utilization_bars(util: dict, warn: float = 0.8) -> go.Figure:
    names = list(util)
    vals = [util[k] for k in names]
    fig = go.Figure(go.Bar(
        x=[min(v, 1.5) * 100 for v in vals], y=names, orientation="h",
        marker_color=[theme.utilization_color(v, warn) for v in vals],
        text=[f"{v:.0%}" for v in vals], textposition="outside", hovertemplate="%{y}: %{text}<extra></extra>"))
    fig.add_vline(x=100, line=dict(color="white", dash="dash"), annotation_text="limit")
    fig.update_xaxes(range=[0, 160], title="% of limit")
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, "Limit utilization", 300)


def exposure_heatmap(heatmap: dict) -> go.Figure:
    tickers = sorted(heatmap)
    cols = sorted({c for row in heatmap.values() for c in row}, key=lambda c: (c != "STK", c))
    z = [[abs(heatmap[t].get(c, 0.0)) for c in cols] for t in tickers]
    text = [[f"{heatmap[t].get(c, 0.0):,.0f}" if c in heatmap[t] else "" for c in cols] for t in tickers]
    fig = go.Figure(go.Heatmap(z=z, x=cols, y=tickers, text=text, texttemplate="%{text}", colorscale="Blues",
                               colorbar=dict(title="|delta $|"), hovertemplate="%{y} / %{x}<br>delta $ %{text}<extra></extra>"))
    return _layout(fig, "Delta-dollar exposure by ticker and expiry", 320)


def scenario_heatmap(scenario: dict) -> go.Figure:
    names = [k for k in scenario if k != "TOTAL"] + (["TOTAL"] if "TOTAL" in scenario else [])
    moves = sorted({float(m) for v in scenario.values() for m in v})
    z = [[scenario[n].get(str(m), scenario[n].get(repr(m), 0.0)) for m in moves] for n in names]
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"{m:+.0%}" for m in moves], y=names, colorscale=[[0, theme.RED], [0.5, "#1c1f26"], [1, theme.GREEN]],
        zmid=0, text=[[f"{v:,.0f}" for v in row] for row in z], texttemplate="%{text}",
        colorbar=dict(title="P&L ($)")))
    fig.update_xaxes(title="Underlying move")
    return _layout(fig, "Scenario grid: P&L under underlying moves", 320)


def equity_compare(runs: dict[str, dict]) -> go.Figure:
    fig = go.Figure()
    for label, run in runs.items():
        eq = run["equity"]["equity"]
        if not eq:
            continue
        base = eq[0][1] or 1.0
        fig.add_trace(go.Scatter(x=[d for d, _ in eq], y=[v / base * 100 for _, v in eq], mode="lines", name=label))
    fig.update_yaxes(title="Equity (start = 100)")
    return _layout(fig, "Equity curves", 360)


def drawdown_compare(runs: dict[str, dict]) -> go.Figure:
    fig = go.Figure()
    for label, run in runs.items():
        dd = run["equity"]["drawdown"]
        fig.add_trace(go.Scatter(x=[d for d, _ in dd], y=[v * 100 for _, v in dd], mode="lines", fill="tozeroy", name=label))
    fig.update_yaxes(title="Drawdown (%)")
    return _layout(fig, "Drawdown", 300)


def iv_history_fig(rows: Sequence[dict], symbol: str) -> go.Figure:
    fig = go.Figure(go.Scatter(x=[r["day"] for r in rows], y=[r["iv"] * 100 for r in rows], mode="lines",
                               line=dict(color=theme.BLUE)))
    fig.update_yaxes(title="ATM IV (%)")
    return _layout(fig, f"{symbol} implied volatility history", 260)
