"""
wheelcli/reports/tables.py — Rich terminal tables for scan output and explain view.

Public API
----------
render_candidates_table(candidates) → rich.table.Table
render_explain(candidate, spot)     → prints a panel to stdout
console                             → shared Console instance
"""

from __future__ import annotations

from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..models import CandidatePut

console = Console()


# =============================================================================
# Formatting helpers
# =============================================================================


def _pct(v: Optional[float], dp: int = 2) -> str:
    return f"{v * 100:.{dp}f}%" if v is not None else "—"


def _f(v: Optional[float], dp: int = 2) -> str:
    return f"{v:.{dp}f}" if v is not None else "—"


def _money(v: Optional[float]) -> str:
    return f"${v:.2f}" if v is not None else "—"


def _event_text(flag: str) -> Text:
    t = Text(flag or "—")
    if "EARNINGS" in flag:
        t.stylize("bold red")
    elif "MACRO" in flag:
        t.stylize("yellow")
    elif "UNKNOWN" in flag:
        t.stylize("dim yellow")
    return t


def _score_text(score: float) -> Text:
    t = Text(f"{score:.4f}")
    if score >= 0.15:
        t.stylize("bold green")
    elif score >= 0.08:
        t.stylize("green")
    else:
        t.stylize("dim")
    return t


# =============================================================================
# Scan results table
# =============================================================================


def render_candidates_table(candidates: list[CandidatePut]) -> Table:
    """
    Build a Rich Table from a ranked list of CandidatePut objects.

    The caller is responsible for sorting *candidates* before passing them in.
    """
    table = Table(
        title="[bold]Wheel Strategy — CSP Candidates[/bold]",
        show_lines=True,
        highlight=True,
        border_style="dim",
        header_style="bold dim",
    )

    columns = [
        ("#",         "right",  "bold"),
        ("Symbol",    "left",   "bold cyan"),
        ("Expiry",    "center", ""),
        ("DTE",       "right",  ""),
        ("Strike",    "right",  ""),
        ("Delta",     "right",  ""),
        ("IV",        "right",  ""),
        ("σ-Dist",    "right",  ""),
        ("Bid",       "right",  ""),
        ("Ask",       "right",  ""),
        ("Mid",       "right",  ""),
        ("Ann.ROC",   "right",  "green"),
        ("Spread%",   "right",  ""),
        ("SkewRatio", "right",  ""),
        ("Event",     "left",   ""),
        ("Score",     "right",  "bold"),
    ]

    for name, justify, style in columns:
        table.add_column(name, justify=justify, style=style or None)

    for i, c in enumerate(candidates, start=1):
        table.add_row(
            str(i),
            c.symbol,
            str(c.expiry),
            str(c.dte),
            _money(c.strike),
            _f(c.delta, 4) if c.delta is not None else "—",
            _pct(c.iv),
            _f(c.sigma_distance) if c.sigma_distance is not None else "—",
            _money(c.bid),
            _money(c.ask),
            _money(c.mid),
            _pct(c.annualized_roc),
            _pct(c.spread_pct) if c.spread_pct is not None else "—",
            _f(c.skew_ratio) if c.skew_ratio is not None else "—",
            _event_text(c.event_flag),
            _score_text(c.final_score),
        )

    return table


# =============================================================================
# Explain panel
# =============================================================================


def render_explain(candidate: CandidatePut, spot: float) -> None:
    """
    Print a detailed scoring breakdown for a single candidate to stdout.
    """
    lines: list[str] = [
        f"[bold]Symbol:[/bold]             {candidate.symbol}",
        f"[bold]Current spot:[/bold]       {_money(spot)}",
        f"[bold]Strike:[/bold]             {_money(candidate.strike)}",
        f"[bold]Expiry:[/bold]             {candidate.expiry}  ({candidate.dte} DTE)",
        "",
        "[bold underline]Market data[/bold underline]",
        f"  Bid / Ask / Mid:    {_money(candidate.bid)} / {_money(candidate.ask)} / {_money(candidate.mid)}",
        f"  Open interest:      {candidate.open_interest if candidate.open_interest is not None else '—'}",
        f"  Volume:             {candidate.volume if candidate.volume is not None else '—'}",
        "",
        "[bold underline]Analytics[/bold underline]",
        (
            f"  Delta (abs):        {_f(candidate.delta, 4)}"
            if candidate.delta is not None
            else "  Delta:              n/a"
        ),
        (
            f"  IV (annualised):    {_pct(candidate.iv)}  "
            f"(e.g. 30% = 0.30)"
            if candidate.iv is not None
            else "  IV:                 n/a"
        ),
        (
            f"  Sigma distance:     {_f(candidate.sigma_distance, 3)}σ  "
            f"(≥ 2.0 = far OTM)"
            if candidate.sigma_distance is not None
            else "  Sigma distance:     n/a  (IV missing)"
        ),
        (
            f"  Skew ratio:         {_f(candidate.skew_ratio, 3)}  "
            f"(> 1.10 = put premium elevated)"
            if candidate.skew_ratio is not None
            else "  Skew ratio:         n/a"
        ),
        (
            f"  Skew diff:          {_f(candidate.skew_diff, 3)}  "
            f"(> 0.03 = IV_otm significantly above ATM)"
            if candidate.skew_diff is not None
            else "  Skew diff:          n/a"
        ),
        f"  Skew bonus:         {'YES (+1 × weight)' if candidate.skew_bonus > 0 else 'no'}",
        "",
        "[bold underline]Scoring components[/bold underline]",
        f"  ROC:                {_pct(candidate.mid / candidate.strike if candidate.mid and candidate.strike else None)}  "
        f"(mid / strike, per expiry)",
        f"  Annualised ROC:     {_pct(candidate.annualized_roc)}",
        f"  Spread %:           {_pct(candidate.spread_pct)}",
        f"  Liquidity factor:   {_f(candidate.liquidity_factor, 3)}  "
        f"(1.0 = tight spread, 0.2 = very wide)",
        f"  Event flag:         {candidate.event_flag or 'none'}",
        f"  Event multiplier:   {_f(candidate.event_multiplier, 3)}",
        "",
        f"  [bold green]Final score:        {candidate.final_score:.4f}[/bold green]",
    ]

    # Valuation context
    if candidate.ibkr_fair_value or candidate.morningstar_fair_value:
        lines += [
            "",
            "[bold underline]Valuation context[/bold underline]",
        ]
        if candidate.ibkr_fair_value:
            upside = (candidate.ibkr_fair_value / spot - 1) * 100 if spot else 0
            lines.append(
                f"  IBKR fair value:    {_money(candidate.ibkr_fair_value)}"
                f"  ({upside:+.1f}% vs spot)"
            )
        if candidate.morningstar_fair_value:
            upside = (candidate.morningstar_fair_value / spot - 1) * 100 if spot else 0
            lines.append(
                f"  MS fair value:      {_money(candidate.morningstar_fair_value)}"
                f"  ({upside:+.1f}% vs spot)"
            )

    # Warnings
    if candidate.warnings:
        lines += [
            "",
            f"  [yellow]⚠  Warnings: {', '.join(candidate.warnings)}[/yellow]",
        ]

    title = (
        f"[bold]Explain: {candidate.symbol} "
        f"${candidate.strike:.0f}P  {candidate.expiry}[/bold]"
    )
    console.print(Panel("\n".join(lines), title=title, expand=False, padding=(1, 2)))
