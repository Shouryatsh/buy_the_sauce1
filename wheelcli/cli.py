"""
wheelcli/cli.py — Typer CLI entry point.

Commands
--------
  wheel init      Create starter data files (universe.csv, earnings_calendar.csv,
                  macro_events.csv) if they do not already exist.

  wheel scan      Connect to IBKR and scan the universe for CSP candidates.
                  Outputs a Rich table ranked by final_score (descending).

  wheel explain   Show a detailed scoring breakdown for the best put candidate
                  on a single symbol.

Configuration
-------------
All scan parameters can be overridden via env vars (prefix WHEEL_) or a .env
file in the working directory.  Run ``wheel scan --help`` for the full list.

Example
-------
  wheel init
  wheel scan --max-dte 30 --weekly-only
  wheel explain AAPL
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .analytics.scoring import score_candidate
from .analytics.skew import compute_skew
from .config import WheelConfig
from .data.cache import WheelCache
from .data.earnings import CSVEarningsProvider, load_macro_events
from .data.ibkr import IBKRClient
from .models import CandidatePut, UniverseEntry
from .reports.tables import console, render_candidates_table, render_explain

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="wheel",
    help="Read-only wheel-strategy CSP scanner backed by Interactive Brokers.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)

_log = logging.getLogger("wheelcli")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        level=level,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UNIVERSE_HEADER = "symbol,ibkr_fair_value,morningstar_fair_value"
_EARNINGS_HEADER = "symbol,earnings_date"
_MACRO_HEADER = "date,description"

_SAMPLE_UNIVERSE = """\
symbol,ibkr_fair_value,morningstar_fair_value
AAPL,,
MSFT,,
NVDA,,
AMZN,,
GOOGL,,
META,,
V,,
MA,,
JPM,,
KO,,
"""

_SAMPLE_EARNINGS = """\
symbol,earnings_date
AAPL,2026-07-31
MSFT,2026-07-28
NVDA,2026-08-20
AMZN,2026-08-01
GOOGL,2026-07-29
META,2026-07-30
V,2026-07-22
MA,2026-07-24
JPM,2026-07-15
KO,2026-07-22
"""

_SAMPLE_MACRO = """\
date,description
2026-06-11,FOMC Meeting
2026-07-30,FOMC Meeting
2026-09-16,FOMC Meeting
2026-11-05,FOMC Meeting
2026-06-12,CPI Release
2026-07-10,CPI Release
2026-08-12,CPI Release
"""


def _write_if_missing(path: str, content: str, label: str) -> None:
    p = Path(path)
    if p.exists():
        console.print(f"  [dim]Already exists — skipping:[/dim] {path}")
    else:
        p.write_text(content, encoding="utf-8")
        console.print(f"  [green]Created:[/green] {path}  ({label})")


def _load_universe(path: str) -> list[UniverseEntry]:
    entries = UniverseEntry.load_csv(path)
    if not entries:
        console.print(
            f"[red]Universe file '{path}' is empty or missing.\n"
            "Run [bold]wheel init[/bold] first, then add your tickers.[/red]"
        )
    return entries


def _scan_symbol(
    symbol: str,
    entry: UniverseEntry,
    ibkr: IBKRClient,
    cache: WheelCache,
    earnings_provider: CSVEarningsProvider,
    macro_events: list,
    config: WheelConfig,
    max_dte: int,
    weekly_only: bool,
    no_cache: bool,
) -> list[CandidatePut]:
    """Run the full scan pipeline for one symbol. Returns all passing candidates."""
    candidates: list[CandidatePut] = []

    # ── Spot price ──────────────────────────────────────────────────────────
    spot_key = f"spot:{symbol}"
    spot: Optional[float] = None if no_cache else cache.get(spot_key)
    if spot is None:
        spot = ibkr.get_underlying_price(symbol)
        if spot:
            cache.set(spot_key, spot)

    if not spot:
        console.print(f"  [yellow]⚠ {symbol}: no price available, skipping.[/yellow]")
        return candidates

    # ── Expirations ─────────────────────────────────────────────────────────
    exp_key = f"exps:{symbol}:{max_dte}:{weekly_only}"
    exps: Optional[list] = None if no_cache else cache.get(exp_key)
    if exps is None:
        exps = ibkr.get_option_expirations(symbol, max_dte, weekly_only)
        cache.set(exp_key, exps)

    if not exps:
        console.print(f"  [yellow]⚠ {symbol}: no expirations in range.[/yellow]")
        return candidates

    # ── Per-expiry scan ─────────────────────────────────────────────────────
    for exp_str in exps:
        # Strikes
        strikes_key = f"strikes:{symbol}:{exp_str}"
        all_strikes: Optional[list] = None if no_cache else cache.get(strikes_key)
        if all_strikes is None:
            all_strikes = ibkr.get_strikes_for_expiry(symbol, exp_str)
            cache.set(strikes_key, all_strikes)

        # Filter to configured range around spot
        strikes = [
            K
            for K in all_strikes
            if config.strike_low_frac * spot <= K <= config.strike_high_frac * spot
        ]
        if not strikes:
            continue

        # Option market data
        chain_key = f"chain:{symbol}:{exp_str}"
        contracts: Optional[list] = None if no_cache else cache.get(chain_key)
        if contracts is None:
            contracts = ibkr.get_put_contracts(symbol, exp_str, spot, strikes)
            cache.set(chain_key, contracts)

        if not contracts:
            continue

        # Skew for this expiry
        skew_data = compute_skew(contracts, spot)

        # Score each contract
        for contract in contracts:
            candidate = score_candidate(
                contract=contract,
                spot=spot,
                skew_data=skew_data,
                earnings_provider=earnings_provider,
                macro_events=macro_events,
                config=config,
                universe_entry=entry,
            )
            if candidate is not None:
                candidates.append(candidate)

    return candidates


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def init(
    universe: str = typer.Option("universe.csv", help="Path for universe CSV"),
    earnings: str = typer.Option("earnings_calendar.csv", help="Path for earnings CSV"),
    macro: str = typer.Option("macro_events.csv", help="Path for macro events CSV"),
) -> None:
    """
    Create starter data files with sample tickers and placeholder values.

    Safe to re-run — existing files are never overwritten.
    """
    console.print("\n[bold]wheel init[/bold] — creating starter files\n")
    _write_if_missing(universe, _SAMPLE_UNIVERSE, "universe / watchlist")
    _write_if_missing(earnings, _SAMPLE_EARNINGS, "earnings calendar")
    _write_if_missing(macro, _SAMPLE_MACRO, "macro events calendar")
    console.print(
        "\n[dim]Edit these CSV files to add your tickers, fair values,"
        " and upcoming earnings dates.[/dim]\n"
    )


@app.command()
def scan(
    max_dte: int = typer.Option(45, help="Maximum days to expiry"),
    weekly_only: bool = typer.Option(True, help="Friday expirations only (weekly options)"),
    sigma_threshold: float = typer.Option(2.0, help="Minimum sigma distance"),
    max_delta: float = typer.Option(0.05, help="Maximum absolute put delta"),
    max_candidates: int = typer.Option(50, help="Maximum rows to display"),
    universe_file: str = typer.Option("universe.csv", help="Path to universe CSV"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass disk cache"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable debug logging"),
) -> None:
    """
    Scan the universe for wheel-strategy CSP candidates.

    Connects to IB Gateway / TWS, fetches live option chains, computes
    sigma distance, skew, event-risk discounts, and composite scores,
    then prints the top candidates ranked by score.
    """
    _setup_logging(verbose)

    cfg = WheelConfig(
        max_dte=max_dte,
        weekly_only=weekly_only,
        sigma_threshold=sigma_threshold,
        max_delta=max_delta,
        max_candidates=max_candidates,
        universe_file=universe_file,
    )

    entries = _load_universe(universe_file)
    if not entries:
        raise typer.Exit(1)

    earnings_provider = CSVEarningsProvider(cfg.earnings_file)
    macro_events = load_macro_events(cfg.macro_events_file)

    all_candidates: list[CandidatePut] = []

    try:
        with WheelCache(cfg.cache_dir, cfg.cache_ttl) as cache:
            with IBKRClient(cfg) as ibkr:
                console.print(
                    f"\n[bold]Scanning {len(entries)} symbols[/bold]"
                    f"  max_dte={max_dte}  weekly_only={weekly_only}"
                    f"  σ≥{sigma_threshold}  δ≤{max_delta}\n"
                )

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("{task.completed}/{task.total}"),
                    TimeElapsedColumn(),
                    console=console,
                    transient=True,
                ) as progress:
                    task = progress.add_task("Scanning…", total=len(entries))

                    for entry in entries:
                        progress.update(task, description=f"Scanning {entry.symbol}…")
                        try:
                            found = _scan_symbol(
                                symbol=entry.symbol,
                                entry=entry,
                                ibkr=ibkr,
                                cache=cache,
                                earnings_provider=earnings_provider,
                                macro_events=macro_events,
                                config=cfg,
                                max_dte=max_dte,
                                weekly_only=weekly_only,
                                no_cache=no_cache,
                            )
                            all_candidates.extend(found)
                        except Exception as exc:
                            console.print(
                                f"  [red]✗ {entry.symbol}: unexpected error — {exc}[/red]"
                            )
                            _log.debug("Error scanning %s", entry.symbol, exc_info=True)
                        finally:
                            progress.advance(task)

    except ConnectionRefusedError:
        console.print(
            f"\n[red bold]Cannot connect to IBKR at "
            f"{cfg.ibkr_host}:{cfg.ibkr_port}.[/red bold]\n"
            "Make sure IB Gateway or TWS is running and API connections are enabled.\n"
            "Paper account default port: [bold]7497[/bold]  "
            "Live account TWS: [bold]7496[/bold]  "
            "IB Gateway live: [bold]4002[/bold]\n"
        )
        raise typer.Exit(1)

    if not all_candidates:
        console.print("[yellow]No candidates found matching the criteria.[/yellow]")
        raise typer.Exit(0)

    all_candidates.sort(key=lambda c: c.final_score, reverse=True)
    top = all_candidates[:max_candidates]

    table = render_candidates_table(top)
    console.print()
    console.print(table)
    console.print(
        f"\n[dim]Scanned {len(entries)} symbols  |  "
        f"{len(all_candidates)} candidates passed filters  |  "
        f"Showing top {len(top)}[/dim]\n"
    )


@app.command()
def explain(
    symbol: str = typer.Argument(..., help="Ticker to explain (e.g. AAPL)"),
    max_dte: int = typer.Option(45, help="Maximum days to expiry for the lookup"),
    weekly_only: bool = typer.Option(True, help="Friday expirations only"),
    universe_file: str = typer.Option("universe.csv"),
    no_cache: bool = typer.Option(False, "--no-cache"),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    """
    Show a detailed scoring breakdown for the best CSP candidate on SYMBOL.

    Connects to IBKR, fetches the option chain for SYMBOL, scores all
    candidates, and prints an explanation panel for the top-ranked put.
    """
    _setup_logging(verbose)
    symbol = symbol.strip().upper()

    cfg = WheelConfig(max_dte=max_dte, weekly_only=weekly_only)
    entries = UniverseEntry.load_csv(universe_file)
    entry = next((e for e in entries if e.symbol == symbol), None)

    earnings_provider = CSVEarningsProvider(cfg.earnings_file)
    macro_events = load_macro_events(cfg.macro_events_file)

    try:
        with WheelCache(cfg.cache_dir, cfg.cache_ttl) as cache:
            with IBKRClient(cfg) as ibkr:
                console.print(f"\n[bold]Fetching data for {symbol}…[/bold]")

                candidates = _scan_symbol(
                    symbol=symbol,
                    entry=entry or UniverseEntry(symbol=symbol),
                    ibkr=ibkr,
                    cache=cache,
                    earnings_provider=earnings_provider,
                    macro_events=macro_events,
                    config=cfg,
                    max_dte=max_dte,
                    weekly_only=weekly_only,
                    no_cache=no_cache,
                )

    except ConnectionRefusedError:
        console.print(
            f"\n[red bold]Cannot connect to IBKR at {cfg.ibkr_host}:{cfg.ibkr_port}.[/red bold]"
        )
        raise typer.Exit(1)

    if not candidates:
        console.print(f"[yellow]No qualifying candidates found for {symbol}.[/yellow]")
        raise typer.Exit(0)

    candidates.sort(key=lambda c: c.final_score, reverse=True)
    best = candidates[0]

    # Re-fetch spot for the explain panel (may come from cache)
    with WheelCache(cfg.cache_dir, cfg.cache_ttl) as cache:
        spot: Optional[float] = cache.get(f"spot:{symbol}")
    if not spot:
        spot = 0.0  # fallback if cache expired between runs

    render_explain(best, spot)

    if len(candidates) > 1:
        console.print(
            f"\n[dim]{len(candidates) - 1} additional candidates not shown. "
            "Run [bold]wheel scan[/bold] to see the full ranked list.[/dim]\n"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
