"""CLI: python -m optionsdesk serve | kill-switch | backtest"""

from __future__ import annotations

import json
import os
from datetime import date
from typing import Optional

import typer

app = typer.Typer(help="optionsdesk: multi-agent wheel + straddle system (paper by default).", no_args_is_help=True)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind address (keep local; the API can place/confirm orders)."),
    port: int = typer.Option(8000),
    agents: Optional[str] = typer.Option(None, help="Comma list: market_data,iv,wheel,straddle,risk,execution,backtest,narrative"),
    simulated: bool = typer.Option(False, "--simulated", help="Offline: simulated market data and simulated broker."),
    config: Optional[str] = typer.Option(None, help="Path to config.yaml"),
    i_understand_live: bool = typer.Option(False, "--i-understand-live", help="Required when mode=live."),
) -> None:
    """Start the orchestrator (all selected agents) and the REST/SSE API."""
    import uvicorn
    from .settings import Mode, Settings
    if config:
        os.environ["OPTIONSDESK_CONFIG"] = config
    if agents:
        os.environ["OPTIONSDESK_AGENTS"] = json.dumps([a.strip() for a in agents.split(",") if a.strip()])
    if simulated:
        os.environ["OPTIONSDESK_DATA_SOURCE"] = "simulated"
        os.environ["OPTIONSDESK_EXECUTION_BACKEND"] = "simulated"
    s = Settings()
    if s.mode is Mode.LIVE and not i_understand_live:
        typer.secho("mode=live: re-run with --i-understand-live. Live orders will still need manual confirmation.",
                    fg=typer.colors.RED)
        raise typer.Exit(2)
    typer.secho(f"mode={s.mode.value} data_source={s.data_source} execution={s.execution_backend} "
                f"port={s.ibkr_port} agents={s.agents}", fg=typer.colors.GREEN if s.mode is Mode.PAPER else typer.colors.RED)
    uvicorn.run("optionsdesk.api.app:app_factory", factory=True, host=host, port=port, log_level="warning")


@app.command("kill-switch")
def kill_switch(state: str = typer.Argument(..., help="on | off"), reason: str = typer.Option("cli")) -> None:
    """Engage/release the global kill switch (works while the server runs: it is checked before every order)."""
    from .db import Database
    from .settings import Settings
    if state not in ("on", "off"):
        raise typer.BadParameter("state must be 'on' or 'off'")
    Database(Settings().db_path).set_kill_switch(state == "on", reason)
    typer.echo(f"kill switch {'ENGAGED' if state == 'on' else 'released'}")


@app.command()
def backtest(
    strategy: str = typer.Argument(..., help="wheel | straddle_long | straddle_short"),
    tickers: str = typer.Option(..., help="Comma list"),
    start: str = typer.Option(...),
    end: str = typer.Option(str(date.today())),
    simulated: bool = typer.Option(False, "--simulated", help="Use synthetic prices."),
) -> None:
    """Run a model-priced backtest headless and print the metrics."""
    from .agents.backtest import simulated_loader, yfinance_loader
    from .backtest import run_straddle_backtest, run_wheel_backtest
    from .earnings import load_earnings
    from .settings import Settings
    s = Settings()
    t = [x.strip().upper() for x in tickers.split(",")]
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    prices = (simulated_loader if simulated else yfinance_loader)(t, d0, d1)
    ev = load_earnings(s.earnings_file)
    res = run_wheel_backtest(prices, d0, d1, s, ev) if strategy == "wheel" else \
        run_straddle_backtest(prices, d0, d1, s, ev, mode=strategy.split("_")[1])
    typer.echo(json.dumps(res.metrics, indent=2, default=str))


if __name__ == "__main__":
    app()
