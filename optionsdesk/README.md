# optionsdesk — multi-agent wheel + straddle system

Live by default (see `config.yaml`): `mode: live`, IBKR port 7496, every order needs a manual confirmation. `--simulated` forces an offline paper demo.
Agents are asyncio modules that talk only through a typed event bus and a SQLite store.
A FastAPI service exposes REST + Server-Sent Events; the Streamlit dashboard reads that API and never talks to IBKR.

```
IBKR (7497 paper) <-> MarketDataAgent -> [event bus] -> IVAgent -> WheelAgent / StraddleAgent -> Signal
Signal -> RiskAgent (approve / flag / block) -> ExecutionAgent (paper default, kill switch) -> OrderUpdate (fills)
fills -> WheelAgent / StraddleAgent update state      BacktestAgent, NarrativeAgent run beside them
Orchestrator supervises every agent (restart with backoff, health); FastAPI + SSE -> Streamlit dashboard
```

Every signal carries a correlation id from the market-data snapshot that produced it, through the risk decision and
order, into the JSON logs and the journal.

## Setup

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest tests/optionsdesk -q     # all tests are mocked; no orders can be placed
```

## Run it

Backend (orchestrator + all agents + API on http://127.0.0.1:8000):

```powershell
python -m optionsdesk serve --i-understand-live   # live: IBKR data + orders on 7496, each order confirmed by you in the dashboard
python -m optionsdesk serve --simulated           # offline paper demo (separate behaviour, forces mode=paper)
```

Dashboard (new, port 8502; your existing wheel dashboard on 8501 is untouched):

```powershell
python -m optionsdesk dashboard
```

VS Code tasks with the same commands are in `.vscode/tasks.json`. Interactive API docs: http://127.0.0.1:8000/docs.

### Starting individual agents

`python -m optionsdesk serve --agents market_data,iv,risk` runs only those agents in this process
(`market_data, iv, wheel, straddle, risk, execution, backtest, narrative`), or set `agents:` in `config.yaml`.
Agents share an in-process bus, so run the ones that depend on each other together
(the strategy agents need `market_data` + `iv`; `execution` needs `risk`).

### Other commands

```powershell
python -m optionsdesk kill-switch on --reason "manual"   # also: off. Checked before every order, even while serving
python -m optionsdesk backtest wheel --tickers AAPL,MSFT --start 2023-01-01        # add --simulated for synthetic prices
```

## Paper trading end to end

1. TWS or IB Gateway: log in to a **paper** account, enable API (File > Global Configuration > API > Settings), socket port 7497.
2. `config.yaml`: keep `mode: paper`; set `data_source: ibkr` (default) and, to send paper orders to IBKR, `execution_backend: ibkr`.
   Market data type defaults to delayed (`ibkr.market_data_type: 3`); set `1` if you have an OPRA subscription.
3. `python -m optionsdesk serve`, then `python -m optionsdesk dashboard`.
4. Watch Overview > Agent status. Signals appear in Journal with the reasoning; paper orders auto-execute
   (`auto_execute_paper: false` makes them wait for confirmation on Orders & controls).
5. Engage the kill switch from Orders & controls, the API, or the CLI at any time.

Safety rails: orders only reach IBKR when `execution_backend: ibkr`; the IBKR broker refuses to trade if the connected account
type (paper accounts start with `DU`) disagrees with `mode`; `mode: live` additionally requires `allow_live: true`, the
`--i-understand-live` start flag, and a manual confirmation (with live acknowledgement) per order. Set `api_token` to require
`X-API-Token` on all mutating endpoints, and keep the API bound to localhost.

## Configuration

One file: `config.yaml` (override path with `OPTIONSDESK_CONFIG`; any key via env, e.g. `OPTIONSDESK_RISK__MAX_NET_DELTA=300`).
Sections: tickers, `ibkr`, `wheel` (target deltas 0.25-0.30, DTE 30-45, roll at 21 DTE, 50% profit take, no calls below cost basis),
`straddle` (long/short, wings, IV-rank thresholds, exits), `risk` (limits), `backtest` (commission, slippage, IV proxy).
Defaults are in `optionsdesk/settings.py`. `earnings_file` (`symbol,earnings_date`) feeds straddle event filters.

## Strategies

- **Wheel**: `SEARCHING -> CSP_OPEN -> ASSIGNED -> CC_OPEN -> CALLED_AWAY -> SEARCHING`. Entry needs liquidity (spread, open interest) and the IV filter.
  Management: close at 50% of premium, roll at 21 DTE or when tested. Cost basis is net of premium; calls are never sold below it unless enabled.
- **Straddle**: long (low IV rank, event ahead, IV not rich versus realized) or short (high IV rank, no event before expiry), optional wings,
  profit target / stop on premium / time exit. Breakevens, max loss and net Greeks are computed for every candidate.

## Backtests

Model-priced (see below), commissions and slippage included, signals on bar *t* fill on bar *t+1*, indicators use only past data
(tests alter the future and assert the past is unchanged). Outputs: equity, drawdown, win rate, Sharpe/Sortino, annualized return
on capital and on deployed capital. Run them from the dashboard's Backtest lab (runs in a worker thread) or the CLI.

## Not verified / known limits

I could not run against a live IBKR session, so these are untested outside mocks:

- IBKR behaviour: connection/reconnect handling, pacing limits, streaming option ticks in batches, `modelGreeks` and open-interest availability
  (liquidity filters reject quotes with unknown open interest, so delayed data may yield few signals), `OPTION_IMPLIED_VOLATILITY` history used to
  bootstrap IV rank, multi-leg rolls sent as separate limit orders (a half-filled roll is possible), order fill tracking, the account-type guard.
- Assignment and expiry are inferred from the stored close on the expiry date, not reconciled against broker positions.
- Backtests use modelled option prices (Black-Scholes on trailing realized vol x 1.10, flat skew, synthetic spreads); real historical chains would be
  needed to validate edge. The risk gate is not applied inside backtests; cash-secured collateral is.
- Live data quality is tagged live / delayed / estimated, but the live path has not been observed with real ticks.
- `earnings_calendar.csv` dates are all in the past as of today; long straddles require an upcoming event, so update it.
- Portfolio value is `starting_capital + realized + unrealized`; it is not read from the IBKR account.
- Streamlit refresh is polling via fragments plus an SSE feed for the event list, not a push-rendered UI.
- Existing code is unchanged except `config.py` (dip bot port 7496 -> 7497) and `pyproject.toml`/`requirements.txt`.
  `tests/test_valuation.py::test_no_shares_returns_insufficient` was already failing before this work.
