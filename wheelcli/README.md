# wheelcli — Wheel-Strategy CSP Scanner (IBKR)

A modular, read-only Python CLI tool that connects to Interactive Brokers,
scans an option universe for attractive cash-secured put (CSP) candidates,
and ranks them by a composite score incorporating annualised ROC, liquidity,
put-skew bonus, and event-risk discounts.

---

## Quick start

```bash
# 1. Requires Python 3.11+
python3.11 -m venv .venv-wheel
source .venv-wheel/bin/activate

# 2. Install (editable)
pip install -e ".[dev]"

# 3. Create starter data files
wheel init

# 4. Edit universe.csv with your tickers (and optional fair values)
# Edit earnings_calendar.csv with upcoming earnings dates
# Edit macro_events.csv with FOMC / CPI dates

# 5. Start IB Gateway or TWS (paper account, port 7497)
# Enable: File → Global Configuration → API → Settings → Enable ActiveX and Socket Clients

# 6. Scan
wheel scan

# 7. Explain the best put for one symbol
wheel explain AAPL
```

---

## IBKR setup

### Port reference

| Session      | Port  |
|-------------|-------|
| TWS paper   | 7497  |
| TWS live    | 7496  |
| IB Gateway live | 4002 |

### Enabling the API

In TWS or IB Gateway:
**File → Global Configuration → API → Settings**

- [x] Enable ActiveX and Socket Clients
- Socket port: `7497` (paper) or `7496` (live)
- [x] Allow connections from localhost only
- Master API client ID: leave empty (wheelcli uses client ID 10 by default)

### Trusted IPs

If running wheelcli on the same machine as TWS, no additional IP configuration
is needed.  For a remote machine, add its IP to the trusted list.

---

## Configuration

All config fields can be overridden via **environment variables** (prefix `WHEEL_`)
or a **`.env` file** in your working directory.

### Key environment variables

| Variable | Default | Description |
|---|---|---|
| `WHEEL_IBKR_HOST` | `127.0.0.1` | IB Gateway / TWS host |
| `WHEEL_IBKR_PORT` | `7497` | API port |
| `WHEEL_IBKR_CLIENT_ID` | `10` | Must be unique per simultaneous connection |
| `WHEEL_MAX_DTE` | `45` | Maximum days to expiry |
| `WHEEL_WEEKLY_ONLY` | `true` | Only Friday expirations |
| `WHEEL_SIGMA_THRESHOLD` | `2.0` | Min sigma distance to pass filter |
| `WHEEL_MAX_DELTA` | `0.05` | Max absolute put delta to pass filter |
| `WHEEL_EARNINGS_PENALTY` | `0.6` | Score multiplier when trade crosses earnings |
| `WHEEL_MACRO_PENALTY` | `0.85` | Score multiplier when trade crosses a macro event |
| `WHEEL_UNKNOWN_EARNINGS_PENALTY` | `0.9` | Penalty for symbols absent from the earnings calendar |
| `WHEEL_SKEW_RATIO_THRESHOLD` | `1.10` | Skew ratio for bonus |
| `WHEEL_SPREAD_CAP` | `0.10` | Max bid/ask spread % for full liquidity credit |
| `WHEEL_CACHE_TTL` | `900` | Cache time-to-live in seconds (default 15 min) |
| `WHEEL_UNIVERSE_FILE` | `universe.csv` | Path to universe CSV |
| `WHEEL_EARNINGS_FILE` | `earnings_calendar.csv` | Path to earnings calendar |
| `WHEEL_MACRO_EVENTS_FILE` | `macro_events.csv` | Path to macro events |

### Example `.env`

```env
WHEEL_IBKR_PORT=7497
WHEEL_MAX_DTE=30
WHEEL_SIGMA_THRESHOLD=1.8
WHEEL_MAX_DELTA=0.05
WHEEL_WEEKLY_ONLY=true
```

---

## Data files

### `universe.csv`

One ticker per row.  The valuation columns are optional — leave blank if not
available.  Fair values are displayed in the `explain` output but do not
affect scoring.

```csv
symbol,ibkr_fair_value,morningstar_fair_value
AAPL,195.00,210.00
MSFT,,
NVDA,130.00,
```

### `earnings_calendar.csv`

Maintain this file regularly — stale earnings dates reduce the accuracy of
the event-risk discount.  Use ISO-8601 dates (`YYYY-MM-DD`).

```csv
symbol,earnings_date
AAPL,2026-07-31
MSFT,2026-07-28
```

**Tip:** You can have multiple rows for the same symbol (e.g., future quarters):

```csv
AAPL,2026-07-31
AAPL,2026-10-30
```

**What happens if a symbol is missing?**
The scanner treats it as "unknown earnings" and applies the
`unknown_earnings_penalty` (default 0.9×) to its final score.

### `macro_events.csv`

Add FOMC meetings, CPI releases, NFP dates, and any other events that
could cause a significant market move.  Dates within the option's holding
period trigger the `macro_penalty` (default 0.85×).

```csv
date,description
2026-06-11,FOMC Meeting (June)
2026-07-10,CPI Release (June)
```

---

## Scoring formula

```
roc              = mid_premium / strike
annualized_roc   = roc × (365 / DTE)

spread_pct       = (ask − bid) / mid
liquidity_factor = clamp(1 − spread_pct / spread_cap,  min_lf, 1.0)

final_score      = annualized_roc
                   × liquidity_factor
                   × (1 + skew_bonus_weight × skew_bonus)
                   × event_multiplier
```

**Sigma distance** (used only for filtering, not scoring):

```
sigma_distance = (spot − strike) / (spot × IV × √T)
```

A put passes the filter if `|delta| ≤ max_delta` OR `sigma_distance ≥ sigma_threshold`.

---

## CLI reference

```
wheel --help
wheel init --help
wheel scan --help
wheel explain --help
```

Common scan flags:

```bash
wheel scan \
  --max-dte 30 \
  --weekly-only \
  --sigma-threshold 1.8 \
  --max-delta 0.05 \
  --max-candidates 25 \
  --no-cache          # bypass disk cache
```

---

## Running tests

```bash
pytest                          # all tests
pytest -v wheelcli/tests/       # verbose
pytest --cov=wheelcli           # with coverage
```

The test suite covers all analytics modules without requiring an IBKR
connection (pure-function unit tests with mock earnings providers).

---

## Architecture

```
wheelcli/
├── cli.py             Typer commands: init, scan, explain
├── config.py          Pydantic-Settings WheelConfig (env-overridable)
├── models.py          Pydantic models: UniverseEntry, OptionContract, CandidatePut
├── data/
│   ├── ibkr.py        IBKRClient — market data via ib_insync
│   ├── cache.py       WheelCache — diskcache (SQLite) with TTL
│   └── earnings.py    EarningsProvider ABC + CSVEarningsProvider
├── analytics/
│   ├── sigma.py       compute_sigma_distance, passes_filter
│   ├── skew.py        find_atm_iv, compute_skew, compute_skew_bonus
│   ├── events.py      compute_event_multiplier
│   └── scoring.py     score_candidate (composite score)
├── reports/
│   └── tables.py      Rich terminal tables + explain panel
└── tests/
    ├── test_sigma.py
    ├── test_skew.py
    └── test_events.py
```

---

## Security notes

- Never hardcode API keys or credentials.  Use environment variables.
- The tool is **read-only** — it never places or modifies IBKR orders.
- Set `Allow connections from localhost only` in TWS/Gateway to minimise
  attack surface.
- The disk cache (`.wheelcache/`) contains market data snapshots.
  It does not store credentials.
