# buy_the_sauce1 — Investment-Grade Dip-Buying & Swing-Trading System

An automated trading system that scans a watchlist of ~50 fundamentally
strong US equities daily, identifies genuine dip-buying opportunities using
a multi-signal scoring model with **ML ensemble predictions**, and manages
the full trade lifecycle — from **scaled entry** through **trailing stops**
and **partial profit-taking** — through Interactive Brokers (IBKR) via
**ib_insync**.

---

## Architecture

```
run.py                  ← CLI entry point (once / scheduled)
 └─ trader.py           ← Orchestration (screen → detect → size → execute → manage)
     ├─ screener.py     ← Fundamental quality filter (FCF, ROIC, D/E, accruals)
     ├─ dip_detector.py ← Technical dip-scoring (RSI, MA50, MA200, 52-wk range)
     ├─ ml_predictor.py ← Ensemble ML (LightGBM + XGBoost + LogReg), multi-horizon
     ├─ risk_manager.py ← Position sizing, stops, trailing stops, partials, scaled entry
     └─ broker.py       ← IBKR order execution (bracket, scaled, partial sell, stop update)

dashboard.py            ← Live Dash-based web dashboard (5 tabs)
backtest_ml.py          ← Walk-forward ML ensemble backtest with trailing stop simulation
config.py               ← All tuneable parameters (single source of truth)
watchlist.py            ← ~50 tickers to scan
edgar.py                ← SEC EDGAR fundamentals + price history (IBKR / yfinance / Stooq)
```

---

## Features

### ML Ensemble Predictor (`ml_predictor.py`)

- **~77 features**: Technical, calendar, price-pattern, macro (SPY/QQQ/TLT/GLD/VIX),
  HMM regime, Monte Carlo DCF, fundamental (ROIC, FCF margin, accruals, cash conversion),
  cross-sectional peer rankings.
- **Multi-horizon forecasts**: 1W (5d), 1M (21d), 1Y (252d) — each with independent AUROC.
- **Alpha-vs-SPY labels**: Model predicts excess return over SPY, not raw direction.
- **Ensemble**: LightGBM (50%) + XGBoost (35%) + Logistic Regression (15%), soft-voting.
- **Swing sell metrics**: ATR stops/targets, RSI/Bollinger/MACD signals, composite sell score.

### Risk Management (`risk_manager.py`)

| Feature | Description |
|---------|-------------|
| **ATR-based stops** | Stop = entry − 2×ATR, clamped to 3–12% |
| **ATR-based targets** | Target = entry + 3×ATR (≥1.5:1 R/R minimum) |
| **Fixed-risk sizing** | Risk 1% of equity per trade |
| **Kelly criterion** | ¼-Kelly fractional sizing (conservative) |
| **Max-position cap** | 5% of equity per stock |
| **Portfolio-level sizing** | Greedy allocation with capital budget tracking |
| **Volatility scaling** | Shrink in HIGH vol, expand in LOW vol |
| **Sector correlation** | 25% haircut for same-sector additions |
| **Trailing stop** | 3-stage adaptive: breakeven → profit-lock → tight trail |
| **Partial exits** | Sell ⅓ at +2×ATR, another ⅓ at +3×ATR; remainder rides trail |
| **Time stop** | Exit at market after 30 calendar days (configurable) |
| **Scaled entry** | 3-tranche buy ladder at −0/1/2×ATR with RSI abort & expiry |

### Execution (`broker.py`, `trader.py`)

- **Buy-side**: Scaled entry (buy ladder) with per-tranche limit orders + shared protective stop.
- **Sell-side**: Trailing stop updates pushed to IBKR, partial exit orders, market close for time stops.
- **Position management loop**: `manage_open_positions()` runs every scan cycle — updates trailing stops, evaluates partial exits, checks time stops, and aborts stale scaled entry tranches.
- **State persistence**: Position state serialised to `results/position_state.json` — survives restarts.
- **Safety**: Pre-trade checks (duplicate guard, position-size cap, equity check), kill switch.

### Dashboard (`dashboard.py`)

5-tab interactive web dashboard (Dash + Plotly):

1. **📡 Screener** — Live dip detector + ML signals (1W/1M/1Y), fundamentals, swing sell metrics.
2. **💰 Risk & Capital** — Portfolio sizing, capital allocation pie, R/R bars, scaled entry ladder, exit strategy parameters, live position management state.
3. **📈 Back-test** — Equity curve + trade log from walk-forward ML backtest.
4. **🗂 Screen Log** — Historical screener runs.
5. **📒 My Trades** — Manual transaction journal with P&L tracking.

### Backtesting (`backtest_ml.py`)

- Walk-forward with expanding training window, re-training every 21 bars.
- Bar-by-bar simulation of trailing stops, partial exits, and time stops.
- Per-ticker and portfolio-level metrics: AUROC, Win%, Sharpe, Max Drawdown, Precision@10.
- SPY buy-and-hold benchmark comparison.

---

## Requirements

- Python 3.9+
- IBKR account (paper or live) with **TWS** or **IB Gateway** running locally
- The packages listed in `requirements.txt`

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/Shouryatsh/buy_the_sauce1.git
cd buy_the_sauce1

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start TWS or IB Gateway
#    Paper-trading port: 7497   (TWS)
#    Live-trading port:  7496   (TWS) | 4002 (IB Gateway)
```

---

## Configuration

Edit **`config.py`** to adjust. Key parameters:

### Risk Management

| Setting | Default | Description |
|---------|---------|-------------|
| `USE_ATR_STOPS` | `True` | ATR-based stops (vs fixed %) |
| `ATR_STOP_MULTIPLIER` | `2.0` | Stop = entry − 2×ATR |
| `ATR_TARGET_MULTIPLIER` | `3.0` | Target = entry + 3×ATR |
| `ATR_MIN_STOP_PCT` | `0.03` | Hard floor: stop ≥ 3% |
| `ATR_MAX_STOP_PCT` | `0.12` | Hard cap: stop ≤ 12% |
| `RISK_PER_TRADE_PCT` | `0.01` | Risk 1% of equity per trade |
| `MAX_POSITIONS` | `10` | Max concurrent positions |
| `MAX_POSITION_PCT` | `0.05` | Max 5% of equity per stock |
| `KELLY_FRACTION` | `0.25` | ¼-Kelly (0 = disabled) |
| `MAX_CAPITAL_DEPLOYED_PCT` | `0.80` | Deploy at most 80% of capital |

### Exit Strategy

| Setting | Default | Description |
|---------|---------|-------------|
| `TRAILING_STOP_ENABLED` | `True` | 3-stage adaptive trailing stop |
| `PARTIAL_EXIT_ENABLED` | `True` | Staged profit-taking |
| `PARTIAL_EXIT_1_FRACTION` | `0.33` | Sell ⅓ at +2×ATR |
| `PARTIAL_EXIT_2_FRACTION` | `0.33` | Sell ⅓ at +3×ATR |
| `TIME_STOP_ENABLED` | `True` | Exit after max hold period |
| `TIME_STOP_DAYS` | `30` | Max holding period |

### Scaled Entry

| Setting | Default | Description |
|---------|---------|-------------|
| `SCALED_ENTRY_ENABLED` | `True` | Buy ladder (3 tranches) |
| `SCALED_ENTRY_FRACTIONS` | `(0.40, 0.35, 0.25)` | T1/T2/T3 qty split |
| `SCALED_ENTRY_ATR_OFFSETS` | `(0.0, 1.0, 2.0)` | Price offsets below T1 |
| `SCALED_ENTRY_RSI_ABORT_LEVEL` | `50.0` | Cancel T2/T3 if RSI > 50 |
| `SCALED_ENTRY_EXPIRY_DAYS` | `5` | Unfilled limits expire after 5d |

Edit **`watchlist.py`** to customise the list of tickers scanned.

---

## Usage

### Dry run (no orders — review signals)

```bash
python run.py --dry-run
```

### Live scan (connects to IBKR, places orders, manages positions)

```bash
python run.py
```

### Scheduled daily scan at 09:45 US-Eastern

```bash
python run.py --schedule 09:45
```

### Dashboard

```bash
python dashboard.py              # opens on http://127.0.0.1:8050
python dashboard.py --port 8080  # custom port
```

### ML Backtest

```bash
python backtest_ml.py                         # full watchlist, 5y, 1M horizon
python backtest_ml.py --tickers AAPL MSFT     # specific tickers
python backtest_ml.py --horizon 5             # 1W swing
python backtest_ml.py --trades                # print per-ticker trade log
```

### Options

```
--dry-run             Evaluate signals and sizes only — no orders placed
--schedule HH:MM      Run every day at this time (24-hour, US-Eastern)
--log-level LEVEL     DEBUG | INFO | WARNING | ERROR  (default: INFO)
```

Logs are written to both stdout and `trading.log`.

---

## How It Works

### 1. Fundamental screen (`screener.py`)

Each ticker is checked against live EDGAR data:

| Criterion | Default threshold |
|-----------|-------------------|
| Free Cash Flow | Strictly positive, growing 2+ of last 3 years |
| FCF Yield | ≥ 1.5% |
| ROE | ≥ 10% |
| Capex / FCF | ≤ 75% |
| D/E | ≤ 3.0× (≤ 15.0× for financials) |
| P/E | 0 < P/E ≤ 60 |
| Net Margin | ≥ 5% |

### 2. Dip scoring (`dip_detector.py`)

Each passing stock is scored 0–4:

| Signal | Condition | Score |
|--------|-----------|-------|
| RSI oversold | RSI-14 < 35 | +1 |
| Below MA-50 | Price ≥ 5% below 50d MA | +1 |
| Below MA-200 | Price below 200d MA | +1 |
| 52-week range | Price in bottom 25% | +1 |

Score ≥ 3 → dip candidate.

### 3. ML ensemble (`ml_predictor.py`)

If dip detected, the ensemble predicts alpha-vs-SPY probability across three horizons.
ML must confirm bullish direction (`ML_GATE_BUY_SIGNAL = True`) for a full BUY signal.

### 4. Position sizing (`risk_manager.py`)

Portfolio-level simultaneous sizing: candidates ranked by signal quality, sized with the
most conservative constraint (fixed-risk, Kelly, cap, remaining cash), then scaled by
volatility regime and sector correlation.

### 5. Execution lifecycle (`trader.py`, `broker.py`)

```
BUY ENTRY                           SELL EXIT
─────────                           ─────────
T1: limit @ current price ──────┐   Trailing stop (3-stage adaptive)
T2: limit @ −1×ATR ────────────┤   ├── Stage 1: breakeven lock (+1×ATR)
T3: limit @ −2×ATR ────────────┤   ├── Stage 2: profit lock (+2×ATR)
  ↕ RSI abort / 5d expiry      │   ├── Stage 3: tight trail (+3×ATR)
                                │   │
  Protective stop @ entry−2×ATR ┘   ├── Partial exit ⅓ @ +2×ATR
                                    ├── Partial exit ⅓ @ +3×ATR
                                    ├── Remainder rides trailing stop
                                    └── Time stop @ 30 calendar days
```

Every scan cycle, `manage_open_positions()` evaluates all tracked positions and
pushes trailing stop updates, partial exit orders, and time-stop closes to IBKR.

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

**Test suites:**
- `tests/test_risk_manager.py` — 50 tests: order sizing, scaled entry, trailing stops, partial exits, time stops.
- `tests/test_trader.py` — 18 tests: position state persistence, register/unregister, manage_open_positions dry-run.
- `tests/test_screener.py` — Fundamental filter tests.
- `tests/test_dip_detector.py` — Dip scoring tests.

---

## File Descriptions

| File | Purpose |
|------|---------|
| `run.py` | CLI entry point (once / scheduled) |
| `trader.py` | Pipeline orchestration + position management loop |
| `screener.py` | Fundamental quality screen (EDGAR-based) |
| `dip_detector.py` | Technical dip scoring (RSI, MA, 52w range) |
| `ml_predictor.py` | ML ensemble + swing sell metrics |
| `risk_manager.py` | Position sizing, stops, trailing, partials, scaled entry |
| `broker.py` | IBKR order execution (bracket, scaled, partial, close) |
| `config.py` | All tuneable parameters |
| `watchlist.py` | Ticker universe (~50 stocks) |
| `edgar.py` | SEC EDGAR data fetching + price history |
| `dashboard.py` | Interactive web dashboard (Dash + Plotly) |
| `backtest_ml.py` | Walk-forward ML backtest with trailing stop sim |
| `backtest.py` | Simpler non-ML backtest |
| `results/` | Backtest CSVs, screen logs, position state JSON |

---

## Disclaimer

This software is provided for educational and research purposes only.
Trading involves substantial risk of loss. Past performance does not
guarantee future results. Always paper-trade and review signals before
connecting to a live account.