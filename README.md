# buy_the_sauce1 — Automated Dip-Buying System (US Stocks / IBKR)

An automated trading system that scans a watchlist of ~50 fundamentally
strong US equities daily, identifies genuine dip-buying opportunities using
a multi-signal scoring model, and places bracket orders (entry + stop-loss +
take-profit) through Interactive Brokers (IBKR) via **ib_insync**.

---

## Architecture

```
run.py          ← CLI entry point (once / scheduled)
 └─ trader.py   ← Orchestration (screen → detect → size → execute)
     ├─ screener.py      ← Fundamental quality filter (P/E, margin, D/E, growth)
     ├─ dip_detector.py  ← Technical dip-scoring (RSI, MA50, MA200, 52-wk range)
     ├─ risk_manager.py  ← Position sizing, stop-loss, take-profit calculation
     └─ broker.py        ← IBKR bracket-order execution (ib_insync)

config.py     ← All tuneable parameters
watchlist.py  ← ~50 tickers to scan
```

---

## Requirements

- Python 3.11+
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

Edit **`config.py`** to adjust:

| Setting | Default | Description |
|---------|---------|-------------|
| `IBKR_PORT` | `7497` | `7497` paper · `7496` live · `4002` Gateway live |
| `STOP_LOSS_PCT` | `0.07` | 7% below entry |
| `TAKE_PROFIT_PCT` | `0.15` | 15% above entry |
| `RISK_PER_TRADE_PCT` | `0.01` | Risk 1% of equity per trade |
| `MAX_POSITIONS` | `10` | Max concurrent positions |
| `MAX_POSITION_PCT` | `0.05` | Max 5% of equity per stock |
| `RSI_OVERSOLD` | `35.0` | RSI threshold for oversold signal |
| `DIP_FROM_MA50_PCT` | `0.05` | Min % below 50-day MA for signal |
| `MIN_DIP_SCORE` | `2` | Min signals to trigger a buy |
| `MAX_PE_RATIO` | `60.0` | Fundamental filter: max P/E |
| `MIN_PROFIT_MARGIN` | `0.05` | Fundamental filter: min net margin |
| `MAX_DEBT_TO_EQUITY` | `2.5` | Fundamental filter: max D/E |

Edit **`watchlist.py`** to customise the list of tickers scanned.

---

## Usage

### Dry run (no orders placed — great for reviewing signals)

```bash
python run.py --dry-run
```

### Live scan (connects to IBKR and places bracket orders)

```bash
python run.py
```

### Scheduled daily scan at 09:45 US-Eastern

```bash
python run.py --schedule 09:45
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

### 1. Fundamental screen (screener.py)

Each ticker is checked against live data from Yahoo Finance:

| Criterion | Default threshold |
|-----------|-------------------|
| Forward/trailing P/E | 0 < P/E ≤ 60 |
| Net profit margin | ≥ 5% |
| Debt-to-equity | ≤ 2.5× |
| Revenue growth (YoY) | ≥ −10% |

Stocks that fail any criterion are excluded from dip detection.

### 2. Dip scoring (dip_detector.py)

Each passing stock is scored 0–4 based on technical signals:

| Signal | Condition | Score |
|--------|-----------|-------|
| RSI oversold | 14-day RSI < 35 | +1 |
| Below MA-50 | Price ≥ 5% below 50-day MA | +1 |
| Below MA-200 | Price below 200-day MA | +1 |
| 52-week range | Price in bottom 30% of 52-wk range | +1 |

A stock needs **score ≥ 2** (configurable via `MIN_DIP_SCORE`) to be
considered a dip opportunity.

### 3. Position sizing (risk_manager.py)

For each dip candidate:

- **Stop-loss price** = entry × (1 − 7%)
- **Take-profit price** = entry × (1 + 15%)
- **Risk per share** = entry − stop-loss
- **Quantity** = min(
    `account_equity × 1% / risk_per_share`,
    `account_equity × 5% / entry`
  )

This gives a risk/reward ratio of ~2.1:1 (15% profit ÷ 7% risk).

### 4. Bracket order (broker.py)

Each position is entered as an IBKR **bracket order** with three legs:

1. **Parent limit order** — BUY at the dip price
2. **Take-profit limit** — SELL at +15%
3. **Protective stop** — SELL STOP at −7%

IBKR automatically cancels the remaining exit leg when one fills (OCA group).

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Disclaimer

This software is provided for educational and research purposes only.
Trading involves substantial risk of loss. Past performance does not
guarantee future results. Always paper-trade and review signals before
connecting to a live account.