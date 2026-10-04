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

---

## Options Wheel Dashboard (Phase 1)

A second, independent tool in this repo: a **decision-support dashboard**
for a cash-secured-put → covered-call wheel strategy run on IBKR. It
screens, scores, and recommends — a human clicks execute. It does not share
config, watchlist, or capital targets with the dip-buying bot above.

```
wheel_dashboard.py         ← Streamlit UI tying everything together
 ├─ wheel_fundamentals.py  ← Quality Score (0-100) + FCF/ROE/margin screen (SEC EDGAR)
 ├─ wheel_options.py       ← Option chain pull, Black-Scholes greeks, richness proxy
 ├─ wheel_risk.py          ← Raw / delta-adjusted / stress exposure, concentration checks
 ├─ wheel_recommend.py     ← Composite ranking + capital sizing
 ├─ wheel_ibkr.py          ← Read-only account/positions/orders/trade-history pull
 └─ wheel_db.py            ← SQLite archive for screener history + trade log

wheel_config.py            ← Capital targets, thresholds, score weights
wheel_watchlist.py         ← Tickers to screen (must be <= MAX_ASSIGNMENT_PRICE)

wheel_skew.py              ← [Phase 2 stub] full IV smile/skew diagnostics
wheel_jump_diffusion.py    ← [Phase 3 stub] Merton jump-diffusion fair-value IV
wheel_execution.py         ← [Phase 5 stub] manual-confirm order staging
```

### Run it

```bash
pip install -r requirements.txt
streamlit run wheel_dashboard.py
```

### Dependencies / known limitations

- **TWS or IB Gateway must be running locally with API access enabled.** The
  dashboard connects read-only, on its own client id (`wheel_config.IBKR_CLIENT_ID`)
  so it can run alongside the dip bot against the same TWS instance.
- **Live/delayed options data (OPRA subscription) is needed for real
  greeks/IV.** Without one, IBKR reports no model greeks and the dashboard
  falls back to a Black-Scholes delta computed from realized volatility —
  shown as an estimate, never mixed in with broker-reported values silently.
- **Fundamentals come from SEC EDGAR, not yfinance**, matching the decision
  already made for the dip bot in this repo (see `requirements.txt`) — EDGAR
  is free, official, and doesn't rate-limit as aggressively. EDGAR's TTM data
  can still be noisy or incomplete for some names; any symbol with missing
  FCF/margin/ROE is scored but flagged `data_reliable=False` rather than
  silently treated as a fail.
- **Realized P&L / win rate are approximate.** IBKR's execution API
  (`reqExecutions`) only retains recent fills; `wheel_db.py` archives fills
  locally as they're seen so history survives past that window, but a full
  backfill would need IBKR's Flex Query API (paid-account feature).
- **Phase 1 exposure and richness figures are Black-Scholes proxies**, not
  broker-reported or Merton-calibrated. They're labeled as such in the
  dashboard and get refined in Phase 3 (jump-diffusion fair value) and
  Phase 4 (live greeks, full risk module).
- **Nothing in this tool submits an order.** The Ranked Recommendations
  panel's "Execute" button is disabled until Phase 5.

---

## Disclaimer

This software is provided for educational and research purposes only.
Trading involves substantial risk of loss. Past performance does not
guarantee future results. Always paper-trade and review signals before
connecting to a live account.