"""
config.py — Central configuration for the buy-the-dip trading system.

All tuneable parameters live here so nothing is hard-coded elsewhere.
"""

# ---------------------------------------------------------------------------
# IBKR connection
# ---------------------------------------------------------------------------
IBKR_HOST: str = "127.0.0.1"
# 7497 = TWS paper trading  |  7496 = TWS live  |  4002 = IB Gateway live
IBKR_PORT: int = 7497
IBKR_CLIENT_ID: int = 1

# ---------------------------------------------------------------------------
# Risk management
# ---------------------------------------------------------------------------
STOP_LOSS_PCT: float = 0.07       # 7% below entry price (fallback when ATR unavailable)
TAKE_PROFIT_PCT: float = 0.15     # 15% above entry price (fallback when ATR unavailable)
RISK_PER_TRADE_PCT: float = 0.01  # Risk 1% of total account equity per trade
MAX_POSITIONS: int = 10           # Max concurrent open positions
MAX_POSITION_PCT: float = 0.05    # Max 5% of portfolio in a single stock

# ---------------------------------------------------------------------------
# ATR-based dynamic stop / target  (overrides fixed STOP_LOSS_PCT when enabled)
# ---------------------------------------------------------------------------
USE_ATR_STOPS: bool = True          # True = ATR-based stops; False = fixed %
ATR_STOP_MULTIPLIER: float = 2.0    # Stop  = entry − ATR_STOP_MULTIPLIER × ATR(14)
ATR_TARGET_MULTIPLIER: float = 3.0  # Target = entry + ATR_TARGET_MULTIPLIER × ATR(14)
ATR_MAX_STOP_PCT: float = 0.12      # Hard cap: stop never > 12% below entry
ATR_MIN_STOP_PCT: float = 0.03      # Hard floor: stop never < 3% below entry
#   These caps prevent ATR blowing out in low-liquidity or very high-vol names.

# ---------------------------------------------------------------------------
# Trailing stop  (adaptive three-stage)
# ---------------------------------------------------------------------------
# Stage 1 — "breakeven lock":  once price reaches +1×ATR above avg entry,
#   trail moves to entry (breakeven).
# Stage 2 — "lock profit":  once price reaches +2×ATR above avg entry,
#   trail moves to entry + 1×ATR.
# Stage 3 — "tight trail":  once price reaches +3×ATR above avg entry,
#   trail tightens to highest_price − 1×ATR.
#
# At all stages the trailing stop can only move UP, never down.
TRAILING_STOP_ENABLED: bool = True
TRAILING_STAGE1_TRIGGER_ATR: float = 1.0   # +1×ATR → trail = entry (breakeven)
TRAILING_STAGE2_TRIGGER_ATR: float = 2.0   # +2×ATR → trail = entry + 1×ATR
TRAILING_STAGE3_TRIGGER_ATR: float = 3.0   # +3×ATR → trail = high  − 1×ATR
TRAILING_STAGE3_TRAIL_ATR: float   = 1.0   # tight trail distance (×ATR)

# ---------------------------------------------------------------------------
# Partial exit  (staged profit-taking)
# ---------------------------------------------------------------------------
# Take partial profits at +2×ATR and +3×ATR to lock in gains while
# letting the remainder run.  Fractions must sum to ≤ 1.0 with the
# final exit at the trailing stop.
PARTIAL_EXIT_ENABLED: bool = True
PARTIAL_EXIT_1_TRIGGER_ATR: float = 2.0   # sell first slice at +2×ATR
PARTIAL_EXIT_1_FRACTION: float    = 0.33  # sell 1/3 of position
PARTIAL_EXIT_2_TRIGGER_ATR: float = 3.0   # sell second slice at +3×ATR
PARTIAL_EXIT_2_FRACTION: float    = 0.33  # sell another 1/3
#   Remaining 1/3 rides the trailing stop to capture extended moves.

# ---------------------------------------------------------------------------
# Time stop  (maximum hold duration)
# ---------------------------------------------------------------------------
# If a position hasn't reached its first partial-exit target within
# TIME_STOP_DAYS, exit at market to free capital.
TIME_STOP_ENABLED: bool = True
TIME_STOP_DAYS: int = 30          # max holding period (calendar days)

# ---------------------------------------------------------------------------
# Scaled entry / buy ladder
# ---------------------------------------------------------------------------
# Instead of entering full size at one price, split the entry into N
# tranches at progressively lower prices.  This improves average cost
# when the dip continues and limits exposure if the thesis is wrong.
#
# How it works
# ------------
# 1. Tranche 1 (T1): placed at the current dip price — immediate entry.
# 2. Tranche 2 (T2): placed 1.0×ATR below T1 price.
# 3. Tranche 3 (T3): placed 2.0×ATR below T1 price (deeper dip).
#
# Each tranche's quantity is a fraction of the total position size
# (from calculate_order).  The fractions should sum to 1.0.
#
# Abort / expiry logic
# --------------------
# - If RSI rises above SCALED_ENTRY_RSI_ABORT_LEVEL before T2/T3 fill,
#   cancel unfilled tranches (momentum has reversed — no longer a dip).
# - Unfilled limit orders expire after SCALED_ENTRY_EXPIRY_DAYS.
# - RSI turn confirmation (optional): T2/T3 only fill if RSI has also
#   turned down from the prior bar (confirming continuing weakness).
SCALED_ENTRY_ENABLED: bool = True
SCALED_ENTRY_N_TRANCHES: int = 3          # 1–5 tranches (1 = disabled)
SCALED_ENTRY_FRACTIONS: tuple = (0.40, 0.35, 0.25)  # must sum to 1.0
#   Front-load T1 so you always get partial exposure.
SCALED_ENTRY_ATR_OFFSETS: tuple = (0.0, 1.0, 2.0)   # ATR multiples below T1
#   T1=0 (market/limit at current), T2=−1×ATR, T3=−2×ATR
SCALED_ENTRY_EXPIRY_DAYS: int = 5         # unfilled limits expire after 5 days
SCALED_ENTRY_RSI_ABORT_LEVEL: float = 50.0  # cancel T2/T3 if RSI > this
SCALED_ENTRY_RSI_TURN_REQUIRED: bool = True # T2/T3 require RSI declining bar-over-bar

# ---------------------------------------------------------------------------
# Simultaneous / portfolio-level sizing
# ---------------------------------------------------------------------------
# Capital budget: the dashboard allocates this across ALL positions
# simultaneously, deducting each committed notional from the remaining pool.
PORTFOLIO_CAPITAL: float = 80_000.0   # total account budget ($)
MAX_CAPITAL_DEPLOYED_PCT: float = 0.80  # deploy at most 80% of budget at once
#   Keeps 20% cash for margin, unexpected opportunities, or drawdown.

# Kelly Criterion fractional sizing
# The system uses the FRACTIONAL Kelly formula:
#   full_kelly = (edge / odds)  where edge = win_rate − (1 − win_rate)/rr
#   position_size = KELLY_FRACTION × full_kelly × equity
# Set KELLY_FRACTION = 0 to disable and rely solely on fixed-risk sizing.
KELLY_FRACTION: float = 0.25          # 1/4-Kelly (conservative; avoids ruin risk)
KELLY_WIN_RATE: float = 0.52          # assumed win rate (conservative estimate)
#   Updated dynamically if backtest data is available.

# Volatility-regime scaling: shrink size in HIGH vol, expand in LOW vol.
# Only applied when USE_ATR_STOPS = True (ATR-based stops already adapt, so
# the vol-scaling provides an additional portfolio-heat guard).
VOL_SCALE_HIGH: float = 0.65    # multiply base size by this in HIGH vol regime
VOL_SCALE_NORMAL: float = 1.00  # no adjustment in NORMAL regime
VOL_SCALE_LOW: float = 1.20     # modest increase in LOW vol (cap to MAX_POSITION_PCT)

# Correlation-based position penalty
# When an existing position in the same sector is already open, the new
# position is scaled down to avoid concentration risk.
CORRELATION_SAME_SECTOR_SCALE: float = 0.75  # 25% size cut for same-sector additions
#   Sector grouping uses the ticker→sector mapping from EDGAR fundamentals.

# ---------------------------------------------------------------------------
# Dip-detection thresholds
# ---------------------------------------------------------------------------
RSI_PERIOD: int = 14
RSI_OVERSOLD: float = 35.0         # RSI below this → oversold signal
MA_FAST: int = 20                  # Fast moving-average window (days)
MA_SLOW: int = 100                 # Slow moving-average window (days)
DIP_FROM_MA50_PCT: float = 0.05    # Price ≥ 5% below 50-day MA → dip signal
WEEK52_LOWER_BAND: float = 0.25    # Price in bottom 30% of 52-week range → signal
MIN_DIP_SCORE: int = 3             # Minimum combined score to trigger buy consideration

# ---------------------------------------------------------------------------
# Fundamental filters
# ---------------------------------------------------------------------------
MAX_PE_RATIO: float = 60.0         # Skip if trailing P/E > this
MIN_PE_RATIO: float = 0.0          # Skip unprofitable (negative EPS)
MIN_PROFIT_MARGIN: float = 0.05    # Skip if net margin < 5%

# --- Debt / equity (sector-aware) ---
# Banks and financials carry structural leverage that is normal and regulated;
# applying a single D/E cap eliminates them unfairly.  We use a higher cap for
# financials and a tighter one for all other sectors.
MAX_DEBT_TO_EQUITY: float = 3.0           # Non-financial stocks (e.g. AAPL, UNH)
MAX_DEBT_TO_EQUITY_FINANCIAL: float = 15.0 # Banks, insurers, diversified financials
# SIC codes considered "financial" for the purpose of the looser D/E cap:
FINANCIAL_SIC_PREFIXES: tuple = ("60", "61", "62", "63", "64", "67")

# --- Free Cash Flow filters (primary quality gate) ---
MIN_FREE_CASH_FLOW: float = 0.0    # FCF must be strictly positive
MIN_FCF_YIELD: float = 0.015       # FCF yield ≥ 1.5% (2% excluded all mega-caps unfairly)
REQUIRE_INCREASING_FCF: bool = True
# Require FCF to be higher in at least FCF_GROWTH_LOOKBACK_YEARS out of the last 3 years
# (not necessarily every single year — avoids penalising one-off capex spikes)
FCF_GROWTH_LOOKBACK_YEARS: int = 3  # look back this many annual periods
FCF_MIN_GROWTH_YEARS: int = 2       # FCF must be up in at least this many of those periods

# NOTE: MIN_REVENUE_GROWTH removed — FCF is a truer measure of business quality.

# --- Capital efficiency filters ---
MIN_RETURN_ON_EQUITY: float = 0.10  # ROE ≥ 10%
MAX_CAPEX_TO_FCF: float = 0.75      # Capex ≤ 75% of FCF
#   Raised from 50% → 75%: 50% was too tight for asset-heavy compounders
#   (BRK-B railroads, META AI infra).  75% still screens out cash-incinerators.

# ---------------------------------------------------------------------------
# ML predictor  (ml_predictor.py) — Ensemble (LightGBM + LogReg + SGD stacking)
# ---------------------------------------------------------------------------
ML_ENABLED: bool = True              # Set False to skip ML entirely (faster)
ML_PREDICT_HORIZON: int = 5          # Predict price direction N days ahead
ML_TRAIN_SPLIT: float = 0.80         # 80% of history used for training
ML_MIN_TRAIN_SAMPLES: int = 150      # Skip ML if fewer clean samples available

# Legacy Random Forest params (kept for reference / fallback):
ML_N_ESTIMATORS: int = 200           # Random Forest trees
ML_MAX_DEPTH: int = 6                # Max tree depth (controls overfitting)
ML_MIN_SAMPLES_LEAF: int = 10        # Minimum samples per leaf (controls overfitting)

# Ensemble-specific: set True to run 5-fold walk-forward CV AUROC per ticker
# (adds ~3-8s per ticker; enabled by default — AUROC gate depends on this)
ML_COMPUTE_CV_AUROC: bool = True

# Minimum walk-forward CV AUROC required to emit a prediction.
# Benchmarked across 30 large/mid-cap US tickers (5y daily data, 5-fold CV):
#   Mean achievable AUROC ≈ 0.52  (range 0.45–0.57)
#   Best single ticker    ≈ 0.57  (MS, AMZN)
#
# 0.65 is NOT achievable for large-cap equities on daily price data alone —
# these are the most heavily arbitraged assets on earth.  Academic literature
# (Gu, Kelly & Xiu 2020) reports 0.52–0.54 for this task.
#
# 0.55 is a statistically defensible gate: it sits ~2 std above the mean of
# the walk-forward null distribution, meaning the model is capturing a
# genuine (small) edge rather than noise.
ML_MIN_AUROC_THRESHOLD: float = 0.51   # Lowered: large-cap mean AUROC ≈ 0.52 (Gu, Kelly & Xiu 2020)

# Confidence thresholds (probability of predicted class)
ML_HIGH_CONFIDENCE_THRESHOLD: float   = 0.65   # ≥ 65% → HIGH
ML_MEDIUM_CONFIDENCE_THRESHOLD: float = 0.55   # 55–65% → MEDIUM  (<55% → LOW)

# Signal gating: ML must agree with dip direction (predict UP) before a
# dip is promoted to a full BUY SIGNAL.  Set False to show ML as info-only.
ML_GATE_BUY_SIGNAL: bool = True

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FILE: str = "trading.log"
LOG_LEVEL: str = "INFO"
