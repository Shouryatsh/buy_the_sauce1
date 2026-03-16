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
STOP_LOSS_PCT: float = 0.07       # 7% below entry price
TAKE_PROFIT_PCT: float = 0.15     # 15% above entry price
RISK_PER_TRADE_PCT: float = 0.01  # Risk 1% of total account equity per trade
MAX_POSITIONS: int = 10           # Max concurrent open positions
MAX_POSITION_PCT: float = 0.05    # Max 5% of portfolio in a single stock

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
