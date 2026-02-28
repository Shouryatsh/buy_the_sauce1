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
MA_FAST: int = 50                  # Fast moving-average window (days)
MA_SLOW: int = 200                 # Slow moving-average window (days)
DIP_FROM_MA50_PCT: float = 0.05    # Price ≥ 5% below 50-day MA → dip signal
WEEK52_LOWER_BAND: float = 0.30    # Price in bottom 30% of 52-week range → signal
MIN_DIP_SCORE: int = 2             # Minimum combined score to trigger buy consideration

# ---------------------------------------------------------------------------
# Fundamental filters
# ---------------------------------------------------------------------------
MAX_PE_RATIO: float = 60.0         # Skip if forward P/E > this
MIN_PE_RATIO: float = 0.0          # Skip unprofitable (negative EPS)
MAX_DEBT_TO_EQUITY: float = 2.5    # Skip if D/E ratio > this
MIN_PROFIT_MARGIN: float = 0.05    # Skip if net margin < 5%
MIN_REVENUE_GROWTH: float = -0.10  # Allow up to -10% YoY revenue decline

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FILE: str = "trading.log"
LOG_LEVEL: str = "INFO"
