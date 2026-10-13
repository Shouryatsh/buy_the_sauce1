"""
wheel_config.py — Configuration for the options wheel decision-support dashboard.

Kept separate from config.py, which governs the unrelated buy-the-dip equity
bot in this repo. The two strategies run against different capital pools and
different rules; nothing in this file (or anything it configures) auto-submits
an order — see wheel_execution.py.
"""

# ---------------------------------------------------------------------------
# Account & targets
# ---------------------------------------------------------------------------
INVESTED_CAPITAL: float = 70_000        # already deployed
SPARE_CAPITAL: float = 30_000           # available for new cash-secured puts
TARGET_MONTHLY_INCOME: float = 2_000
MAX_ASSIGNMENT_PRICE: float = 300       # never sell puts on strikes above this
MAX_DELTA_ABS: float = 0.10             # ceiling; lower preferred, scored vs premium
FCF_YIELD_MIN: float = 0.05             # standard bar
FCF_YIELD_MIN_ELITE: float = 0.02       # lower bar granted to "elite" names
MAX_NAME_CONCENTRATION: float = 0.15    # % of total book (invested + spare) per underlying

TOTAL_BOOK: float = INVESTED_CAPITAL + SPARE_CAPITAL

# ---------------------------------------------------------------------------
# "Elite" quality bar — lets a name use FCF_YIELD_MIN_ELITE instead of FCF_YIELD_MIN
# ---------------------------------------------------------------------------
ELITE_MIN_ROE: float = 0.25
ELITE_MIN_NET_MARGIN: float = 0.20
ELITE_MIN_REVENUE_GROWTH: float = 0.15

# ---------------------------------------------------------------------------
# Composite Quality Score (0-100) weights — must sum to 1.0
# Net-margin trend and FCF trend weighted highest, then FCF yield,
# then ROE and revenue growth roughly equal.
# ---------------------------------------------------------------------------
QUALITY_WEIGHTS: dict = {
    "net_margin_trend": 0.30,
    "fcf_trend": 0.30,
    "fcf_yield": 0.20,
    "roe": 0.10,
    "revenue_growth": 0.10,
}

# Normalization caps used to rescale each raw metric to a 0-1 score before
# weighting (e.g. an FCF yield >= FCF_YIELD_NORM_CAP scores the max 1.0).
FCF_YIELD_NORM_CAP: float = 0.15
ROE_NORM_CAP: float = 0.30
REVENUE_GROWTH_NORM_CAP: float = 0.25
NET_MARGIN_TREND_NORM_CAP: float = 0.10   # +/-10pp swing maps to +/-1.0

# ---------------------------------------------------------------------------
# One-time universe scan (wheel_universe.py / scan_universe.py)
#
# Builds a shortlist from a much larger universe than WHEEL_WATCHLIST's
# ~44 names, using SEC EDGAR's own company_tickers.json ordering (roughly
# market-cap descending) to approximate "top N US stocks" for free -- the
# usual paid/scraped index-constituent sources (iShares fund holdings,
# Wikipedia) are blocked by anti-bot measures in this environment.
# ---------------------------------------------------------------------------
UNIVERSE_SIZE: int = 1200                  # pulled before filtering -- some names won't have clean EDGAR data
UNIVERSE_MIN_NET_MARGIN: float = 0.10      # net margin >= 10%
UNIVERSE_MIN_FCF: float = 0.0              # FCF strictly positive
UNIVERSE_SCAN_WORKERS: int = 10            # concurrent EDGAR fetches -- edgar.py's shared rate limiter caps aggregate rate
UNIVERSE_PRICE_BATCH_SIZE: int = 150       # tickers per yfinance bulk-download call
SHORTLIST_PATH: str = "wheel_shortlist.json"

# ---------------------------------------------------------------------------
# Options selection engine
# ---------------------------------------------------------------------------
DEFAULT_DTE_MIN: int = 30
DEFAULT_DTE_MAX: int = 60
RICHNESS_LOOKBACK_DAYS: int = 252          # ~1 trading year for realized vol
RICHNESS_OVERPRICED_RATIO: float = 1.20    # market IV / realized vol above this -> "overpriced to sell"
RISK_FREE_RATE: float = 0.05               # proxy for r in Black-Scholes; refine later with a live curve

# Minimum premium worth collecting per contract -- filters out trades that
# clear every other screen but pay too little to justify the assignment
# risk and commissions. $0.50/share = $50/contract (options are 100 shares).
MIN_PREMIUM_PER_CONTRACT: float = 50.0

# Scope limits for the option chain pull -- each snapshot market-data
# request has IBKR's own ~11s server-side hold regardless of contract
# count, so bounding *how many requests* get made per symbol matters far
# more than trimming contracts within one request. MAX_EXPIRIES_PER_SYMBOL
# caps how many separate reqTickers calls happen per symbol (one per
# expiry); the strike band narrows the search to strikes actually plausible
# for a MAX_DELTA_ABS=0.10-ish put (deep-ITM/ATM strikes are never useful
# for this strategy and were previously fetched and quoted for nothing).
MAX_EXPIRIES_PER_SYMBOL: int = 2
STRIKE_BAND_LOW_PCT: float = 0.65          # search strikes from 65%..92% of spot
STRIKE_BAND_HIGH_PCT: float = 0.92

# ---------------------------------------------------------------------------
# Stress scenario (Phase 1 proxy via Black-Scholes; refine in Phase 4 with a
# proper repricing model / Phase 3's Merton calibration)
# ---------------------------------------------------------------------------
STRESS_UNDERLYING_SHOCK_PCT: float = -0.10   # underlying -10%
STRESS_IV_SHOCK_PCT: float = 0.30            # IV +30% (relative)

# ---------------------------------------------------------------------------
# Composite recommendation weights (quality x richness x capital efficiency)
# ---------------------------------------------------------------------------
RECOMMENDATION_WEIGHTS: dict = {
    "quality": 0.40,
    "richness": 0.35,
    "capital_efficiency": 0.25,
}

# ---------------------------------------------------------------------------
# IBKR connection — distinct client id from config.IBKR_CLIENT_ID (dip bot)
# so both tools can connect to the same TWS/Gateway instance simultaneously.
# ---------------------------------------------------------------------------
IBKR_HOST: str = "127.0.0.1"
IBKR_PAPER_PORT: int = 7497
IBKR_LIVE_PORT: int = 7496
IBKR_GATEWAY_LIVE_PORT: int = 4002
IBKR_CLIENT_ID: int = 7
USE_PAPER_ACCOUNT: bool = False  # live account (port 7496); the dashboard no longer offers paper

# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
WHEEL_DB_PATH: str = "wheel.db"
