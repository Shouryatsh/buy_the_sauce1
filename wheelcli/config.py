"""
wheelcli/config.py — Pydantic-Settings configuration model.

All parameters can be overridden via:
  • Environment variables prefixed with WHEEL_  (e.g. WHEEL_IBKR_PORT=7496)
  • A .env file in the working directory
  • Direct keyword arguments at instantiation

Example .env
------------
  WHEEL_IBKR_HOST=127.0.0.1
  WHEEL_IBKR_PORT=7497
  WHEEL_IBKR_CLIENT_ID=10
  WHEEL_MAX_DTE=45
  WHEEL_SIGMA_THRESHOLD=2.0
  WHEEL_MAX_DELTA=0.05
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class WheelConfig(BaseSettings):
    """Central configuration for the wheel-strategy scanner."""

    model_config = SettingsConfigDict(
        env_prefix="WHEEL_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── IBKR connectivity ─────────────────────────────────────────────────
    ibkr_host: str = Field("127.0.0.1", description="IB Gateway / TWS host")
    ibkr_port: int = Field(
        7497,
        description="7497=TWS paper  |  7496=TWS live  |  4002=IB Gateway live",
    )
    ibkr_client_id: int = Field(
        10,
        description="Client ID — must be unique among all connections to the same TWS/Gateway",
    )
    ibkr_connect_timeout: float = Field(
        15.0, description="Connection timeout in seconds"
    )

    # ── Scan parameters ───────────────────────────────────────────────────
    max_dte: int = Field(45, ge=1, le=365, description="Maximum days to expiry")
    weekly_only: bool = Field(
        True, description="If True, keep only Friday expirations (weekly options)"
    )
    sigma_threshold: float = Field(
        2.0, ge=0.0, description="Minimum sigma distance to pass filter"
    )
    max_delta: float = Field(
        0.05, ge=0.0, le=1.0, description="Maximum absolute put delta to pass filter"
    )

    # Put strike search range as fraction of spot
    strike_low_frac: float = Field(
        0.68, ge=0.0, le=1.0, description="Lowest strike to consider = spot × this"
    )
    strike_high_frac: float = Field(
        1.01, ge=0.0, le=2.0, description="Highest strike to consider = spot × this"
    )

    # ── Event-risk discount ───────────────────────────────────────────────
    event_window_days: int = Field(
        3,
        ge=0,
        description="Penalise trades whose expiry falls within N days after earnings",
    )
    earnings_penalty: float = Field(
        0.6, ge=0.0, le=1.0, description="Score multiplier when trade crosses earnings"
    )
    macro_penalty: float = Field(
        0.85, ge=0.0, le=1.0, description="Score multiplier when trade crosses a macro event"
    )
    unknown_earnings_penalty: float = Field(
        0.9,
        ge=0.0,
        le=1.0,
        description="Conservative penalty for symbols absent from earnings calendar",
    )

    # ── Put skew ─────────────────────────────────────────────────────────
    skew_ratio_threshold: float = Field(
        1.10, ge=1.0, description="skew_ratio >= this → skew_bonus = 1"
    )
    skew_diff_threshold: float = Field(
        0.03, ge=0.0, description="skew_diff (IV_otm − IV_atm) >= this → skew_bonus = 1"
    )
    skew_bonus_weight: float = Field(
        0.3, ge=0.0, description="Weight of skew bonus in composite score"
    )

    # ── Liquidity ─────────────────────────────────────────────────────────
    spread_cap: float = Field(
        0.10,
        gt=0.0,
        description="Spread % cap used in liquidity_factor formula (10% default)",
    )
    min_liquidity_factor: float = Field(
        0.2, ge=0.0, le=1.0, description="Floor for liquidity_factor"
    )

    # ── IBKR pacing ───────────────────────────────────────────────────────
    ibkr_batch_size: int = Field(
        40,
        ge=1,
        le=50,
        description="Max simultaneous market-data subscriptions per batch",
    )
    ibkr_batch_delay: float = Field(
        1.0, ge=0.0, description="Seconds to wait between request batches"
    )
    ibkr_mktdata_wait: float = Field(
        3.0, ge=0.5, description="Seconds to wait for model greeks after reqMktData"
    )

    # ── File paths ────────────────────────────────────────────────────────
    universe_file: str = Field("universe.csv")
    earnings_file: str = Field("earnings_calendar.csv")
    macro_events_file: str = Field("macro_events.csv")
    cache_dir: str = Field(".wheelcache", description="Directory for disk cache")
    cache_ttl: int = Field(900, ge=1, description="Cache time-to-live in seconds")

    # ── Output ────────────────────────────────────────────────────────────
    max_candidates: int = Field(50, ge=1, description="Rows shown in scan output")
