"""Single source of truth for configuration.

Precedence (highest first): init kwargs, environment (OPTIONSDESK_*, nested with
"__", e.g. OPTIONSDESK_RISK__MAX_NET_DELTA=300), config.yaml (path override via
OPTIONSDESK_CONFIG), defaults below.
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class Mode(str, Enum):
    PAPER = "paper"
    LIVE = "live"


class IBKRSettings(BaseModel):
    host: str = "127.0.0.1"
    paper_port: int = 7497
    live_port: int = 7496
    client_id: int = 21  # distinct from dip bot (1), wheel dashboard (7), wheelcli (10)
    connect_timeout: float = 10.0
    # 1=live, 3=delayed. Delayed is requested when no OPRA subscription exists.
    market_data_type: Literal[1, 2, 3, 4] = 3
    max_requests_per_second: float = 20.0
    batch_size: int = 40  # simultaneous option market-data lines per batch
    mktdata_wait_seconds: float = 3.0  # let model greeks arrive before reading
    reconnect_backoff_seconds: float = 5.0
    reconnect_max_backoff_seconds: float = 120.0
    quote_cache_ttl_seconds: float = 30.0


class WheelSettings(BaseModel):
    put_delta_min: float = 0.25
    put_delta_max: float = 0.30
    call_delta_min: float = 0.25
    call_delta_max: float = 0.30
    dte_min: int = 30
    dte_max: int = 45
    roll_dte: int = 21
    profit_take_pct: float = 0.50  # close when this fraction of premium is captured
    tested_buffer_pct: float = 0.02  # "tested" when spot <= strike * (1 + buffer) for puts
    min_iv_rank: float = 20.0
    require_iv_rank: bool = False  # False: allow entries (flagged) while IV history is still < 30 days
    max_spread_pct: float = 0.10
    min_open_interest: int = 100
    allow_cc_below_cost_basis: bool = False
    contracts_per_trade: int = 1


class StraddleSettings(BaseModel):
    mode: Literal["long", "short"] = "long"
    use_wings: bool = False
    wing_width_pct: float = 0.05
    dte_min: int = 7
    dte_max: int = 45
    long_max_iv_rank: float = 30.0
    short_min_iv_rank: float = 60.0
    min_iv_over_rv: float = 1.1  # short entries need IV/RV at least this
    max_iv_over_rv: float = 1.0  # long entries need IV/RV at most this
    event_window_days_min: int = 1  # long: earnings must be at least this far ahead
    event_window_days_max: int = 14  # long: and no further than this
    long_requires_event: bool = True
    min_edge_vs_expected_move: float = 0.0
    profit_target_pct: float = 0.50
    stop_loss_pct: float = 0.50  # of premium paid (long) / of credit (short)
    exit_dte: int = 2
    max_spread_pct: float = 0.15
    min_open_interest: int = 100


class RiskSettings(BaseModel):
    max_capital_at_risk: float = 100_000.0
    max_ticker_concentration: float = 0.15  # fraction of max_capital_at_risk
    max_net_delta: float = 500.0  # share-equivalents, absolute
    max_net_vega: float = 2_000.0
    max_net_theta_abs: float = 1_000.0
    scenario_moves: list[float] = Field(default_factory=lambda: [-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20])
    max_scenario_loss_pct: float = 0.20  # worst grid P&L / max_capital_at_risk
    warn_utilization: float = 0.80  # flag (amber) above this fraction of any limit


class BacktestSettings(BaseModel):
    commission_per_contract: float = 0.65
    slippage_pct_of_spread: float = 0.50  # fill this far through the bid/ask spread
    risk_free_rate: float = 0.045
    initial_capital: float = 100_000.0
    # Option prices are modelled (no historical chains): IV proxy = trailing realized vol x premium.
    iv_window: int = 20
    rv_window: int = 60
    iv_premium: float = 1.10
    event_iv_uplift: float = 1.20  # IV multiplier for expiries that span a known event
    contracts: int = 1


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OPTIONSDESK_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    mode: Mode = Mode.PAPER
    allow_live: bool = False  # must be true AND mode=live for live trading to be possible
    # "simulated" never touches IBKR; "ibkr" needs TWS/Gateway with API enabled.
    data_source: Literal["ibkr", "simulated"] = "ibkr"
    execution_backend: Literal["ibkr", "simulated"] = "simulated"
    auto_execute_paper: bool = True  # False: paper orders also wait for manual confirmation
    order_fill_timeout_seconds: float = 60.0
    heartbeat_interval_seconds: float = 15.0
    iv_wait_seconds: float = 2.0  # strategy agents wait this long for a chain's IV metrics before deciding without them
    signal_cooldown_seconds: float = 1800.0  # don't re-emit an identical signal that risk blocked or the broker rejected
    strategies: list[Literal["wheel", "straddle"]] = Field(default_factory=lambda: ["wheel", "straddle"])
    # Agents to run in this process (market_data, iv, wheel, straddle, risk, execution, backtest, narrative).
    agents: list[str] = Field(default_factory=lambda: [
        "market_data", "iv", "wheel", "straddle", "risk", "execution", "backtest", "narrative"])
    starting_capital: float = 100_000.0  # base for portfolio value = base + realized + unrealized
    api_token: Optional[str] = None  # if set, mutating API calls need header X-API-Token
    narrative_interval_seconds: float = 3600.0
    db_path: str = "optionsdesk.db"
    tickers: list[str] = Field(default_factory=lambda: ["AAPL", "MSFT", "KO"])
    earnings_file: str = "earnings_calendar.csv"
    poll_interval_seconds: float = 60.0
    log_level: str = "INFO"

    ibkr: IBKRSettings = Field(default_factory=IBKRSettings)
    wheel: WheelSettings = Field(default_factory=WheelSettings)
    straddle: StraddleSettings = Field(default_factory=StraddleSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    backtest: BacktestSettings = Field(default_factory=BacktestSettings)

    @model_validator(mode="after")
    def _live_needs_flag(self) -> "Settings":
        if self.mode is Mode.LIVE and not self.allow_live:
            raise ValueError("mode=live requires allow_live=true in config")
        return self

    @property
    def ibkr_port(self) -> int:
        return self.ibkr.live_port if self.mode is Mode.LIVE else self.ibkr.paper_port

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        path = os.environ.get("OPTIONSDESK_CONFIG", "config.yaml")
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings]
        if os.path.exists(path):
            sources.append(YamlConfigSettingsSource(settings_cls, yaml_file=path))
        return tuple(sources)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
