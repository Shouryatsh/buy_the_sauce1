"""Typed events exchanged over the bus. Every event carries a correlation_id so a
trade decision can be traced from market data through signal, risk and order."""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DataQuality(str, Enum):
    LIVE = "live"
    DELAYED = "delayed"
    ESTIMATED = "estimated"  # model-derived (e.g. Black-Scholes), not broker-reported


class Event(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=new_id)
    correlation_id: str = Field(default_factory=new_id)
    ts: datetime = Field(default_factory=utcnow)

    @classmethod
    def topic(cls) -> str:
        return cls.__name__

    def derive(self, event_cls: type["Event"], **fields: Any) -> "Event":
        """Create a downstream event that inherits this event's correlation_id."""
        return event_cls(correlation_id=self.correlation_id, **fields)


class Quote(Event):
    symbol: str
    price: float
    quality: DataQuality = DataQuality.DELAYED


class OptionQuote(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    expiry: date
    strike: float
    right: Literal["C", "P"]
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    iv: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    open_interest: Optional[int] = None
    volume: Optional[int] = None
    quality: DataQuality = DataQuality.DELAYED

    @property
    def mid(self) -> Optional[float]:
        if self.bid is not None and self.ask is not None and self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        if self.last is not None and self.last > 0:
            return self.last
        return None

    @property
    def spread_pct(self) -> Optional[float]:
        m = self.mid
        if m and self.bid is not None and self.ask is not None and self.bid > 0:
            return (self.ask - self.bid) / m
        return None


class ChainSnapshot(Event):
    symbol: str
    underlying_price: float
    quotes: list[OptionQuote]
    quality: DataQuality = DataQuality.DELAYED


class IVMetrics(Event):
    symbol: str
    iv_current: float
    iv_rank: Optional[float] = None  # 0-100, None until enough history
    iv_percentile: Optional[float] = None
    history_days: int = 0
    expected_move_abs: Optional[float] = None
    expected_move_pct: Optional[float] = None
    expected_move_expiry: Optional[date] = None
    realized_vol: Optional[float] = None  # annualized, from stored daily closes
    rv_days: int = 0


class Leg(BaseModel):
    model_config = ConfigDict(frozen=True)

    right: Literal["C", "P", "STK"]
    strike: float = 0.0
    expiry: Optional[date] = None
    quantity: int  # signed: + long, - short (STK: shares)
    price: Optional[float] = None  # per-share reference price
    delta: Optional[float] = None
    iv: Optional[float] = None


class SignalAction(str, Enum):
    OPEN = "open"
    CLOSE = "close"
    ROLL = "roll"


class Signal(Event):
    strategy: Literal["wheel", "straddle"]
    symbol: str
    action: SignalAction
    legs: list[Leg]
    reason: str
    score: float = 0.0
    capital_required: float = 0.0
    max_loss: Optional[float] = None
    meta: dict[str, Any] = Field(default_factory=dict)


class Verdict(str, Enum):
    APPROVE = "approve"
    FLAG = "flag"  # allowed, but warn (amber)
    BLOCK = "block"


class RiskDecision(Event):
    signal_id: str
    verdict: Verdict
    reasons: list[str] = Field(default_factory=list)
    utilization: dict[str, float] = Field(default_factory=dict)


class OrderStatus(str, Enum):
    PENDING_CONFIRMATION = "pending_confirmation"
    SUBMITTED = "submitted"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class OrderIntent(Event):
    signal_id: str
    symbol: str
    strategy: str
    legs: list[Leg]
    limit_price: Optional[float] = None
    reason: str


class OrderUpdate(Event):
    order_id: str
    signal_id: str
    status: OrderStatus
    detail: str = ""
    paper: bool = True
    fill_price: Optional[float] = None
    leg_fills: list[float] = Field(default_factory=list)  # per-share fill for each OrderIntent leg, same order


class AgentStatus(str, Enum):
    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"
    ERROR = "error"
    STOPPED = "stopped"


class Heartbeat(Event):
    agent: str
    status: AgentStatus
    last_error: Optional[str] = None


class KillSwitchChanged(Event):
    engaged: bool
    reason: str = ""
