"""BacktestAgent: runs historical simulations in a worker thread so the API/UI never block.

Jobs run one at a time. Price data: synthetic walk when data_source=simulated, otherwise
yfinance daily closes (free). Option prices in the simulation are modelled - see backtest.py.
"""

from __future__ import annotations

import asyncio
import math
import zlib
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from ..backtest import run_straddle_backtest, run_wheel_backtest
from ..earnings import load_earnings
from ..events import AgentStatus, new_id
from ..settings import Settings
from .base import BaseAgent, Context

PriceLoader = Callable[[list[str], date, date], dict[str, pd.Series]]
WARMUP_DAYS = 400  # calendar days of history before `start` for vol/IV-rank warm-up


def simulated_loader(tickers: list[str], start: date, end: date) -> dict[str, pd.Series]:
    idx = pd.bdate_range(pd.Timestamp(start) - pd.Timedelta(days=WARMUP_DAYS), pd.Timestamp(end))
    out = {}
    for t in tickers:
        rng = np.random.default_rng(zlib.crc32(t.encode()))
        r = rng.normal(0.08 / 252, 0.27 / math.sqrt(252), len(idx))
        out[t] = pd.Series(100 * np.exp(np.cumsum(r)), index=idx)
    return out


def yfinance_loader(tickers: list[str], start: date, end: date) -> dict[str, pd.Series]:
    import yfinance as yf
    begin = pd.Timestamp(start) - pd.Timedelta(days=WARMUP_DAYS)
    out = {}
    for t in tickers:
        df = yf.download(t, start=begin.date().isoformat(), end=(pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat(),
                         progress=False, auto_adjust=True)
        if df is None or df.empty:
            raise LookupError(f"no price history for {t}")
        close = df["Close"]
        close = close.iloc[:, 0] if hasattr(close, "columns") else close
        out[t] = close.dropna()
    return out


def default_loader(settings: Settings) -> PriceLoader:
    return simulated_loader if settings.data_source == "simulated" else yfinance_loader


def merge_overrides(settings: Settings, overrides: Optional[dict[str, Any]]) -> Settings:
    data = settings.model_dump()
    for section, values in (overrides or {}).items():
        if section not in ("wheel", "straddle", "backtest"):
            raise ValueError(f"cannot override '{section}'")
        data[section] = {**data[section], **values}
    return Settings(**data)


@dataclass
class Job:
    run_id: str
    strategy: str
    tickers: list[str]
    start: date
    end: date
    overrides: Optional[dict[str, Any]]


class BacktestAgent(BaseAgent):
    name = "backtest"

    def __init__(self, ctx: Context, loader: Optional[PriceLoader] = None) -> None:
        super().__init__(ctx)
        self.loader = loader or default_loader(self.settings)
        self.queue: asyncio.Queue[Job] = asyncio.Queue()
        self.current: Optional[str] = None

    def submit(self, strategy: str, tickers: list[str], start: date, end: date,
               overrides: Optional[dict[str, Any]] = None) -> str:
        if strategy not in ("wheel", "straddle_long", "straddle_short"):
            raise ValueError("strategy must be wheel, straddle_long or straddle_short")
        if not tickers or start >= end:
            raise ValueError("need at least one ticker and start < end")
        merge_overrides(self.settings, overrides)  # validate early
        run_id = new_id()
        params = {"tickers": tickers, "start": start.isoformat(), "end": end.isoformat(), "overrides": overrides or {}}
        self.db.create_backtest(run_id, strategy, params)
        self.queue.put_nowait(Job(run_id, strategy, tickers, start, end, overrides))
        return run_id

    def execute(self, job: Job) -> dict:
        """Blocking; runs in a worker thread."""
        s = merge_overrides(self.settings, job.overrides)
        prices = self.loader(job.tickers, job.start, job.end)
        events = load_earnings(s.earnings_file)
        if job.strategy == "wheel":
            res = run_wheel_backtest(prices, job.start, job.end, s, events)
        else:
            res = run_straddle_backtest(prices, job.start, job.end, s, events, mode=job.strategy.split("_")[1])
        return res.to_json()

    async def run_job(self, job: Job) -> None:
        self.current = job.run_id
        try:
            out = await asyncio.to_thread(self.execute, job)
            self.db.finish_backtest(job.run_id, {**out["metrics"], "n_trades": len(out["trades"])},
                                    {"equity": out["equity"], "drawdown": out["drawdown"], "trades": out["trades"]})
            self.beat(AgentStatus.RUNNING, ran=True)
        except Exception as exc:  # noqa: BLE001
            self.db.finish_backtest(job.run_id, {}, [], error=f"{type(exc).__name__}: {exc}")
            self.degrade(exc, f"backtest {job.run_id}")
        finally:
            self.current = None

    async def run(self) -> None:
        self.beat(AgentStatus.RUNNING)
        while True:
            try:
                job = await asyncio.wait_for(self.queue.get(), timeout=self.settings.heartbeat_interval_seconds)
            except asyncio.TimeoutError:
                self.beat(AgentStatus.RUNNING if self.status is not AgentStatus.ERROR else None)
                continue
            await self.run_job(job)
