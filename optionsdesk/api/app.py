"""FastAPI layer: REST + Server-Sent Events. The dashboard talks only to this API (never to IBKR)."""

from __future__ import annotations

import json
import secrets
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from ..agents.backtest import BacktestAgent
from ..agents.execution import ExecutionAgent
from ..agents.narrative import build_narrative
from ..events import (
    Heartbeat, IVMetrics, KillSwitchChanged, OrderUpdate, RiskDecision, Signal,
)
from ..logs import configure_logging
from ..runtime import Runtime, build_runtime
from ..settings import Settings, get_settings
from . import views

STREAM_TYPES = (Signal, RiskDecision, OrderUpdate, Heartbeat, IVMetrics, KillSwitchChanged)


class BacktestRequest(BaseModel):
    strategy: str
    tickers: list[str] = Field(min_length=1, max_length=20)
    start: date
    end: date
    overrides: Optional[dict[str, dict[str, Any]]] = None


class KillSwitchRequest(BaseModel):
    engaged: bool
    reason: str = ""


class ConfirmRequest(BaseModel):
    acknowledge_live: bool = False


def _rt(request: Request) -> Runtime:
    return request.app.state.rt


def _agent(rt: Runtime, name: str, cls):
    a = rt.agents.get(name)
    if not isinstance(a, cls):
        raise HTTPException(503, f"{name} agent is not running in this process")
    return a


def require_token(request: Request, x_api_token: Optional[str] = Header(default=None)) -> None:
    token = _rt(request).ctx.settings.api_token
    if token and not (x_api_token and secrets.compare_digest(x_api_token, token)):
        raise HTTPException(401, "missing or invalid X-API-Token")


async def event_stream(rt: Runtime):
    """Yield SSE messages for decision-relevant events (chains and raw quotes are too chatty)."""
    sub = rt.ctx.bus.subscribe(*STREAM_TYPES, maxsize=500)
    try:
        while True:
            ev = await sub.get()
            yield {"event": type(ev).__name__, "id": ev.id,
                   "data": json.dumps(ev.model_dump(mode="json"), default=str)}
    finally:
        sub.close()


def create_app(settings: Optional[Settings] = None, runtime: Optional[Runtime] = None,
               start_agents: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        s = settings or get_settings()
        rt = runtime or build_runtime(s)
        app.state.rt = rt
        if start_agents:
            configure_logging(rt.ctx.settings.log_level)
            await rt.start()
        try:
            yield
        finally:
            if start_agents:
                await rt.stop()

    app = FastAPI(title="optionsdesk", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health(request: Request):
        rt = _rt(request)
        s = rt.ctx.settings
        return {"status": "ok", "mode": s.mode.value, "data_source": s.data_source,
                "execution_backend": s.execution_backend, "kill_switch": rt.ctx.db.kill_switch_engaged(),
                "agents": list(rt.agents)}

    @app.get("/agents")
    def agents(request: Request):
        return _rt(request).orchestrator.health()

    @app.get("/overview")
    def overview(request: Request):
        return views.overview(_rt(request))

    @app.get("/wheel")
    def wheel(request: Request):
        return views.wheel_view(_rt(request))

    @app.get("/straddle")
    def straddle(request: Request, mode: Optional[str] = Query(default=None, pattern="^(long|short)$")):
        return views.straddle_view(_rt(request), mode)

    @app.get("/scanner")
    def scanner(request: Request, strategy: str = Query("all", pattern="^(all|wheel_put|wheel_call|straddle)$"),
                min_delta: float = 0.0, max_delta: float = 1.0, min_dte: int = 0, max_dte: int = 365,
                min_iv_rank: float = 0.0, max_spread_pct: float = 1.0, min_open_interest: int = 0):
        f = views.ScannerFilters(strategy, min_delta, max_delta, min_dte, max_dte, min_iv_rank, max_spread_pct,
                                 min_open_interest)
        return views.scanner(_rt(request), f)

    @app.get("/risk")
    def risk(request: Request):
        return views.risk_view(_rt(request))

    @app.get("/journal")
    def journal(request: Request, limit: int = Query(200, ge=1, le=1000)):
        return views.journal_view(_rt(request), limit)

    @app.get("/orders")
    def orders(request: Request, limit: int = Query(200, ge=1, le=1000)):
        return _rt(request).ctx.db.list_orders(limit)

    @app.get("/iv/{symbol}")
    def iv_history(request: Request, symbol: str):
        rows = _rt(request).ctx.db.iv_history(symbol.upper())
        return [{"day": d, "iv": v} for d, v in rows]

    @app.get("/narrative")
    def narrative(request: Request):
        rt = _rt(request)
        text = build_narrative(rt.ctx.db, rt.ctx.today(), rt.ctx.db.kill_switch_engaged())
        return {"text": text, "generated_at": datetime.now(timezone.utc).isoformat()}

    @app.get("/kill-switch")
    def kill_switch(request: Request):
        db = _rt(request).ctx.db
        return {"engaged": db.kill_switch_engaged(), "reason": db.kv_get("kill_switch_reason", "")}

    @app.post("/kill-switch", dependencies=[Depends(require_token)])
    async def set_kill_switch(request: Request, body: KillSwitchRequest):
        rt = _rt(request)
        ex = rt.agents.get("execution")
        if isinstance(ex, ExecutionAgent):
            await ex.set_kill_switch(body.engaged, body.reason)
        else:  # still persist the flag; any ExecutionAgent in another process reads it before each order
            rt.ctx.db.set_kill_switch(body.engaged, body.reason)
        return {"engaged": body.engaged}

    @app.post("/orders/{order_id}/confirm", dependencies=[Depends(require_token)])
    async def confirm(request: Request, order_id: str, body: ConfirmRequest):
        ex = _agent(_rt(request), "execution", ExecutionAgent)
        try:
            await ex.confirm(order_id, body.acknowledge_live)
        except KeyError as exc:
            raise HTTPException(404, str(exc))
        except PermissionError as exc:
            raise HTTPException(403, str(exc))
        return {"order_id": order_id, "status": "confirmed"}

    @app.post("/orders/{order_id}/reject", dependencies=[Depends(require_token)])
    def reject(request: Request, order_id: str):
        ex = _agent(_rt(request), "execution", ExecutionAgent)
        try:
            ex.reject(order_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc))
        return {"order_id": order_id, "status": "cancelled"}

    @app.post("/backtests", status_code=202, dependencies=[Depends(require_token)])
    def submit_backtest(request: Request, body: BacktestRequest):
        bt = _agent(_rt(request), "backtest", BacktestAgent)
        try:
            run_id = bt.submit(body.strategy, [t.upper() for t in body.tickers], body.start, body.end, body.overrides)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        return {"run_id": run_id, "status": "running"}

    @app.get("/backtests")
    def list_backtests(request: Request):
        return _rt(request).ctx.db.list_backtests()

    @app.get("/backtests/{run_id}")
    def get_backtest(request: Request, run_id: str):
        row = _rt(request).ctx.db.get_backtest(run_id)
        if row is None:
            raise HTTPException(404, "unknown run")
        return row

    @app.get("/stream")
    async def stream(request: Request):
        return EventSourceResponse(event_stream(_rt(request)), ping=15)

    return app


def app_factory() -> FastAPI:  # for `uvicorn optionsdesk.api.app:app_factory --factory`
    return create_app()
