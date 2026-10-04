"""
wheel_ibkr.py — Read-only IBKR account/position/order/trade-history pull for
the options wheel dashboard, via ib_insync.

Threading model
----------------
ib_insync drives its socket I/O through asyncio, and its transport/protocol
plus every pending-request Future are permanently bound to whichever event
loop was *running* when IB.connect() executed. Streamlit re-executes the
whole script on a new native thread for every rerun (same thread name,
"ScriptRunner.scriptThread", but a different Thread object each time), so a
WheelIBKRClient kept alive across reruns via st.session_state would
otherwise be asked to make requests from a thread with no relationship to
the loop its connection actually lives on. Just giving that new thread its
own fresh event loop doesn't fix it either -- that loop has never seen the
live socket, so any request submitted to it hangs forever waiting for a
response that's actually arriving on a different loop nobody is polling.

The fix: every bit of ib_insync work happens on one dedicated background
thread (_IBLoopThread) that owns a single event loop for the life of the
process. WheelIBKRClient.run() submits coroutines to it via
asyncio.run_coroutine_threadsafe() and blocks whichever thread is calling
until the result comes back -- so callers never need to think about which
thread they're on. Plain cache reads that ib_insync documents as
non-blocking (positions(), portfolio(), accountValues(), openTrades()) are
called directly since they're just synchronized dict/list reads, not new
requests, and don't touch the loop at all.

This is also deliberately separate from broker.py (the buy-the-dip bot's
order executor): the wheel dashboard is read-only through Phase 4, and uses
its own client id (wheel_config.IBKR_CLIENT_ID) so it can connect to the
same TWS/Gateway instance alongside the dip bot without a client-id
collision.

Known limitation: IBKR's execution/fills API (reqExecutions) has turned out
to retain essentially nothing beyond the current session in practice
(confirmed empty even on a 90-day request against a live account) -- nowhere
near a typical wheel position's holding period. Two consequences, both
handled in wheel_db.py: (1) get_trade_stats() below archives every fill it
sees via wheel_db.record_fill() so a history accumulates locally going
forward, since IBKR's API can't supply one after the fact; (2) a worthless
expiration generates *no execution at all* -- the position just disappears
from the account -- so it's invisible to fill-netting no matter how far back
you can query. For anything that closed before archiving was wired up, or
that will never appear as an execution, wheel_db.record_manual_trade() is
the only way to capture it. IBKR's Flex Query Web Service (free on every
account tier, not a paid feature -- enable it under Account Management >
Reporting > Flex Queries) is the real fix for complete automatic history;
not yet implemented here.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import wheel_config as wc
import wheel_db

logger = logging.getLogger(__name__)


class _IBLoopThread:
    """One persistent background thread running one asyncio event loop,
    shared by every WheelIBKRClient in this process (see module docstring)."""

    _instance: Optional["_IBLoopThread"] = None
    _create_lock = threading.Lock()

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, name="wheel-ibkr-loop", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    @classmethod
    def instance(cls) -> "_IBLoopThread":
        with cls._create_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def run(self, coro, timeout: float = 30.0):
        """Run *coro* on the dedicated loop and block the calling thread
        until it completes. Safe to call from any thread."""
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result(timeout=timeout)


@dataclass
class AccountSummary:
    net_liquidation: float = 0.0
    total_cash: float = 0.0
    buying_power: float = 0.0
    gross_position_value: float = 0.0
    margin_used: float = 0.0
    is_paper: bool = True


@dataclass
class PositionRecord:
    symbol: str
    sec_type: str            # "STK" or "OPT"
    right: Optional[str] = None       # "P" / "C" for options
    strike: Optional[float] = None
    expiry: Optional[str] = None
    quantity: float = 0.0
    avg_cost: float = 0.0
    market_price: Optional[float] = None
    market_value: Optional[float] = None
    unrealized_pnl: Optional[float] = None


@dataclass
class TradeStats:
    realized_pnl: float = 0.0
    win_rate: Optional[float] = None
    avg_premium_per_contract: Optional[float] = None
    avg_days_to_close: Optional[float] = None
    trade_count: int = 0


class WheelIBKRClient:
    """Thread-safe (see module docstring), read-only wrapper around ib_insync."""

    def __init__(
        self,
        host: str = wc.IBKR_HOST,
        port: Optional[int] = None,
        client_id: int = wc.IBKR_CLIENT_ID,
        use_paper: bool = wc.USE_PAPER_ACCOUNT,
    ) -> None:
        self._host = host
        self._use_paper = use_paper
        self._port = port or (wc.IBKR_PAPER_PORT if use_paper else wc.IBKR_LIVE_PORT)
        self._client_id = client_id
        self._loop_thread = _IBLoopThread.instance()
        # IB() must be constructed on the dedicated loop thread too:
        # eventkit's Event objects (which ib_insync builds many of at
        # construction time) bind to whatever loop is current when created.
        self._ib = self._loop_thread.run(self._make_ib())

    @staticmethod
    async def _make_ib():
        from ib_insync import IB
        return IB()

    def __enter__(self) -> "WheelIBKRClient":
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.disconnect()

    def run(self, factory, timeout: float = 30.0):
        """Call *factory* (a zero-arg callable) and await its result if
        needed -- entirely on the dedicated loop thread, including the
        initial call itself, not just the awaiting.

        This matters because several of ib_insync's "...Async" methods
        (e.g. reqExecutionsAsync, reqSecDefOptParamsAsync) are plain
        functions, not real coroutines: they send the actual request over
        the socket the instant they're *called* and merely return a Future
        to await later. Passing an already-evaluated call like
        `client.run(ib.reqExecutionsAsync(filt))` evaluates that call on
        the caller's thread before this method ever runs -- which crashes
        under Streamlit before run_coroutine_threadsafe is even involved.
        Wrapping it in a zero-arg callable defers the call itself until the
        loop thread runs it: `client.run(lambda: ib.reqExecutionsAsync(filt))`.
        """
        async def _runner():
            result = factory()
            if inspect.isawaitable(result):
                result = await result
            return result
        return self._loop_thread.run(_runner(), timeout=timeout)

    def run_many(self, factories: list, timeout: float = 120.0) -> list:
        """Like `.run()`, but runs every factory in *factories* concurrently
        on the dedicated loop instead of one at a time.

        This is the actual fix for "pulling N symbols' option chains takes
        forever": each snapshot-market-data request has IBKR's own ~11
        second server-side hold before it returns full tick data, no matter
        how few/many contracts are in it. Doing that serially for 20+
        symbols is 20+ x 11s+ of pure waiting; since it's I/O-bound (not
        CPU-bound), asyncio.gather lets all of those waits overlap on this
        one loop, so the wall-clock cost is closer to the slowest single
        symbol, not the sum of all of them.

        Returns a list of (result, exception) pairs, same order as
        *factories* -- one symbol's failure doesn't abort the batch.
        """
        async def _one(factory):
            try:
                result = factory()
                if inspect.isawaitable(result):
                    result = await result
                return (result, None)
            except Exception as exc:
                return (None, exc)

        async def _runner():
            return await asyncio.gather(*(_one(f) for f in factories))

        return self._loop_thread.run(_runner(), timeout=timeout)

    @property
    def ib(self):
        """The underlying ib_insync IB object. Safe to read its
        already-synchronized caches (positions(), portfolio(),
        accountValues(), openTrades()) or to construct Contract objects
        directly from any thread; route anything that issues a new
        request through `.run(...)` instead of calling it directly."""
        return self._ib

    def connect(self, timeout: float = 8.0) -> None:
        logger.info(
            "Connecting to IBKR (wheel dashboard) at %s:%d (client_id=%d, paper=%s)",
            self._host, self._port, self._client_id, self._use_paper,
        )
        self.run(
            lambda: self._ib.connectAsync(
                self._host, self._port, clientId=self._client_id,
                timeout=timeout, readonly=True,
            ),
            timeout=timeout + 5,
        )

    def disconnect(self) -> None:
        if self._ib.isConnected():
            self.run(lambda: self._ib.disconnect())

    def is_connected(self) -> bool:
        return self._ib.isConnected()

    # ------------------------------------------------------------------
    # Account summary
    # ------------------------------------------------------------------

    def get_account_summary(self) -> AccountSummary:
        values = {av.tag: av for av in self._ib.accountValues() if av.currency == "USD"}
        summary = AccountSummary(is_paper=self._use_paper)
        if "NetLiquidation" in values:
            summary.net_liquidation = float(values["NetLiquidation"].value)
        if "TotalCashValue" in values:
            summary.total_cash = float(values["TotalCashValue"].value)
        if "BuyingPower" in values:
            summary.buying_power = float(values["BuyingPower"].value)
        if "GrossPositionValue" in values:
            summary.gross_position_value = float(values["GrossPositionValue"].value)
        if "MaintMarginReq" in values:
            summary.margin_used = float(values["MaintMarginReq"].value)
        return summary

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def get_positions(self) -> list[PositionRecord]:
        records: list[PositionRecord] = []
        portfolio = {item.contract.conId: item for item in self._ib.portfolio()}
        for pos in self._ib.positions():
            c = pos.contract
            item = portfolio.get(c.conId)
            records.append(
                PositionRecord(
                    symbol=c.symbol,
                    sec_type=c.secType,
                    right=getattr(c, "right", None) or None,
                    strike=getattr(c, "strike", None) or None,
                    expiry=getattr(c, "lastTradeDateOrContractMonth", None) or None,
                    quantity=pos.position,
                    avg_cost=pos.avgCost,
                    market_price=item.marketPrice if item else None,
                    market_value=item.marketValue if item else None,
                    unrealized_pnl=item.unrealizedPNL if item else None,
                )
            )
        return records

    def get_open_orders(self) -> list[dict]:
        orders = []
        for trade in self._ib.openTrades():
            orders.append({
                "symbol": trade.contract.symbol,
                "sec_type": trade.contract.secType,
                "action": trade.order.action,
                "quantity": trade.order.totalQuantity,
                "order_type": trade.order.orderType,
                "limit_price": getattr(trade.order, "lmtPrice", None),
                "status": trade.orderStatus.status,
            })
        return orders

    # ------------------------------------------------------------------
    # Trade history / realized P&L
    # ------------------------------------------------------------------

    def get_recent_fills(self, days: int = 30) -> list:
        """Return ib_insync Fill objects from the last *days* (subject to
        IBKR's execution-report retention window — see module docstring)."""
        from ib_insync import ExecutionFilter

        cutoff = datetime.now() - timedelta(days=days)
        filt = ExecutionFilter(time=cutoff.strftime("%Y%m%d %H:%M:%S"))
        return self.run(lambda: self._ib.reqExecutionsAsync(filt))

    def get_trade_stats(self, days: int = 30) -> TradeStats:
        """Derive realized P&L / win rate / avg premium from recent fills
        (plus, separately, manually-logged trades -- see wheel_dashboard.py's
        Monthly P&L tab, which adds wheel_db.get_manual_realized_pnl_since()
        to whatever this returns).

        Options wheel P&L convention: selling a put/call collects premium
        (credit); closing (buy-to-close, assignment, or expiry) realizes the
        gain/loss. This groups fills by contract and nets them, which is a
        reasonable approximation for a single-leg wheel but won't reconcile
        exactly against IBKR's own realized-PNL report for multi-leg or
        rolled positions — cross-check against the TWS P&L tab.

        A worthless expiration is handled specially: it generates *no*
        execution at all (the position just disappears from the account),
        so a contract that never nets back to flat but whose expiry has
        already passed is treated as expired worthless -- the premium
        already collected is realized in full, rather than the position
        silently never counting as "closed".
        """
        fills = self.get_recent_fills(days=days)

        # Archive every fill seen -- see module docstring on why IBKR's own
        # API can't be trusted to still have this next time we ask.
        for f in fills:
            wheel_db.record_fill(
                symbol=f.contract.symbol,
                sec_type=f.contract.secType,
                action=f.execution.side,
                quantity=f.execution.shares,
                price=f.execution.price,
                conid=f.contract.conId,
                closed_date=f.execution.time.date().isoformat() if f.execution.time else None,
            )

        if not fills:
            return TradeStats()

        by_contract: dict = {}
        for f in fills:
            key = f.contract.conId
            by_contract.setdefault(key, []).append(f)

        realized_total = 0.0
        premiums = []
        durations = []
        wins = 0
        closed_trades = 0
        today = datetime.now().date()

        for key, contract_fills in by_contract.items():
            contract_fills.sort(key=lambda f: f.execution.time)
            net_qty = 0.0
            net_cash = 0.0
            open_time = contract_fills[0].execution.time
            for f in contract_fills:
                qty = f.execution.shares if f.execution.side == "BOT" else -f.execution.shares
                price = f.execution.price
                multiplier = float(contract_fills[0].contract.multiplier or 1)
                cash_flow = -qty * price * multiplier
                net_qty += qty
                net_cash += cash_flow
                if f.execution.side == "SLD":
                    premiums.append(price * multiplier)
                if abs(net_qty) < 1e-6 and abs(cash_flow) > 0:
                    # position flattened -> a closed trade
                    closed_trades += 1
                    realized_total += net_cash
                    if net_cash > 0:
                        wins += 1
                    durations.append((f.execution.time - open_time).days)
                    net_cash = 0.0
                    open_time = f.execution.time

            # Still "open" per the fills we have, but its expiry has already
            # passed -> nothing will ever arrive to net against (expiration
            # isn't an execution). Realize what was collected instead of
            # silently dropping it.
            contract = contract_fills[-1].contract
            expiry_str = getattr(contract, "lastTradeDateOrContractMonth", None)
            if abs(net_qty) > 1e-6 and expiry_str:
                try:
                    expiry_date = datetime.strptime(expiry_str[:8], "%Y%m%d").date()
                except ValueError:
                    expiry_date = None
                if expiry_date and expiry_date < today:
                    closed_trades += 1
                    realized_total += net_cash
                    if net_cash > 0:
                        wins += 1
                    durations.append((expiry_date - open_time.date()).days)

        stats = TradeStats(
            realized_pnl=round(realized_total, 2),
            trade_count=closed_trades,
        )
        if closed_trades:
            stats.win_rate = wins / closed_trades
        if premiums:
            stats.avg_premium_per_contract = sum(premiums) / len(premiums)
        if durations:
            stats.avg_days_to_close = sum(durations) / len(durations)
        return stats
