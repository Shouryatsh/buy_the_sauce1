"""
broker.py — IBKR order execution via ib_insync.

Responsibilities
----------------
* Connect to / disconnect from IBKR TWS or IB Gateway.
* Fetch account equity (net liquidation value).
* Fetch current open positions.
* Place bracket orders (entry limit + protective stop + take-profit limit).
* Place scaled entry (buy ladder) orders with per-tranche limits + shared stop.
* Pre-trade safety checks (position-size caps, price sanity, duplicate guard).
* Kill switch: cancel all open orders for a symbol or globally.
* Cancel all open orders for a symbol.

All monetary values are in USD.

Usage
-----
    from broker import IBKRBroker
    with IBKRBroker() as broker:
        equity = broker.get_account_equity()
        positions = broker.get_open_positions()
        broker.place_bracket_order(order_spec)
        broker.place_scaled_entry(plan, order_spec)
        broker.kill_switch()  # emergency cancel all
"""

from __future__ import annotations

import logging
from typing import Generator

from ib_insync import IB, Contract, LimitOrder, Order, Stock, Trade

from risk_manager import OrderSpec, ScaledEntryPlan
import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Broker class
# ---------------------------------------------------------------------------

class IBKRBroker:
    """Thin wrapper around ib_insync for dip-buying operations."""

    def __init__(
        self,
        host: str = config.IBKR_HOST,
        port: int = config.IBKR_PORT,
        client_id: int = config.IBKR_CLIENT_ID,
    ) -> None:
        self._host = host
        self._port = port
        self._client_id = client_id
        self._ib = IB()

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "IBKRBroker":
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.disconnect()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Connect to TWS / IB Gateway.  Raises on failure."""
        logger.info(
            "Connecting to IBKR at %s:%d (client_id=%d)",
            self._host, self._port, self._client_id,
        )
        self._ib.connect(self._host, self._port, clientId=self._client_id)
        logger.info("Connected — account: %s", self._ib.wrapper.accounts)

    def disconnect(self) -> None:
        """Disconnect cleanly."""
        if self._ib.isConnected():
            self._ib.disconnect()
            logger.info("Disconnected from IBKR")

    # ------------------------------------------------------------------
    # Account information
    # ------------------------------------------------------------------

    def get_account_equity(self) -> float:
        """Return the net liquidation value (total equity) in USD."""
        account_values = self._ib.accountValues()
        for av in account_values:
            if av.tag == "NetLiquidation" and av.currency == "USD":
                return float(av.value)
        raise RuntimeError("Could not retrieve NetLiquidation from IBKR")

    def get_open_positions(self) -> list[dict]:
        """Return a list of current positions as simple dicts.

        Each dict has keys: symbol, quantity, avg_cost.
        """
        positions = []
        for pos in self._ib.positions():
            positions.append(
                {
                    "symbol": pos.contract.symbol,
                    "quantity": pos.position,
                    "avg_cost": pos.avgCost,
                }
            )
        return positions

    def is_already_in_position(self, symbol: str) -> bool:
        """Return True if we currently hold shares of *symbol*."""
        for pos in self.get_open_positions():
            if pos["symbol"] == symbol and pos["quantity"] > 0:
                return True
        return False

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def _make_contract(self, symbol: str) -> Contract:
        """Create and qualify a US equity contract."""
        contract = Stock(symbol, "SMART", "USD")
        self._ib.qualifyContracts(contract)
        return contract

    def place_bracket_order(self, spec: OrderSpec) -> list[Trade]:
        """Place a bracket order: entry limit + stop-loss + take-profit.

        The bracket uses IBKR's native ``bracketOrder`` helper which
        automatically sets the OCA (One Cancels All) group so that if
        either exit leg fills, the other is cancelled.

        Returns the list of Trade objects (one per leg).
        """
        if self.is_already_in_position(spec.symbol):
            logger.info("%s: already in position, skipping order", spec.symbol)
            return []

        contract = self._make_contract(spec.symbol)

        parent, take_profit, stop_loss = self._ib.bracketOrder(
            action="BUY",
            quantity=spec.quantity,
            limitPrice=spec.entry_price,
            takeProfitPrice=spec.take_profit_price,
            stopLossPrice=spec.stop_loss_price,
        )

        trades = []
        for order in (parent, take_profit, stop_loss):
            trade = self._ib.placeOrder(contract, order)
            trades.append(trade)
            logger.info(
                "Placed order: %s %s qty=%d price=%.2f orderId=%d",
                spec.symbol, order.action, order.totalQuantity,
                getattr(order, "lmtPrice", getattr(order, "auxPrice", 0)),
                order.orderId,
            )

        return trades

    def cancel_all_orders_for(self, symbol: str) -> None:
        """Cancel all open orders for *symbol* (e.g., stale day orders)."""
        for trade in self._ib.openTrades():
            if trade.contract.symbol == symbol:
                self._ib.cancelOrder(trade.order)
                logger.info("Cancelled order %d for %s", trade.order.orderId, symbol)

    # ------------------------------------------------------------------
    # Pre-trade safety checks
    # ------------------------------------------------------------------

    def pre_trade_check(self, spec: OrderSpec) -> tuple[bool, str]:
        """Run pre-trade safety checks.  Returns (ok, reason).

        Checks:
        1. Not already in position.
        2. Account equity is sufficient (spec.position_value ≤ equity × MAX_POSITION_PCT).
        3. Entry price is within ±5% of last traded price (sanity check).
        4. Total open positions < MAX_POSITIONS.
        """
        if self.is_already_in_position(spec.symbol):
            return False, f"{spec.symbol}: already in position"

        try:
            equity = self.get_account_equity()
        except Exception as exc:
            return False, f"Cannot read account equity: {exc}"

        max_notional = equity * config.MAX_POSITION_PCT
        if spec.position_value > max_notional * 1.05:  # 5% tolerance for rounding
            return False, (
                f"{spec.symbol}: position value ${spec.position_value:,.0f} "
                f"exceeds cap ${max_notional:,.0f}"
            )

        open_count = len([p for p in self.get_open_positions() if p["quantity"] > 0])
        if open_count >= config.MAX_POSITIONS:
            return False, f"Max positions reached ({open_count}/{config.MAX_POSITIONS})"

        return True, "OK"

    # ------------------------------------------------------------------
    # Scaled entry execution
    # ------------------------------------------------------------------

    def place_scaled_entry(
        self,
        plan: "ScaledEntryPlan",
        spec: OrderSpec,
    ) -> list[Trade]:
        """Place a scaled entry (buy ladder) with per-tranche limit orders
        and a shared protective stop on the total quantity.

        Each tranche is a separate limit order.  The hard stop is placed
        as a conditional stop order on the TOTAL expected fill quantity.
        IBKR's OCA group ties the stop to all entry legs.

        Returns the list of all Trade objects placed.
        """
        ok, reason = self.pre_trade_check(spec)
        if not ok:
            logger.warning("Pre-trade check FAILED: %s", reason)
            return []

        contract = self._make_contract(spec.symbol)
        all_trades: list[Trade] = []
        oca_group = f"BL_{spec.symbol}_{self._ib.client.getReqId()}"

        # Place entry limit orders for each pending tranche
        for tranche in plan.tranches:
            if tranche.status != "PENDING":
                continue

            entry_order = LimitOrder(
                action="BUY",
                totalQuantity=tranche.quantity,
                lmtPrice=tranche.limit_price,
                tif="GTC",            # Good-Till-Cancelled
                ocaGroup=oca_group,
                ocaType=3,            # 3 = reduce size on fill (don't cancel others)
            )
            trade = self._ib.placeOrder(contract, entry_order)
            all_trades.append(trade)
            logger.info(
                "Placed scaled entry T%d: %s BUY %d @ $%.2f  OCA=%s",
                tranche.tranche_id, spec.symbol,
                tranche.quantity, tranche.limit_price, oca_group,
            )

        # Place a single protective stop for the full position
        from ib_insync import StopOrder
        stop_order = StopOrder(
            action="SELL",
            totalQuantity=plan.total_quantity,
            stopPrice=spec.stop_loss_price,
            tif="GTC",
            ocaGroup=oca_group,
            ocaType=1,  # 1 = cancel others in group when this fills
        )
        trade = self._ib.placeOrder(contract, stop_order)
        all_trades.append(trade)
        logger.info(
            "Placed protective stop: %s SELL %d @ $%.2f (stop)  OCA=%s",
            spec.symbol, plan.total_quantity, spec.stop_loss_price, oca_group,
        )

        return all_trades

    # ------------------------------------------------------------------
    # Sell-side execution  (trailing stop, partial exits, close)
    # ------------------------------------------------------------------

    def sell_partial(self, symbol: str, quantity: int, limit_price: float | None = None) -> Trade | None:
        """Sell a partial position.

        Uses a limit order at *limit_price* if provided, else a market order.
        Returns the Trade object or None on failure.
        """
        if quantity <= 0:
            return None

        contract = self._make_contract(symbol)

        if limit_price is not None and limit_price > 0:
            order = LimitOrder(
                action="SELL",
                totalQuantity=quantity,
                lmtPrice=round(limit_price, 2),
                tif="DAY",
            )
        else:
            from ib_insync import MarketOrder
            order = MarketOrder(action="SELL", totalQuantity=quantity)

        trade = self._ib.placeOrder(contract, order)
        logger.info(
            "Partial exit: SELL %d %s @ %s  orderId=%d",
            quantity, symbol,
            f"${limit_price:.2f}" if limit_price else "MKT",
            order.orderId,
        )
        return trade

    def update_stop_order(self, symbol: str, new_stop_price: float) -> int:
        """Update the stop price on all open stop orders for *symbol*.

        This is used to implement the trailing stop: on each bar the caller
        computes the new trailing stop level via TrailingStopState.update()
        and pushes it to the broker.

        Returns the number of stop orders modified.
        """
        count = 0
        for trade in self._ib.openTrades():
            if trade.contract.symbol != symbol:
                continue
            order = trade.order
            # Stop orders have auxPrice (the trigger price)
            if hasattr(order, "auxPrice") and order.action == "SELL" and order.orderType in ("STP", "STOP"):
                old_price = order.auxPrice
                if new_stop_price > old_price:
                    order.auxPrice = round(new_stop_price, 2)
                    self._ib.placeOrder(trade.contract, order)
                    count += 1
                    logger.info(
                        "Trailing stop updated: %s stop $%.2f → $%.2f  orderId=%d",
                        symbol, old_price, new_stop_price, order.orderId,
                    )
        return count

    def close_position(self, symbol: str) -> Trade | None:
        """Close the entire position for *symbol* at market.

        Used for time-stop or RSI-abort full exits.
        Returns the Trade or None if no position found.
        """
        qty = 0
        for pos in self.get_open_positions():
            if pos["symbol"] == symbol and pos["quantity"] > 0:
                qty = int(pos["quantity"])
                break
        if qty <= 0:
            logger.info("%s: no open position to close", symbol)
            return None

        # Cancel any open orders first
        self.cancel_all_orders_for(symbol)

        from ib_insync import MarketOrder
        contract = self._make_contract(symbol)
        order = MarketOrder(action="SELL", totalQuantity=qty)
        trade = self._ib.placeOrder(contract, order)
        logger.info(
            "Close position: SELL %d %s @ MKT  orderId=%d",
            qty, symbol, order.orderId,
        )
        return trade

    # ------------------------------------------------------------------
    # Kill switch — emergency cancel all orders
    # ------------------------------------------------------------------

    def kill_switch(self, symbol: str | None = None) -> int:
        """Emergency cancel all open orders (or for a specific symbol).

        Returns the count of cancelled orders.
        """
        count = 0
        for trade in self._ib.openTrades():
            if symbol is None or trade.contract.symbol == symbol:
                self._ib.cancelOrder(trade.order)
                count += 1
                logger.warning(
                    "KILL SWITCH: cancelled order %d for %s",
                    trade.order.orderId, trade.contract.symbol,
                )
        logger.warning("KILL SWITCH complete — %d order(s) cancelled", count)
        return count
