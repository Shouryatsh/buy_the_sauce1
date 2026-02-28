"""
broker.py — IBKR order execution via ib_insync.

Responsibilities
----------------
* Connect to / disconnect from IBKR TWS or IB Gateway.
* Fetch account equity (net liquidation value).
* Fetch current open positions.
* Place bracket orders (entry limit + protective stop + take-profit limit).
* Cancel all open orders for a symbol.

All monetary values are in USD.

Usage
-----
    from broker import IBKRBroker
    with IBKRBroker() as broker:
        equity = broker.get_account_equity()
        positions = broker.get_open_positions()
        broker.place_bracket_order(order_spec)
"""

from __future__ import annotations

import logging
from typing import Generator

from ib_insync import IB, Contract, LimitOrder, Order, Stock, Trade

from risk_manager import OrderSpec
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
