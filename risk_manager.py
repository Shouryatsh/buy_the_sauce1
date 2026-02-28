"""
risk_manager.py — Position sizing, stop-loss, and take-profit calculations.

Design
------
* Risk a fixed fraction of account equity per trade (RISK_PER_TRADE_PCT).
* Stop-loss is placed STOP_LOSS_PCT below the entry price.
* Take-profit is placed TAKE_PROFIT_PCT above the entry price.
* Position size (# shares) is derived so that hitting the stop-loss
  costs exactly RISK_PER_TRADE_PCT of equity.
* Additional guardrails:
  - No single position may exceed MAX_POSITION_PCT of equity.
  - Total open positions are capped at MAX_POSITIONS.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
class OrderSpec:
    """All the parameters needed to place a bracket order."""
    symbol: str
    quantity: int
    entry_price: float        # limit price for the entry leg
    stop_loss_price: float    # stop price for the protective stop
    take_profit_price: float  # limit price for the take-profit leg
    risk_amount: float        # dollar risk for this trade
    position_value: float     # notional value of the position

    def __str__(self) -> str:
        return (
            f"{self.symbol}: qty={self.quantity} "
            f"entry={self.entry_price:.2f} "
            f"stop={self.stop_loss_price:.2f} (-{config.STOP_LOSS_PCT*100:.0f}%) "
            f"target={self.take_profit_price:.2f} (+{config.TAKE_PROFIT_PCT*100:.0f}%) "
            f"risk=${self.risk_amount:.2f} "
            f"notional=${self.position_value:.2f}"
        )


# ---------------------------------------------------------------------------
# Main calculation
# ---------------------------------------------------------------------------

def calculate_order(
    symbol: str,
    entry_price: float,
    account_equity: float,
    open_positions: int = 0,
    stop_loss_pct: float = config.STOP_LOSS_PCT,
    take_profit_pct: float = config.TAKE_PROFIT_PCT,
    risk_per_trade_pct: float = config.RISK_PER_TRADE_PCT,
    max_position_pct: float = config.MAX_POSITION_PCT,
    max_positions: int = config.MAX_POSITIONS,
) -> OrderSpec | None:
    """Calculate position size and bracket-order prices.

    Parameters
    ----------
    symbol:
        Ticker symbol.
    entry_price:
        Expected entry price (e.g. last close or mid-spread).
    account_equity:
        Total account net liquidation value in USD.
    open_positions:
        Number of positions currently held (to check MAX_POSITIONS).
    stop_loss_pct:
        Fraction below entry to place the stop.
    take_profit_pct:
        Fraction above entry to place the take-profit.
    risk_per_trade_pct:
        Fraction of equity to risk per trade.
    max_position_pct:
        Maximum fraction of equity in a single position.
    max_positions:
        Maximum total concurrent positions allowed.

    Returns
    -------
    OrderSpec or None if the trade should be skipped.
    """
    if entry_price <= 0:
        logger.warning("%s: invalid entry price %.4f", symbol, entry_price)
        return None

    if account_equity <= 0:
        logger.warning("Account equity %.2f is non-positive", account_equity)
        return None

    if open_positions >= max_positions:
        logger.info(
            "%s: skipped — already at max positions (%d/%d)",
            symbol, open_positions, max_positions,
        )
        return None

    stop_loss_price = round(entry_price * (1.0 - stop_loss_pct), 2)
    take_profit_price = round(entry_price * (1.0 + take_profit_pct), 2)

    # Dollar risk per share
    risk_per_share = entry_price - stop_loss_price

    # Max dollar risk for this trade
    max_risk_dollars = account_equity * risk_per_trade_pct

    # Shares based on risk budget
    quantity_by_risk = int(max_risk_dollars / risk_per_share) if risk_per_share > 0 else 0

    # Shares based on position-size cap
    max_position_value = account_equity * max_position_pct
    quantity_by_cap = int(max_position_value / entry_price) if entry_price > 0 else 0

    quantity = min(quantity_by_risk, quantity_by_cap)

    if quantity <= 0:
        logger.info(
            "%s: skipped — position size rounded to 0 "
            "(equity=%.0f, risk_budget=%.0f, risk_per_share=%.2f)",
            symbol, account_equity, max_risk_dollars, risk_per_share,
        )
        return None

    position_value = quantity * entry_price
    risk_amount = quantity * risk_per_share

    spec = OrderSpec(
        symbol=symbol,
        quantity=quantity,
        entry_price=round(entry_price, 2),
        stop_loss_price=stop_loss_price,
        take_profit_price=take_profit_price,
        risk_amount=round(risk_amount, 2),
        position_value=round(position_value, 2),
    )
    logger.info("OrderSpec: %s", spec)
    return spec
