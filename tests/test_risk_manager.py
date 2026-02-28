"""
tests/test_risk_manager.py — Unit tests for risk_manager.py.

All tests are offline; no market data or broker connections required.
"""

from __future__ import annotations

import pytest

import config
from risk_manager import calculate_order, OrderSpec


# ---------------------------------------------------------------------------
# Basic order calculation
# ---------------------------------------------------------------------------

class TestCalculateOrder:
    def test_returns_order_spec(self):
        spec = calculate_order("AAPL", entry_price=150.0, account_equity=100_000.0)
        assert spec is not None
        assert isinstance(spec, OrderSpec)

    def test_stop_loss_below_entry(self):
        spec = calculate_order("AAPL", entry_price=150.0, account_equity=100_000.0)
        assert spec is not None
        assert spec.stop_loss_price < spec.entry_price

    def test_take_profit_above_entry(self):
        spec = calculate_order("AAPL", entry_price=150.0, account_equity=100_000.0)
        assert spec is not None
        assert spec.take_profit_price > spec.entry_price

    def test_stop_loss_percentage(self):
        entry = 200.0
        spec = calculate_order("TEST", entry_price=entry, account_equity=200_000.0,
                               stop_loss_pct=0.07)
        assert spec is not None
        expected_stop = round(entry * (1 - 0.07), 2)
        assert spec.stop_loss_price == pytest.approx(expected_stop)

    def test_take_profit_percentage(self):
        entry = 200.0
        spec = calculate_order("TEST", entry_price=entry, account_equity=200_000.0,
                               take_profit_pct=0.15)
        assert spec is not None
        expected_tp = round(entry * (1 + 0.15), 2)
        assert spec.take_profit_price == pytest.approx(expected_tp)

    def test_risk_amount_matches_budget(self):
        """Risk amount should be ≤ risk_per_trade_pct * equity."""
        equity = 100_000.0
        spec = calculate_order("TEST", entry_price=100.0, account_equity=equity,
                               risk_per_trade_pct=0.01)
        assert spec is not None
        assert spec.risk_amount <= equity * 0.01 + 0.01  # tiny rounding tolerance

    def test_position_value_within_cap(self):
        """Position notional should not exceed max_position_pct * equity."""
        equity = 100_000.0
        spec = calculate_order("TEST", entry_price=50.0, account_equity=equity,
                               max_position_pct=0.05)
        assert spec is not None
        assert spec.position_value <= equity * 0.05 + 50.0  # allow one share rounding

    def test_quantity_positive_integer(self):
        spec = calculate_order("TEST", entry_price=100.0, account_equity=50_000.0)
        assert spec is not None
        assert isinstance(spec.quantity, int)
        assert spec.quantity > 0

    # ------------------------------------------------------------------
    # Edge cases — should return None
    # ------------------------------------------------------------------

    def test_returns_none_for_zero_entry_price(self):
        result = calculate_order("TEST", entry_price=0.0, account_equity=100_000.0)
        assert result is None

    def test_returns_none_for_negative_entry_price(self):
        result = calculate_order("TEST", entry_price=-10.0, account_equity=100_000.0)
        assert result is None

    def test_returns_none_for_zero_equity(self):
        result = calculate_order("TEST", entry_price=100.0, account_equity=0.0)
        assert result is None

    def test_returns_none_when_max_positions_reached(self):
        result = calculate_order(
            "TEST",
            entry_price=100.0,
            account_equity=100_000.0,
            open_positions=config.MAX_POSITIONS,
            max_positions=config.MAX_POSITIONS,
        )
        assert result is None

    def test_returns_none_for_very_small_equity(self):
        # Equity so small that position size rounds to 0
        result = calculate_order("TEST", entry_price=10_000.0, account_equity=10.0)
        assert result is None

    # ------------------------------------------------------------------
    # Risk / reward ratio sanity check
    # ------------------------------------------------------------------

    def test_risk_reward_ratio(self):
        """Default config: take-profit 15%, stop-loss 7% → R:R > 2."""
        entry = 100.0
        spec = calculate_order("TEST", entry_price=entry, account_equity=100_000.0)
        assert spec is not None
        reward = spec.take_profit_price - entry
        risk = entry - spec.stop_loss_price
        assert risk > 0
        rr = reward / risk
        assert rr > 2.0  # 15% / 7% ≈ 2.14

    # ------------------------------------------------------------------
    # __str__ smoke test
    # ------------------------------------------------------------------

    def test_str_representation(self):
        spec = calculate_order("MSFT", entry_price=300.0, account_equity=200_000.0)
        assert spec is not None
        s = str(spec)
        assert "MSFT" in s
        assert "entry=" in s
        assert "stop=" in s
        assert "target=" in s
