"""
tests/test_risk_manager.py — Unit tests for risk_manager.py.

All tests are offline; no market data or broker connections required.
"""

from __future__ import annotations

import datetime

import pytest

import config
from risk_manager import (
    calculate_order, OrderSpec,
    plan_scaled_entry, ScaledEntryPlan, Tranche,
    TrailingStopState,
    evaluate_partial_exits, PartialExitSignal,
    check_time_stop,
)


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


# ---------------------------------------------------------------------------
# Scaled entry (buy ladder)
# ---------------------------------------------------------------------------

class TestScaledEntry:
    """Tests for plan_scaled_entry() and ScaledEntryPlan."""

    @pytest.fixture
    def base_spec(self) -> OrderSpec:
        """A simple OrderSpec for testing (100 shares @ $100)."""
        return OrderSpec(
            symbol="TEST", quantity=100, entry_price=100.0,
            stop_loss_price=94.0, take_profit_price=109.0,
            risk_amount=600.0, position_value=10_000.0,
            stop_method="ATR", stop_pct=0.06, target_pct=0.09,
            reward_risk_ratio=1.5, kelly_qty=0, risk_qty=100,
            vol_scale=1.0, sector_scale=1.0, sizing_method="risk",
        )

    def test_plan_creates_correct_number_of_tranches(self, base_spec):
        plan = plan_scaled_entry(base_spec, atr_14=0.02)
        assert plan.n_tranches == config.SCALED_ENTRY_N_TRANCHES

    def test_plan_total_quantity_equals_spec(self, base_spec):
        plan = plan_scaled_entry(base_spec, atr_14=0.02)
        total = sum(t.quantity for t in plan.tranches)
        assert total == base_spec.quantity

    def test_tranche_prices_decrease(self, base_spec):
        plan = plan_scaled_entry(base_spec, atr_14=0.02)
        prices = [t.limit_price for t in plan.tranches]
        for i in range(1, len(prices)):
            assert prices[i] <= prices[i - 1], f"T{i+1} price should be ≤ T{i}"

    def test_t1_price_equals_entry(self, base_spec):
        plan = plan_scaled_entry(base_spec, atr_14=0.02)
        assert plan.tranches[0].limit_price == base_spec.entry_price

    def test_all_tranches_start_pending(self, base_spec):
        plan = plan_scaled_entry(base_spec, atr_14=0.02)
        for t in plan.tranches:
            assert t.status == "PENDING"

    def test_abort_unfilled_cancels_pending(self, base_spec):
        plan = plan_scaled_entry(base_spec, atr_14=0.02)
        plan.tranches[0].status = "FILLED"
        n = plan.abort_unfilled("test")
        assert n == plan.n_tranches - 1
        assert plan.tranches[0].status == "FILLED"
        for t in plan.tranches[1:]:
            assert t.status == "CANCELLED"

    def test_rsi_abort_cancels_t2_plus(self, base_spec):
        plan = plan_scaled_entry(base_spec, atr_14=0.02)
        plan.tranches[0].status = "FILLED"
        n = plan.check_rsi_abort(current_rsi=55.0)  # > 50 default abort
        assert n == plan.n_tranches - 1
        assert plan.tranches[0].status == "FILLED"

    def test_rsi_below_abort_keeps_tranches(self, base_spec):
        plan = plan_scaled_entry(base_spec, atr_14=0.02)
        n = plan.check_rsi_abort(current_rsi=30.0)
        assert n == 0
        for t in plan.tranches:
            assert t.status == "PENDING"

    def test_expiry_cancels_past_date(self, base_spec):
        plan = plan_scaled_entry(base_spec, atr_14=0.02)
        future = plan.expiry_date + datetime.timedelta(days=1)
        n = plan.check_expiry(today=future)
        assert n == plan.n_tranches

    def test_expiry_keeps_before_date(self, base_spec):
        plan = plan_scaled_entry(base_spec, atr_14=0.02)
        before = plan.expiry_date - datetime.timedelta(days=1)
        n = plan.check_expiry(today=before)
        assert n == 0

    def test_should_fill_t1_at_limit(self, base_spec):
        plan = plan_scaled_entry(base_spec, atr_14=0.02)
        t1 = plan.tranches[0]
        assert plan.should_fill_tranche(t1, current_price=99.0)

    def test_should_not_fill_above_limit(self, base_spec):
        plan = plan_scaled_entry(base_spec, atr_14=0.02)
        t1 = plan.tranches[0]
        assert not plan.should_fill_tranche(t1, current_price=101.0)

    def test_t2_requires_rsi_decline_when_enabled(self, base_spec):
        plan = plan_scaled_entry(base_spec, atr_14=0.02)
        t2 = plan.tranches[1]
        # Price at T2 limit, RSI rising → should NOT fill
        assert not plan.should_fill_tranche(
            t2, current_price=t2.limit_price - 0.01,
            current_rsi=35.0, prior_rsi=33.0,  # RSI going up
        )
        # Price at T2 limit, RSI declining → should fill
        assert plan.should_fill_tranche(
            t2, current_price=t2.limit_price - 0.01,
            current_rsi=33.0, prior_rsi=35.0,  # RSI going down
        )

    def test_avg_fill_price_empty(self, base_spec):
        plan = plan_scaled_entry(base_spec, atr_14=0.02)
        assert plan.avg_fill_price == 0.0

    def test_avg_fill_price_partial(self, base_spec):
        plan = plan_scaled_entry(base_spec, atr_14=0.02)
        plan.tranches[0].status = "FILLED"
        plan.tranches[1].status = "FILLED"
        avg = plan.avg_fill_price
        assert avg > 0
        assert avg <= plan.tranches[0].limit_price  # T2 is lower

    def test_summary_table(self, base_spec):
        plan = plan_scaled_entry(base_spec, atr_14=0.02)
        table = plan.summary_table()
        assert len(table) == plan.n_tranches + 1  # +1 for TOTAL row
        assert table[-1]["Tranche"] == "TOTAL"

    def test_disabled_returns_single_tranche(self, base_spec, monkeypatch):
        monkeypatch.setattr(config, "SCALED_ENTRY_ENABLED", False)
        plan = plan_scaled_entry(base_spec, atr_14=0.02)
        assert plan.n_tranches == 1
        assert plan.tranches[0].quantity == base_spec.quantity

    def test_str_representation(self, base_spec):
        plan = plan_scaled_entry(base_spec, atr_14=0.02)
        s = str(plan)
        assert "TEST" in s
        assert "T1" in s


# ---------------------------------------------------------------------------
# Trailing stop
# ---------------------------------------------------------------------------

class TestTrailingStop:
    """Tests for the three-stage adaptive trailing stop."""

    def test_initial_stop_is_set(self):
        ts = TrailingStopState(entry_price=100.0, atr_14_abs=2.0, initial_stop=96.0)
        assert ts.current_stop == 96.0
        assert ts.stage == 0

    def test_stage_1_breakeven(self):
        ts = TrailingStopState(entry_price=100.0, atr_14_abs=2.0, initial_stop=96.0)
        # Price reaches entry + 1×ATR = 102
        ts.update(102.0)
        assert ts.stage == 1
        assert ts.current_stop == 100.0  # breakeven

    def test_stage_2_profit_lock(self):
        ts = TrailingStopState(entry_price=100.0, atr_14_abs=2.0, initial_stop=96.0)
        ts.update(104.0)  # entry + 2×ATR
        assert ts.stage == 2
        assert ts.current_stop == 102.0  # entry + 1×ATR

    def test_stage_3_tight_trail(self):
        ts = TrailingStopState(entry_price=100.0, atr_14_abs=2.0, initial_stop=96.0)
        ts.update(106.0)  # entry + 3×ATR
        assert ts.stage == 3
        # high=106, trail = 106 - 1×ATR = 104
        assert ts.current_stop == 104.0

    def test_trail_only_moves_up(self):
        ts = TrailingStopState(entry_price=100.0, atr_14_abs=2.0, initial_stop=96.0)
        ts.update(106.0)  # stage 3, stop=104
        stop_after_up = ts.current_stop
        ts.update(103.0)  # price drops but stop must stay
        assert ts.current_stop == stop_after_up

    def test_trail_advances_with_higher_highs(self):
        ts = TrailingStopState(entry_price=100.0, atr_14_abs=2.0, initial_stop=96.0)
        ts.update(106.0)  # stop = 104
        ts.update(108.0)  # higher high → stop = 108 - 2 = 106
        assert ts.current_stop == 106.0

    def test_disabled_trailing_stop(self, monkeypatch):
        monkeypatch.setattr(config, "TRAILING_STOP_ENABLED", False)
        ts = TrailingStopState(entry_price=100.0, atr_14_abs=2.0, initial_stop=96.0)
        ts.update(110.0)
        assert ts.current_stop == 96.0  # unchanged
        assert ts.stage == 0

    def test_stage_label(self):
        ts = TrailingStopState(entry_price=100.0, atr_14_abs=2.0, initial_stop=96.0)
        assert ts.stage_label == "INITIAL"
        ts.update(102.0)
        assert ts.stage_label == "BREAKEVEN"
        ts.update(104.0)
        assert ts.stage_label == "PROFIT_LOCK"
        ts.update(106.0)
        assert ts.stage_label == "TIGHT_TRAIL"

    def test_str_representation(self):
        ts = TrailingStopState(entry_price=100.0, atr_14_abs=2.0, initial_stop=96.0)
        s = str(ts)
        assert "INITIAL" in s
        assert "100.00" in s


# ---------------------------------------------------------------------------
# Partial exits
# ---------------------------------------------------------------------------

class TestPartialExits:
    """Tests for evaluate_partial_exits()."""

    def test_no_exits_below_trigger(self):
        signals = evaluate_partial_exits(
            current_price=103.0, entry_price=100.0, atr_14_abs=2.0,
        )
        assert len(signals) == 0

    def test_exit_1_triggered(self):
        # Trigger at entry + 2×ATR = 104
        signals = evaluate_partial_exits(
            current_price=104.5, entry_price=100.0, atr_14_abs=2.0,
        )
        assert len(signals) == 1
        assert signals[0].exit_id == 1
        assert signals[0].fraction == config.PARTIAL_EXIT_1_FRACTION

    def test_both_exits_triggered(self):
        # Trigger at entry + 3×ATR = 106
        signals = evaluate_partial_exits(
            current_price=107.0, entry_price=100.0, atr_14_abs=2.0,
        )
        assert len(signals) == 2

    def test_already_taken_exit_not_repeated(self):
        signals = evaluate_partial_exits(
            current_price=107.0, entry_price=100.0, atr_14_abs=2.0,
            exits_already_taken={1},
        )
        assert len(signals) == 1
        assert signals[0].exit_id == 2

    def test_disabled_returns_empty(self, monkeypatch):
        monkeypatch.setattr(config, "PARTIAL_EXIT_ENABLED", False)
        signals = evaluate_partial_exits(
            current_price=110.0, entry_price=100.0, atr_14_abs=2.0,
        )
        assert len(signals) == 0


# ---------------------------------------------------------------------------
# Time stop
# ---------------------------------------------------------------------------

class TestTimeStop:
    def test_not_triggered_within_period(self):
        entry = datetime.date.today() - datetime.timedelta(days=10)
        assert not check_time_stop(entry)

    def test_triggered_after_period(self):
        entry = datetime.date.today() - datetime.timedelta(days=config.TIME_STOP_DAYS + 1)
        assert check_time_stop(entry)

    def test_disabled_never_triggers(self, monkeypatch):
        monkeypatch.setattr(config, "TIME_STOP_ENABLED", False)
        entry = datetime.date.today() - datetime.timedelta(days=999)
        assert not check_time_stop(entry)
