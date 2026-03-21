"""
tests/test_trader.py — Unit tests for trader.py position management.

Tests the position state lifecycle: register, persist, load, manage,
and unregister.  All tests are offline (no broker or market data).
"""

from __future__ import annotations

import datetime
import json
import os
import tempfile

import pytest

import config
from risk_manager import (
    TrailingStopState,
    ScaledEntryPlan,
    Tranche,
)


# ---------------------------------------------------------------------------
# PositionState serialisation
# ---------------------------------------------------------------------------

class TestPositionStateSerde:
    """Test PositionState.to_dict() / from_dict() round-trip."""

    def _make_state(self):
        from trader import PositionState
        ts = TrailingStopState(
            entry_price=150.0, atr_14_abs=3.0, initial_stop=144.0,
        )
        ts.update(155.0)  # advance to stage 1
        return PositionState(
            symbol="AAPL",
            entry_price=150.0,
            entry_date=datetime.date(2026, 1, 15),
            quantity=50,
            atr_14_abs=3.0,
            trailing_stop=ts,
            partial_exits_taken={1},
        )

    def test_round_trip(self):
        state = self._make_state()
        d = state.to_dict()
        # Ensure it's JSON-serialisable
        json_str = json.dumps(d)
        d2 = json.loads(json_str)
        from trader import PositionState
        restored = PositionState.from_dict(d2)

        assert restored.symbol == state.symbol
        assert restored.entry_price == state.entry_price
        assert restored.entry_date == state.entry_date
        assert restored.quantity == state.quantity
        assert restored.atr_14_abs == state.atr_14_abs
        assert restored.trailing_stop.current_stop == state.trailing_stop.current_stop
        assert restored.trailing_stop.stage == state.trailing_stop.stage
        assert restored.trailing_stop.highest_price == state.trailing_stop.highest_price
        assert restored.trailing_stop.n_updates == state.trailing_stop.n_updates
        assert restored.partial_exits_taken == state.partial_exits_taken

    def test_to_dict_keys(self):
        state = self._make_state()
        d = state.to_dict()
        assert "symbol" in d
        assert "entry_price" in d
        assert "entry_date" in d
        assert "trailing_stop" in d
        assert "partial_exits_taken" in d
        assert isinstance(d["partial_exits_taken"], list)

    def test_from_dict_with_empty_partials(self):
        from trader import PositionState
        ts = TrailingStopState(entry_price=100.0, atr_14_abs=2.0, initial_stop=96.0)
        state = PositionState(
            symbol="MSFT", entry_price=100.0,
            entry_date=datetime.date(2026, 2, 1),
            quantity=30, atr_14_abs=2.0, trailing_stop=ts,
        )
        d = state.to_dict()
        restored = PositionState.from_dict(d)
        assert restored.partial_exits_taken == set()


# ---------------------------------------------------------------------------
# Persistence (save/load)
# ---------------------------------------------------------------------------

class TestStatePersistence:
    """Test _save_state / _load_state with a temp file."""

    def test_save_and_load(self, monkeypatch, tmp_path):
        import trader

        state_file = str(tmp_path / "test_position_state.json")
        monkeypatch.setattr(trader, "_STATE_FILE", state_file)

        # Clear and register a position
        trader._open_positions.clear()
        trader.register_position(
            symbol="TEST",
            entry_price=100.0,
            quantity=25,
            atr_14_abs=2.0,
            stop_loss_price=96.0,
        )

        assert "TEST" in trader._open_positions
        assert os.path.exists(state_file)

        # Verify file contents
        with open(state_file) as f:
            data = json.load(f)
        assert "TEST" in data
        assert data["TEST"]["entry_price"] == 100.0

        # Clear in-memory and reload
        trader._open_positions.clear()
        assert len(trader._open_positions) == 0

        trader._load_state()
        assert "TEST" in trader._open_positions
        assert trader._open_positions["TEST"].entry_price == 100.0
        assert trader._open_positions["TEST"].quantity == 25

    def test_unregister_saves_state(self, monkeypatch, tmp_path):
        import trader

        state_file = str(tmp_path / "test_position_state.json")
        monkeypatch.setattr(trader, "_STATE_FILE", state_file)
        trader._open_positions.clear()

        trader.register_position("A", 50.0, 10, 1.0, 48.0)
        trader.register_position("B", 60.0, 20, 1.5, 57.0)
        assert len(trader._open_positions) == 2

        trader.unregister_position("A")
        assert "A" not in trader._open_positions

        # Reload from disk — should only have B
        trader._open_positions.clear()
        trader._load_state()
        assert "A" not in trader._open_positions
        assert "B" in trader._open_positions

    def test_load_nonexistent_file(self, monkeypatch, tmp_path):
        import trader

        state_file = str(tmp_path / "nonexistent.json")
        monkeypatch.setattr(trader, "_STATE_FILE", state_file)
        trader._open_positions.clear()
        trader._load_state()
        assert len(trader._open_positions) == 0

    def test_load_corrupted_file(self, monkeypatch, tmp_path):
        import trader

        state_file = str(tmp_path / "corrupted.json")
        monkeypatch.setattr(trader, "_STATE_FILE", state_file)
        with open(state_file, "w") as f:
            f.write("{invalid json")
        trader._open_positions.clear()
        trader._load_state()
        assert len(trader._open_positions) == 0


# ---------------------------------------------------------------------------
# Register / unregister
# ---------------------------------------------------------------------------

class TestRegisterUnregister:
    """Test register_position / unregister_position."""

    def setup_method(self):
        import trader
        trader._open_positions.clear()

    def test_register_creates_state(self, monkeypatch, tmp_path):
        import trader
        monkeypatch.setattr(trader, "_STATE_FILE", str(tmp_path / "state.json"))

        state = trader.register_position("GOOG", 140.0, 15, 2.5, 135.0)
        assert state.symbol == "GOOG"
        assert state.entry_price == 140.0
        assert state.quantity == 15
        assert state.atr_14_abs == 2.5
        assert state.trailing_stop.initial_stop == 135.0
        assert state.trailing_stop.current_stop == 135.0
        assert state.trailing_stop.stage == 0
        assert state.partial_exits_taken == set()
        assert state.entry_date == datetime.date.today()

    def test_register_overwrites_existing(self, monkeypatch, tmp_path):
        import trader
        monkeypatch.setattr(trader, "_STATE_FILE", str(tmp_path / "state.json"))

        trader.register_position("GOOG", 140.0, 15, 2.5, 135.0)
        trader.register_position("GOOG", 145.0, 20, 3.0, 139.0)
        assert trader._open_positions["GOOG"].entry_price == 145.0
        assert trader._open_positions["GOOG"].quantity == 20

    def test_unregister_removes(self, monkeypatch, tmp_path):
        import trader
        monkeypatch.setattr(trader, "_STATE_FILE", str(tmp_path / "state.json"))

        trader.register_position("GOOG", 140.0, 15, 2.5, 135.0)
        assert "GOOG" in trader._open_positions
        trader.unregister_position("GOOG")
        assert "GOOG" not in trader._open_positions

    def test_unregister_nonexistent_is_noop(self, monkeypatch, tmp_path):
        import trader
        monkeypatch.setattr(trader, "_STATE_FILE", str(tmp_path / "state.json"))

        trader.unregister_position("DOESNOTEXIST")  # should not raise

    def test_get_open_positions_snapshot(self, monkeypatch, tmp_path):
        import trader
        monkeypatch.setattr(trader, "_STATE_FILE", str(tmp_path / "state.json"))

        trader.register_position("A", 50.0, 10, 1.0, 48.0)
        trader.register_position("B", 60.0, 20, 1.5, 57.0)
        snap = trader.get_open_positions_snapshot()
        assert "A" in snap
        assert "B" in snap
        # Mutating snapshot shouldn't affect module state
        del snap["A"]
        assert "A" in trader._open_positions


# ---------------------------------------------------------------------------
# manage_open_positions (dry_run mode — no broker needed)
# ---------------------------------------------------------------------------

class TestManageOpenPositions:
    """Test manage_open_positions in dry-run mode (no broker)."""

    def setup_method(self):
        import trader
        trader._open_positions.clear()

    def test_empty_positions_returns_empty_result(self, monkeypatch, tmp_path):
        import trader
        monkeypatch.setattr(trader, "_STATE_FILE", str(tmp_path / "state.json"))

        result = trader.manage_open_positions(broker=None, dry_run=True)
        assert result["trailing_updates"] == []
        assert result["partial_exits"] == []
        assert result["time_stops"] == []
        assert result["full_exits"] == []

    def test_time_stop_triggers_in_dry_run(self, monkeypatch, tmp_path):
        import trader
        monkeypatch.setattr(trader, "_STATE_FILE", str(tmp_path / "state.json"))

        # Register a position with an entry date far in the past
        state = trader.register_position("TSLA", 200.0, 10, 4.0, 192.0)
        state.entry_date = datetime.date.today() - datetime.timedelta(days=config.TIME_STOP_DAYS + 5)

        # We need to mock _fetch_history to return a simple price
        import pandas as pd
        mock_hist = pd.DataFrame({
            "Close": [210.0],
            "High": [212.0],
            "Low": [208.0],
        }, index=pd.to_datetime(["2026-03-01"]))

        monkeypatch.setattr(trader, "_fetch_history", lambda sym, period_years=1: mock_hist)

        result = trader.manage_open_positions(broker=None, dry_run=True)
        assert "TSLA" in result["time_stops"]
        assert ("TSLA", "time_stop") in result["full_exits"]
        assert "TSLA" not in trader._open_positions

    def test_trailing_stop_updates_in_dry_run(self, monkeypatch, tmp_path):
        import trader
        monkeypatch.setattr(trader, "_STATE_FILE", str(tmp_path / "state.json"))

        # Register a position; price will go up enough to trigger stage 1
        state = trader.register_position("NVDA", 100.0, 20, 2.0, 96.0)

        import pandas as pd
        mock_hist = pd.DataFrame({
            "Close": [103.0],  # entry + 1.5×ATR → triggers stage 1 (breakeven)
            "High": [103.5],
            "Low": [101.0],
        }, index=pd.to_datetime(["2026-03-01"]))
        monkeypatch.setattr(trader, "_fetch_history", lambda sym, period_years=1: mock_hist)

        result = trader.manage_open_positions(broker=None, dry_run=True)
        # Stage 1 should have moved stop to 100.0 (breakeven)
        assert len(result["trailing_updates"]) == 1
        sym, old, new = result["trailing_updates"][0]
        assert sym == "NVDA"
        assert new == 100.0  # breakeven

    def test_trailing_stop_hit_exits_position(self, monkeypatch, tmp_path):
        import trader
        monkeypatch.setattr(trader, "_STATE_FILE", str(tmp_path / "state.json"))

        state = trader.register_position("META", 100.0, 20, 2.0, 96.0)
        # First advance to stage 1 (breakeven at 100)
        state.trailing_stop.update(102.5)

        import pandas as pd
        # Now price drops below breakeven stop
        mock_hist = pd.DataFrame({
            "Close": [99.0],
            "High": [100.5],
            "Low": [98.5],
        }, index=pd.to_datetime(["2026-03-01"]))
        monkeypatch.setattr(trader, "_fetch_history", lambda sym, period_years=1: mock_hist)

        result = trader.manage_open_positions(broker=None, dry_run=True)
        assert any("META" in r[0] and "trailing" in r[1] for r in result["full_exits"])
        assert "META" not in trader._open_positions

    def test_no_price_data_skips_management(self, monkeypatch, tmp_path):
        import trader
        monkeypatch.setattr(trader, "_STATE_FILE", str(tmp_path / "state.json"))

        trader.register_position("BADTICKER", 50.0, 10, 1.0, 48.0)
        monkeypatch.setattr(trader, "_fetch_history", lambda sym, period_years=1: None)

        result = trader.manage_open_positions(broker=None, dry_run=True)
        # Should skip, not crash; position remains
        assert "BADTICKER" in trader._open_positions
        assert result["full_exits"] == []


# ---------------------------------------------------------------------------
# Evaluate symbol (integration — offline mock)
# ---------------------------------------------------------------------------

class TestEvaluateSymbol:
    """Smoke test for evaluate_symbol with mocked data sources."""

    def test_returns_tuple(self, monkeypatch):
        import trader
        # Mock external calls to avoid network
        monkeypatch.setattr(trader, "_fetch_info", lambda sym: {})
        from screener import FundamentalProfile
        fp = FundamentalProfile(symbol="TEST")
        fp.passes = False
        fp.fail_reasons = ["mocked"]
        monkeypatch.setattr(trader, "screen_fundamental", lambda sym, info: fp)

        fund, dip = trader.evaluate_symbol("TEST")
        assert fund is not None
        assert not fund.passes
        assert dip is None  # should skip dip detection when fund fails
