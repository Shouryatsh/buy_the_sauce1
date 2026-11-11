"""The Overview alerts panel surfaces unread screener alerts, and never breaks if the screener store is unusable."""

from __future__ import annotations

import wheel_config as wc
import wheel_watchlist as ww
import wheel_ui
from stockscreen import store
from wheel_ibkr import AccountSummary


class StubClient:
    def get_account_summary(self):
        return AccountSummary(net_liquidation=100_000, total_cash=50_000, buying_power=100_000, margin_used=1_000, is_paper=False)

    def get_positions(self):
        return []


def test_unread_screener_alerts_appear_on_overview(tmp_path, monkeypatch):
    monkeypatch.setattr(wc, "WHEEL_DB_PATH", str(tmp_path / "o.db"))
    monkeypatch.setattr(ww, "EXCLUSIONS_PATH", tmp_path / "exclusions.json")
    store.add_alert({"symbol": "ABC", "name": "ABC Corp", "kind": "low_price", "price": 10.0, "trigger_text": "in the low-price zone on: 52w",
                     "fund_status": "PASSED (Stable)", "tech_status": "Healthy pullback", "reason": "r"})
    titles = [a.title for a in wheel_ui.compute_alerts(StubClient())]
    assert any(t.startswith("Screener: ABC") for t in titles)
    store.mark_alerts_read()
    assert not any(a.title.startswith("Screener:") for a in wheel_ui.compute_alerts(StubClient()))


def test_a_broken_screener_store_never_breaks_the_overview(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "list_alerts", lambda **k: (_ for _ in ()).throw(RuntimeError("db locked")))
    assert isinstance(wheel_ui.compute_alerts(StubClient()), list)
