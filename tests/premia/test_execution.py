"""Phase 7: the paper executor's refusals, rule gates, kill switch, audit log, and the daily report."""

from __future__ import annotations

import pandas as pd
import pytest

from premia import risk
from premia.execution import PaperExecutor, Proposal, Refused
from premia.report import append_marks, daily_report


class FakeBroker:
    def __init__(self, accounts=("DU123456",), quote=None):
        self.accounts = list(accounts)
        self.q = quote or {"bid": 1.00, "ask": 1.10, "open_interest": 2000}
        self.orders = []
        self.connected_to = None
        self.disconnected = False

    def connect(self, host, port, client_id):
        self.connected_to = port

    def managed_accounts(self):
        return self.accounts

    def quote(self, *a):
        return dict(self.q)

    def place_limit(self, symbol, expiry, strike, right, qty, limit, ref):
        self.orders.append((symbol, expiry, strike, right, qty, limit, ref))
        return {"order_id": len(self.orders), "status": "Submitted"}

    def disconnect(self):
        self.disconnected = True


def _ex(tmp_path, **kw):
    b = FakeBroker(**kw)
    return PaperExecutor(b, base_dir=tmp_path), b


def test_live_port_is_refused_before_any_connection(tmp_path):
    ex, b = _ex(tmp_path)
    for port in (7496, 4001, 1234):
        with pytest.raises(Refused):
            ex.connect(port=port)
    assert b.connected_to is None


def test_a_live_account_on_the_paper_port_is_refused_and_disconnected(tmp_path):
    ex, b = _ex(tmp_path, accounts=("U1234567",))
    with pytest.raises(Refused):
        ex.connect()
    assert b.disconnected and not ex.connected
    ex2, _ = _ex(tmp_path, accounts=("DU1", "U2"))
    with pytest.raises(Refused):
        ex2.connect()


SELL = Proposal("SPY", "P", 600.0, "20261218", -2, "csp", "model", 0.12, 1.0,
                impact={"net_vega": -200.0, "margin": 120_000.0, "by_name": {"SPY": 120_000.0}})


def test_orders_pass_every_rule_then_are_limit_priced(tmp_path):
    ex, b = _ex(tmp_path)
    ex.connect()
    rec = ex.submit(SELL, risk.Exposures(), 1_000_000)
    assert rec["status"] == "placed" and b.orders and b.orders[0][5] == pytest.approx(1.025, abs=0.006)    # mid 1.05 - 0.5 x 0.05, to the cent
    events = [e["event"] for e in ex.log.read()]
    assert events[:2] == ["connected", "proposal"] and "decision" in events


def test_rules_block_illiquid_and_limit_breaching_orders(tmp_path):
    ex, b = _ex(tmp_path, quote={"bid": 1.0, "ask": 1.5, "open_interest": 2000})
    ex.connect()
    assert ex.submit(SELL, risk.Exposures(), 1_000_000)["status"] == "blocked"
    ex2, b2 = _ex(tmp_path)
    ex2.connect()
    big = Proposal(**{**SELL.__dict__, "impact": {"margin": 700_000.0}})
    rec = ex2.submit(big, risk.Exposures(), 1_000_000)
    assert rec["status"] == "blocked" and any("margin" in r for r in rec["reasons"]) and not b2.orders


def test_kill_switch_blocks_openings_but_not_closings(tmp_path):
    ex, b = _ex(tmp_path)
    ex.connect()
    assert ex.daily_check(nav=960_000, prev_nav=1_000_000, hwm=1_000_000)          # 4 % daily loss trips it
    assert ex.kill.engaged
    assert ex.submit(SELL, risk.Exposures(), 960_000)["status"] == "blocked"
    close = Proposal("SPY", "P", 600.0, "20261218", +2, "close", "kill switch", closing=True)
    assert ex.submit(close, risk.Exposures(), 960_000)["status"] == "placed"
    ex.kill.set(False)
    assert not ex.kill.engaged


def test_daily_report_attributes_pnl_from_stored_marks(tmp_path):
    rows = []
    for d, S, iv in (("2026-11-02", 600.0, 0.20), ("2026-11-03", 606.0, 0.19)):
        rows.append({"date": d, "contract": "SPY 20261218 580P", "symbol": "SPY", "right": "P", "K": 580.0,
                     "expiry": "2026-12-18", "T": 45 / 365, "S": S, "iv": iv, "r": 0.04, "q": 0.012, "qty": -2,
                     "trade_type": "csp", "bid": 5.0, "ask": 5.1})
    marks = append_marks(rows, tmp_path / "marks.csv")
    rep = daily_report("2026-11-03", marks, {"SPY": -60.0}, 1_000_000, out_dir=tmp_path / "reports")
    assert rep["positions"] == 1 and rep["by_symbol"][0]["stock_shares"] == -60.0
    a = rep["attribution_today"]
    assert a["total"] > 0 and abs(a["residual"]) < 0.2 * abs(a["total"])           # short put gains on a rally + vol drop
    assert (tmp_path / "reports" / "2026-11-03.md").exists()


def test_paper_loop_proposals_need_a_live_quote_that_confirms_the_edge():
    from premia import paper
    from .fixtures import underlying
    u = underlying("TST", seed=21, n=900, iv_premium=0.06)
    U = {"TST": u}
    good = lambda *a: {"bid": 50.0, "ask": 50.2, "open_interest": 5000}          # far above any fair value
    bad = lambda *a: {"bid": 0.01, "ask": 0.02, "open_interest": 5000}
    props = paper.wheel_proposals(U, ["TST"], 0.0, 1_000_000, good, held=set())
    assert props and props[0].qty < 0 and props[0].edge_per_share > 0 and props[0].impact["margin"] > 0
    assert paper.wheel_proposals(U, ["TST"], 0.0, 1_000_000, bad, held=set()) == []
    assert paper.wheel_proposals(U, ["TST"], 0.0, 1_000_000, good, held={"TST"}) == []


def test_paper_hedge_orders_respect_band_and_cap():
    from premia import paper
    pos = [{"symbol": "X", "delta_shares": 400.0}]
    out = paper.hedge_orders(pos, {"X": 100.0}, {"X": -20.0}, {"X": 0.0}, 1.0, 1_000_000)
    assert out["X"] < 0
    assert paper.hedge_orders([{"symbol": "X", "delta_shares": 1.0}], {"X": 100.0}, {"X": -20.0}, {}, 1.0, 1_000_000) == {}
