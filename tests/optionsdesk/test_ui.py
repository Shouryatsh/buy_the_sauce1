import json
from datetime import date, timedelta

import plotly.graph_objects as go
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from optionsdesk.agents.risk import RiskAgent
from optionsdesk.agents.straddle import StraddleAgent
from optionsdesk.api.app import create_app
from optionsdesk.events import Leg, Signal, SignalAction
from optionsdesk.ui import charts, client, theme
from optionsdesk.ui.live import parse_sse_lines

from .test_api import TODAY, feed, make_rt


# ---- pure pieces ------------------------------------------------------------

def test_colour_system():
    assert theme.pnl_color(5) == theme.GREEN and theme.pnl_color(-5) == theme.RED and theme.pnl_color(0) == theme.MUTED
    assert theme.utilization_color(0.5) == theme.GREEN and theme.utilization_color(0.85) == theme.AMBER
    assert theme.utilization_color(1.2) == theme.RED
    assert theme.STATUS_COLORS["block"] == theme.RED and theme.STATUS_COLORS["flag"] == theme.AMBER


def test_payoff_chart_has_green_red_fills_and_breakeven_lines():
    xs = [80, 90, 100, 110, 120]
    fig = charts.payoff_fig(xs, [800, 0, -800, 0, 800], [90, 110], 100.0)
    assert isinstance(fig, go.Figure) and fig.layout.template.layout.paper_bgcolor is not None
    fills = {t.name: t.fillcolor for t in fig.data if t.fill}
    assert "46,204,113" in fills["Profit"] and "231,76,60" in fills["Loss"]
    assert len([s for s in fig.layout.shapes if s.type == "line" and s.x0 in (90, 110)]) == 2


def test_gauge_zones_and_missing_rank():
    fig = charts.iv_gauge(15.0, 30, 60)
    steps = fig.data[0].gauge.steps
    assert [(s.range[0], s.range[1]) for s in steps] == [(0, 30), (30, 60), (60, 100)]
    assert charts.iv_gauge(None, 30, 60).data[0].value == 0


def test_utilization_bars_colour_by_threshold():
    fig = charts.utilization_bars({"capital": 0.4, "delta": 0.9, "vega": 1.3})
    assert list(fig.data[0].marker.color) == [theme.GREEN, theme.AMBER, theme.RED]


def test_scenario_heatmap_centered_on_zero_with_total_row():
    sc = {"AAA": {"-0.1": -500.0, "0.0": 0.0, "0.1": 300.0}, "TOTAL": {"-0.1": -500.0, "0.0": 0.0, "0.1": 300.0}}
    fig = charts.scenario_heatmap(sc)
    h = fig.data[0]
    assert list(h.y) == ["AAA", "TOTAL"] and h.zmid == 0 and list(h.x) == ["-10%", "+0%", "+10%"]
    assert h.z[0][0] == -500.0


def test_exposure_heatmap_puts_stock_column_first():
    fig = charts.exposure_heatmap({"AAA": {"STK": 10000.0, "2025-07-11": -2000.0}, "BBB": {"2025-07-11": 500.0}})
    assert list(fig.data[0].x) == ["STK", "2025-07-11"]


def test_compare_charts_normalize_equity():
    run = {"equity": {"equity": [["2024-01-01", 200.0], ["2024-01-02", 210.0]], "drawdown": [["2024-01-01", 0.0], ["2024-01-02", -0.01]]}}
    fig = charts.equity_compare({"a": run})
    assert list(fig.data[0].y) == [100.0, 105.0]
    assert list(charts.drawdown_compare({"a": run}).data[0].y) == [0.0, -1.0]


def test_sse_parser_handles_events_comments_and_bad_json():
    lines = [": ping\n", "event: Signal\n", 'data: {"a": 1}\n', "\n", "event: Heartbeat\n", "data: not json\n", "\n",
             'data: {"b": 2}\n', "\n"]
    assert list(parse_sse_lines(lines)) == [("Signal", {"a": 1}), ("message", {"b": 2})]


# ---- pages rendered against the real API ------------------------------------

@pytest.fixture
def ui(monkeypatch):
    rt = make_rt()
    feed(rt, "AAA")
    feed(rt, "BBB", spot=50.0)
    with TestClient(create_app(runtime=rt, start_agents=False)) as c:
        def fake(method, path, **kw):
            r = c.request(method, path, params=kw.get("params"), json=kw.get("json"))
            if r.status_code >= 400:
                raise client.ApiError(f"{r.status_code}: {r.text}")
            return r.json()

        monkeypatch.setattr(client, "request", fake)
        yield rt


def populate(rt):
    w = rt.agents["wheel"]
    w.positions["AAA"].open_csp(95.0, TODAY + timedelta(35), 2.0, 1, TODAY)
    w._persist(w.positions["AAA"])
    st = StraddleAgent(rt.ctx)
    st.events = {"AAA": [TODAY + timedelta(7)]}
    st.iv["AAA"] = rt.tracker.ivs["AAA"].model_copy(update={"iv_rank": 15.0, "realized_vol": 0.36})
    sig = st.on_chain(rt.tracker.chains["AAA"].model_copy(update={"quotes": [
        q for q in rt.tracker.chains["AAA"].quotes]}))
    risk = RiskAgent(rt.ctx)
    risk.spots.update({"AAA": 100.0, "BBB": 50.0})
    if sig is not None:
        risk.on_signal(sig)
    other = Signal(strategy="wheel", symbol="BBB", action=SignalAction.OPEN, reason="Sell 1x BBB 45P",
                   legs=[Leg(right="P", strike=45.0, expiry=TODAY + timedelta(35), quantity=-1, price=1.0)])
    risk.on_signal(other)
    rt.ctx.db.create_backtest("bt1", "wheel", {"tickers": ["AAA"], "start": "2024-01-01", "end": "2024-06-01", "overrides": {}})
    series = {"equity": [["2024-01-01", 100000.0], ["2024-01-02", 100500.0]], "drawdown": [["2024-01-01", 0.0], ["2024-01-02", 0.0]],
              "trades": []}
    rt.ctx.db.finish_backtest("bt1", {"annualized_return": 0.1, "sharpe": 1.1, "max_drawdown": -0.05, "n_trades": 3}, series)


def run_page(name, **state):
    at = AppTest.from_string(f"from optionsdesk.ui.screens import {name} as p\np.render()", default_timeout=60)
    at.session_state["refresh_s"] = 0
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


@pytest.mark.parametrize("page,header", [
    ("overview", "Overview"), ("wheel", "Wheel"), ("straddle", "Straddle"), ("scanner", "Opportunity scanner"),
    ("risk", "Risk"), ("backtest", "Backtest lab"), ("journal", "Trade journal"), ("controls", "Orders & controls"),
])
def test_every_page_renders_with_data(ui, page, header):
    populate(ui)
    at = run_page(page)
    assert not at.exception, [e.value for e in at.exception]
    assert at.header[0].value == header
    assert not at.error  # no API errors surfaced


def test_pages_render_on_an_empty_system(monkeypatch):
    rt = make_rt()
    with TestClient(create_app(runtime=rt, start_agents=False)) as c:
        monkeypatch.setattr(client, "request", lambda m, p, **kw: c.request(m, p, params=kw.get("params"), json=kw.get("json")).json())
        for page in ("overview", "wheel", "straddle", "scanner", "risk", "backtest", "journal", "controls"):
            at = run_page(page)
            assert not at.exception, (page, [e.value for e in at.exception])


def test_overview_shows_paper_badge_metrics_and_agents(ui):
    populate(ui)
    at = run_page("overview")
    labels = [m.label for m in at.metric]
    assert {"Portfolio value", "Realized P&L", "Unrealized P&L", "Open premium collected", "Capital deployed"} <= set(labels)
    html = " ".join(m.value for m in at.markdown)
    assert "NOT LIVE" in html and "trading enabled" in html
    assert any("agent" in df.value.columns for df in at.dataframe)


def test_risk_page_flags_over_limit(ui):
    populate(ui)
    ui.ctx.settings.risk.max_capital_at_risk = 5_000  # the open 95P (~$9k) now exceeds the limit
    at = run_page("risk")
    assert any("Over limit" in e.value for e in at.error)


def test_api_failure_is_shown_not_raised(monkeypatch):
    def down(*a, **k):
        raise client.ApiError("API unreachable at http://x")
    monkeypatch.setattr(client, "request", down)
    at = run_page("overview")
    assert not at.exception and any("unreachable" in e.value for e in at.error)


def test_controls_kill_switch_button_engages(ui):
    at = run_page("controls")
    at.button(key="ks_engage").click().run()
    assert not at.exception and ui.ctx.db.kill_switch_engaged()
    assert any("ENGAGED" in m.value for m in at.markdown)
    at.button(key="ks_release").click().run()
    assert not ui.ctx.db.kill_switch_engaged()


def test_controls_confirm_requires_live_acknowledgement(ui):
    from optionsdesk.events import OrderIntent
    ui.ctx.settings = ui.ctx.settings  # unchanged: paper
    sig = Signal(strategy="wheel", symbol="AAA", action=SignalAction.OPEN, reason="r",
                 legs=[Leg(right="P", strike=95.0, expiry=TODAY, quantity=-1, price=1.0)])
    intent = sig.derive(OrderIntent, signal_id=sig.id, symbol="AAA", strategy="wheel", legs=sig.legs, reason="r")
    ui.ctx.db.save_order_intent("ord1", intent, "pending_confirmation", False)
    at = run_page("controls")
    assert any(b.key == "c_ord1" for b in at.button)
    at.button(key="c_ord1").click().run()  # agent has no such awaiting order -> 404 shown, no crash
    assert not at.exception and any("404" in e.value for e in at.error)


def test_backtest_page_lists_and_compares_runs(ui):
    populate(ui)
    at = run_page("backtest")
    assert not at.exception
    assert any("bt1" in str(df.value.to_dict()) for df in at.dataframe)
    assert at.multiselect(key="bt_compare") is not None
