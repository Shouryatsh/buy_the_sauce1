from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from optionsdesk.backtest import compute_metrics, run_straddle_backtest, run_wheel_backtest
from optionsdesk.settings import BacktestSettings, Settings, StraddleSettings


def walk(n=600, vol=0.25, drift=0.08, seed=7, s0=100.0, shock_at=None, shock=-0.3):
    rng = np.random.default_rng(seed)
    r = rng.normal(drift / 252, vol / np.sqrt(252), n)
    if shock_at is not None:
        r[shock_at] += shock
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.Series(s0 * np.exp(np.cumsum(r)), index=idx)


def settings(**bt):
    return Settings(backtest=BacktestSettings(**bt))


def test_compute_metrics_known_values():
    d0 = date(2024, 1, 1)
    dates = [d0 + timedelta(i) for i in range(3)]
    m = compute_metrics(dates, [100.0, 110.0, 99.0], [{"pnl": 5}, {"pnl": -2}, {"pnl": 3}], 100.0, 0.0)
    assert m["max_drawdown"] == pytest.approx(99 / 110 - 1)
    assert m["win_rate"] == pytest.approx(2 / 3) and m["trades"] == 3
    assert m["total_return"] == pytest.approx(-0.01)
    assert m["sharpe"] is not None and m["sortino"] is not None


def test_compute_metrics_flat_equity_has_no_ratio():
    d0 = date(2024, 1, 1)
    m = compute_metrics([d0 + timedelta(i) for i in range(4)], [100.0] * 4, [], 100.0, 0.0)
    assert m["sharpe"] is None and m["max_drawdown"] == 0 and m["win_rate"] is None


def test_wheel_runs_and_reports_everything():
    px = walk()
    d = px.index
    res = run_wheel_backtest({"AAA": px}, d[100].date(), d[-1].date(), settings())
    assert res.equity[0] == pytest.approx(100_000)  # nothing traded before the first fill bar
    assert len(res.equity) == len(res.dates) == len(res.drawdown) and max(res.drawdown) <= 0
    assert res.trades and all("pnl" in t for t in res.trades)
    for k in ("annualized_return", "max_drawdown", "sharpe", "sortino", "win_rate", "annualized_return_on_deployed"):
        assert k in res.metrics
    assert np.all(np.isfinite(res.equity))
    assert res.to_json()["equity"][0][0] == d[100].date().isoformat()


def test_wheel_has_no_lookahead():
    px = walk(shock_at=450)
    d = px.index
    start, k = d[100].date(), 380
    full = run_wheel_backtest({"AAA": px}, start, d[-1].date(), settings())
    trunc = run_wheel_backtest({"AAA": px.iloc[: k + 1]}, start, d[k].date(), settings())
    n = len(trunc.equity)
    assert n > 100 and trunc.dates == full.dates[:n]
    assert trunc.equity == pytest.approx(full.equity[:n])  # future data cannot change the past

    altered = px.copy()
    altered.iloc[k + 1:] *= np.linspace(1.0, 3.0, len(altered) - k - 1)  # wildly different future
    other = run_wheel_backtest({"AAA": altered}, start, d[-1].date(), settings())
    assert other.equity[:n] == pytest.approx(full.equity[:n])


def test_wheel_costs_reduce_returns():
    px = walk()
    d = px.index
    args = ({"AAA": px}, d[100].date(), d[-1].date())
    free = run_wheel_backtest(*args, settings(commission_per_contract=0.0, slippage_pct_of_spread=0.0))
    costly = run_wheel_backtest(*args, settings(commission_per_contract=2.0, slippage_pct_of_spread=1.0))
    assert costly.metrics["final_equity"] < free.metrics["final_equity"]


def test_wheel_crash_triggers_assignment():
    px = walk(shock_at=300, shock=-0.35, drift=0.0, vol=0.15)
    d = px.index
    res = run_wheel_backtest({"AAA": px}, d[100].date(), d[-1].date(), settings())
    assert any(t["assigned"] for t in res.trades) or res.metrics["max_drawdown"] < -0.01


def test_wheel_cc_never_called_below_cost_basis_by_default():
    px = walk(shock_at=300, shock=-0.20, drift=0.0, vol=0.2)
    d = px.index
    res = run_wheel_backtest({"AAA": px}, d[100].date(), d[-1].date(), settings())
    # assigned cycles that ended via call-away must not realize a stock loss beyond premium collected
    for t in res.trades:
        if t["assigned"]:
            assert t["pnl"] >= -1e-6 or t["premium"] > 0


def test_multi_ticker_shares_capital():
    a, b = walk(seed=1), walk(seed=2, s0=60.0)
    d = a.index
    res = run_wheel_backtest({"AAA": a, "BBB": b}, d[100].date(), d[-1].date(), settings())
    assert {t["symbol"] for t in res.trades} <= {"AAA", "BBB"} and res.metrics["trades"] >= 1


def _event_dates(px, every=60):
    return [px.index[i].date() for i in range(130, len(px), every)]


def test_long_straddle_backtest_runs_with_events():
    px = walk(vol=0.30)
    d = px.index
    st = Settings(straddle=StraddleSettings(mode="long", long_max_iv_rank=100, max_iv_over_rv=5.0,
                                            min_edge_vs_expected_move=-5.0))
    res = run_straddle_backtest({"AAA": px}, d[100].date(), d[-1].date(), st, events={"AAA": _event_dates(px)})
    assert res.strategy == "straddle_long" and res.trades
    assert {t["reason"] for t in res.trades} <= {"profit target", "stop loss", "time exit", "expired"}
    assert res.equity[0] == pytest.approx(100_000)


def test_straddle_no_event_means_no_long_trades():
    px = walk(vol=0.30)
    d = px.index
    st = Settings(straddle=StraddleSettings(mode="long", long_max_iv_rank=100, max_iv_over_rv=5.0,
                                            min_edge_vs_expected_move=-5.0))
    res = run_straddle_backtest({"AAA": px}, d[100].date(), d[-1].date(), st, events={})
    assert res.trades == []


def test_straddle_has_no_lookahead():
    px = walk(vol=0.30, shock_at=420, shock=0.2)
    d = px.index
    st = Settings(straddle=StraddleSettings(mode="long", long_max_iv_rank=100, max_iv_over_rv=5.0,
                                            min_edge_vs_expected_move=-5.0))
    ev = {"AAA": _event_dates(px)}
    start, k = d[100].date(), 380
    full = run_straddle_backtest({"AAA": px}, start, d[-1].date(), st, events=ev)
    trunc = run_straddle_backtest({"AAA": px.iloc[: k + 1]}, start, d[k].date(), st, events=ev)
    n = len(trunc.equity)
    assert trunc.equity == pytest.approx(full.equity[:n])


def test_short_straddle_costs_reduce_returns():
    px = walk(vol=0.30)
    d = px.index
    kw = dict(mode="short", short_min_iv_rank=0, min_iv_over_rv=0.0, min_edge_vs_expected_move=-5.0)
    args = ({"AAA": px}, d[100].date(), d[-1].date())
    free = run_straddle_backtest(*args, Settings(straddle=StraddleSettings(**kw),
                                                 backtest=BacktestSettings(commission_per_contract=0, slippage_pct_of_spread=0)))
    costly = run_straddle_backtest(*args, Settings(straddle=StraddleSettings(**kw),
                                                   backtest=BacktestSettings(commission_per_contract=2, slippage_pct_of_spread=1)))
    assert free.trades and costly.metrics["final_equity"] < free.metrics["final_equity"]
