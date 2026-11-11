"""Phases 4, 6, 7: hedging engine, risk limits and kill switch, and the backtester's accounting and timing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from premia import backtest as bt, hedging as H, risk
from premia.config import LIMITS

from .fixtures import underlying


# ---------------------------------------------------------------------------- hedging
def test_ww_band_grows_with_cost_and_gamma_and_shrinks_with_risk_aversion():
    a = H.ww_half_width(100, 50, 0.1, 0.02, 1e6, 1.0)
    assert H.ww_half_width(100, 100, 0.1, 0.02, 1e6, 1.0) > a                  # more gamma -> wider band
    assert H.ww_half_width(100, 50, 0.1, 0.02, 1e7, 1.0) > a                   # gamma_ra = 1000 / NAV: bigger book tolerates more
    assert H.ww_half_width(100, 50, 0.1, 0.02, 1e6, 2.0) == pytest.approx(2 * a)


def test_hedge_policies_and_the_hard_delta_cap():
    band = H.HedgePolicy("band", band_mult=1.0)
    w = H.ww_half_width(100, 50, 0.1, 0.02, 1e6, 1.0)
    assert H.hedge_trade(band, w * 0.5, 0, 100, 50, 0.1, 0.02, 1e6) == 0              # inside the band: no trade
    assert H.hedge_trade(band, w + 100, 0, 100, 50, 0.1, 0.02, 1e6) == pytest.approx(-100, abs=1)   # back to the edge
    assert H.hedge_trade(H.HedgePolicy("time", every=5), 300, 3, 100, 50, 0.1, 0.02, 1e6) == 0
    assert H.hedge_trade(H.HedgePolicy("time", every=5), 300, 5, 100, 50, 0.1, 0.02, 1e6) == -300
    assert H.hedge_trade(H.HedgePolicy("none"), 5000, 0, 100, 50, 0.1, 0.02, 1e6, max_delta_shares=1000) == -4000


def test_hedge_frequency_trades_cost_against_variance_on_gbm():
    """Taleb's frontier on simulated GBM: daily hedging has less P&L variance than weekly and trades more shares."""
    from premia import strategies as st
    rng = np.random.default_rng(0)
    n, days, S0, sig, T = 4000, 30, 100.0, 0.2, 30 / 252
    z = rng.normal(0, sig * np.sqrt(1 / 252), (n, days))
    paths = np.column_stack([np.full(n, S0), S0 * np.exp(np.cumsum(z - 0.5 * sig ** 2 / 252, axis=1))])
    pnl_d, turn_d = st.hedged_paths_pnl(paths, [97.0], T, 0.0, 0.0, [sig], "P", [1.0])
    weekly = paths[:, ::5]
    pnl_w, turn_w = st.hedged_paths_pnl(weekly, [97.0], T, 0.0, 0.0, [sig], "P", [1.0])
    assert pnl_d.var() < pnl_w.var() and turn_d.mean() > turn_w.mean()


def test_tail_hedge_quantity():
    assert H.tail_hedge_quantity(50_000, 100_000, 1000, 0, -10, 1) == 0
    assert H.tail_hedge_quantity(150_000, 100_000, 1000, 0, -10, 1) == 50
    assert H.tail_hedge_quantity(0, 100_000, 1000, -30, -10, 1) == 20                 # volga floor alone


# ---------------------------------------------------------------------------- risk
NAV = 1e6


@pytest.mark.parametrize("attr,val", [("net_delta", 0.06 * NAV), ("net_vega", 0.005 * NAV), ("net_vanna", 0.003 * NAV),
                                      ("net_volga", 0.003 * NAV), ("margin", 0.6 * NAV)])
def test_every_greek_and_margin_limit_blocks(attr, val):
    ok, why = risk.check_open(risk.Exposures(), risk.Exposures(**{attr: val}), NAV)
    assert not ok and any(attr in w for w in why)


def test_name_and_sector_concentration_block_and_risk_reducing_trades_pass():
    before = risk.Exposures(by_name={"X": 0.14 * NAV}, by_sector={"tech": 0.14 * NAV})
    ok, why = risk.check_open(before, risk.Exposures(by_name={"X": 0.02 * NAV}), NAV)
    assert not ok and any("X notional" in w for w in why)
    breached = risk.Exposures(net_vega=-0.01 * NAV)
    ok, _ = risk.check_open(breached, risk.Exposures(net_vega=+0.002 * NAV), NAV)           # reduces an existing breach
    assert ok


def test_kill_switch_trips_on_daily_loss_and_drawdown_and_stays_until_released():
    k = risk.KillSwitch(hwm=NAV)
    assert not k.update(0.98 * NAV, NAV, "d1")
    assert k.update(0.98 * NAV * 0.965, 0.98 * NAV, "d2") and "daily loss" in k.reason
    k2 = risk.KillSwitch(hwm=NAV)
    prev = 1.0
    for i, f in enumerate([0.98, 0.96, 0.94, 0.92, 0.90, 0.88, 0.86, 0.84]):        # 2 % a day: never a daily-loss trip
        k2.update(f * NAV, prev * NAV, f"d{i}")
        prev = f
    assert k2.engaged and "drawdown" in k2.reason
    assert not k2.maybe_release_after_cooldown(0.84 * NAV)
    k2._days_engaged = 21
    assert k2.maybe_release_after_cooldown(0.84 * NAV) and not k2.engaged and k2.hwm == 0.84 * NAV


def test_liquidity_rules():
    assert not risk.liquidity_ok(1000, None, 1.0)[0]
    assert not risk.liquidity_ok(1000, 1.0, 1.3)[0]                     # spread 26 % of mid
    assert not risk.liquidity_ok(100, 1.0, 1.05)[0]
    assert not risk.liquidity_ok(None, 1.0, 1.05)[0]
    assert risk.liquidity_ok(1000, 1.0, 1.05)[0]


# ---------------------------------------------------------------------------- backtester
@pytest.fixture(scope="module")
def world():
    u = underlying("TST", seed=11, n=1100)
    spy = underlying("SPY", seed=12, n=1100, vol=0.15)
    return {"TST": u, "SPY": spy}


def _run(U, sleeve="wheel", policy=None, start_i=600, end_i=880, **kw):
    dates = U["SPY"].df.index[start_i:end_i]
    cfg = bt.RunConfig(sleeve=sleeve, policy=policy or H.HedgePolicy("band", band_mult=1.0), lam=0.0,
                       straddle_threshold=-0.5, tail_hedge=False, n_names=1, stock_weight=0.10, **kw)
    b = bt.Backtest(cfg, U, lambda t: ["TST"], dates)
    betas = {s: 1.0 for s in U}
    d = b.run(lambda t: betas)
    return b, d


def test_accounting_reconciles_every_day(world):
    b, d = _run(world)
    assert len(b.trades) > 4
    dn = d["nav"].diff().iloc[1:]
    explained = (d["a_total_opt"] + d["a_stock"] + d["dividends"] - d["cost_opt_comm"] - d["cost_opt_spread"]
                 - d["cost_stock"] - d["cost_borrow"]).iloc[1:]
    assert np.allclose(dn.values, explained.values, atol=0.05), (dn - explained).abs().max()
    per_sym = pd.DataFrame(b.sym_rows).groupby("date")["pnl"].sum().reindex(d.index, fill_value=0.0).iloc[1:]
    assert np.allclose(per_sym.values, dn.values, atol=0.05)              # per-underlying P&L adds up to the book
    assert abs(d["net_delta"]).mean() < abs(_run(world, policy=H.HedgePolicy("none"))[1]["net_delta"]).mean()


def test_straddle_sleeve_runs_and_reconciles(world):
    b, d = _run(world, sleeve="straddle")
    assert any(t["kind"] == "straddle" for t in b.trades)
    dn = d["nav"].diff().iloc[1:]
    explained = (d["a_total_opt"] + d["a_stock"] + d["dividends"] - d["cost_opt_comm"] - d["cost_opt_spread"]
                 - d["cost_stock"] - d["cost_borrow"]).iloc[1:]
    assert np.allclose(dn.values, explained.values, atol=0.05)


def test_no_lookahead_changing_the_future_leaves_the_past_unchanged(world):
    import copy
    from premia import market
    from .fixtures import panel, SKEWED
    cut_i = 800
    b1, d1 = _run(world)
    raw = panel(seed=11, n=1100)
    k = raw.index > raw.index[cut_i]
    raw.loc[k, ["close", "open", "high", "low"]] *= np.linspace(1, 0.5, k.sum())[:, None]
    raw.loc[k, "iv"] *= 1.8
    u2 = market.prepare("TST", raw, [], SKEWED, min_train=500)
    U2 = {"TST": u2, "SPY": world["SPY"]}
    b2, d2 = _run(U2)
    cut = raw.index[cut_i]
    t1 = [t for t in b1.trades if t["date"] <= cut]
    t2 = [t for t in b2.trades if t["date"] <= cut]
    assert len(t1) == len(t2) and all(a["K"] == b["K"] and a["qty"] == b["qty"] and a["date"] == b["date"] for a, b in zip(t1, t2))
    pd.testing.assert_series_equal(d1.loc[:cut, "nav"], d2.loc[:cut, "nav"])


def test_orders_fill_on_the_next_bar_with_spread_and_commission(world):
    b, d = _run(world)
    opens = [t for t in b.trades if t["action"] == "open"]
    first = opens[0]
    decision_days = d.index[d.index < first["date"]]
    assert len(decision_days) >= 1                                       # filled after the day it was decided
    assert d.loc[first["date"], "cost_opt_comm"] >= 0.65 * abs(first["qty"]) - 1e-9


def test_assignment_delivers_shares_at_the_strike():
    from premia.backtest import Backtest, OptPos, RunConfig
    from .fixtures import underlying as mk
    u = mk("TST", seed=13, n=700)
    U = {"TST": u, "SPY": mk("SPY", seed=14, n=700)}
    dates = u.df.index[600:640]
    b = Backtest(RunConfig(tail_hedge=False), U, lambda t: [], dates)
    t = dates[5]
    S = float(u.df.at[t, "close"])
    b.opts[1] = OptPos(1, "TST", "P", round(S * 1.2), t, -2, "csp", 1, dates[0], 1.0, 0.3, mark=S * 0.2)
    rec = {k: 0.0 for k in ("assignments", "early_assignments", "a_total_opt")}
    cash0 = b.cash
    b._settle(t, rec)
    assert b.stock["TST"] == 200 and b.cash == pytest.approx(cash0 - 200 * round(S * 1.2)) and rec["assignments"] == 1
    assert b.lots["TST"] == 200


def test_early_assignment_only_when_time_value_is_gone():
    from premia.backtest import Backtest, OptPos, RunConfig
    from .fixtures import underlying as mk
    u = mk("TST", seed=13, n=700)
    U = {"TST": u, "SPY": mk("SPY", seed=14, n=700)}
    dates = u.df.index[600:640]
    t, expiry = dates[5], dates[25]                                      # ~20 trading days left
    S = float(u.df.at[t, "close"])
    K = round(S * 1.3)
    rec = {k: 0.0 for k in ("assignments", "early_assignments", "a_total_opt")}
    b = Backtest(RunConfig(tail_hedge=False), U, lambda t: [], dates)
    b.opts[1] = OptPos(1, "TST", "P", K, expiry, -1, "csp", 1, dates[0], 1.0, 0.3, mark=K - S + 0.01)
    b._settle(t, rec)
    assert rec["early_assignments"] == 1 and b.stock["TST"] == 100 and 1 not in b.opts
    rec = {k: 0.0 for k in ("assignments", "early_assignments", "a_total_opt")}
    b = Backtest(RunConfig(tail_hedge=False), U, lambda t: [], dates)
    b.opts[1] = OptPos(1, "TST", "P", K, expiry, -1, "csp", 1, dates[0], 1.0, 0.3, mark=K - S + 0.25 * S * 0.1)
    b._settle(t, rec)
    assert rec["early_assignments"] == 0 and 1 in b.opts                  # real time value left: not exercised


def test_mcmillan_sleeve_trades_reconciles_and_never_hedges(world, monkeypatch):
    from premia import mcmillan as mm
    from premia.research.mcmillan_bt import prepare
    monkeypatch.setattr(mm, "EVER_PROB_MIN", 0.0)                   # let the synthetic world trade
    U = {k: prepare(v, term=False) for k, v in world.items()}
    dates = U["SPY"].df.index[600:1000]
    cfg = bt.RunConfig(sleeve="mcmillan", policy=H.HedgePolicy("none"), tail_hedge=False, mm_hist_freq=0.0,
                       mm_budget=0.02, mm_one_contract_max=0.05)
    b = bt.Backtest(cfg, U, lambda t: ["TST"], dates)
    d = b.run(lambda t: {"TST": 1.0, "SPY": 1.0})
    opens = [t for t in b.trades if t["action"] == "open" and t["kind"] == "mcm"]
    assert opens and {t["right"] for t in opens} == {"C", "P"}
    assert all(t["date"] > s["date"] for t in opens for s in b.mm_signals[:1])           # filled after a signal
    assert d["hedge_shares"].sum() == 0
    dn = d["nav"].diff().iloc[1:]
    explained = (d["a_total_opt"] + d["a_stock"] + d["dividends"] - d["cost_opt_comm"] - d["cost_opt_spread"]
                 - d["cost_stock"] - d["cost_borrow"]).iloc[1:]
    assert np.allclose(dn.values, explained.values, atol=0.05)
    closes = [t for t in b.trades if t["action"] == "close" and t["kind"] == "mcm"]
    assert closes and all(t["reason"] for t in closes)
    for t in opens:                                                    # expiry chosen per the book: at least 90 days
        assert (t["expiry"] - t["date"]).days >= 85


def test_partial_close_keeps_the_rest_of_the_position(world):
    from premia.backtest import Backtest, Order, OptPos, RunConfig
    U = dict(world)
    dates = U["TST"].df.index[600:610]
    b = Backtest(RunConfig(tail_hedge=False), U, lambda t: [], dates)
    t0 = dates[2]
    S = float(U["TST"].df.at[t0, "close"])
    exp = dates[-1] + pd.Timedelta(days=120)
    b.opts[1] = OptPos(1, "TST", "C", round(S), exp, 3, "mcm", 1, dates[0], 5.0, 0.3, mark=5.0)
    b.pending.append(Order("TST", "C", round(S), exp, -1, "mcm", 1, 1, "partial"))
    rec = {k: 0.0 for k in ("cost_opt_comm", "cost_opt_spread")}
    b._execute_pending(t0, rec)
    assert b.opts[1].qty == 2 and b.trades[-1]["partial"]
    b.pending.append(Order("TST", "C", round(S), exp, -2, "mcm", 1, 1, "rest"))
    b._execute_pending(t0, rec)
    assert 1 not in b.opts
