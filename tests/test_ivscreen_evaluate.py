"""
tests/test_ivscreen_evaluate.py — store round-trips and evaluation: data-availability statuses (unknown is
never low), freshness, short history, realized vol, exclusions, and the end-to-end classification of
synthetic symbols whose percentiles are known by construction.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import wheel_config as wc
from ivscreen import evaluate as ev
from ivscreen import metrics as mx
from ivscreen import store
from ivscreen.config import IVConfig

TODAY = date(2026, 10, 21)


def bdays(n, end=TODAY):
    return pd.bdate_range(end=pd.Timestamp(end), periods=n)


def iv_series(values, end=TODAY):
    return pd.Series(values, index=bdays(len(values), end), dtype=float, name="IV")


def closes(n=700, end=TODAY, calm_tail=False, wild_tail=False, seed=0):
    """Daily closes whose recent volatility is lowest / highest of the window when asked."""
    rng = np.random.default_rng(seed)
    vol = np.full(n, 0.012)
    if calm_tail:
        vol[-40:] = 0.002
    if wild_tail:
        vol[-40:] = 0.05
    rets = rng.normal(0, 1, n) * vol
    return pd.Series(100 * np.exp(np.cumsum(rets)), index=bdays(n, end))


def meta(sym="AAA", **kw):
    base = {"symbol": sym, "name": f"{sym} Corp", "category": "Common stock", "exchange": "NYSE", "sector": "Technology",
            "industry": "Software", "sources": "Wheel universe", "fetch_status": "ok", "fetch_detail": "",
            "iv_checked_at": "2026-10-21T10:00:00", "next_earnings": None}
    base.update(kw)
    return base


CFG = IVConfig()


def one(m, iv, close=None, cfg=CFG):
    return ev.evaluate_symbol(m, iv, close, cfg, TODAY)


# ---------------------------------------------------------------------------
# classification of ranked symbols
# ---------------------------------------------------------------------------

def test_rising_iv_is_the_top_of_its_history():
    row = one(meta(), iv_series(np.linspace(0.2, 0.6, 600)), closes())
    assert row["status"] == ev.OK and row["iv_pct"] == 100.0 and row["iv_band"] == mx.VERY_HIGH
    assert row["iv_obs"] == 504 and not row["partial_window"]


def test_falling_iv_is_the_bottom_of_its_history():
    row = one(meta(), iv_series(np.linspace(0.6, 0.2, 600)), closes())
    assert row["iv_pct"] == 0.0 and row["iv_band"] == mx.VERY_LOW


def test_realized_vol_is_ranked_against_its_own_history():
    wild = one(meta(), iv_series(np.linspace(0.3, 0.3, 600) + np.random.default_rng(1).normal(0, 0.01, 600)), closes(wild_tail=True))
    calm = one(meta(), iv_series(np.linspace(0.3, 0.3, 600) + np.random.default_rng(1).normal(0, 0.01, 600)), closes(calm_tail=True))
    assert wild["rv_pct"] >= 95 and wild["rv_band"] == mx.VERY_HIGH
    assert calm["rv_pct"] <= 5 and calm["rv_band"] == mx.VERY_LOW


def test_relationship_and_gap_use_the_two_percentiles_not_raw_levels():
    # IV at a 2-year high, stock very calm: raw levels differ wildly (60% vs ~3%) but only the percentiles are compared.
    row = one(meta(), iv_series(np.linspace(0.2, 0.6, 600)), closes(calm_tail=True))
    assert row["relationship"] == mx.REL_IV_ELEVATED and row["gap"] == pytest.approx(row["iv_pct"] - row["rv_pct"])
    row2 = one(meta(), iv_series(np.linspace(0.6, 0.2, 600)), closes(wild_tail=True))
    assert row2["relationship"] == mx.REL_RV_ELEVATED and row2["gap"] < 0


def test_trend_columns_follow_percentile_change():
    noisy = np.random.default_rng(2).normal(0.3, 0.02, 595)
    jump = iv_series(np.concatenate([noisy, [0.35, 0.40, 0.45, 0.50, 0.60]]))     # a spike in the last 5 sessions
    row = one(meta(), jump, closes())
    assert row["iv_trend"] == "Rising" and row["iv_chg"] >= 5 and row["iv_pct"] == 100.0
    fell = iv_series(np.concatenate([noisy, [0.25, 0.2, 0.15, 0.1, 0.05]]))
    assert one(meta(), fell, closes())["iv_trend"] == "Falling"
    flat = iv_series(np.random.default_rng(5).normal(0.3, 0.02, 600))
    assert one(meta(), flat, closes())["iv_trend"] in ("Flat", "Rising", "Falling")


def test_lookback_and_rv_window_are_configurable_without_a_rescan():
    iv = iv_series(np.concatenate([np.linspace(0.6, 0.2, 400), np.linspace(0.2, 0.3, 200)]))
    a = one(meta(), iv, closes(), IVConfig(lookback_days=504))
    b = one(meta(), iv, closes(), IVConfig(lookback_days=100))
    assert a["iv_pct"] != b["iv_pct"] and b["iv_obs"] == 100
    assert one(meta(), iv, closes(), IVConfig(rv_window=50))["rv_pct"] is not None


# ---------------------------------------------------------------------------
# data availability: unknown is never low
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fetch,expected", [("no_data", ev.NO_OPTIONS_DATA), ("unresolved", ev.UNRESOLVED),
                                            ("unsupported", ev.UNSUPPORTED), ("error", ev.FETCH_ERROR),
                                            ("duplicate", ev.DUPLICATE), (None, ev.NOT_SCANNED)])
def test_unavailable_statuses_carry_no_percentile_and_no_band(fetch, expected):
    row = one(meta(fetch_status=fetch, fetch_detail="why"), None, None)
    assert row["status"] == expected and row["iv_pct"] is None and row["iv_band"] == mx.NA
    assert row["relationship"] == mx.REL_NA and row["rv_pct"] is None


def test_unavailable_symbols_are_not_in_the_very_low_band():
    df = ev.evaluate_all({"A": meta("A", fetch_status="no_data"), "B": meta("B", fetch_status="error")}, {}, lambda s: None, CFG, TODAY)
    assert (df["iv_band"] == mx.VERY_LOW).sum() == 0 and set(df["status"]) == {ev.NO_OPTIONS_DATA, ev.FETCH_ERROR}


def test_fetch_ok_but_no_rows_is_no_options_data_not_zero():
    row = one(meta(), iv_series([]), closes())
    assert row["status"] == ev.NO_OPTIONS_DATA and row["iv_pct"] is None


def test_short_history_is_not_ranked_and_says_how_much_there_is():
    row = one(meta(), iv_series(np.linspace(0.2, 0.4, 60)), closes())
    assert row["status"] == ev.SHORT_HISTORY and row["iv_pct"] is None and row["iv_obs"] == 59
    assert "126 needed" in row["detail"]


def test_partial_window_is_ranked_but_flagged():
    row = one(meta(), iv_series(np.linspace(0.2, 0.4, 300)), closes())
    assert row["status"] == ev.OK and row["partial_window"] and row["iv_obs"] == 299


def test_stale_iv_is_flagged_and_not_ranked_as_current():
    old = iv_series(np.linspace(0.2, 0.6, 600), end=TODAY - timedelta(days=20))
    row = one(meta(), old, closes())
    assert row["status"] == ev.STALE and row["iv_age"] >= 20 and "latest implied-vol reading" in row["detail"]


def test_missing_prices_keep_the_iv_percentile_and_mark_rv_unavailable():
    row = one(meta(), iv_series(np.linspace(0.2, 0.6, 600)), None)
    assert row["status"] == ev.OK and row["iv_pct"] == 100.0
    assert row["rv_pct"] is None and row["relationship"] == mx.REL_NA and row["rv_status"] == "none"


def test_stale_prices_do_not_produce_a_current_rv_percentile():
    row = one(meta(), iv_series(np.linspace(0.2, 0.6, 600)), closes(end=TODAY - timedelta(days=30)))
    assert row["rv_pct"] is None and row["rv_status"] == "stale"


def test_short_price_history_gives_no_rv_percentile():
    row = one(meta(), iv_series(np.linspace(0.2, 0.6, 600)), closes(n=100))
    assert row["rv_pct"] is None and row["rv_status"] == "none"


def test_earnings_inside_the_window_is_flagged_and_a_past_date_is_ignored():
    soon = one(meta(next_earnings=(TODAY + timedelta(days=12)).isoformat()), iv_series(np.linspace(0.2, 0.6, 600)), closes())
    assert soon["earnings_in_window"] and soon["days_to_earnings"] == 12
    far = one(meta(next_earnings=(TODAY + timedelta(days=80)).isoformat()), iv_series(np.linspace(0.2, 0.6, 600)), closes())
    assert not far["earnings_in_window"]
    past = one(meta(next_earnings=(TODAY - timedelta(days=3)).isoformat()), iv_series(np.linspace(0.2, 0.6, 600)), closes())
    assert past["next_earnings"] is None and not past["earnings_in_window"]


def test_one_bad_symbol_does_not_hide_the_rest():
    def loader(sym):
        if sym == "BAD":
            raise RuntimeError("boom")
        return closes()
    ivs = {"BAD": iv_series(np.linspace(0.2, 0.6, 600)), "OK": iv_series(np.linspace(0.2, 0.6, 600))}
    df = ev.evaluate_all({"BAD": meta("BAD"), "OK": meta("OK")}, ivs, loader, CFG, TODAY)
    assert dict(zip(df["symbol"], df["status"])) == {"BAD": ev.FETCH_ERROR, "OK": ev.OK}


def test_excluded_symbols_are_never_evaluated():
    ivs = {s: iv_series(np.linspace(0.2, 0.6, 600)) for s in ("A", "B")}
    df = ev.evaluate_all({"A": meta("A"), "B": meta("B")}, ivs, lambda s: closes(), CFG, TODAY, exclude={"b"})
    assert list(df["symbol"]) == ["A"]


def test_percentile_history_for_the_chart():
    h = ev.percentile_history(iv_series(np.linspace(0.2, 0.6, 600)), closes(), CFG)
    assert {"iv_pct", "rv_pct"} <= set(h.columns) and h["iv_pct"].dropna().iloc[-1] == 100.0
    assert ev.percentile_history(iv_series(np.linspace(0.2, 0.6, 60)), None, CFG).empty


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------

@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(wc, "WHEEL_DB_PATH", str(tmp_path / "iv.db"))
    return str(tmp_path / "iv.db")


def test_iv_roundtrip_overwrites_overlap_and_drops_non_positive(db):
    store.put_iv("AAA", iv_series([0.2, 0.3, 0.0, 0.4, float("nan")]))
    assert list(store.load_iv("AAA").round(2)) == [0.2, 0.3, 0.4]
    store.put_iv("AAA", iv_series([0.9], end=date(2026, 10, 20)))                  # same last date: replaced, not duplicated
    s = store.load_iv("AAA")
    assert len(s) == 3 and s.iloc[-1] == 0.9
    assert store.last_iv_date("AAA") == "2026-10-20" and store.last_iv_date("ZZZ") is None


def test_symbol_upsert_updates_only_given_columns_and_rejects_unknown(db):
    store.upsert_symbol("aaa", name="A Corp", exchange="NYSE")
    store.upsert_symbol("AAA", fetch_status="ok")
    row = store.load_symbols()["AAA"]
    assert row["name"] == "A Corp" and row["fetch_status"] == "ok"
    with pytest.raises(ValueError):
        store.upsert_symbol("AAA", nonsense=1)


def test_purge_removes_symbol_and_history(db):
    for s in ("AAA", "BBB"):
        store.upsert_symbol(s, fetch_status="ok")
        store.put_iv(s, iv_series([0.2, 0.3]))
    assert store.purge_symbols(["bbb"]) == {"symbols": 1, "iv_rows": 2}
    assert list(store.load_symbols()) == ["AAA"] and "BBB" not in store.load_all_iv()


def test_stamp_changes_when_data_changes(db):
    a = store.stamp()
    store.upsert_symbol("AAA", fetch_status="ok", iv_checked_at="2026-10-21T10:00:00")
    b = store.stamp()
    store.put_iv("AAA", iv_series([0.2, 0.3]))
    assert len({a, b, store.stamp()}) == 3


def test_orphaned_running_scan_is_recorded_as_partial(db):
    rid = store.start_run("x", 5)
    assert store.interrupt_orphans() == 1
    assert store.latest_run()["id"] == rid and store.latest_run()["status"] == "partial"
