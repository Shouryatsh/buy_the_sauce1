"""
tests/test_ivscreen_ui.py — the IV screener page end to end with AppTest: the three sections, the
data-availability handling, the instrument-type / ticker / earnings filters, absolute exclusions, and
editing the thresholds. Synthetic data in a throwaway DB; no network, no IBKR.
"""

from __future__ import annotations

import json
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import wheel_config as wc
import wheel_watchlist as ww
from ivscreen import config as iconfig
from ivscreen import store
from stockscreen import config as sconfig
from stockscreen import prices as pr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TODAY = date.today()
IDX = pd.bdate_range(end=pd.Timestamp(TODAY), periods=700)


def ivs(values):
    return pd.Series(values, index=IDX[-len(values):], name="IV")


def px(calm=False, wild=False, seed=0):
    rng = np.random.default_rng(seed)
    vol = np.full(len(IDX), 0.012)
    if calm:
        vol[-40:] = 0.002
    if wild:
        vol[-40:] = 0.05
    close = 100 * np.exp(np.cumsum(rng.normal(0, 1, len(IDX)) * vol))
    return pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1e6}, index=IDX)


def noisy(n=600, seed=1):
    return np.random.default_rng(seed).normal(0.3, 0.02, n)


def seed_symbol(sym, iv=None, price=None, category="Common stock", status="ok", earnings=None, exchange="NYSE",
                sector="Technology", sources="Wheel universe", detail=""):
    store.upsert_symbol(sym, name=f"{sym} Corp", category=category, exchange=exchange, sector=sector, sources=sources,
                        fetch_status=status, fetch_detail=detail, iv_checked_at=store.now_iso(), details_at=store.now_iso(),
                        next_earnings=earnings.isoformat() if earnings else None, con_id=abs(hash(sym)) % 10**6)
    if iv is not None:
        store.put_iv(sym, iv)
    if price is not None:
        pr._save(sym, price)


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(wc, "WHEEL_DB_PATH", str(tmp_path / "ui.db"))
    monkeypatch.setattr(ww, "EXCLUSIONS_PATH", tmp_path / "exclusions.json")
    monkeypatch.setattr(sconfig, "CONFIG_PATH", tmp_path / "scfg.json")
    monkeypatch.setattr(iconfig, "CONFIG_PATH", tmp_path / "ivcfg.json")
    monkeypatch.setattr(pr, "PRICE_DIR", tmp_path / "prices")
    st.cache_data.clear()
    st.cache_resource.clear()

    seed_symbol("LOWV", ivs(np.linspace(0.6, 0.2, 600)), px(calm=True))                       # IV pct 0, RV low
    seed_symbol("LOW2", ivs(np.concatenate([noisy(595), [0.29, 0.288, 0.286, 0.285, 0.28]])), px(), exchange="NASDAQ", sector="Energy")
    seed_symbol("VHIGH", ivs(np.linspace(0.2, 0.6, 600)), px(calm=True), earnings=TODAY + timedelta(days=9))  # IV pct 100, RV calm
    seed_symbol("HIGHX", ivs(np.concatenate([noisy(595), [0.31, 0.312, 0.315, 0.318, 0.32]])), px(wild=True), sources="Holdings")
    seed_symbol("MIDX", ivs(np.concatenate([noisy(599), [0.30]])), px(seed=9))
    seed_symbol("ETFX", ivs(np.linspace(0.2, 0.6, 600)), px(calm=True), category="ETF", exchange="ARCA", sector="", sources="ETFs")
    seed_symbol("NOPTS", None, None, status="no_data", detail="IBKR returned no implied-vol history for this symbol")
    seed_symbol("BROKE", None, None, status="error", detail="request timed out after 60s")
    seed_symbol("SHORTH", ivs(np.linspace(0.2, 0.4, 60)), px())
    seed_symbol("OLDIV", ivs(np.linspace(0.2, 0.6, 600))[:-40], px())
    seed_symbol("NOPX", ivs(np.linspace(0.2, 0.6, 600)), None)                                  # IV ranked, no prices -> RV unavailable
    seed_symbol("EXCL", ivs(np.linspace(0.2, 0.6, 600)), px(calm=True))                         # excluded below
    ww.save_exclusions({"EXCL"})
    rid = store.start_run("Wheel universe + Holdings", 12, "note")
    store.finish_run(rid, "complete", {"ok": 9, "no_options": 1, "errors": 1}, "ok")

    at = AppTest.from_file(os.path.join(ROOT, "wheel_dashboard.py"), default_timeout=90).run()
    at.switch_page("app_pages/iv_screener.py").run()
    assert not at.exception, [e.message for e in at.exception]
    return at


def tables(at):
    return [d.value for d in at.dataframe]


def table_a(at):
    return next((t for t in tables(at) if "IV trend" in t.columns and "Event note" not in t.columns and "Gap (IV−RV)" not in t.columns), None)


def table_b(at):
    return next((t for t in tables(at) if "Event note" in t.columns), None)


def table_c(at):
    return next((t for t in tables(at) if "Gap (IV−RV)" in t.columns), None)


def metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def text(at):
    return " ".join([m.value for m in at.markdown] + [c.value for c in at.caption] + [s.value for s in at.success]
                    + [w.value for w in at.warning] + [i.value for i in at.info])


# ---------------------------------------------------------------------------

def test_low_iv_watchlist_lists_only_low_ranked_stocks_lowest_first(app):
    a = table_a(app)
    assert list(a["Symbol"]) == ["LOWV", "LOW2"]
    assert list(a["IV band"]) == ["Very low", "Low"]
    assert a["IV pct"].iloc[0] == 0.0 and a["IV pct"].is_monotonic_increasing


def test_high_iv_watchlist_has_very_high_first_and_event_note(app):
    b = table_b(app)
    assert list(b["Symbol"]) == ["NOPX", "VHIGH", "HIGHX"]            # NOPX has no prices but its IV percentile is still valid
    assert list(b["IV band"]) == ["Very high", "Very high", "High"] and b["IV pct"].iloc[0] == 100.0
    assert "Earnings" in b.loc[b["Symbol"] == "VHIGH", "Event note"].iloc[0]
    assert b.loc[b["Symbol"] == "HIGHX", "Event note"].iloc[0] == ""


def test_no_table_shows_raw_volatility_levels(app):
    for t in tables(app):
        assert not any(c in ("IV", "RV", "Implied vol", "Realized vol", "HV") for c in t.columns)


def test_comparison_classifies_each_pair_and_shows_trend_following(app):
    c = table_c(app).set_index("Symbol")
    assert c.loc["VHIGH", "Relationship"] == "IV elevated vs RV" and c.loc["VHIGH", "Gap (IV−RV)"] > 25
    assert c.loc["LOWV", "Relationship"] == "Both subdued"
    assert c.loc["HIGHX", "Relationship"] == "Both elevated" and c.loc["HIGHX", "IV trend"].startswith("▲")
    assert "RV following IV?" in c.columns and c["RV following IV?"].notna().all()
    assert "NOPX" not in c.index                                            # IV only: listed separately, not compared
    counts = {m.label: m.value for m in app.metric}
    assert counts["IV elevated vs RV"] == "1" and counts["Both subdued"] == "1" and counts["Both elevated"] == "1"


def test_missing_prices_listed_under_rv_unavailable_not_dropped(app):
    nopx = next(t for t in tables(app) if "RV data" in t.columns)
    assert list(nopx["Symbol"]) == ["NOPX"] and nopx["RV data"].iloc[0] == "none"


def test_unknown_is_never_shown_as_low_and_is_explained(app):
    a, b, c = table_a(app), table_b(app), table_c(app)
    for sym in ("NOPTS", "BROKE", "SHORTH", "OLDIV"):
        assert all(sym not in set(t["Symbol"]) for t in (a, b, c))
    avail = next(t for t in tables(app) if "Why not ranked" in t.columns).set_index("Symbol")
    assert "unknown" in avail.loc["NOPTS", "Why not ranked"] and "No options data" in avail.loc["NOPTS", "Why not ranked"]
    assert "retry" in avail.loc["BROKE", "Why not ranked"].lower()
    assert "Too little" in avail.loc["SHORTH", "Why not ranked"] and "stale" in avail.loc["OLDIV", "Why not ranked"].lower()
    assert "ETFX" not in avail.index                                         # a hidden instrument type is not a data problem
    assert "their volatility is unknown, not low" in text(app)


def test_coverage_summary_and_counts(app):
    t = text(app)
    assert "Last scan" in t and "**7 ranked**" in t and "status **complete**" in t and "shortlist you control" in t
    assert metric(app, "Ranked symbols") == "6"                              # ETFX is ranked too but hidden by the instrument filter
    assert metric(app, "Very low IV") == "1" and metric(app, "Low IV") == "1"
    assert metric(app, "High IV") == "1" and metric(app, "Very high IV") == "2"          # VHIGH and NOPX
    assert metric(app, "Not ranked") == "4"


def test_partial_scan_warns(app):
    store.finish_run(store.start_run("x", 3), "partial", {}, "cancelled")
    app.run()
    assert any("partial" in w.value for w in app.warning)


def test_etfs_are_a_separate_category_hidden_until_selected(app):
    assert "ETFX" not in set(table_b(app)["Symbol"])
    types = next(m for m in app.multiselect if m.key == "iv_f_types")
    assert "ETF" in types.options and "ETF" not in types.value
    types.set_value(["Common stock", "ETF"]).run()
    assert "ETFX" in set(table_b(app)["Symbol"])
    types = next(m for m in app.multiselect if m.key == "iv_f_types")
    types.set_value(["ETF"]).run()
    assert set(table_b(app)["Symbol"]) == {"ETFX"} and table_a(app) is None


def test_ticker_exchange_sector_watchlist_and_earnings_filters(app):
    next(i for i in app.text_input if i.key == "iv_f_text").set_value("low").run()
    assert list(table_a(app)["Symbol"]) == ["LOWV", "LOW2"] and table_b(app) is None
    next(i for i in app.text_input if i.key == "iv_f_text").set_value("").run()
    next(m for m in app.multiselect if m.key == "iv_f_exch").set_value(["NASDAQ"]).run()
    assert list(table_a(app)["Symbol"]) == ["LOW2"]
    next(m for m in app.multiselect if m.key == "iv_f_exch").set_value([]).run()
    next(m for m in app.multiselect if m.key == "iv_f_sect").set_value(["Energy"]).run()
    assert list(table_a(app)["Symbol"]) == ["LOW2"]
    next(m for m in app.multiselect if m.key == "iv_f_sect").set_value([]).run()
    next(m for m in app.multiselect if m.key == "iv_f_lists").set_value(["Holdings"]).run()
    assert list(table_b(app)["Symbol"]) == ["HIGHX"]
    next(m for m in app.multiselect if m.key == "iv_f_lists").set_value([]).run()
    next(s for s in app.selectbox if s.key == "iv_f_earn").set_value("Within the 30-day IV window").run()
    assert list(table_b(app)["Symbol"]) == ["VHIGH"] and table_a(app) is None


def test_excluded_ticker_is_hidden_everywhere_and_purged(app):
    t = text(app) + " ".join(" ".join(str(v) for v in df.to_numpy().ravel()) for df in tables(app))
    assert "EXCL" not in t
    assert "EXCL" not in store.load_symbols() and "EXCL" not in store.load_all_iv()
    ww.save_exclusions({"EXCL", "VHIGH"})                                    # adding one later removes it on the next render
    app.run()
    assert "VHIGH" not in set(table_b(app)["Symbol"]) and "VHIGH" not in store.load_symbols()


def test_detail_view_explains_the_relationship_without_calling_options_overpriced(app):
    app.session_state["iv_pick"] = "VHIGH"
    app.run()
    t = text(app)
    assert "IV elevated vs RV." in t and "not by itself evidence that the options are overpriced" in t
    assert "fall inside the 30-day IV window" in t
    assert metric(app, "IV percentile") == "100" and metric(app, "Gap (IV − RV)").startswith("+")
    assert len(app.get("plotly_chart")) >= 2                                  # scatter + percentile history


def test_page_states_the_mcmillan_rule_and_the_disclaimer(app):
    t = text(app)
    assert "never compared with each other" in t and "not proof that options are overpriced" in t


def test_changing_the_thresholds_reclassifies_instantly_and_persists(app, tmp_path):
    next(n for n in app.number_input if n.label == "High ≥").set_value(95.0)
    next(n for n in app.number_input if n.label == "Very high ≥").set_value(99.0)
    next(b for b in app.button if b.label == "Save settings").click().run()
    assert json.loads((tmp_path / "ivcfg.json").read_text())["high_min"] == 95.0
    b = table_b(app)
    assert list(b["Symbol"]) == ["NOPX", "VHIGH"] and set(b["IV band"]) == {"Very high"}     # HIGHX (~88th) no longer qualifies as high


def test_inconsistent_thresholds_are_refused(app, tmp_path):
    next(n for n in app.number_input if n.label == "Very low ≤").set_value(40.0)
    next(b for b in app.button if b.label == "Save settings").click().run()
    assert any("Bands must satisfy" in e.value for e in app.error) and not (tmp_path / "ivcfg.json").exists()


def test_empty_state_before_any_scan(tmp_path, monkeypatch):
    monkeypatch.setattr(wc, "WHEEL_DB_PATH", str(tmp_path / "empty.db"))
    monkeypatch.setattr(ww, "EXCLUSIONS_PATH", tmp_path / "exclusions.json")
    monkeypatch.setattr(sconfig, "CONFIG_PATH", tmp_path / "scfg.json")
    monkeypatch.setattr(iconfig, "CONFIG_PATH", tmp_path / "ivcfg.json")
    monkeypatch.setattr(pr, "PRICE_DIR", tmp_path / "prices")
    st.cache_data.clear()
    st.cache_resource.clear()
    at = AppTest.from_file(os.path.join(ROOT, "wheel_dashboard.py"), default_timeout=60).run()
    at.switch_page("app_pages/iv_screener.py").run()
    assert not at.exception
    assert any("No scan has been run yet" in i.value for i in at.info)
    assert any("Results appear here" in c.value for c in at.caption)
    run_btn = next(b for b in at.button if b.label == "Run scan")
    assert run_btn.disabled                                                  # not connected: cannot scan
