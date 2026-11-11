"""
tests/test_stockscreen_ui.py — the Fundamentals page end to end with AppTest:
summary counts, presets, filters, sorting, stock-detail navigation, threshold
editing, alerts and the partial-scan warning. Synthetic records, throwaway DB.
"""

from __future__ import annotations

import os

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import wheel_config as wc
import wheel_watchlist as ww
from stockscreen import config as sconfig
from stockscreen import store
from tests.stockscreen_fixtures import good_fund, positions, record

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOW = dict(pos=positions(8, 10, 12, pc=6))


def seed():
    flat_growth = good_fund("Flat Growth", rev=(25,) * 12)
    deteriorating = good_fund("Deteriorating Margin", rev=(25,) * 8 + (30,) * 4, ni=(14,) * 8 + (13, 11, 9, 7.5))
    recs = [
        record("PASSHI", name="Steady Compounder"),
        record("PASSPB", tech="pullback", name="Pullback Corp", **LOW),
        record("PASSREV", tech="reversal", name="Reversal Corp", **LOW),
        record("PASSDWN", tech="downtrend", name="Downtrend Corp", **LOW),
        record("FLATGR", fund=flat_growth),
        record("DETER", fund=deteriorating, tech="pullback", **LOW),
        record("ETFX", fund={"fetch": "no_statements", "error": "fund or ETF"}),
    ]
    rid = store.start_run("Wheel universe (shortlist)", 7, "The wheel universe is a shortlist, not the whole market.")
    for r in recs:
        store.upsert_result(r["symbol"], rid, r)
    store.finish_run(rid, "complete", 6, 1, 0, "ok")
    return rid


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(wc, "WHEEL_DB_PATH", str(tmp_path / "ui.db"))
    monkeypatch.setattr(sconfig, "CONFIG_PATH", tmp_path / "cfg.json")
    monkeypatch.setattr(ww, "EXCLUSIONS_PATH", tmp_path / "exclusions.json")      # never read the real exclusion list
    st.cache_data.clear()
    seed()
    at = AppTest.from_file(os.path.join(ROOT, "wheel_dashboard.py"), default_timeout=60).run()
    at.switch_page("app_pages/screener.py").run()
    assert not at.exception, [e.message for e in at.exception]
    return at


def results_symbols(at):
    for df in at.dataframe:
        if "symbol" in df.value.columns and "status" in df.value.columns:
            return list(df.value["symbol"])
    return []


def metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def test_summary_counts_and_coverage_banner(app):
    assert metric(app, "Stocks scanned") == "7"
    assert metric(app, "Pass all fundamentals") == "4"
    assert metric(app, "In the low-price zone") == "3"
    assert metric(app, "Healthy pullbacks") == "1" and metric(app, "Reversal confirmation") == "1"
    assert metric(app, "Declining fundamentals") == "1"
    assert metric(app, "Data / coverage issues") == "1"                       # the ETF without statements
    banner = " ".join(x.value for x in app.success) + " ".join(c.value for c in app.caption)
    assert "status **complete**" in banner and "not a market-wide scan" in banner and "6 of 7" not in banner
    assert "7 of 7 tickers have results" in banner


def test_default_preset_lists_only_passing_stocks(app):
    assert sorted(results_symbols(app)) == ["PASSDWN", "PASSHI", "PASSPB", "PASSREV"]


def test_presets_filter_the_table(app):
    app.segmented_control(key="preset").set_value("Healthy pullbacks").run()
    assert results_symbols(app) == ["PASSPB"]
    app.segmented_control(key="preset").set_value("Potential reversals").run()
    assert results_symbols(app) == ["PASSREV"]
    app.segmented_control(key="preset").set_value("Quality stocks near lows").run()
    assert sorted(results_symbols(app)) == ["PASSDWN", "PASSPB", "PASSREV"]
    app.segmented_control(key="preset").set_value("Strong uptrends").run()
    assert results_symbols(app) == ["PASSHI"]
    app.segmented_control(key="preset").set_value("Fundamental deterioration").run()
    assert results_symbols(app) == ["DETER"]
    assert not app.exception


def test_the_dangerous_lookalike_is_kept_separate_and_visible(app):
    # class E (low price but declining fundamentals / downtrend) is findable, not silently dropped
    app.segmented_control(key="preset").set_value("Quality stocks").run()
    app.multiselect(key="flt_Quality stocks_status").set_value([]).run()                 # every status
    app.multiselect(key="flt_Quality stocks_opp").set_value(["E"]).run()
    assert sorted(results_symbols(app)) == ["DETER", "PASSDWN"]


def test_extra_filters_narrow_the_results(app):
    key = "flt_Quality stocks"
    app.number_input(key=key + "_cap").set_value(1000.0).run()                           # $1,000B minimum market cap
    assert results_symbols(app) == []
    assert any("No stocks match" in i.value for i in app.info)
    app.number_input(key=key + "_cap").set_value(0.0).run()
    app.slider(key=key + "_m52").set_value(20).run()                                    # max 52-week range position
    assert sorted(results_symbols(app)) == ["PASSDWN", "PASSPB", "PASSREV"]
    app.slider(key=key + "_m52").set_value(100).run()
    app.slider(key=key + "_rsi").set_value((0, 35)).run()                                # RSI between 0 and 35
    assert results_symbols(app) == ["PASSDWN"]


def test_results_are_sorted_by_rank_score_by_default(app):
    app.multiselect(key="flt_Quality stocks_status").set_value([]).run()
    df = next(d.value for d in app.dataframe if "composite" in d.value.columns)
    scores = [s for s in df["composite"] if s == s]
    assert scores == sorted(scores, reverse=True)
    app.segmented_control(key="sort_by").set_value("Mkt cap").run()
    assert not app.exception


def test_column_groups_can_be_hidden(app):
    app.pills(key="col_groups").set_value(["Company", "Classification"]).run()
    df = next(d.value for d in app.dataframe if "symbol" in d.value.columns and "status" in d.value.columns)
    assert "fcf_yield" not in df.columns and "opportunity" in df.columns


def test_stock_detail_navigation_shows_calculations_and_explanations(app):
    app.selectbox(key="detail_pick").set_value("PASSPB").run()
    assert not app.exception, [e.message for e in app.exception]
    assert any("PASSPB" in s.value for s in app.subheader)
    tables = [d.value for d in app.dataframe]
    crit = next(t for t in tables if "Criterion" in t.columns)
    assert set(crit["Result"]) == {"PASS"} and len(crit) == 6
    obs = next(t for t in tables if "Period" in t.columns)
    assert len(obs) == 6                                                                 # 4 quarters + 2 fiscal years
    assert obs["Period"].str.startswith("Quarter ended").sum() == 4 and obs["Period"].str.startswith("Fiscal year ended").sum() == 2
    assert any(t.get("Item") is not None and "FCF yield" in list(t["Item"]) for t in tables if "Item" in t.columns)
    assert any("Healthy pullback" in b.value for b in app.markdown) or any("healthy pullback" in w.value.lower() for w in app.markdown)


def test_failed_stock_detail_explains_why_it_was_excluded(app):
    app.multiselect(key="flt_Quality stocks_status").set_value([]).run()
    app.selectbox(key="detail_pick").set_value("FLATGR").run()
    assert any("Why it was excluded" in w.value and "Revenue growth" in w.value for w in app.warning)
    app.selectbox(key="detail_pick").set_value("ETFX").run()
    assert any("fund or ETF" in w.value for w in app.warning)


def test_threshold_change_reclassifies_without_a_rescan(app, tmp_path):
    assert metric(app, "Pass all fundamentals") == "4"
    app.number_input[[n.label for n in app.number_input].index("Revenue growth must exceed (%)")].set_value(25.0)
    next(b for b in app.button if b.label == "Save profile").click().run()
    assert metric(app, "Pass all fundamentals") == "0"                                   # 20% growth no longer clears 25%
    assert (tmp_path / "cfg.json").exists()
    next(b for b in app.button if b.label == "Reset to defaults").click().run()
    assert metric(app, "Pass all fundamentals") == "4"


def test_alerts_listed_and_marked_read(app):
    store.add_alert({"symbol": "PASSPB", "name": "Pullback Corp", "kind": "healthy_pullback", "price": 10.0,
                     "trigger_text": "in the low-price zone on: 52w", "percentile": 6.0, "fcf_yield": 0.053,
                     "fund_status": "PASSED (Stable)", "tech_status": "Healthy pullback", "reason": "r",
                     "data_asof": "2026-06-30", "price_asof": "2026-10-20"})
    app.run()
    assert any("1 unread" in s.value for s in app.subheader)
    next(b for b in app.button if b.label == "Mark all as read").click().run()
    assert any("0 unread" in s.value for s in app.subheader)


def test_partial_scan_is_not_presented_as_full_coverage(tmp_path, monkeypatch):
    monkeypatch.setattr(wc, "WHEEL_DB_PATH", str(tmp_path / "p.db"))
    monkeypatch.setattr(sconfig, "CONFIG_PATH", tmp_path / "cfg.json")
    st.cache_data.clear()
    rid = store.start_run("Wheel universe (shortlist)", 100, "")
    store.upsert_result("PASSHI", rid, record("PASSHI"))
    store.finish_run(rid, "partial", 1, 0, 2, "cancelled")
    at = AppTest.from_file(os.path.join(ROOT, "wheel_dashboard.py"), default_timeout=60).run()
    at.switch_page("app_pages/screener.py").run()
    assert any("partial" in w.value.lower() for w in at.warning)
    shown = [e.value for group in (at.warning, at.markdown, at.success, at.info) for e in group]
    assert any("1 of 100 tickers have results" in v for v in shown)


def test_empty_database_shows_a_first_run_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(wc, "WHEEL_DB_PATH", str(tmp_path / "empty.db"))
    monkeypatch.setattr(sconfig, "CONFIG_PATH", tmp_path / "cfg.json")
    st.cache_data.clear()
    at = AppTest.from_file(os.path.join(ROOT, "wheel_dashboard.py"), default_timeout=60).run()
    at.switch_page("app_pages/screener.py").run()
    assert not at.exception and any("No scan has been run yet" in i.value for i in at.info)
