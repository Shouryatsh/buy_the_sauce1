"""
tests/test_volsurface_ui.py — the Vol surface page end to end with AppTest. The network is replaced by a synthetic option
chain generated from a known SVI surface; the database is a throwaway.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import wheel_config as wc
import wheel_watchlist as ww
from stockscreen import config as sconfig
from tests.volsurface_fixtures import ASOF, SPOT, make_quotes, settle_expiry
from volsurface import analysis as an
from volsurface import data, model, store, surface
from volsurface import ui as vui

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAYS = (14, 30, 60, 120, 240)
LISTING = [settle_expiry(d) for d in range(1, 300)]


def fake_vix(n=620, last_high=True) -> pd.DataFrame:
    rng = np.random.default_rng(4)
    idx = pd.bdate_range(end="2026-10-23", periods=n)
    vix = 16 + np.cumsum(rng.normal(0, 0.15, n))
    vix = np.clip(vix, 11, 25)
    if last_high:
        vix[-1] = vix.max() + 3
    return pd.DataFrame({"^VIX": vix, "^VIX3M": vix * 0.93 + 1, "^SKEW": 130 + rng.normal(0, 4, n), "^VVIX": 90 + rng.normal(0, 6, n),
                         "^GSPC": 5000 * np.exp(np.cumsum(rng.normal(0, 0.008, n)))}, index=idx)


def qset(quotes=None, asof=ASOF, source="Yahoo (delayed)"):
    return data.QuoteSet("SPX", source, asof, SPOT, make_quotes(DAYS) if quotes is None else quotes, LISTING, [])


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(wc, "WHEEL_DB_PATH", str(tmp_path / "vs.db"))
    monkeypatch.setattr(ww, "EXCLUSIONS_PATH", tmp_path / "exclusions.json")
    monkeypatch.setattr(sconfig, "CONFIG_PATH", tmp_path / "scfg.json")
    monkeypatch.setattr(data, "now_et", lambda: ASOF)
    monkeypatch.setattr(data, "yahoo_listing", lambda s: (LISTING, SPOT))
    monkeypatch.setattr(data, "fetch_yahoo", lambda symbol, expiries=None, workers=6: qset())
    monkeypatch.setattr(vui, "vix_history", lambda: fake_vix())
    st.cache_data.clear()
    st.cache_resource.clear()
    return tmp_path


def open_page(timeout=180):
    at = AppTest.from_file(os.path.join(ROOT, "wheel_dashboard.py"), default_timeout=timeout).run()
    at.switch_page("app_pages/vol_surface.py").run()
    return at


@pytest.fixture()
def app(env):
    at = open_page()
    assert not at.exception, [e.message for e in at.exception]
    return at


def metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def metric_delta(at, label):
    return next(m.delta for m in at.metric if m.label == label)


def text(at):
    return " ".join([m.value for m in at.markdown] + [c.value for c in at.caption] + [s.value for s in at.success] + [w.value for w in at.warning]
                    + [i.value for i in at.info] + [e.value for e in at.error])


# ---------------------------------------------------------------------------

def test_page_loads_with_all_seven_sections_and_the_disclaimer(app):
    heads = [s.value for s in app.subheader]
    for n in range(1, 8):
        assert any(h.startswith(f"{n} ·") for h in heads), heads
    assert "never places, changes or cancels an order" in text(app)
    assert metric(app, "Underlying") == "SPX" and metric(app, "Expiries used") == "5"
    assert len(app.get("plotly_chart")) >= 8


def test_loaded_quotes_are_saved_as_a_snapshot_automatically(app):
    snaps = store.list_snapshots("SPX")
    assert len(snaps) == 1 and int(snaps.iloc[0]["n_quotes"]) == len(make_quotes(DAYS))


def test_smile_defaults_to_the_30_day_expiry_and_switches(app):
    sel = next(s for s in app.selectbox if s.key == "vs_smile_exp")
    assert sel.value == settle_expiry(30)
    sel.set_value(settle_expiry(120)).run()
    assert not app.exception
    assert float(metric(app, "Days to expiry")) == pytest.approx(120, abs=1)
    assert "ATM implied vol" in [m.label for m in app.metric]


def test_term_structure_table_lists_every_expiry_with_a_fit(app):
    t = next(d.value for d in app.dataframe if "ATM IV (fit)" in d.value.columns)
    assert len(t) == 5 and (t["Fit"] == "ok").all() and (t["ATM total variance"].diff().dropna() > 0).all()      # total variance must grow with maturity
    assert (t["Forward from"] == "parity").all() and (t["Fit error (vol pts)"] < 0.5).all()


def test_total_variance_section_confirms_atm_variance_is_monotone(app):
    assert any("ATM total variance against maturity" in m.value and ":green[it is]" in m.value for m in app.markdown)


# ---------------------------------------------------------------------------
# the live SVI lab
# ---------------------------------------------------------------------------

def lab_key(name, days=30):
    return f"vsp_{settle_expiry(days)}_{name}"


def test_lab_starts_at_the_automatic_fit_with_five_parameter_sliders(app):
    sliders = {s.key: s for s in app.slider if s.key.startswith("vsp_")}
    assert {lab_key(n) for n in ("a", "b", "rho", "m", "sigma")} <= set(sliders)
    assert sliders[lab_key("rho")].value < 0                                       # equity skew
    assert float(metric(app, "Fit error, your curve").split()[0]) == pytest.approx(float(metric_delta(app, "Fit error, your curve").split()[-1]), abs=0.01)


def test_moving_a_slider_changes_the_fit_error_and_reset_restores_it(app):
    before = float(metric(app, "Fit error, your curve").split()[0])
    app.slider(key=lab_key("rho")).set_value(0.6).run()
    after = float(metric(app, "Fit error, your curve").split()[0])
    assert after > before * 2
    next(b for b in app.button if b.key == "vs_lab_reset").click().run()
    assert float(metric(app, "Fit error, your curve").split()[0]) == pytest.approx(before, abs=0.01)


def test_an_arbitrageable_curve_shows_the_negative_density_warning(app):
    app.slider(key=lab_key("a")).set_value(-0.0410)
    app.slider(key=lab_key("b")).set_value(0.1331)
    app.slider(key=lab_key("rho")).set_value(0.306)
    app.slider(key=lab_key("m")).set_value(0.3586)
    app.slider(key=lab_key("sigma")).set_value(0.4153)
    app.run()
    assert not app.exception
    assert metric(app, "Butterfly") in ("extrapolation only", "ARBITRAGE in data")
    assert any("Negative risk-neutral density" in w.value for w in app.warning)


def test_a_non_positive_variance_curve_is_called_out(app):
    app.slider(key=lab_key("a")).set_value(-0.1)
    app.slider(key=lab_key("b")).set_value(0.01)
    app.run()
    assert any("not positive" in e.value for e in app.error)


def test_automatic_fit_style_can_be_switched_per_slice(app):
    style = next(s for s in app.get("button_group") if s.key == "vs_lab_style")
    assert style is not None
    app.get("button_group")                                                       # smoke: the control exists; switching is exercised below
    app.session_state["vs_lab_style"] = "Plain least squares"
    app.run()
    assert not app.exception and float(metric(app, "Fit error, your curve").split()[0]) < 1.0


# ---------------------------------------------------------------------------
# arbitrage
# ---------------------------------------------------------------------------

def test_clean_synthetic_surface_reports_no_arbitrage(app):
    assert metric(app, "Butterfly, in the quoted range") == "5 of 5 clean"
    assert metric(app, "Calendar, in the common range") == "4 of 4 clean"
    assert metric(app, "ATM variance monotone") == "yes"
    assert any("No static arbitrage where the market actually quotes" in s.value for s in app.success)
    bt = next(d.value for d in app.dataframe if "Lee wings L/R" in d.value.columns)
    assert (bt["Butterfly"].isin(["clean", "extrapolation only"])).all() and (bt["Lee ≤ 2"] == "ok").all()
    ct = next(d.value for d in app.dataframe if "Common data k" in d.value.columns)
    assert len(ct) == 4 and (ct["Calendar"] != "ARBITRAGE in data").all()
    qc = next(d.value for d in app.dataframe if "Monotonicity violations" in d.value.columns)
    assert len(qc) == 5 and (qc["Monotonicity violations"] == 0).all() and (qc["Convexity (butterfly) violations"] == 0).all()


def test_crossing_expiries_are_reported_as_calendar_arbitrage(env, monkeypatch):
    def crossing(T):                                                              # the 120-day smile has LESS variance than the 60-day one
        from tests.volsurface_fixtures import true_params
        p = true_params(T)
        return p if abs(T * 365 - 120) > 1 else type(p)(p.a * 0.2, p.b * 0.4, p.rho, p.m, p.sigma)
    monkeypatch.setattr(data, "fetch_yahoo", lambda symbol, expiries=None, workers=6: qset(make_quotes(DAYS, params_fn=crossing)))
    at = open_page()
    assert not at.exception
    assert not metric(at, "ATM variance monotone") == "yes"
    ct = next(d.value for d in at.dataframe if "Common data k" in d.value.columns)
    assert (ct["Calendar"] == "ARBITRAGE in data").any()
    assert any("arbitrage inside the quoted region" in w.value for w in at.warning)


# ---------------------------------------------------------------------------
# dynamics and decisions
# ---------------------------------------------------------------------------

def test_dynamics_without_history_says_so_and_still_shows_market_indicators(app):
    t = text(app)
    assert "No earlier snapshot of this underlying yet" in t
    labels = [m.label for m in app.metric]
    assert {"VIX", "VIX / VIX3M", "CBOE SKEW", "VVIX", "SPX 20d realized vol"} <= set(labels)
    assert "th pct" in metric_delta(app, "VIX")


def test_dynamics_compares_with_an_earlier_snapshot_and_charts_history(env, monkeypatch):
    shifted = lambda T: __import__("volsurface.svi", fromlist=["SVI"]).SVI(0.002 * T + 0.0008 * T / 0.25, 0.09 * np.sqrt(T) + 0.01, -0.55, 0.01, 0.12 * np.sqrt(T) + 0.03)   # noqa: E731
    old = qset(make_quotes(DAYS, params_fn=shifted), asof=ASOF - pd.Timedelta(days=1))
    sid = store.save_snapshot(old)
    surf = surface.build_surface(old.quotes, old.spot, old.asof)
    store.update_metrics(sid, an.snapshot_metrics(model.fit_surface(surf, model.FitSettings(weights="equal", nstarts=6))))
    at = open_page()
    assert not at.exception, [e.message for e in at.exception]
    assert "No earlier snapshot" not in text(at)
    assert at.selectbox(key="vs_dyn_pick").value == sid
    assert any("Implied-vol change, now minus" in c.value for c in at.caption)
    assert len(store.snapshot_metrics_df("SPX")) == 2                           # the new snapshot's metrics were cached too


def test_saved_snapshot_source_works_without_fetching(env, monkeypatch):
    sid = store.save_snapshot(qset(asof=ASOF - pd.Timedelta(days=2), source="IBKR (frozen)"))
    monkeypatch.setattr(data, "fetch_yahoo", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no network in this test")))
    at = AppTest.from_file(os.path.join(ROOT, "wheel_dashboard.py"), default_timeout=180).run()
    at.switch_page("app_pages/vol_surface.py").run()
    assert any("RuntimeError" in e.value for e in at.error)                       # the automatic Yahoo load failed, visibly
    at.session_state["vs_source"] = "Saved snapshot"
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    assert at.selectbox(key="vs_snap_pick").value == sid
    assert metric(at, "Source") == "IBKR (frozen)" and metric(at, "Expiries used") == "5"


def test_decisions_section_has_scorecard_and_a_rule_respecting_put_scan(app):
    t = text(app)
    assert "Volatility level: VIX" in t and "Term structure: VIX / VIX3M" in t and "never places" in t
    scan = next(d.value for d in app.dataframe if "Passes my rules" in d.value.columns)
    assert len(scan) > 3 and scan["Passes my rules"].all()                       # default view: only what passes
    assert (scan["|Delta|"] <= wc.MAX_DELTA_ABS + 1e-9).all() and (scan["Spread %"] <= wc.MAX_SPREAD_PCT * 100 + 1e-9).all()
    assert set(scan["Signal"]) <= {"RICH", "FAIR", "CHEAP", "UNKNOWN"} and (scan["Strike"] < SPOT * 1.01).all()
    assert scan["Annualised %"].is_monotonic_decreasing                          # default order: highest yield first
    app.session_state["vs_dec_show"] = "All"
    app.run()
    full = next(d.value for d in app.dataframe if "Passes my rules" in d.value.columns)
    assert len(full) > len(scan) and not full["Passes my rules"].all()


def test_ibkr_source_without_a_connection_explains_what_to_do(env):
    at = open_page()
    at.session_state["vs_source"] = "IBKR (live)"
    at.run()
    assert not at.exception
    assert any("Connect to IBKR" in c.value for c in at.caption)


def test_fetch_failure_is_shown_not_hidden(env, monkeypatch):
    monkeypatch.setattr(data, "fetch_yahoo", lambda *a, **k: (_ for _ in ()).throw(ValueError("Yahoo lists no options for SPX")))
    at = open_page()
    assert not at.exception
    assert any("Yahoo lists no options for SPX" in e.value for e in at.error)


def test_a_closed_market_with_no_bids_gives_a_clear_message(env, monkeypatch):
    q = make_quotes(DAYS)
    q["bid"], q["ask"] = 0.0, 0.0
    monkeypatch.setattr(data, "fetch_yahoo", lambda *a, **k: qset(q))
    at = open_page()
    assert not at.exception
    assert any("No expiry has enough usable quotes" in e.value for e in at.error)


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------

def test_scorecard_reads_percentiles_without_predicting():
    pcts = {"VIX": {"value": 28.0, "pct": 93.0}, "VIX / VIX3M": {"value": 1.08, "pct": 90.0}, "CBOE SKEW": {"value": 150, "pct": 80.0},
            "SPX 20d realized vol": {"value": 0.2, "pct": 40.0}}
    cm = {30: {"atm": 0.2, "skew90": 0.05}, 90: {"atm": 0.18, "skew90": 0.04}}
    items = an.scorecard(pcts, cm)
    titles = " ".join(i["title"] for i in items)
    assert "VIX 28.0, 93th percentile (very high)" in titles and "backwardation" in titles and "−2.0 vol pts".replace("−", "-") in titles
    assert items[0]["tone"] == "warn" and any(i["tone"] == "warn" for i in items)
    assert not any(w in " ".join(i["text"] for i in items).lower() for w in ("will rise", "will fall", "guarantee", "buy now", "sell now"))
    calm = an.scorecard({"VIX": {"value": 12.0, "pct": 8.0}, "VIX / VIX3M": {"value": 0.85, "pct": 15.0}}, {})
    assert calm[0]["tone"] == "info" and "cheap" in calm[0]["text"] and "contango" in calm[1]["title"]
    assert an.scorecard({}, {})[0]["title"] == "Market-wide indicators are unavailable"


def test_vix_percentiles_use_each_series_own_history():
    out = vui.vix_percentiles(fake_vix())
    assert out["VIX"]["pct"] >= 99 and 0 <= out["VIX / VIX3M"]["pct"] <= 100
    assert set(out) == {"VIX", "VIX / VIX3M", "CBOE SKEW", "VVIX", "SPX 20d realized vol"}
    assert vui.vix_percentiles(pd.DataFrame()) == {}


def test_check_positions_locates_a_short_put_on_its_own_smile(env, monkeypatch):
    expiry = settle_expiry(30)
    q30 = make_quotes(DAYS)
    q30 = q30[(q30["expiry"] == expiry) & (q30["right"] == "P")]
    F = SPOT * np.exp(0.027 * 30 / 365)
    strike = float(q30.iloc[(q30["strike"] - F * 0.93).abs().argsort().iloc[0]]["strike"])
    pos = [SimpleNamespace(symbol="SPX", sec_type="OPT", right="P", strike=strike, expiry=expiry.replace("-", ""), quantity=-1.0),
           SimpleNamespace(symbol="SPX", sec_type="OPT", right="P", strike=strike, expiry=expiry.replace("-", ""), quantity=+1.0),       # long: ignored
           SimpleNamespace(symbol="SPX", sec_type="STK", right=None, strike=None, expiry=None, quantity=-5.0)]
    client = SimpleNamespace(get_positions=lambda: pos)
    out = vui.check_positions(client, model.FitSettings(weights="equal", nstarts=6))
    assert len(out) == 1
    r = out.iloc[0]
    assert r["Strike"] == strike and r["Qty"] == -1.0 and r["Note"] == ""
    assert r["Market IV"] == pytest.approx(r["Fit IV"], abs=0.01) and r["Signal"] in ("FAIR", "RICH", "CHEAP")
    assert r["P(ITM) smile %"] > 0 and r["Model price"] > 0 and r["Days"] == pytest.approx(30, abs=1)


def test_check_positions_reports_a_failed_lookup_in_the_row(env, monkeypatch):
    monkeypatch.setattr(data, "fetch_yahoo", lambda *a, **k: (_ for _ in ()).throw(ValueError("Yahoo lists no options for ZZZ")))
    pos = [SimpleNamespace(symbol="ZZZ", sec_type="OPT", right="P", strike=50.0, expiry="20261120", quantity=-2.0)]
    out = vui.check_positions(SimpleNamespace(get_positions=lambda: pos), model.FitSettings())
    assert len(out) == 1 and "Yahoo lists no options" in out.iloc[0]["Note"]
