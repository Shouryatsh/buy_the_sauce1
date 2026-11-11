"""Phase 1 data rules: corporate actions, bad prints, carry-forward limits, membership, point-in-time fundamentals."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from premia import data, fundamentals as F
from premia import store


def _px(closes, start="2024-01-01"):
    idx = pd.bdate_range(start, periods=len(closes))
    c = np.asarray(closes, float)
    return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "adj_close": c, "volume": 1e6}, index=idx)


def test_unadjusted_close_undoes_later_splits_only():
    px = _px([50.0] * 6)
    acts = pd.DataFrame({"date": [px.index[3]], "kind": ["split"], "value": [4.0]})
    un = data.unadjusted_close(px, acts)
    assert list(un.iloc[:3]) == [200.0] * 3 and list(un.iloc[3:]) == [50.0] * 3     # pre-split prices were 4x higher


def test_dividend_yield_uses_only_dividends_already_paid():
    px = _px([100.0] * 400)
    d1, d2 = px.index[100], px.index[300]
    acts = pd.DataFrame({"date": [d1, d2], "kind": ["div", "div"], "value": [1.0, 1.0]})
    q = data.dividend_yield_ttm(px, acts)
    assert q.iloc[99] == 0 and q.iloc[100] == pytest.approx(0.01) and q.iloc[300] == pytest.approx(0.02)
    # a year after the first dividend it drops out of the trailing window
    later = px.index.searchsorted(d1 + pd.Timedelta(days=366))
    assert q.iloc[later] == pytest.approx(0.01)


def test_bad_iv_prints_that_reverse_are_removed_and_real_jumps_kept():
    idx = pd.bdate_range("2024-01-01", periods=10)
    iv = pd.Series([0.2, 0.2, 0.6, 0.2, 0.2, 0.2, 0.4, 0.42, 0.41, 0.4], index=idx)
    clean, bad = data.clean_iv(iv)
    assert bad.iloc[2] and not bad.iloc[6]                 # spike that reverses = bad print; a level shift that stays = real
    assert idx[2] not in clean.index and idx[6] in clean.index


def test_iv_is_carried_forward_at_most_three_days_never_invented():
    px = _px([100.0] * 10)
    iv = pd.Series([0.2], index=[px.index[0]])
    a = data.align(px, iv)
    assert a["iv"].iloc[:4].notna().all() and a["iv"].iloc[4:].isna().all()
    assert list(a["iv_age"].iloc[:4]) == [0, 1, 2, 3]


def test_membership_intervals_are_point_in_time():
    m = pd.DataFrame({"ticker": ["A", "B", "C"], "start": ["2010-01-01", "2012-06-01", "2009-01-01"],
                      "end": [None, "2015-01-01", "2011-01-01"]})
    assert data.members_on("2011-06-30", m) == {"A"}
    assert data.members_on("2013-01-01", m) == {"A", "B"}
    assert data.members_on("2015-01-01", m) == {"A"}           # end date exclusive


def _facts(entries_by_tag: dict, shares=None) -> dict:
    g = {tag: {"units": {"USD": entries}} for tag, entries in entries_by_tag.items()}
    out = {"cik": 1, "entityName": "X", "facts": {"us-gaap": g, "dei": {}}}
    if shares:
        out["facts"]["dei"]["EntityCommonStockSharesOutstanding"] = {"units": {"shares": shares}}
    return out


def _q(start, end, val, filed, form="10-Q"):
    return {"start": start, "end": end, "val": val, "filed": filed, "form": form, "accn": f"a{filed}{end}", "fy": 0, "fp": ""}


def test_facts_asof_hides_filings_made_later_including_restatements():
    f = _facts({"Revenues": [_q("2020-01-01", "2020-03-31", 100, "2020-05-01"),
                             _q("2020-01-01", "2020-03-31", 80, "2021-05-01")]})       # restated a year later
    early = F.facts_asof(f, "2020-06-30")["facts"]["us-gaap"]["Revenues"]["units"]["USD"]
    assert [e["val"] for e in early] == [100]
    assert F.facts_asof(f, "2020-04-30")["facts"]["us-gaap"] == {}


def test_share_count_filed_before_a_split_is_split_adjusted():
    shares = [{"end": "2024-04-20", "val": 2.5e9, "filed": "2024-05-20", "form": "10-Q", "accn": "x1"}]
    f = _facts({}, shares)
    sh, end = F.shares_asof(f, "2024-06-28")
    assert sh == 2.5e9 and end == "2024-04-20"
    assert F.split_ratio_between([("2024-06-10", 10.0)], end, "2024-06-28") == 10.0
    assert F.split_ratio_between([("2024-06-10", 10.0)], end, "2024-06-01") == 1.0


def test_pit_screen_end_to_end_on_synthetic_company():
    """16 quarters of growing FCF and a 20 % margin: passes; with flat FCF it fails 'increasing'."""
    def company(growth: float):
        ent = {"NetCashProvidedByUsedInOperatingActivities": [], "PaymentsToAcquirePropertyPlantAndEquipment": [],
               "Revenues": [], "NetIncomeLoss": []}
        for i in range(16):
            q0 = pd.Timestamp("2017-01-01") + pd.DateOffset(months=3 * i)
            q1 = q0 + pd.DateOffset(months=3) - pd.Timedelta(days=1)
            filed = (q1 + pd.Timedelta(days=35)).strftime("%Y-%m-%d")
            s, e = q0.strftime("%Y-%m-%d"), q1.strftime("%Y-%m-%d")
            g = (1 + growth) ** (i / 4)
            ent["NetCashProvidedByUsedInOperatingActivities"].append(_q(s, e, 300 * g, filed))
            ent["PaymentsToAcquirePropertyPlantAndEquipment"].append(_q(s, e, 100 * g, filed))
            ent["Revenues"].append(_q(s, e, 1000, filed))
            ent["NetIncomeLoss"].append(_q(s, e, 200, filed))
        return _facts(ent, [{"end": "2020-09-30", "val": 1e8, "filed": "2020-11-05", "form": "10-Q", "accn": "s"}])
    ok = F.evaluate("X", company(0.15), "2020-12-31", close_unadj=150.0)    # mcap 15 bn: large cap, needs > 5 %
    assert ok.status == "pass", ok
    assert ok.net_margin == pytest.approx(0.2) and ok.cap_bucket == "large" and ok.fcf_growth == pytest.approx(0.15, abs=0.01)
    flat = F.evaluate("X", company(0.0), "2020-12-31", close_unadj=150.0)
    assert flat.status == "fail" and flat.pass_fcf_increasing is False
    # one day before the Q3 filing the latest quarter is Q2: nothing filed later leaks in
    before = F.evaluate("X", company(0.15), "2020-11-03", close_unadj=150.0)
    assert before.latest_quarter == "2020-06-30"


def test_store_round_trip(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    px = _px([10.0, 11.0, 12.0])
    store.put_prices("ZZZ", px, path=db)
    store.put_iv("ZZZ", pd.Series([0.3, 0.31], index=px.index[:2]), path=db)
    assert store.load_prices("ZZZ", path=db)["close"].tolist() == [10.0, 11.0, 12.0]
    assert store.load_iv("ZZZ", path=db).tolist() == [0.3, 0.31]
