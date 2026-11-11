"""
tests/test_stockscreen_sec.py — SEC XBRL → quarterly/annual series, on synthetic
company-facts JSON (no network).
"""

from __future__ import annotations

import pytest

from stockscreen import sec_data as sd


def entry(start, end, val, filed, form="10-Q", accn="a"):
    return {"start": start, "end": end, "val": val, "filed": filed, "form": form, "accn": accn}


def facts_with(tag, entries, unit="USD", taxonomy="us-gaap"):
    return {"facts": {taxonomy: {tag: {"units": {unit: entries}}}}}


# A December fiscal year, cash-flow style: only year-to-date values are filed.
OCF_2024 = [
    entry("2024-01-01", "2024-03-31", 100, "2024-05-01"),                    # YTD3  -> Q1 = 100
    entry("2024-01-01", "2024-06-30", 250, "2024-08-01"),                    # YTD6  -> Q2 = 150
    entry("2024-01-01", "2024-09-30", 420, "2024-11-01"),                    # YTD9  -> Q3 = 170
    entry("2024-01-01", "2024-12-31", 600, "2025-02-15", form="10-K"),       # FY    -> Q4 = 180
]


def test_quarters_are_derived_from_year_to_date():
    f = facts_with("NetCashProvidedByUsedInOperatingActivities", OCF_2024)
    q, y = sd.build_series(sd._collect(f, sd.TAGS["ocf"]))
    assert [(o.end, o.value) for o in q] == [("2024-03-31", 100), ("2024-06-30", 150), ("2024-09-30", 170), ("2024-12-31", 180)]
    assert [o.basis for o in q] == ["reported 3M", "derived YTD", "derived YTD", "derived Q4 = FY - 9M"]
    assert [(o.end, o.value) for o in y] == [("2024-12-31", 600)]
    assert sum(o.value for o in q) == y[0].value            # the quarters add back to the fiscal year


def test_reported_three_month_value_beats_derived():
    income = [
        entry("2024-01-01", "2024-03-31", 40, "2024-05-01"),
        entry("2024-04-01", "2024-06-30", 55, "2024-08-01"),                 # discrete Q2 reported directly
        entry("2024-01-01", "2024-06-30", 95, "2024-08-01"),
    ]
    q, _ = sd.build_series(sd._collect(facts_with("NetIncomeLoss", income), sd.TAGS["net_income"]))
    q2 = next(o for o in q if o.end == "2024-06-30")
    assert q2.value == 55 and q2.basis == "reported 3M"


def test_latest_filing_wins_and_first_publication_is_kept():
    restated = [entry("2024-01-01", "2024-03-31", 100, "2024-05-01"),
                entry("2024-01-01", "2024-03-31", 90, "2025-05-01")]          # restated a year later
    p = sd._collect(facts_with("NetIncomeLoss", restated), sd.TAGS["net_income"])[("2024-01-01", "2024-03-31")]
    assert p.value == 90 and p.filed == "2025-05-01" and p.first_filed == "2024-05-01"


def test_capex_tag_priority_and_non_cash_tag_is_ignored():
    f = {"facts": {"us-gaap": {
        "CapitalExpendituresIncurredButNotYetPaid": {"units": {"USD": [entry("2024-01-01", "2024-03-31", 999, "2024-05-01")]}},
        "PaymentsToAcquireProductiveAssets": {"units": {"USD": [entry("2024-01-01", "2024-03-31", 30, "2024-05-01")]}},
        "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": [entry("2024-01-01", "2024-03-31", 25, "2024-05-01")]}},
    }}}
    p = sd._collect(f, sd.TAGS["capex"])[("2024-01-01", "2024-03-31")]
    assert p.value == 25 and p.tag == "PaymentsToAcquirePropertyPlantAndEquipment"


def test_missing_quarter_is_absent_not_zero():
    f = facts_with("NetCashProvidedByUsedInOperatingActivities", [OCF_2024[0], OCF_2024[2]])  # YTD6 missing
    q, _ = sd.build_series(sd._collect(f, sd.TAGS["ocf"]))
    assert [o.end for o in q] == ["2024-03-31"]               # Q2 and Q3 cannot be derived, so they do not appear


def test_non_standard_forms_are_ignored():
    f = facts_with("NetIncomeLoss", [entry("2024-01-01", "2024-03-31", 10, "2024-05-01", form="8-K")])
    assert sd._collect(f, sd.TAGS["net_income"]) == {}


def _obs(ends_values):
    return [sd.Obs(e, v, "x", "", "t") for e, v in ends_values]


def test_ttm_points_need_four_consecutive_quarters():
    obs = _obs([("2024-03-31", 1), ("2024-06-30", 2), ("2024-09-30", 3), ("2024-12-31", 4), ("2025-03-31", 5)])
    assert sd.ttm_points(obs, 5) == [("2024-12-31", 10), ("2025-03-31", 14)]
    gap = _obs([("2024-03-31", 1), ("2024-09-30", 3), ("2024-12-31", 4), ("2025-03-31", 5), ("2025-06-30", 6)])
    # 2024-06-30 is missing: only the window made of four genuinely consecutive quarters is valid, nothing is imputed
    assert sd.ttm_points(gap, 5) == [("2025-06-30", 18)]
    assert sd.contiguous(gap, 5) is None and sd.contiguous(gap, 4) is not None


def test_shares_ladder_prefers_cover_page_and_demotes_inconsistent_count():
    dei = {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
        {"end": "2026-07-15", "val": 100_000_000, "form": "10-Q", "filed": "2026-08-01", "accn": "x"}]}}}
    wavg = {"WeightedAverageNumberOfSharesOutstandingBasic": {"units": {"shares": [
        {"start": "2026-04-01", "end": "2026-06-30", "val": 102_000_000, "form": "10-Q", "filed": "2026-08-01", "accn": "x"}]}}}
    ok = sd.share_candidates({"facts": {"dei": dei, "us-gaap": wavg}})
    assert ok[0]["source"].startswith("SEC cover page") and ok[0]["value"] == 100_000_000
    # one class only on the cover page (e.g. dual-class company): 30M vs 102M weighted average → rejected
    dei["EntityCommonStockSharesOutstanding"]["units"]["shares"][0]["val"] = 30_000_000
    bad = sd.share_candidates({"facts": {"dei": dei, "us-gaap": wavg}})
    assert bad[0]["approx"] is True and bad[-1].get("rejected") is True


def test_dual_class_cover_values_are_summed():
    dei = {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
        {"end": "2026-07-15", "val": 60_000_000, "form": "10-Q", "filed": "2026-08-01", "accn": "x"},
        {"end": "2026-07-15", "val": 40_000_000, "form": "10-Q", "filed": "2026-08-01", "accn": "x"}]}}}
    c = sd.share_candidates({"facts": {"dei": dei}})[0]
    assert c["value"] == 100_000_000 and c["classes"] == 2


def test_twelve_twelve_twelve_sixteen_week_calendar():
    """Domino's-style fiscal year: YTD periods are 84, 168, 252 and 364 days and Q4 is 16 weeks."""
    e = [entry("2024-12-30", "2025-03-23", 100, "2025-05-01"),      # 84 days
         entry("2024-12-30", "2025-06-15", 220, "2025-08-01"),      # 168
         entry("2024-12-30", "2025-09-07", 330, "2025-11-01"),      # 252
         entry("2024-12-30", "2025-12-28", 480, "2026-02-20", form="10-K")]   # 364
    q, y = sd.build_series(sd._collect(facts_with("NetCashProvidedByUsedInOperatingActivities", e), sd.TAGS["ocf"]))
    assert [o.value for o in q] == [100, 120, 110, 150]
    assert sum(o.value for o in q) == y[0].value
    assert sd.contiguous(q, 4) is not None                          # the 112-day Q4 gap is accepted


# --- capex families -----------------------------------------------------------------------------------------------

def _cash_facts(tags_values):
    """Quarterly (3M, reported) values for several tags over four quarters of 2025 plus the same for the quarters before."""
    ends = [("2024-10-01", "2024-12-31"), ("2025-01-01", "2025-03-31"), ("2025-04-01", "2025-06-30"), ("2025-07-01", "2025-09-30")]
    facts = {}
    for tag, vals in tags_values.items():
        facts[tag] = {"units": {"USD": [entry(s, e, v, e.replace("-", "")[:8] and e) for (s, e), v in zip(ends, vals)]}}
    return {"facts": {"us-gaap": facts}}


def test_oil_and_gas_capex_adds_to_other_ppe():
    f = _cash_facts({"PaymentsToAcquireOtherPropertyPlantAndEquipment": [5, 5, 5, 5],
                     "PaymentsToExploreAndDevelopOilAndGasProperties": [100, 110, 120, 130]})
    q, _ = sd.combine_capex(f)
    assert [o.value for o in q] == [105, 115, 125, 135]                      # separate cash-flow lines add
    assert "PaymentsToExploreAndDevelopOilAndGasProperties" in q[-1].tag and "+" in q[-1].tag


def test_first_tag_wins_within_a_family():
    f = _cash_facts({"PaymentsToAcquirePropertyPlantAndEquipment": [10, 10, 10, 10], "PaymentsToAcquireProductiveAssets": [99, 99, 99, 99]})
    q, _ = sd.combine_capex(f)
    assert [o.value for o in q] == [10, 10, 10, 10]                          # alternatives for the same line, not summed


def test_software_capex_is_added_only_on_top_of_a_primary_family():
    both = _cash_facts({"PaymentsToAcquirePropertyPlantAndEquipment": [10, 10, 10, 10], "PaymentsToDevelopSoftware": [3, 3, 3, 3]})
    assert [o.value for o in sd.combine_capex(both)[0]] == [13, 13, 13, 13]
    only_software = _cash_facts({"PaymentsToDevelopSoftware": [3, 3, 3, 3]})
    assert sd.combine_capex(only_software) == ([], [])                       # would omit PP&E purchases: unavailable, not "small"


def test_acquisitions_and_non_cash_accruals_are_never_capex():
    f = _cash_facts({"PaymentsToAcquireRealEstate": [500] * 4, "PaymentsToAcquireOilAndGasProperty": [900] * 4,
                     "CapitalExpendituresIncurredButNotYetPaid": [70] * 4})
    assert sd.combine_capex(f) == ([], [])


def test_a_family_that_stopped_reporting_does_not_drag_the_sum_or_hide_quarters():
    ends = [("2022-10-01", "2022-12-31"), ("2025-04-01", "2025-06-30"), ("2025-07-01", "2025-09-30")]
    f = {"facts": {"us-gaap": {
        "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": [entry(s, e, v, e) for (s, e), v in zip(ends, [10, 10, 10])]}},
        "PaymentsToExploreAndDevelopOilAndGasProperties": {"units": {"USD": [entry(*ends[0], 50, "2023-02-01")]}}}}}   # inactive since 2022
    q, _ = sd.combine_capex(f)
    assert [(o.end, o.value) for o in q][-2:] == [("2025-06-30", 10), ("2025-09-30", 10)]
