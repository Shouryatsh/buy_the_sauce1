"""
stockscreen/sec_data.py — SEC EDGAR XBRL "company facts" → clean quarterly and
annual series for the flow items the screener needs (operating cash flow,
capex, revenue, net income).

Why this is not a one-liner: companies file cash-flow statements as
*year-to-date* figures (a Q3 10-Q reports nine months), and income statements
carry both three-month and YTD columns. So discrete fiscal quarters are
derived by differencing YTD values within a fiscal year:

    Q1 = YTD3        Q2 = YTD6 - YTD3        Q3 = YTD9 - YTD6        Q4 = FY - YTD9

A directly reported 3-month value is preferred over a derived one whenever it
exists. Every observation records its basis ("reported-3M", "derived YTD",
"derived Q4 = FY - 9M"), the SEC filing date, and the XBRL tag used, so the
UI can show exactly where each number came from. Periods are identified by
their (start, end) dates, never by the fy/fp labels (which describe the
filing, not the period).

Nothing is ever imputed: a quarter that cannot be derived is simply absent.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# Priority-ordered XBRL tags per metric. For each period the first tag that
# reports it wins (companies change tags over time).
# Capital expenditure is the sum of distinct *families* of cash-flow lines. Inside a family the first tag a period
# reports wins (companies change tags over time); across families the amounts add, because they are separate lines
# (an oil producer's drilling spend sits beside its "other PP&E" line). Deliberately not capex: asset *acquisitions*
# (PaymentsToAcquireRealEstate, PaymentsToAcquireOilAndGasProperty are M&A) and non-cash accruals
# (CapitalExpendituresIncurredButNotYetPaid).
CAPEX_FAMILIES = [
    {"name": "PP&E", "primary": True, "tags": [
        "PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets", "PaymentsForCapitalImprovements",
        "PaymentsToAcquireOtherPropertyPlantAndEquipment", "PaymentsToAcquireOtherProductiveAssets"]},
    {"name": "oil & gas", "primary": True, "tags": [
        "PaymentsToAcquireOilAndGasPropertyAndEquipment", "PaymentsToExploreAndDevelopOilAndGasProperties"]},
    {"name": "real-estate development", "primary": True, "tags": ["PaymentsToDevelopRealEstateAssets"]},
    # capitalised software is real capex but only ever *added to* a primary family: on its own it would leave out the
    # company's PP&E purchases and overstate FCF, so a software-only company is reported as "capex unavailable".
    {"name": "capitalised software", "primary": False, "tags": [
        "PaymentsToDevelopSoftware", "PaymentsForSoftware", "PaymentsToAcquireSoftware"]},
]

TAGS = {
    "ocf": ["NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": [t for fam in CAPEX_FAMILIES for t in fam["tags"]],     # flat view (kept for reference / tests)
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax",
                "Revenues",
                "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax",
                "SalesRevenueGoodsNet",
                "SalesRevenueServicesNet",
                "RevenuesNetOfInterestExpense",             # banks: net interest income + non-interest income
                "RegulatedAndUnregulatedOperatingRevenue",  # utilities
                "RealEstateRevenueNet"],                    # REITs
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    # reported total of investing activities: used only to detect companies whose investment purchases and
    # sales sit inside operating cash flow (see fundamentals.fcf_distortion)
    "investing": ["NetCashProvidedByUsedInInvestingActivities", "NetCashProvidedByUsedInInvestingActivitiesContinuingOperations"],
    "disc_ops": ["IncomeLossFromDiscontinuedOperationsNetOfTax",
                 "IncomeLossFromDiscontinuedOperationsNetOfTaxAttributableToReportingEntity"],
}

PARSER_VERSION = 5          # bump to invalidate cached parsed payloads when the parsing rules change

FORMS = {"10-K", "10-K/A", "10-KT", "10-Q", "10-Q/A", "10-QT"}

# duration windows in days
# wide enough for 52/53-week and 12-12-12-16-week calendars (e.g. Domino's: YTD of 84, 168, 252 days)
_DUR = {3: (80, 115), 6: (160, 200), 9: (245, 290), 12: (340, 380)}
GAP_DAYS = (80, 120)         # days between consecutive quarter ends (a 16-week Q4 is 112)


def _d(s: str) -> date:
    return datetime.strptime(s[:10], "%Y-%m-%d").date()


def _months(days: int) -> Optional[int]:
    for m, (lo, hi) in _DUR.items():
        if lo <= days <= hi:
            return m
    return None


@dataclass
class Period:
    start: str
    end: str
    value: float
    filed: str
    tag: str
    form: str = ""
    first_filed: str = ""          # earliest filing that reported this period (publication date)

    @property
    def months(self) -> Optional[int]:
        return _months((_d(self.end) - _d(self.start)).days)


def _collect(facts: dict, tags: list[str], unit: str = "USD", taxonomy: str = "us-gaap") -> dict:
    """(start, end) -> Period, most recently filed value per period, first
    tag (in priority order) that reports the period."""
    out: dict[tuple, Period] = {}
    for rank, tag in enumerate(tags):
        try:
            entries = facts["facts"][taxonomy][tag]["units"][unit]
        except KeyError:
            continue
        best: dict[tuple, Period] = {}
        earliest: dict[tuple, str] = {}
        for e in entries:
            if e.get("form") not in FORMS or e.get("val") is None or not e.get("start") or not e.get("end"):
                continue
            key = (e["start"], e["end"])
            filed = e.get("filed", "")
            earliest[key] = min(earliest.get(key, filed), filed)
            p = Period(e["start"], e["end"], float(e["val"]), filed, tag, e.get("form", ""))
            if key not in best or p.filed > best[key].filed:
                best[key] = p
        for key, p in best.items():
            p.first_filed = earliest[key]
        for key, p in best.items():
            if key not in out:          # earlier tag in the priority list wins
                out[key] = p
    return out


@dataclass
class Obs:
    end: str
    value: float
    basis: str
    filed: str
    tag: str
    start: str = ""
    first_filed: str = ""          # publication date: when this period was first reported

    def to_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "value": self.value, "basis": self.basis,
                "filed": self.filed, "first_filed": self.first_filed, "tag": self.tag}


def build_series(periods: dict) -> tuple[list[Obs], list[Obs]]:
    """(quarters, years), each sorted by period end ascending."""
    plist = list(periods.values())
    annual = sorted((p for p in plist if p.months == 12), key=lambda p: p.end)
    # one annual observation per fiscal-year end
    seen, years = set(), []
    for p in annual:
        if p.end not in seen:
            seen.add(p.end)
            years.append(Obs(p.end, p.value, "reported FY", p.filed, p.tag, p.start, p.first_filed))

    # candidate fiscal-year starts: annual starts, starts of 6M/9M YTD entries,
    # and the day after the latest annual end (the in-progress year)
    starts = {p.start for p in annual}
    starts |= {p.start for p in plist if p.months in (6, 9)}
    if annual:
        starts.add((_d(annual[-1].end) + timedelta(days=1)).isoformat())

    discrete3 = {p.end: p for p in plist if p.months == 3}     # directly reported 3-month values
    quarters: dict[str, Obs] = {}

    def ytd(s: str, months: int) -> Optional[Period]:
        # YTD entries start on the fiscal-year start; tolerate a 1-3 day drift
        for p in plist:
            if p.months == months and abs((_d(p.start) - _d(s)).days) <= 3:
                return p
        return None

    for s in sorted(starts):
        y3, y6, y9, y12 = ytd(s, 3), ytd(s, 6), ytd(s, 9), ytd(s, 12)

        def put(end: str, value: float, basis: str, filed: str, tag: str, first: str):
            d3 = discrete3.get(end)
            if d3 is not None:                                  # reported 3M beats derived
                quarters[end] = Obs(end, d3.value, "reported 3M", d3.filed, d3.tag, "", d3.first_filed)
            elif end not in quarters or basis.startswith("reported"):
                quarters[end] = Obs(end, value, basis, filed, tag, "", first)

        if y3:
            put(y3.end, y3.value, "reported 3M", y3.filed, y3.tag, y3.first_filed)
        if y3 and y6:
            put(y6.end, y6.value - y3.value, "derived YTD", max(y3.filed, y6.filed), y6.tag,
                max(y3.first_filed, y6.first_filed))
        elif y6 and y6.end in discrete3:
            d = discrete3[y6.end]
            put(y6.end, d.value, "reported 3M", d.filed, y6.tag, d.first_filed)
        if y6 and y9:
            put(y9.end, y9.value - y6.value, "derived YTD", max(y6.filed, y9.filed), y9.tag,
                max(y6.first_filed, y9.first_filed))
        elif y9 and y9.end in discrete3:
            d = discrete3[y9.end]
            put(y9.end, d.value, "reported 3M", d.filed, y9.tag, d.first_filed)
        if y9 and y12:
            put(y12.end, y12.value - y9.value, "derived Q4 = FY - 9M", max(y9.filed, y12.filed), y12.tag,
                max(y9.first_filed, y12.first_filed))

    # directly reported 3M values for quarters no chain produced (e.g. no YTD entries)
    for end, p in discrete3.items():
        quarters.setdefault(end, Obs(end, p.value, "reported 3M", p.filed, p.tag, "", p.first_filed))

    return sorted(quarters.values(), key=lambda o: o.end), years


def contiguous(obs: list[Obs], n: int) -> Optional[list[Obs]]:
    """The latest *n* observations if consecutive quarter ends are ~3 months apart."""
    if len(obs) < n:
        return None
    tail = obs[-n:]
    for a, b in zip(tail, tail[1:]):
        if not GAP_DAYS[0] <= (_d(b.end) - _d(a.end)).days <= GAP_DAYS[1]:
            return None
    return tail


def ttm_points(obs: list[Obs], count: int) -> list[tuple[str, float]]:
    """Trailing-four-quarter sums at each of the last *count* quarter ends
    (oldest first); a point is omitted if its four quarters aren't consecutive."""
    out: list[tuple[str, float]] = []
    for i in range(len(obs) - 1, 2, -1):
        win = obs[i - 3:i + 1]
        if all(GAP_DAYS[0] <= (_d(b.end) - _d(a.end)).days <= GAP_DAYS[1] for a, b in zip(win, win[1:])):
            out.append((obs[i].end, sum(o.value for o in win)))
        else:
            break                       # an older gap invalidates everything before it
        if len(out) == count:
            break
    return list(reversed(out))


# ---------------------------------------------------------------------------
# shares outstanding
# ---------------------------------------------------------------------------

def _dei_shares(facts: dict) -> Optional[dict]:
    """Cover-page shares (dei) of the latest filing. Class-specific values are
    dimensioned facts that SEC's companyfacts API omits, so for many dual-class
    companies this is missing or very old: callers must check freshness."""
    try:
        entries = facts["facts"]["dei"]["EntityCommonStockSharesOutstanding"]["units"]["shares"]
    except KeyError:
        return None
    entries = [e for e in entries if e.get("val") and e.get("form") in FORMS]
    if not entries:
        return None
    latest_filed = max(e["filed"] for e in entries)
    group = [e for e in entries if e["filed"] == latest_filed]
    accn = max(group, key=lambda e: e["end"])["accn"]
    cover = [e for e in group if e["accn"] == accn]
    vals = sorted({float(e["val"]) for e in cover})
    return {"value": float(sum(vals)), "classes": len(vals), "end": max(e["end"] for e in cover),
            "filed": latest_filed, "source": "SEC cover page (dei)", "approx": False}


def _instant_shares(facts: dict) -> Optional[dict]:
    try:
        entries = facts["facts"]["us-gaap"]["CommonStockSharesOutstanding"]["units"]["shares"]
    except KeyError:
        return None
    entries = [e for e in entries if e.get("val") and e.get("form") in FORMS]
    if not entries:
        return None
    e = max(entries, key=lambda x: (x["end"], x["filed"]))
    return {"value": float(e["val"]), "classes": 1, "end": e["end"], "filed": e["filed"],
            "source": "SEC balance sheet (CommonStockSharesOutstanding)", "approx": False}


def _weighted_shares(facts: dict) -> Optional[dict]:
    """Weighted-average basic shares of the latest single quarter: an
    approximation of the current count (flagged as such)."""
    try:
        entries = facts["facts"]["us-gaap"]["WeightedAverageNumberOfSharesOutstandingBasic"]["units"]["shares"]
    except KeyError:
        return None
    q = [e for e in entries if e.get("val") and e.get("form") in FORMS and e.get("start") and e.get("end")
         and _months((_d(e["end"]) - _d(e["start"])).days) == 3]
    if not q:
        return None
    e = max(q, key=lambda x: (x["end"], x["filed"]))
    return {"value": float(e["val"]), "classes": 1, "end": e["end"], "filed": e["filed"],
            "source": "SEC weighted-average basic shares (approx.)", "approx": True}


def share_candidates(facts: dict) -> list[dict]:
    """Share-count candidates from SEC, most authoritative first. If the
    cover-page number disagrees wildly with the weighted average it is treated
    as a partial (single-class) count and demoted."""
    dei, inst, wavg = _dei_shares(facts), _instant_shares(facts), _weighted_shares(facts)
    out = [c for c in (dei, inst, wavg) if c]
    if dei and wavg and not 0.7 <= dei["value"] / wavg["value"] <= 1.4:
        out = [c for c in out if c is not dei] + [
            dict(dei, source=dei["source"] + " (rejected: inconsistent with weighted average)", rejected=True)]
    return out


# ---------------------------------------------------------------------------
# top level
# ---------------------------------------------------------------------------

@dataclass
class ParsedFacts:
    cik: int
    name: str
    quarters: dict = field(default_factory=dict)    # metric -> [Obs]
    years: dict = field(default_factory=dict)       # metric -> [Obs]
    shares: list = field(default_factory=list)      # candidates, best first
    latest_filed: str = ""
    has_statements: bool = False                    # any us-gaap annual or quarterly cash-flow data


def _active(series: list[Obs], newest: str, window_days: int = 200) -> bool:
    return bool(series) and (_d(newest) - _d(series[-1].end)).days <= window_days


def _sum_obs(group: list[Obs], names: list[str]) -> Obs:
    return Obs(end=group[0].end, value=sum(o.value for o in group),
               basis="; ".join(sorted({o.basis for o in group})), filed=max(o.filed for o in group),
               tag="+".join(o.tag for o in group), start=group[0].start,
               first_filed=max(o.first_filed for o in group))


def combine_capex(facts: dict) -> tuple[list[Obs], list[Obs]]:
    """Capex = sum over the families that are currently reporting. A quarter or year is kept only if every active
    family has it (never a partial sum); software capex counts only on top of a primary family."""
    per = []
    for fam in CAPEX_FAMILIES:
        q, y = build_series(_collect(facts, fam["tags"]))
        per.append((fam, q, y))
    newest_q = max((q[-1].end for _f, q, _y in per if q), default=None)
    newest_y = max((y[-1].end for _f, _q, y in per if y), default=None)
    if newest_q is None:
        return [], []
    active = [(f, q, y) for f, q, y in per if _active(q, newest_q)]
    if not any(f["primary"] for f, _q, _y in active):
        return [], []                                   # software-only (or nothing): capex cannot be measured
    def merge(idx: int, newest: Optional[str]) -> list[Obs]:
        fams = [(f, grp[idx]) for f, *grp in active if grp[idx]]
        if not fams:
            return []
        common = set.intersection(*[{o.end for o in series} for _f, series in fams])
        out = []
        for end in sorted(common):
            group = [next(o for o in series if o.end == end) for _f, series in fams]
            out.append(_sum_obs(group, [f["name"] for f, _s in fams]))
        return out
    return merge(0, newest_q), merge(1, newest_y)


def parse_company_facts(facts: dict) -> ParsedFacts:
    out = ParsedFacts(cik=int(facts.get("cik", 0) or 0), name=facts.get("entityName", ""))
    for metric, tags in TAGS.items():
        if metric == "capex":
            out.quarters[metric], out.years[metric] = combine_capex(facts)
            continue
        periods = _collect(facts, tags)
        q, y = build_series(periods)
        out.quarters[metric], out.years[metric] = q, y
    out.shares = share_candidates(facts)
    out.has_statements = bool(out.quarters["ocf"] or out.years["ocf"])
    filed = [o.filed for m in out.quarters.values() for o in m[-1:]] + [o.filed for m in out.years.values() for o in m[-1:]]
    out.latest_filed = max(filed) if filed else ""
    return out


# ---------------------------------------------------------------------------
# network
# ---------------------------------------------------------------------------

def fetch_company_facts(cik: int, retries: int = 3) -> dict:
    """SEC company facts with the shared 8 req/s limiter and retry/backoff on
    429 and 5xx responses."""
    import requests
    import edgar
    url = edgar.EDGAR_FACTS_URL.format(cik=cik)
    last: Optional[Exception] = None
    for attempt in range(retries):
        edgar._sec_rate_limiter.acquire()
        try:
            resp = requests.get(url, headers=edgar.EDGAR_HEADERS, timeout=30)
            if resp.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last = exc
            if getattr(getattr(exc, "response", None), "status_code", None) == 404:
                raise                                   # no such filer: not retryable
            time.sleep(1.5 * (attempt + 1))
    raise last  # type: ignore[misc]


_sic_slots = __import__("threading").Semaphore(2)     # the submissions file is large: never more than 2 at a time


def fetch_sic(cik: int, retries: int = 3) -> tuple[str, str]:
    """(SIC code, SIC description) from SEC's submissions API, with backoff; ("", "") if unavailable."""
    import requests
    import edgar
    for attempt in range(retries):
        with _sic_slots:
            try:
                edgar._sec_rate_limiter.acquire()
                resp = requests.get(edgar.EDGAR_SUBMISSIONS_URL.format(cik=cik), headers=edgar.EDGAR_HEADERS, timeout=45)
                resp.raise_for_status()
                j = resp.json()
                return str(j.get("sic", "") or ""), str(j.get("sicDescription", "") or "")
            except Exception as exc:
                logger.info("SIC lookup for CIK %s failed (attempt %d): %s", cik, attempt + 1, exc)
        time.sleep(2.0 * (2 ** attempt))
    return "", ""


def get_cik(symbol: str) -> Optional[int]:
    import edgar
    return edgar._load_ticker_map().get(symbol.upper().replace(".", "-")) or edgar._load_ticker_map().get(symbol.upper())
