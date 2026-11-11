"""
premia/fundamentals.py — phase 2 hard filters, evaluated strictly point in time.

At each evaluation date t only SEC XBRL facts *filed on or before t* are used (a later restatement of the same period
is invisible until it is filed). The XBRL parsing (YTD differencing into discrete quarters, capex families, no
imputation) is the dashboard's `stockscreen.sec_data`, unchanged.

Definitions (SPEC.md section 4, items 1-2):
  FCF                = operating cash flow - capital expenditure, as filed; missing capex is never treated as zero
  TTM                = sum of the latest 4 consecutive fiscal quarters
  net margin         = TTM net income / TTM revenue                                   > 10 %
  FCF increasing     = TTM FCF now > 4 quarters ago > 8 quarters ago                  (2 years)
  FCF growth         = TTM FCF now / TTM FCF 4 quarters ago - 1 (requires the base > 0)
                       > 10 % for mid caps, > 5 % for large caps
  market cap (PIT)   = latest filed shares outstanding x unadjusted close at t
  cap buckets        mid: 2-10 $bn, large: >= 10 $bn; below 2 $bn is excluded
  staleness          the latest quarter must end within 200 days of t, else the name is "stale" (not a pass)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import pandas as pd

from stockscreen.fundamentals import ttm_at
from stockscreen.sec_data import parse_company_facts

from .config import SCREEN, Screen

MAX_AGE_DAYS = 200


def facts_asof(facts: dict, asof: str) -> dict:
    """The facts as they were known at the end of *asof* (YYYY-MM-DD): every entry filed later is dropped."""
    out = {"cik": facts.get("cik"), "entityName": facts.get("entityName"), "facts": {}}
    for tax, tags in facts.get("facts", {}).items():
        out["facts"][tax] = {}
        for tag, body in tags.items():
            units = {}
            for unit, entries in body.get("units", {}).items():
                kept = [e for e in entries if e.get("filed", "9999") <= asof]
                if kept:
                    units[unit] = kept
            if units:
                out["facts"][tax][tag] = {"units": units}
    return out


def shares_asof(facts: dict, asof: str) -> Optional[tuple[float, str]]:
    """(shares, as-of date of the count): latest cover-page count filed on or before *asof*; balance sheet as fallback."""
    for tax, tag in (("dei", "EntityCommonStockSharesOutstanding"), ("us-gaap", "CommonStockSharesOutstanding")):
        try:
            entries = facts["facts"][tax][tag]["units"]["shares"]
        except KeyError:
            continue
        entries = [e for e in entries if e.get("val") and e.get("filed", "9999") <= asof]
        if not entries:
            continue
        latest = max(e["filed"] for e in entries)
        group = [e for e in entries if e["filed"] == latest]
        if tax == "dei":
            top = max(group, key=lambda e: e["end"])
            return float(sum({float(e["val"]) for e in group if e["accn"] == top["accn"]})), top["end"]
        top = max(group, key=lambda e: e["end"])
        return float(top["val"]), top["end"]
    return None


def split_ratio_between(splits: list[tuple[str, float]], after: str, upto: str) -> float:
    """Product of split ratios with after < date <= upto (a count reported before a split is in pre-split shares)."""
    r = 1.0
    for d, ratio in splits:
        if after < d <= upto:
            r *= ratio
    return r


@dataclass
class PITResult:
    symbol: str
    asof: str
    status: str                      # pass | fail | stale | insufficient
    net_margin: Optional[float] = None
    fcf0: Optional[float] = None     # TTM FCF now, 4 and 8 quarters ago
    fcf4: Optional[float] = None
    fcf8: Optional[float] = None
    fcf_growth: Optional[float] = None
    mcap: Optional[float] = None
    cap_bucket: str = ""
    latest_quarter: str = ""
    fcf_basis: str = "TTM quarters"
    pass_margin: Optional[bool] = None
    pass_fcf_increasing: Optional[bool] = None
    pass_fcf_growth: Optional[bool] = None
    pass_cap: Optional[bool] = None
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _ttm_series(obs: list[dict]) -> dict[str, float]:
    return {o["end"]: v for o in obs if (v := ttm_at(obs, o["end"])) is not None}


def evaluate(symbol: str, facts: dict, asof: str, close_unadj: Optional[float], cfg: Screen = SCREEN,
             splits: Optional[list[tuple[str, float]]] = None) -> PITResult:
    res = PITResult(symbol, asof, "insufficient")
    pf = parse_company_facts(facts_asof(facts, asof))
    Q = {k: [o.to_dict() for o in v] for k, v in pf.quarters.items()}
    if not Q.get("ocf"):
        res.reason = "no cash-flow statements filed by this date"
        return res
    res.latest_quarter = Q["ocf"][-1]["end"]
    if (pd.Timestamp(asof) - pd.Timestamp(res.latest_quarter)).days > MAX_AGE_DAYS:
        res.status, res.reason = "stale", f"latest quarter {res.latest_quarter} older than {MAX_AGE_DAYS} days"
        return res

    rev, ni = _ttm_series(Q.get("revenue", [])), _ttm_series(Q.get("net_income", []))
    common = sorted(set(rev) & set(ni))
    if common and rev[common[-1]] > 0:
        res.net_margin = ni[common[-1]] / rev[common[-1]]
        res.pass_margin = res.net_margin > cfg.min_net_margin

    ocf, capex = _ttm_series(Q.get("ocf", [])), _ttm_series(Q.get("capex", []))
    ends = sorted(set(ocf) & set(capex))
    fcf = {e: ocf[e] - capex[e] for e in ends}
    if len(ends) >= 9:
        e0 = ends[-1]
        # 4 and 8 quarters back = the TTM points ~1 and ~2 years before (tolerant of 52/53-week calendars)
        def back(days: int) -> Optional[str]:
            target = pd.Timestamp(e0) - pd.Timedelta(days=days)
            cand = [e for e in ends if abs((pd.Timestamp(e) - target).days) <= 20]
            return cand[-1] if cand else None
        e4, e8 = back(364), back(728)
        if e4 and e8:
            res.fcf0, res.fcf4, res.fcf8 = fcf[e0], fcf[e4], fcf[e8]
            res.pass_fcf_increasing = res.fcf0 > res.fcf4 > res.fcf8
            if res.fcf4 > 0:
                res.fcf_growth = res.fcf0 / res.fcf4 - 1
    if res.fcf0 is None:
        # quarterly capex cannot always be derived (tag changes between YTD and FY filings): fall back to the last
        # three fiscal years, which is less timely but uses the same definition
        Y = {k: [o.to_dict() for o in v] for k, v in pf.years.items()}
        yo = {o["end"]: o["value"] for o in Y.get("ocf", [])}
        yc = {o["end"]: o["value"] for o in Y.get("capex", [])}
        yends = sorted(set(yo) & set(yc))
        if len(yends) >= 3 and (pd.Timestamp(asof) - pd.Timestamp(yends[-1])).days <= 365 + MAX_AGE_DAYS - 90:
            f0, f1, f2 = (yo[e] - yc[e] for e in yends[-1:-4:-1])
            res.fcf0, res.fcf4, res.fcf8 = f0, f1, f2
            res.pass_fcf_increasing = f0 > f1 > f2
            res.fcf_growth = f0 / f1 - 1 if f1 > 0 else None
            res.fcf_basis = "fiscal years"
    sh = shares_asof(facts, asof)
    if sh and close_unadj:
        res.mcap = sh[0] * split_ratio_between(splits or [], sh[1], asof) * close_unadj
        res.cap_bucket = "large" if res.mcap >= cfg.largecap_min else "mid" if res.mcap >= cfg.midcap_min else "small"
        res.pass_cap = res.cap_bucket in ("mid", "large")
    if res.fcf4 is not None and res.fcf4 <= 0:
        res.pass_fcf_growth = False                       # growth from a non-positive base is not "FCF > x %"
    elif res.fcf_growth is not None and res.cap_bucket in ("mid", "large"):
        thr = cfg.fcf_growth_largecap if res.cap_bucket == "large" else cfg.fcf_growth_midcap
        res.pass_fcf_growth = res.fcf_growth > thr

    checks = [res.pass_margin, res.pass_fcf_increasing, res.pass_fcf_growth, res.pass_cap]
    if any(c is False for c in checks):
        res.status = "fail"
        res.reason = ", ".join(n for n, c in zip(["margin", "fcf_increasing", "fcf_growth", "cap"], checks) if c is False)
    elif all(c is True for c in checks):
        res.status = "pass"
    else:
        res.status = "insufficient"
        res.reason = "missing: " + ", ".join(n for n, c in zip(["margin", "fcf_increasing", "fcf_growth", "cap"], checks) if c is None)
    return res
