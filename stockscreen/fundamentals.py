"""
stockscreen/fundamentals.py — the fundamental screen: FCF consistency, net
margin, revenue growth, FCF yield, size, and the stable-or-improving trend
test, evaluated from the raw quarterly/annual observations in stockscreen.sec_data.

Everything threshold-dependent happens here at *evaluation* time from stored raw
observations, so changing a threshold re-classifies instantly without a rescan.

Per-criterion status is one of
    pass  — computed and meets the rule
    fail  — computed and does not meet the rule (including exactly-on-the-line
            for the strict ">" rules)
    na    — could not be computed (missing/non-contiguous data). NEVER a pass.

Overall status (first match wins):
    ERROR              data could not be fetched or parsed
    FAILED             any criterion failed
    STALE_DATA         nothing failed but the latest reported quarter is too old
    INSUFFICIENT_DATA  nothing failed but some criterion could not be computed
    PASSED
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Optional

STATUSES = ["PASSED", "FAILED", "INSUFFICIENT_DATA", "STALE_DATA", "ERROR"]
TREND_LABELS = ["Improving", "Stable", "Declining", "Insufficient data"]
_EPS = 1e-9
_TOL = 1e-9          # comparison tolerance so 110/100 - 1 (= 0.10000000000000009) is not "above 10%"


def gt(x: float, thr: float) -> bool:
    """Strictly greater than, immune to float noise at the boundary."""
    return x > thr + _TOL


def ge(x: float, thr: float) -> bool:
    """Greater than or equal, immune to float noise at the boundary."""
    return x >= thr - _TOL


def _d(s: str) -> date:
    return datetime.strptime(s[:10], "%Y-%m-%d").date()


# ---------------------------------------------------------------------------
# period helpers (operate on lists of {"end","value",...} dicts, ascending)
# ---------------------------------------------------------------------------

def _gap_ok(a: str, b: str) -> bool:
    from .sec_data import GAP_DAYS
    return GAP_DAYS[0] <= (_d(b) - _d(a)).days <= GAP_DAYS[1]


def _index(obs: list[dict], end: str) -> Optional[int]:
    for i, o in enumerate(obs):
        if o["end"] == end:
            return i
    return None


def ttm_at(obs: list[dict], end: str) -> Optional[float]:
    """Sum of the four consecutive quarters ending at *end*, else None."""
    i = _index(obs, end)
    if i is None or i < 3:
        return None
    win = obs[i - 3:i + 1]
    if not all(_gap_ok(a["end"], b["end"]) for a, b in zip(win, win[1:])):
        return None
    return sum(o["value"] for o in win)


def prior_end(obs: list[dict], end: str, quarters_back: int = 4) -> Optional[str]:
    i = _index(obs, end)
    if i is None or i - quarters_back < 0:
        return None
    seg = obs[i - quarters_back:i + 1]
    if not all(_gap_ok(a["end"], b["end"]) for a, b in zip(seg, seg[1:])):
        return None
    return obs[i - quarters_back]["end"]


# ---------------------------------------------------------------------------
# trend classification
# ---------------------------------------------------------------------------

def classify_trend(values: list[float], kind: str, tol: float, cfg) -> dict:
    """Improving / Stable / Declining over a chronological series.

    kind "level": relative change of the last value vs the first (revenue, FCF).
    kind "pp":    absolute change (margins, growth rates — already ratios).

    Rules (n points, n-1 sequential moves; a move counts only if it exceeds a
    dead-band of tol/4 so noise doesn't count as direction):
      Declining  change < -tol x severe_mult                       (severe, any path)
                 or change < -tol and at least ceil(0.75 (n-1)) moves down
      Improving  change > +tol and at least ceil(0.75 (n-1)) moves up
      Stable     otherwise — including a one-quarter blip or a change inside tolerance.
    """
    vals = [v for v in values if v is not None and not math.isnan(v)]
    n = len(vals)
    if n < cfg.trend_min_points:
        return {"label": "Insufficient data", "n": n, "change": None, "ups": None, "downs": None}
    first, last = vals[0], vals[-1]
    if kind == "level":
        change = (last - first) / max(abs(first), _EPS)
        band = lambda prev: tol / 4 * max(abs(prev), _EPS)       # noqa: E731
    else:
        change = last - first
        band = lambda prev: tol / 4                              # noqa: E731
    diffs = [b - a for a, b in zip(vals, vals[1:])]
    ups = sum(1 for d, a in zip(diffs, vals) if d > band(a))
    downs = sum(1 for d, a in zip(diffs, vals) if d < -band(a))
    need = math.ceil(0.75 * (n - 1))
    if change < -tol * cfg.trend_severe_mult or (change < -tol and downs >= need):
        label = "Declining"
    elif change > tol and ups >= need:
        label = "Improving"
    else:
        label = "Stable"
    return {"label": label, "n": n, "change": change, "ups": ups, "downs": downs}


# ---------------------------------------------------------------------------
# main evaluation
# ---------------------------------------------------------------------------

def _crit(key: str, label: str, status: str, value, threshold, detail: str) -> dict:
    return {"key": key, "label": label, "status": status, "value": value, "threshold": threshold, "detail": detail}


FINANCIAL_GROUPS = {
    "bank": ("60",),                                    # depository institutions
    "credit": ("61",),                                  # non-depository credit: lenders, card issuers
    "broker/asset manager": ("62",),                    # brokers, dealers, exchanges, investment advice
    "investment holding": ("67",),                      # holding and other investment offices (REITs, 6798, excluded below)
    "insurance": ("63", "64"),
}


def financial_group(sic: Optional[str]) -> Optional[str]:
    """Industry group for SIC codes where operating cash flow can contain investment, lending or trading flows."""
    sic = (sic or "").strip()
    if len(sic) < 2 or sic == "6798":                   # REITs have ordinary operating cash flow
        return None
    for name, prefixes in FINANCIAL_GROUPS.items():
        if sic.startswith(prefixes):
            return name
    return None


def fcf_distortion(fund: dict, capex_year: Optional[dict]) -> Optional[dict]:
    """Detect a financial company whose investment purchases and sales sit inside operating cash flow.

    If the reported investing total equals capex (nothing else flows through investing) the company has nowhere
    else to put its investment trading, so those flows are in operating activities and OCF minus capex is not free
    cash flow. Blackstone FY2025: investing -115.7M equals PP&E purchases -115.7M, while its operating section holds
    -3,807M of investments purchased and +6,679M of sale proceeds (net +2.9B). Data vendors move those lines to
    investing, turning operating cash flow from 4.7B into 1.9B."""
    grp = financial_group(fund.get("sic"))
    if grp is None or not capex_year or capex_year.get("capex") in (None, 0):
        return None
    inv = [o for o in (fund.get("years") or {}).get("investing", []) if o["end"] == capex_year["end"]]
    if not inv:
        return None
    investing, capex = inv[-1]["value"], capex_year["capex"]
    residual = abs(investing + capex)                       # what investing holds beyond capex
    if residual <= max(0.10 * capex, 1_000_000):
        return {"group": grp, "investing": investing, "capex": capex, "fy_end": capex_year["end"],
                "ocf": capex_year["ocf"], "residual": residual}
    return None


def pick_shares(fund: dict, today: date, max_age_days: int) -> Optional[dict]:
    """First fresh, non-rejected share-count candidate (SEC first, then an
    external fallback fetched by the scanner). A stale share count is never used."""
    cands = list(fund.get("shares") or [])
    if fund.get("shares_external"):
        cands.append(fund["shares_external"])
    for c in cands:
        if c.get("rejected"):
            continue
        try:
            age = (today - _d(c["end"])).days
        except Exception:
            continue
        if age <= max_age_days and c.get("value", 0) > 0:
            return dict(c, age_days=age)
    return None


def evaluate_fundamentals(fund: dict, price: Optional[dict], cfg, today: Optional[date] = None) -> dict:
    """Evaluate one company. *fund* is the stored raw payload (quarters, years,
    shares, ...); *price* is {"last","asof","qe_close":{end: close}} or None."""
    today = today or date.today()
    out: dict = {"status": "ERROR", "criteria": [], "reasons": [], "flags": [], "trends": {}, "metrics": {}, "stale": False}
    if not fund or fund.get("fetch") not in ("ok", None):
        out["reasons"] = [fund.get("error") or "fundamentals unavailable"] if fund else ["no fundamentals data"]
        out["status"] = "ERROR" if fund and fund.get("fetch") == "error" else "INSUFFICIENT_DATA"
        return out

    Q, Y = fund["quarters"], fund["years"]
    crit: list[dict] = []
    flags: list[str] = out["flags"]
    m: dict = out["metrics"]

    # ---- align capex with operating cash flow --------------------------------
    capex_by_end = {o["end"]: o["value"] for o in Q.get("capex", [])}
    ocf_q = Q.get("ocf", [])
    capex_missing_ok = cfg.allow_missing_capex_as_zero

    def fcf_value(end: str, ocf: float, capex_map: dict) -> tuple[Optional[float], Optional[float]]:
        if end in capex_map:
            return ocf - capex_map[end], capex_map[end]
        if capex_missing_ok:
            flags.append(f"capex missing for {end}: assumed 0 (allow_missing_capex_as_zero is on)")
            return ocf, 0.0
        return None, None

    fcf_q = []
    for o in ocf_q:
        v, cx = fcf_value(o["end"], o["value"], capex_by_end)
        fcf_q.append({"end": o["end"], "ocf": o["value"], "capex": cx, "fcf": v, "basis": o["basis"],
                      "filed": o.get("first_filed") or o.get("filed")})
    capex_y = {o["end"]: o["value"] for o in Y.get("capex", [])}
    fcf_y = []
    for o in Y.get("ocf", []):
        v, cx = fcf_value(o["end"], o["value"], capex_y)
        fcf_y.append({"start": o.get("start"), "end": o["end"], "ocf": o["value"], "capex": cx, "fcf": v,
                      "filed": o.get("first_filed") or o.get("filed")})
    m["fcf_quarters"], m["fcf_years"] = fcf_q[-8:], fcf_y[-3:]
    m["sic"], m["sic_desc"] = fund.get("sic") or "", fund.get("sic_desc") or ""
    m["financial_group"] = financial_group(fund.get("sic"))
    distortion = None
    if cfg.financial_fcf_policy != "off" and fcf_y and fcf_y[-1]["capex"] is not None:
        distortion = fcf_distortion(fund, fcf_y[-1])
    if m["financial_group"]:
        flags.append(f"financial company ({m['financial_group']}, SIC {m['sic']}): operating cash flow can include investment, "
                     "lending or trading flows, so FCF may not be comparable with other industries")
    if distortion:
        note = (
            f"FCF is not meaningful here: reported investing activities ({distortion['investing'] / 1e6:,.0f}M) equal capex "
            f"({-distortion['capex'] / 1e6:,.0f}M) for the fiscal year ended {distortion['fy_end']}, so this {distortion['group']}'s "
            f"investment purchases and sales must sit inside operating cash flow ({distortion['ocf'] / 1e6:,.0f}M). Data "
            "vendors reclassify them to investing, which can turn FCF negative.")
        m["fcf_distortion"] = distortion
        if cfg.financial_fcf_policy == "na":
            m["fcf_not_meaningful"], m["fcf_note"] = True, note
        else:
            flags.append(note)

    # ---- freshness -----------------------------------------------------------
    latest_q_end = ocf_q[-1]["end"] if ocf_q else None
    age = (today - _d(latest_q_end)).days if latest_q_end else None
    m["latest_quarter_end"], m["data_age_days"] = latest_q_end, age
    m["latest_filed"] = fund.get("latest_filed")
    out["stale"] = bool(age is not None and age > cfg.max_data_age_days)

    # ---- A. FCF consistency ---------------------------------------------------
    nq, ny = cfg.require_fcf_quarters, cfg.require_fcf_years
    qs, ys = fcf_q[-nq:], fcf_y[-ny:]
    q_ok = len(qs) == nq and all(_gap_ok(a["end"], b["end"]) for a, b in zip(qs, qs[1:]))
    y_ok = len(ys) == ny
    q_vals = [x["fcf"] for x in qs]
    y_vals = [x["fcf"] for x in ys]
    if not q_ok or any(v is None for v in q_vals):
        why = "fewer than %d consecutive reported quarters" % nq if not q_ok else "capex not reported for a quarter"
        crit.append(_crit("fcf_consistency", f"FCF positive in each of the last {nq} quarters and {ny} fiscal years",
                          "na", None, f"> 0 x {nq}Q + {ny}FY", why))
    elif not y_ok or any(v is None for v in y_vals):
        why = "fewer than %d reported fiscal years" % ny if not y_ok else "capex not reported for a fiscal year"
        crit.append(_crit("fcf_consistency", f"FCF positive in each of the last {nq} quarters and {ny} fiscal years",
                          "na", None, f"> 0 x {nq}Q + {ny}FY", why))
    else:
        bad = [f"Q {x['end']}" for x in qs if x["fcf"] <= 0] + [f"FY {x['end']}" for x in ys if x["fcf"] <= 0]
        crit.append(_crit("fcf_consistency", f"FCF positive in each of the last {nq} quarters and {ny} fiscal years",
                          "fail" if bad else "pass", min(q_vals + y_vals), f"> 0 x {nq}Q + {ny}FY",
                          ("non-positive: " + ", ".join(bad)) if bad else "all observations positive"))

    # ---- common TTM ends -------------------------------------------------------
    rev_q, ni_q = Q.get("revenue", []), Q.get("net_income", [])
    capex_q_obs = Q.get("capex", [])
    # Revenue / net income do not depend on capex: a bank with no capex line still gets its margin and growth.
    fcf_ends = [o["end"] for o in ocf_q
                if ttm_at(ocf_q, o["end"]) is not None and (capex_missing_ok or ttm_at(capex_q_obs, o["end"]) is not None)]
    rn_ends = [o["end"] for o in rev_q if ttm_at(rev_q, o["end"]) is not None and ttm_at(ni_q, o["end"]) is not None]

    def ttm_fcf(e: str) -> Optional[float]:
        a = ttm_at(ocf_q, e)
        c = ttm_at(capex_q_obs, e)
        if a is None:
            return None
        if c is None:
            return a if capex_missing_ok else None
        return a - c

    latest_f = fcf_ends[-1] if fcf_ends else None            # FCF inputs
    latest = rn_ends[-1] if rn_ends else None                # revenue / net income
    m["ttm_period_end"], m["ttm_fcf_period_end"] = latest, latest_f
    ttm_f = ttm_fcf(latest_f) if latest_f else None
    ttm_rev = ttm_at(rev_q, latest) if latest else None
    ttm_ni = ttm_at(ni_q, latest) if latest else None
    m.update(ttm_fcf=ttm_f, ttm_revenue=ttm_rev, ttm_net_income=ttm_ni)
    if latest_f and latest_f != latest_q_end:
        flags.append(f"TTM FCF ends {latest_f}, but operating cash flow is reported through {latest_q_end} (capex lags)")
    if latest and latest_q_end and latest < latest_q_end:
        flags.append(f"TTM revenue / net income end {latest}, but cash flow is reported through {latest_q_end} (income statement lags)")

    # ---- B. net profit margin --------------------------------------------------
    if ttm_rev is None or ttm_ni is None or ttm_rev <= 0:
        crit.append(_crit("net_margin", "TTM net profit margin", "na", None, f"> {cfg.min_net_margin:.0%}",
                          "TTM revenue or net income unavailable"))
        margin = None
    else:
        margin = ttm_ni / ttm_rev
        crit.append(_crit("net_margin", "TTM net profit margin", "pass" if gt(margin, cfg.min_net_margin) else "fail",
                          margin, f"> {cfg.min_net_margin:.0%}",
                          f"TTM net income {ttm_ni:,.0f} / TTM revenue {ttm_rev:,.0f}"))
    m["net_margin"] = margin

    # ---- C. revenue growth -------------------------------------------------------
    g_ttm = None
    if latest:
        pe = prior_end(rev_q, latest, 4)
        prior_rev = ttm_at(rev_q, pe) if pe else None
        if prior_rev and prior_rev > 0 and ttm_rev is not None:
            g_ttm = ttm_rev / prior_rev - 1
            m["prior_ttm_revenue"] = prior_rev
    ry = Y.get("revenue", [])
    g_fy = None
    if len(ry) >= 2 and ry[-2]["value"] > 0 and 340 <= (_d(ry[-1]["end"]) - _d(ry[-2]["end"])).days <= 380:
        g_fy = ry[-1]["value"] / ry[-2]["value"] - 1
        m["fy_revenue"], m["fy_revenue_prior"] = ry[-1]["value"], ry[-2]["value"]
        m["fy_growth_label"] = f"FY ended {ry[-1]['end']} vs FY ended {ry[-2]['end']}"
    basis = cfg.growth_basis
    if basis == "ttm" or (basis == "auto" and g_ttm is not None):
        growth, g_label = g_ttm, (f"TTM revenue to {latest} vs prior TTM" if latest else "TTM")
    else:
        growth, g_label = g_fy, "fiscal-year revenue growth (TTM unavailable)" if basis == "auto" else "fiscal-year revenue growth"
        if basis == "auto" and g_fy is not None:
            flags.append("revenue growth uses the fiscal-year basis because TTM growth could not be computed")
    m.update(revenue_growth=growth, revenue_growth_ttm=g_ttm, revenue_growth_fy=g_fy, revenue_growth_basis=g_label)
    crit.append(_crit("revenue_growth", "Revenue growth", "na" if growth is None else ("pass" if gt(growth, cfg.min_revenue_growth) else "fail"),
                      growth, f"> {cfg.min_revenue_growth:.0%}", g_label if growth is not None else "growth cannot be computed"))

    # ---- market cap and FCF yield ----------------------------------------------
    shares = pick_shares(fund, today, cfg.max_data_age_days)
    last_price = price.get("last") if price else None
    mcap = shares["value"] * last_price if shares and last_price else None
    m.update(shares=shares, price=last_price, price_asof=price.get("asof") if price else None, market_cap=mcap)
    if shares and shares.get("approx"):
        flags.append(f"share count is an approximation ({shares['source']})")
    if shares and shares.get("classes", 1) > 1:
        flags.append("multiple share classes: market cap sums the listed classes at this ticker's price")
    if not shares:
        flags.append("no current share count available")
    crit.append(_crit("market_cap", "Market capitalization", "na" if mcap is None else ("pass" if ge(mcap, cfg.min_market_cap) else "fail"),
                      mcap, f">= ${cfg.min_market_cap:,.0f}",
                      "no fresh share count or price" if mcap is None else
                      f"{shares['value']:,.0f} shares x ${last_price:,.2f} ({price.get('asof')}; shares: {shares['source']}, as of {shares['end']})"))
    fyield = ttm_f / mcap if (ttm_f is not None and mcap) else None
    m["fcf_yield"] = fyield
    crit.append(_crit("fcf_yield", "TTM FCF yield", "na" if fyield is None else ("pass" if ge(fyield, cfg.min_fcf_yield) else "fail"),
                      fyield, f">= {cfg.min_fcf_yield:.0%}",
                      "TTM FCF or market cap unavailable" if fyield is None else
                      f"TTM FCF {ttm_f:,.0f} / market cap {mcap:,.0f}"))

    # ---- F. trends ---------------------------------------------------------------
    N = 5
    f_ends, r_ends = fcf_ends[-N:], rn_ends[-N:]
    fcf_series = [(e, ttm_fcf(e)) for e in f_ends]
    rev_series = [(e, ttm_at(rev_q, e)) for e in r_ends]
    margin_series = []
    for e in r_ends:
        r, n_ = ttm_at(rev_q, e), ttm_at(ni_q, e)
        margin_series.append((e, n_ / r if r and r > 0 and n_ is not None else None))
    growth_series = []
    for e in r_ends:
        pe = prior_end(rev_q, e, 4)
        r, pr_ = ttm_at(rev_q, e), (ttm_at(rev_q, pe) if pe else None)
        growth_series.append((e, r / pr_ - 1 if r and pr_ and pr_ > 0 else None))
    yield_series = []
    qe = (price or {}).get("qe_close") or {}
    sh_val = shares["value"] if shares else None
    for e in f_ends:
        f_, px = ttm_fcf(e), qe.get(e)
        yield_series.append((e, f_ / (sh_val * px) if f_ is not None and sh_val and px else None))
    m["series"] = {"fcf_ttm": fcf_series, "revenue_ttm": rev_series, "net_margin_ttm": margin_series,
                   "revenue_growth_ttm": growth_series, "fcf_yield_approx": yield_series}

    vals = lambda s: [v for _e, v in s if v is not None]                                   # noqa: E731
    tr = {
        "fcf": classify_trend(vals(fcf_series), "level", cfg.trend_tol_level, cfg),
        "revenue": classify_trend(vals(rev_series), "level", cfg.trend_tol_level, cfg),
        "net_margin": classify_trend(vals(margin_series), "pp", cfg.trend_tol_margin_pp, cfg),
        "revenue_growth": classify_trend(vals(growth_series), "pp", cfg.trend_tol_growth_pp, cfg),
    }
    # FCF yield: separate a price-driven move from a cash-flow-driven one
    ytr = classify_trend(vals(yield_series), "level", cfg.trend_tol_level, cfg)
    ytr["note"] = ""
    if ytr["label"] == "Declining" and tr["fcf"]["label"] != "Declining":
        ytr["label"], ytr["note"] = "Stable", "yield fell because the share price rose; underlying FCF is not deteriorating"
    elif ytr["label"] == "Improving" and tr["fcf"]["label"] == "Declining":
        ytr["label"], ytr["note"] = "Stable", "yield rose because the share price fell while FCF is deteriorating"
    elif tr["fcf"]["label"] == "Declining":
        ytr["note"] = "FCF itself is deteriorating"
    tr["fcf_yield"] = ytr
    out["trends"] = tr

    critical = [k for k in cfg.critical_trend_metrics if k in tr]
    declining = [k for k in critical if tr[k]["label"] == "Declining"]
    insufficient = [k for k in critical if tr[k]["label"] == "Insufficient data"]
    if declining:
        crit.append(_crit("trend", "No critical fundamental metric declining", "fail", None, "none Declining",
                          "declining: " + ", ".join(declining)))
    elif insufficient:
        crit.append(_crit("trend", "No critical fundamental metric declining", "na", None, "none Declining",
                          "not enough history to classify: " + ", ".join(insufficient)))
    else:
        crit.append(_crit("trend", "No critical fundamental metric declining", "pass", None, "none Declining",
                          "critical metrics stable or improving"))

    # ---- flags: one-offs and abrupt moves ------------------------------------------
    disc = Q.get("disc_ops", [])
    if latest and disc:
        d_ttm = ttm_at(disc, latest)
        if d_ttm is not None and ttm_ni and abs(d_ttm) > 0.10 * abs(ttm_ni):
            flags.append(f"discontinued operations are {abs(d_ttm) / abs(ttm_ni):.0%} of TTM net income (margin may not be comparable)")
    if len(fcf_q) >= 5 and fcf_q[-1]["fcf"] is not None and fcf_q[-5]["fcf"] and fcf_q[-5]["fcf"] > 0:
        chg = fcf_q[-1]["fcf"] / fcf_q[-5]["fcf"] - 1
        if chg < -0.5:
            flags.append(f"latest-quarter FCF is down {abs(chg):.0%} year over year")
    ms = vals(margin_series)
    if len(ms) >= 2 and abs(ms[-1] - ms[-2]) > 0.05:
        flags.append(f"TTM net margin moved {abs(ms[-1] - ms[-2]) * 100:.1f} pp in one quarter (possible one-off)")
    if ttm_ni and ttm_ni > 0 and ni_q:
        top = max((o["value"] for o in ni_q[-4:]), default=0)
        if top > 0.6 * ttm_ni and ttm_ni > 0:
            flags.append("one quarter makes up over 60% of TTM net income (possible one-off)")
    if m.get("ttm_revenue") and m.get("prior_ttm_revenue") and m["ttm_revenue"] < m["prior_ttm_revenue"]:
        flags.append("TTM revenue is below the prior TTM (revenue declining)")

    # ---- financial companies whose FCF is distorted by investment flows ---------------------------------
    if m.get("fcf_not_meaningful"):
        for c in crit:
            if c["key"] in ("fcf_consistency", "fcf_yield"):
                c.update(status="na", value=None, detail=m["fcf_note"])
        na_trend = {"label": "Insufficient data", "n": 0, "change": None, "ups": None, "downs": None, "note": "FCF not meaningful"}
        tr["fcf"], tr["fcf_yield"] = dict(na_trend), dict(na_trend)
        m["fcf_yield_as_filed"], m["fcf_yield"] = m.get("fcf_yield"), None
        keep = [k for k in cfg.critical_trend_metrics if k in tr and k not in ("fcf", "fcf_yield")]
        declining = [k for k in keep if tr[k]["label"] == "Declining"]
        insufficient = [k for k in keep if tr[k]["label"] == "Insufficient data"]
        for c in crit:
            if c["key"] == "trend":
                if declining:
                    c.update(status="fail", detail="declining: " + ", ".join(declining))
                elif insufficient:
                    c.update(status="na", detail="not enough history to classify: " + ", ".join(insufficient))
                else:
                    c.update(status="pass", detail="critical metrics other than FCF are stable or improving")

    # ---- overall status -----------------------------------------------------------------
    out["criteria"] = crit
    fails = [c for c in crit if c["status"] == "fail"]
    nas = [c for c in crit if c["status"] == "na"]
    if fails:
        out["status"] = "FAILED"
        out["reasons"] = [f"{c['label']}: {c['detail']}" for c in fails]
    elif out["stale"]:
        out["status"] = "STALE_DATA"
        out["reasons"] = [f"latest reported quarter ended {latest_q_end} ({age} days ago; limit {cfg.max_data_age_days})"]
    elif nas:
        out["status"] = "INSUFFICIENT_DATA"
        out["reasons"] = [f"{c['label']}: {c['detail']}" for c in nas]
    else:
        out["status"] = "PASSED"
    return out
