"""
stockscreen/evaluate.py — turn one stored scan record into a fully classified,
scored, flat result row (plus the nested detail the stock-detail view shows).

A scan record is threshold-free raw data:
    {"symbol","name","scanned_at","fund":{...},"price":{...},"tech":{...}}
Evaluating it with a ScreenConfig is cheap, so the UI re-evaluates every
record whenever a threshold changes.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

import pandas as pd

from . import fundamentals as fd
from . import scoring as sc
from . import technicals as tc

TECH_FRESH_DAYS = 7          # price bar older than this => price/technical data is stale


def _d(s: str) -> date:
    return datetime.strptime(s[:10], "%Y-%m-%d").date()


def fundamental_trend_summary(trends: dict, cfg) -> str:
    crit = [trends.get(k, {}).get("label", "Insufficient data") for k in cfg.critical_trend_metrics if k in trends]
    if not crit:
        return "Insufficient data"
    if "Declining" in crit:
        return "Declining"
    if "Insufficient data" in crit:
        return "Insufficient data"
    return "Improving" if sum(1 for c in crit if c == "Improving") >= 2 else "Stable"


def evaluate_record(rec: dict, cfg, today: Optional[date] = None) -> dict:
    today = today or date.today()
    fund, price, tech = rec.get("fund") or {}, rec.get("price") or {}, rec.get("tech") or {}
    fe = fd.evaluate_fundamentals(fund, price, cfg, today)
    pos = price.get("positions") or {}
    zone = tc.classify_price_zone(pos, cfg) if pos else {"label": "Insufficient history", "in_zone": [], "available": 0}
    trend = tc.classify_trend(tech, cfg) if tech else {"label": "Insufficient data", "signals": [], "reasons": ["no price data"]}
    opp = sc.classify_opportunity(fe["status"], fe.get("criteria", []), zone["label"], trend, cfg)
    fs = sc.fundamental_score(fe, cfg) if fe.get("metrics", {}).get("fcf_quarters") is not None else {"score": None, "coverage": 0}
    ps = sc.price_score(pos, cfg) if pos else {"score": None, "coverage": 0}
    ts = sc.technical_score(tech, trend, cfg) if tech else {"score": None, "coverage": 0}
    comp = sc.composite(fs["score"], ps["score"], ts["score"], cfg)

    m = fe.get("metrics", {})
    price_asof = price.get("asof")
    price_age = (today - _d(price_asof)).days if price_asof else None
    price_stale = price_age is None or price_age > TECH_FRESH_DAYS
    fq, fy = m.get("fcf_quarters", []), m.get("fcf_years", [])
    p52, p2, p3 = pos.get("52w", {}), pos.get("2y", {}), pos.get("3y", {})

    row = {
        "symbol": rec["symbol"], "name": rec.get("name") or "",
        "status": fe["status"], "stale": fe["stale"], "price_stale": price_stale,
        "opportunity": opp["label"], "opp_code": opp["code"], "opp_why": opp["why"],
        "price_zone": zone["label"], "zone_lookbacks": ",".join(zone["in_zone"]),
        "tech_trend": trend["label"], "recovery_signals": ", ".join(trend.get("signals", [])),
        "fund_trend": fundamental_trend_summary(fe.get("trends", {}), cfg),
        "market_cap": m.get("market_cap"), "price": price.get("last"), "price_asof": price_asof,
        "q_fcf": fq[-1]["fcf"] if fq else None, "ttm_fcf": m.get("ttm_fcf"),
        "fy_fcf_latest": fy[-1]["fcf"] if fy else None, "fy_fcf_prior": fy[-2]["fcf"] if len(fy) > 1 else None,
        "net_margin": m.get("net_margin"), "revenue_growth": m.get("revenue_growth"),
        "growth_basis": m.get("revenue_growth_basis"), "fcf_yield": m.get("fcf_yield"),
        "pos_52w": p52.get("range_pos") if p52.get("available") else None,
        "pos_2y": p2.get("range_pos") if p2.get("available") else None,
        "pos_3y": p3.get("range_pos") if p3.get("available") else None,
        "pct_52w": p52.get("percentile") if p52.get("available") else None,
        "pct_2y": p2.get("percentile") if p2.get("available") else None,
        "pct_3y": p3.get("percentile") if p3.get("available") else None,
        "below_high_52w": p52.get("pct_below_high") if p52.get("available") else None,
        "rsi": tech.get("rsi"), "rsi_band": tc.rsi_band(tech.get("rsi"), cfg),
        "macd_status": tc.macd_status(tech) if tech else "n/a",
        "sma50": tech.get("sma50"), "sma200": tech.get("sma200"),
        "fund_score": fs["score"], "price_score": ps["score"], "tech_score": ts["score"], "composite": comp,
        "data_age_days": m.get("data_age_days"), "latest_quarter_end": m.get("latest_quarter_end"),
        "sector": m.get("sic_desc") or "", "fin_group": m.get("financial_group") or "",
        "fcf_not_meaningful": bool(m.get("fcf_not_meaningful")),
        "flags": "; ".join(fe.get("flags", [])), "reasons": "; ".join(fe.get("reasons", [])),
        "scanned_at": rec.get("scanned_at"),
    }
    return {"row": row, "fund_eval": fe, "zone": zone, "trend": trend, "opportunity": opp,
            "scores": {"fundamental": fs, "price": ps, "technical": ts, "composite": comp}}


def evaluate_all(records: list[dict], cfg, today: Optional[date] = None) -> tuple[pd.DataFrame, dict[str, dict]]:
    rows, details = [], {}
    for rec in records:
        try:
            ev = evaluate_record(rec, cfg, today)
        except Exception as exc:                  # one bad record must never hide the rest
            ev = {"row": {"symbol": rec.get("symbol", "?"), "name": rec.get("name", ""), "status": "ERROR",
                          "reasons": f"evaluation error: {exc}", "opportunity": sc.OPPORTUNITY["X"], "opp_code": "X",
                          "price_zone": "Insufficient history", "tech_trend": "Insufficient data",
                          "fund_trend": "Insufficient data", "stale": False, "price_stale": True}, "error": str(exc)}
        rows.append(ev["row"])
        details[rec["symbol"]] = ev
    return pd.DataFrame(rows), details
