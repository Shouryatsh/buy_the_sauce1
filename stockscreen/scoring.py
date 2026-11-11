"""
stockscreen/scoring.py — the opportunity classification and the three
transparent heuristic scores (fundamental quality, price opportunity, technical
trend).

HEURISTIC ONLY. These scores and classes rank how well a stock fits the
screening philosophy; they have not been backtested and are not return
predictions. A "Potential buy zone" label describes price location plus
trend/fundamental context — it is not evidence that a stock is undervalued.

Scores are 0-100 weighted averages over the components that could be computed.
A missing component is excluded and the remaining weights renormalised (never
silently given a good value); the share of weight actually available is reported
as the score's coverage, and a score with coverage below min_score_coverage is None.
"""

from __future__ import annotations

from typing import Optional

OPPORTUNITY = {
    "A": "A. Quality stock - monitor",
    "B": "B. Potential buy zone - strong fundamentals",
    "C": "C. Potential buy zone - healthy pullback",
    "D": "D. Potential buy zone - reversal confirmation",
    "E": "E. Fundamental or technical warning",
    "X": "Not eligible",
}


def classify_opportunity(fund_status: str, criteria: list[dict], zone_label: str, tech: dict, cfg) -> dict:
    low = zone_label in ("Deep value-zone candidate", "Low-price candidate")
    tl = tech.get("label", "Insufficient data")
    signals = tech.get("signals", [])
    eligible = fund_status == "PASSED"

    if eligible:
        if not low:
            why = "passes the fundamental screen; price is not in the low zone" if zone_label != "Insufficient history" \
                else "passes the fundamental screen; not enough price history for the low-zone test"
            return {"code": "A", "label": OPPORTUNITY["A"], "why": why}
        if tl == "Downtrend":
            return {"code": "E", "label": OPPORTUNITY["E"],
                    "why": "passes fundamentals and is in the low-price zone, but the technical trend is a persistent downtrend"}
        if tl in ("Strong uptrend", "Healthy pullback"):
            return {"code": "C", "label": OPPORTUNITY["C"],
                    "why": f"low-price zone with a constructive long-term trend ({tl.lower()})"}
        if tl == "Trend reversal candidate" and len(signals) >= cfg.reversal_confirm_signals:
            return {"code": "D", "label": OPPORTUNITY["D"],
                    "why": f"low-price zone with recovery signals: {', '.join(signals)}"}
        why = ("low-price zone; recovery signals incomplete" if tl == "Trend reversal candidate"
               else f"low-price zone; technical confirmation incomplete ({tl.lower()})")
        return {"code": "B", "label": OPPORTUNITY["B"], "why": why}

    # not eligible: surface the dangerous look-alikes instead of dropping them silently
    others_pass = all(c["status"] == "pass" for c in criteria if c["key"] != "trend") and bool(criteria)
    trend_failed = any(c["key"] == "trend" and c["status"] == "fail" for c in criteria)
    if low and others_pass and trend_failed:
        return {"code": "E", "label": OPPORTUNITY["E"],
                "why": "in the low-price zone, but a critical fundamental metric is declining"}
    return {"code": "X", "label": OPPORTUNITY["X"], "why": "does not pass the fundamental screen"}


# ---------------------------------------------------------------------------
# scores
# ---------------------------------------------------------------------------

def _scale(x: Optional[float], threshold: float, cap: float) -> Optional[float]:
    """<=0 -> 0; threshold -> 50; cap or more -> 100 (linear between)."""
    if x is None:
        return None
    if x <= 0:
        return 0.0
    if x < threshold:
        return 50.0 * x / threshold
    return min(100.0, 50.0 + 50.0 * (x - threshold) / max(cap - threshold, 1e-9))


def _combine(components: dict[str, Optional[float]], weights: dict[str, float], min_cov: float) -> dict:
    total = sum(weights.values())
    avail = {k: v for k, v in components.items() if v is not None and k in weights}
    w_avail = sum(weights[k] for k in avail)
    coverage = w_avail / total if total else 0.0
    if not avail or coverage < min_cov:
        return {"score": None, "coverage": coverage, "components": components,
                "missing": [k for k, v in components.items() if v is None]}
    score = sum(weights[k] * v for k, v in avail.items()) / w_avail
    return {"score": round(score, 1), "coverage": round(coverage, 2), "components": components,
            "missing": [k for k, v in components.items() if v is None]}


_TREND_POINTS = {"Improving": 100.0, "Stable": 60.0, "Declining": 0.0}


def fundamental_score(fe: dict, cfg) -> dict:
    m, tr = fe.get("metrics", {}), fe.get("trends", {})
    fq = [x["fcf"] for x in m.get("fcf_quarters", []) if x["fcf"] is not None]
    meaningful = not m.get("fcf_not_meaningful")
    comp = {
        "fcf_consistency": (100.0 * sum(1 for v in fq if v > 0) / len(fq)) if len(fq) >= 4 and meaningful else None,
        "fcf_trend": _TREND_POINTS.get(tr.get("fcf", {}).get("label")) if meaningful else None,
        "net_margin": _scale(m.get("net_margin"), cfg.min_net_margin, 0.30),
        "revenue_growth": _scale(m.get("revenue_growth"), cfg.min_revenue_growth, 0.30),
        "fcf_yield": _scale(m.get("fcf_yield"), cfg.min_fcf_yield, 0.15),
    }
    classified = [t["label"] for t in tr.values() if t.get("label") in ("Improving", "Stable", "Declining")]
    comp["stability"] = (100.0 * sum(1 for l in classified if l != "Declining") / len(classified)) if classified else None
    return _combine(comp, cfg.weights_fundamental, cfg.min_score_coverage)


def price_score(pos: dict, cfg) -> dict:
    def inv(label):
        p = pos.get(label, {})
        return None if not p.get("available") or p.get("range_pos") is None else max(0.0, 100.0 - p["range_pos"])
    pcts = [100.0 - p["percentile"] for p in pos.values() if p.get("available") and p.get("percentile") is not None]
    p52 = pos.get("52w", {})
    comp = {
        "range_52w": inv("52w"), "range_2y": inv("2y"), "range_3y": inv("3y"),
        "percentiles": sum(pcts) / len(pcts) if pcts else None,
        "from_high": min(100.0, p52["pct_below_high"] / 40.0 * 100.0) if p52.get("available") else None,
    }
    return _combine(comp, cfg.weights_price, cfg.min_score_coverage)


_TREND_SCORE = {"Strong uptrend": 100.0, "Healthy pullback": 75.0, "Trend reversal candidate": 55.0,
                "Sideways/consolidating": 40.0, "Downtrend": 10.0}


def _rsi_points(r: Optional[float]) -> Optional[float]:
    if r is None:
        return None
    if r < 30:
        return 25.0
    if r < 40:
        return 40.0
    if r < 50:
        return 70.0
    if r <= 65:
        return 100.0
    if r <= 70:
        return 80.0
    return 60.0


def _macd_points(t: dict) -> Optional[float]:
    h, hp = t.get("macd_hist"), t.get("macd_hist_prev")
    if h is None:
        return None
    if t.get("macd_cross") == "bullish":
        return 100.0
    if h > 0:
        return 85.0 if hp is not None and h > hp else 60.0
    return 35.0 if hp is not None and h > hp else 10.0


def technical_score(t: dict, trend: dict, cfg) -> dict:
    p = t.get("price")
    above = [1.0 if p > t[k] else 0.0 for k in ("sma20", "sma50", "sma200") if t.get(k) is not None and p is not None]
    slopes = [1.0 if t[k] > 0 else 0.0 for k in ("sma50_slope", "sma200_slope") if t.get(k) is not None]
    comp = {
        "trend": _TREND_SCORE.get(trend.get("label")),
        "price_vs_sma": 100.0 * sum(above) / len(above) if above else None,
        "sma_cross": (100.0 if t["sma50"] > t["sma200"] else 0.0) if t.get("sma50") is not None and t.get("sma200") is not None else None,
        "slopes": 100.0 * sum(slopes) / len(slopes) if slopes else None,
        "rsi": _rsi_points(t.get("rsi")),
        "macd": _macd_points(t),
    }
    return _combine(comp, cfg.weights_technical, cfg.min_score_coverage)


def composite(fs: Optional[float], ps: Optional[float], ts: Optional[float], cfg) -> Optional[float]:
    """Heuristic rank score: weighted mean of the available scores. Needs the
    fundamental score (the screen's anchor); price/technical renormalise if absent."""
    if fs is None:
        return None
    parts = {"fundamental": fs, "price": ps, "technical": ts}
    w = cfg.weights_rank
    avail = {k: v for k, v in parts.items() if v is not None}
    tot = sum(w[k] for k in avail)
    return round(sum(w[k] * v for k, v in avail.items()) / tot, 1) if tot else None
