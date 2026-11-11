"""
volsurface/analysis.py — from a fitted surface to numbers you can act on (read-only analysis, never an order).

* constant-maturity ATM vol and skew (interpolated in total variance, so they do not jump between expiries)
* risk-neutral probability of finishing below a strike, read off the SMILE: for a put struck at K,
      P(F_T < K) = d(put)/dK = N(-d2) + phi(d1) e^{-k} w'(k) / (2 sqrt(w)),
  i.e. the Black-Scholes term N(-d2) at the LOCAL vol sigma(K), plus a correction from the smile's slope. For an
  equity-style (negative) skew the distribution is left-skewed with its mean pinned at the forward, so near the money
  the smile gives LESS probability of finishing below the strike than a flat ATM vol (the median sits above the
  forward), while on deep out-of-the-money puts, where the local vol is high, it gives far MORE.
  It is a risk-neutral (pricing) probability, not a forecast.
* spread-aware "rich / cheap versus the smooth surface" for each quote: a put is RICH only if even the BID implied
  vol is above the fitted vol by a quarter vol point, CHEAP only if the ASK implied vol is that far below it. Anything inside the
  bid-ask band (or within that margin of it) is indistinguishable from noise and is called FAIR.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.special import ndtr

from . import bs
from .model import FittedSlice, FittedSurface
from .svi import SVI

SQRT_2PI = np.sqrt(2.0 * np.pi)


# ---------------------------------------------------------------------------
# constant maturity
# ---------------------------------------------------------------------------

def surface_range(fs: FittedSurface) -> tuple[float, float]:
    ts = [s.T for s in fs.fitted]
    return (min(ts), max(ts)) if ts else (0.0, 0.0)


def constant_maturity(fs: FittedSurface, days=(30, 60, 90)) -> dict:
    """ATM implied vol, 95% and 90% moneyness skews (IV at K=0.95F / 0.90F minus ATM, in vol units), and the 105% call
    wing, at constant maturities. A maturity outside the fitted expiries is skipped (never extrapolated silently)."""
    lo, hi = surface_range(fs)
    out: dict = {}
    for d in days:
        T = d / 365.0
        if not (lo * 0.999 <= T <= hi * 1.001):
            continue
        atm = float(fs.iv(0.0, T))
        out[d] = {"T": T, "atm": atm, "skew95": float(fs.iv(np.log(0.95), T)) - atm,
                  "skew90": float(fs.iv(np.log(0.90), T)) - atm, "call105": float(fs.iv(np.log(1.05), T)) - atm}
    return out


def snapshot_metrics(fs: FittedSurface) -> dict:
    cm = constant_maturity(fs)
    m: dict = {"asof": str(fs.surface.asof), "spot": fs.surface.spot, "n_slices": len(fs.fitted)}
    for d, v in cm.items():
        m[f"atm{d}"], m[f"skew95_{d}"], m[f"skew90_{d}"] = v["atm"], v["skew95"], v["skew90"]
    if 30 in cm and 90 in cm:
        m["term_30_90"] = cm[90]["atm"] - cm[30]["atm"]
    near = min(fs.fitted, key=lambda s: abs(s.T - 30 / 365)) if fs.fitted else None
    if near is not None:
        p = near.params
        m.update({"svi_expiry": near.expiry, "svi_a": p.a, "svi_b": p.b, "svi_rho": p.rho, "svi_m": p.m, "svi_sigma": p.sigma})
    return m


# ---------------------------------------------------------------------------
# risk-neutral probability from the smile
# ---------------------------------------------------------------------------

def prob_below(params: SVI, k, T: float) -> np.ndarray:
    """Risk-neutral P(F_T < K), K = F e^k, read off an SVI smile (see module docstring)."""
    k = np.asarray(k, dtype=float)
    w = np.maximum(params.w(k), 1e-12)
    sw = np.sqrt(w)
    d1 = (-k + 0.5 * w) / sw
    d2 = d1 - sw
    phi = np.exp(-0.5 * d1 ** 2) / SQRT_2PI
    return np.clip(ndtr(-d2) + phi * np.exp(-k) * params.dw(k) / (2.0 * sw), 0.0, 1.0)


def prob_below_flat(vol: float, k, T: float) -> np.ndarray:
    """The same probability if one flat volatility priced every strike."""
    k = np.asarray(k, dtype=float)
    sw = vol * np.sqrt(T)
    return ndtr(-((-k + 0.5 * sw ** 2) / sw - sw))


# ---------------------------------------------------------------------------
# rich / cheap against the smooth surface
# ---------------------------------------------------------------------------

MIN_GAP = 0.0025          # a quote must clear the fit by this much (0.25 vol points) beyond its bid-ask band to be called rich or cheap


def classify_vs_fit(iv_bid, iv_ask, iv_fit, min_gap: float = MIN_GAP) -> np.ndarray:
    """RICH: bid IV above the fit by at least *min_gap* (you could sell above the surface even at the bid). CHEAP: ask IV below it by
    at least *min_gap*. FAIR: otherwise — the fit lies inside the bid-ask band or within a quarter vol point of it, which is
    smaller than what a smooth parametric curve can be trusted to. UNKNOWN: no usable bid or ask IV."""
    iv_bid, iv_ask, iv_fit = (np.asarray(x, dtype=float) for x in np.broadcast_arrays(iv_bid, iv_ask, iv_fit))
    out = np.full(iv_fit.shape, "FAIR", dtype=object)
    out[np.isfinite(iv_bid) & (iv_bid - iv_fit > min_gap)] = "RICH"
    out[np.isfinite(iv_ask) & (iv_fit - iv_ask > min_gap)] = "CHEAP"
    out[~np.isfinite(iv_bid) & ~np.isfinite(iv_ask)] = "UNKNOWN"
    return out


def quote_table(fsl: FittedSlice, spot: Optional[float] = None) -> pd.DataFrame:
    """Every usable OTM quote of one expiry against its fitted smile."""
    sl, p = fsl.slice, fsl.params
    u = sl.usable.copy()
    u["iv_fit"] = p.iv(u["k"].to_numpy(), sl.T)
    u["diff_vp"] = (u["iv"] - u["iv_fit"]) * 100
    u["signal"] = classify_vs_fit(u["iv_bid"], u["iv_ask"], u["iv_fit"])
    u["p_below"] = prob_below(p, u["k"].to_numpy(), sl.T)
    u["p_below_flat"] = prob_below_flat(float(p.iv(0.0, sl.T)), u["k"].to_numpy(), sl.T)
    u["delta"] = bs.delta(sl.F, u["K"].to_numpy(), sl.T, u["iv_fit"].to_numpy(), u["cp"].to_numpy())
    return u


def put_scan(fsl: FittedSlice, spot: Optional[float], max_delta: float = 0.15, max_spread: float = 0.40,
             min_bid: float = 0.05, sort_by: str = "yield") -> pd.DataFrame:
    """Cash-secured-put view of one expiry's OTM puts. *sort_by*: "yield" (highest annualised yield first — what a premium seller
    scans), "rich" (RICH signals first, then largest bid-vs-fit gap) or "strike" (highest strike first)."""
    sl = fsl.slice
    t = quote_table(fsl, spot)
    t = t[t["cp"] < 0].copy()
    ref = spot or sl.F
    t["days"] = sl.days
    t["abs_delta"] = t["delta"].abs()
    t["yield"] = t["bid"] / t["K"]
    t["annual_yield"] = t["yield"] * 365.0 / max(sl.days, 1e-9)
    t["breakeven"] = t["K"] - t["bid"]
    t["cushion"] = (ref - t["breakeven"]) / ref
    t["edge"] = t["bid"] - bs.price(sl.F, t["K"].to_numpy(), sl.T, t["iv_fit"].to_numpy(), -1.0, sl.D)      # bid vs the fitted value
    t["passes"] = (t["abs_delta"] <= max_delta) & (t["rel_spread"] <= max_spread) & (t["bid"] >= min_bid)
    if sort_by == "rich":
        order = {"RICH": 0, "FAIR": 1, "UNKNOWN": 2, "CHEAP": 3}
        t["_o"] = t["signal"].map(order)
        return t.sort_values(["_o", "diff_vp"], ascending=[True, False]).drop(columns="_o").reset_index(drop=True)
    if sort_by == "strike":
        return t.sort_values("K", ascending=False).reset_index(drop=True)
    return t.sort_values("annual_yield", ascending=False).reset_index(drop=True)


def check_position(fsl: FittedSlice, strike: float, right: str, spot: Optional[float] = None) -> dict:
    """Where one option position sits on its expiry's fitted smile."""
    sl, p = fsl.slice, fsl.params
    k = float(np.log(strike / sl.F))
    cp = 1.0 if right.upper().startswith("C") else -1.0
    fit_iv = float(p.iv(k, sl.T))
    q = sl.points[(np.isclose(sl.points["K"], strike)) & (sl.points["cp"] == cp)]
    row = q.iloc[0] if len(q) else None
    mkt_iv = float(row["iv"]) if row is not None and np.isfinite(row["iv"]) else None
    sig = None
    if row is not None:
        sig = str(classify_vs_fit(row["iv_bid"], row["iv_ask"], fit_iv))
    pb = float(prob_below(p, k, sl.T))
    flat = float(prob_below_flat(float(p.iv(0.0, sl.T)), k, sl.T))
    itm_smile, itm_flat = (pb, flat) if cp < 0 else (1 - pb, 1 - flat)
    model_px = float(bs.price(sl.F, strike, sl.T, fit_iv, cp, sl.D))
    return {"k": k, "days": sl.days, "fit_iv": fit_iv, "market_iv": mkt_iv,
            "diff_vp": None if mkt_iv is None else (mkt_iv - fit_iv) * 100, "signal": sig,
            "bid": None if row is None else float(row["bid"]), "ask": None if row is None else float(row["ask"]),
            "model_price": model_px, "p_itm": itm_smile, "p_itm_flat": itm_flat,
            "moneyness_pct": (strike / (spot or sl.F) - 1) * 100,
            "inside_data": bool(sl.usable["k"].min() <= k <= sl.usable["k"].max())}


# ---------------------------------------------------------------------------
# dynamics
# ---------------------------------------------------------------------------

def surface_difference(a: FittedSurface, b: FittedSurface, k_grid=None, days_grid=(14, 30, 60, 90, 180, 365)) -> pd.DataFrame:
    """IV(b) - IV(a) in vol points on a common (maturity, log-moneyness) grid; NaN where either surface has no maturity."""
    k_grid = np.linspace(-0.3, 0.15, 10) if k_grid is None else np.asarray(k_grid)
    ra, rb = surface_range(a), surface_range(b)
    rows = {}
    for d in days_grid:
        T = d / 365.0
        if ra[0] * 0.999 <= T <= ra[1] * 1.001 and rb[0] * 0.999 <= T <= rb[1] * 1.001:
            rows[d] = (b.iv(k_grid, T) - a.iv(k_grid, T)) * 100
    df = pd.DataFrame(rows, index=np.round(np.exp(k_grid) * 100, 1)).T
    df.index.name, df.columns.name = "days", "% of forward"
    return df


def percentile_of_last(series: pd.Series, lookback: int = 504, min_obs: int = 60) -> Optional[float]:
    """McMillan-style: share of the prior lookback readings below the latest one (own history only)."""
    from ivscreen import metrics as mx
    pct, _ = mx.rank(series.dropna(), lookback, min_obs)
    return pct


# ---------------------------------------------------------------------------
# scorecard: plain-language reading of today's surface and the market-wide indicators
# ---------------------------------------------------------------------------

def scorecard(pcts: dict, cm: dict) -> list[dict]:
    """Observations for a premium seller, each {title, text, tone}. *pcts* maps indicator name -> {"value", "pct"} (own-history
    percentiles, McMillan: never one series against another's level); *cm* is constant_maturity(). Descriptive and conditional by
    design — none of it forecasts anything."""
    from ivscreen import metrics as imx
    out: list[dict] = []

    def band(p):
        return imx.classify(p).lower() if p is not None else "n/a"

    vix = pcts.get("VIX")
    if vix and vix["pct"] is not None:
        p = vix["pct"]
        tone = "warn" if p >= 90 else "good" if p >= 75 else "info"
        out.append({"title": f"Volatility level: VIX {vix['value']:.1f}, {p:.0f}th percentile ({band(p)})", "tone": tone,
                    "text": ("Index options are expensive against their own two-year history. Premium sellers are paid more than usual, but this is also when "
                             "markets tend to move most — size for the larger swings the price is telling you about." if p >= 75 else
                             "Index options are cheap against their own history: premium is thin, so selling it pays less than usual for the same risk." if p <= 25 else
                             "Index implied volatility is inside its normal range for the last two years.")})
    ratio = pcts.get("VIX / VIX3M")
    if ratio:
        backward = ratio["value"] > 1.0
        out.append({"title": f"Term structure: VIX / VIX3M = {ratio['value']:.2f} ({'backwardation' if backward else 'contango'})"
                             + (f", {ratio['pct']:.0f}th percentile" if ratio["pct"] is not None else ""), "tone": "warn" if backward else "info",
                    "text": ("Near-term volatility is priced above the next quarter's — the market is paying up for immediate protection, typical of stress. "
                             "Short-dated premium is rich for a reason." if backward else
                             "Near-term volatility is priced below the next quarter's, the usual calm configuration: selling shorter-dated options "
                             "benefits from the upward slope of the curve as they age.")})
    if 30 in cm and 90 in cm:
        t = (cm[90]["atm"] - cm[30]["atm"]) * 100
        out.append({"title": f"This surface: 30-day ATM vol {cm[30]['atm'] * 100:.1f}%, 90-day {cm[90]['atm'] * 100:.1f}% ({t:+.1f} vol pts)", "tone": "info",
                    "text": "Fitted, constant-maturity ATM implied vols. A positive gap is contango in this underlying's own surface; a negative one, backwardation."})
    if 30 in cm:
        sk = cm[30]["skew90"] * 100
        sk_p = pcts.get("CBOE SKEW")
        out.append({"title": f"Skew: the 90%-strike put carries {sk:+.1f} vol pts over ATM at 30 days"
                             + (f"; CBOE SKEW {sk_p['value']:.0f}, {sk_p['pct']:.0f}th percentile" if sk_p and sk_p["pct"] is not None else ""), "tone": "info",
                    "text": "Steeper skew means out-of-the-money puts are priced richly relative to the at-the-money option, so selling them collects more per "
                            "unit of delta — and the market is saying a large drop is more likely or more costly than a symmetric model assumes."})
    rv, iv = pcts.get("SPX 20d realized vol"), pcts.get("VIX")
    if rv and iv and rv["pct"] is not None and iv["pct"] is not None:
        rel = imx.relationship(iv["pct"], rv["pct"])
        out.append({"title": f"Implied vs realized (each against its own history): VIX {iv['pct']:.0f}th pct, 20-day realized {rv['pct']:.0f}th pct — {rel}",
                    "tone": "info", "text": imx.REL_NOTE[rel]})
    if not out:
        out.append({"title": "Market-wide indicators are unavailable", "tone": "info",
                    "text": "Could not load VIX history, so only the surface itself can be read. Check the connection and reload."})
    return out
