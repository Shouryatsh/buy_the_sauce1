"""
premia/strategies.py — phase 4 strategy models: they turn the market model, the RV forecast and the physical
distribution into proposed trades. They never place orders; the backtester / executor applies the risk layer.

Forecast variance over an option's life (both strategies):
    sigma_f^2 = max(fc - n_events_per_year * E[m^2], 0.5 fc) + 1{earnings in life} * E[m^2] / T
fc = the walk-forward RV forecast (annualised variance, phase 3 choice), E[m^2] = mean squared earnings-day
reaction over the last 8 releases (point in time). Removing the average earnings contribution from fc and adding back
the event only where one falls inside the option's life is the standard event-variance decomposition (used by
practitioners for earnings straddles; cf. Gatheral ch. 3 on term-structure bumps).

Wheel (short premium, McMillan structure, model-chosen strike/expiry):
    candidates  monthly expiries near 21/30/45 days x |delta| in {0.10..0.30} (puts; calls once assigned)
    edge/share  fill - D E_P[payoff] - commission/100 - expected hedge cost per share
    risk        CVaR_95 of the daily-delta-hedged P&L on FHS paths
    score       (edge - lambda * CVaR) / K  x 365 / days      -> best score, only if edge > 0 and score > 0
    guardrails  no expiry spanning earnings; calls never below cost basis; close before an earnings date that
                appears inside the life; max loss per position (premia.risk)
    hold/close  close when buying back is cheaper than the expected cost of holding: ask + commission <
                D E_P[remaining payoff] + lambda * CVaR_remaining  (this is what "take profit early / roll when
                tested" becomes once the model decides)

Long straddle (long volatility):
    entry       ATM straddle on the first monthly expiry 21-45 days out when sigma_f - IV_ask >= threshold [WF]
                and the model value V(sigma_f) exceeds the ask plus costs; earnings inside the life are the usual
                catalyst (they enter through sigma_f)
    exits       remaining edge < 0 (sigma_f over the remaining life below IV_bid); the day after the event; 7 days
                to expiry (time stop); IV 5 vol points below entry with no event left (vega stop)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from . import pricing
from .config import COSTS, PARAMS
from .hedging import stock_cost_frac
from .market import Underlying, monthly_expiries, strike_step, year_frac
from .physical import distribution

EVENTS_PER_YEAR = 4


def earnings_move_var(close: pd.Series, earnings, t: pd.Timestamp, n: int = 8) -> float:
    """Mean squared earnings-day reaction over the last n releases before t (daily-variance units). The reaction is
    the larger of the release day's and the next day's squared log return (before-open vs after-close timing)."""
    r = np.log(close).diff()
    vals = []
    for e in [pd.Timestamp(e) for e in earnings if pd.Timestamp(e) <= t][-n:]:
        i = close.index.searchsorted(e)
        if 0 < i < len(close) - 1 and close.index[i + 1] <= t:
            vals.append(max(r.iloc[i] ** 2, r.iloc[i + 1] ** 2))
    return float(np.mean(vals)) if len(vals) >= 3 else 0.0


def event_var(u: Underlying, t: pd.Timestamp, n: int = 8) -> float:
    return earnings_move_var(u.df["close"], u.earnings, t, n)


def event_adjusted_var(fc: float, m2: float, T: float, event_in_window: bool) -> float:
    """Split a forecast's annual variance into diffusion + earnings jumps: remove the average jump content
    (EVENTS_PER_YEAR x m2, floored at half the forecast), then add the jump back only if a release falls before expiry."""
    base = max(fc - EVENTS_PER_YEAR * m2, 0.5 * fc)
    if event_in_window and T > 0:
        base += m2 / T
    return float(base)


def forecast_var(u: Underlying, t: pd.Timestamp, T: float, expiry: pd.Timestamp, model: str = "har_iv") -> float:
    """sigma_f^2 (annual) over the life of an option expiring at *expiry*."""
    row = u.df.loc[t]
    fc = row.get(f"fc_{model}", np.nan)
    for alt in ("fc_har", "fc_naive"):
        if not np.isfinite(fc):
            fc = row.get(alt, np.nan)
    if not np.isfinite(fc):
        return np.nan
    if u.etf:
        return float(fc)
    return event_adjusted_var(float(fc), event_var(u, t), T, bool(u.earnings_between(t, expiry)))


def trading_days(t: pd.Timestamp, expiry: pd.Timestamp) -> int:
    return int(max(np.busday_count(t.date(), expiry.date()), 1))


def _bs_delta(S, K, tau, r, q, sig, right):
    from scipy.special import ndtr
    vt = sig * np.sqrt(tau)
    d1 = (np.log(S / K) + (r - q + 0.5 * sig * sig) * tau) / vt
    dq = np.exp(-q * tau)
    return dq * ndtr(d1) if right == "C" else -dq * ndtr(-d1)


def hedged_paths_pnl(paths: np.ndarray, K, T: float, r: float, q: float, sig, right: str, premium,
                     short: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """P&L per share of short (or long) options at strikes K (vector), each delta-hedged daily at its own strike vol
    (sticky strike) along the simulated price paths. Returns (pnl, turnover in shares), both shaped (n_strikes, n_paths)."""
    K = np.atleast_1d(np.asarray(K, float))[:, None]
    sig = np.atleast_1d(np.asarray(sig, float))[:, None]
    prem = np.atleast_1d(np.asarray(premium, float))[:, None]
    n, m = paths.shape
    days = m - 1
    sign = -1.0 if short else 1.0
    h_prev = np.zeros((len(K), n))
    hedge_pnl = np.zeros((len(K), n))
    turnover = np.zeros((len(K), n))
    for j in range(days):
        S = paths[None, :, j]
        tau = max(T * (1 - j / days), 1e-5)
        h = -sign * _bs_delta(S, K, tau, r, q, sig, right)      # shares that neutralise the option position's delta
        turnover += np.abs(h - h_prev)
        hedge_pnl += h * (paths[None, :, j + 1] - S)
        h_prev = h
    turnover += np.abs(h_prev)                                     # unwind at expiry
    ST = paths[None, :, -1]
    pay = np.maximum(K - ST, 0) if right == "P" else np.maximum(ST - K, 0)
    pnl = (prem - pay) if short else (pay - prem)
    return pnl + hedge_pnl, turnover


def _cvar(pnl: np.ndarray, level: float = PARAMS.cvar_level) -> np.ndarray:
    """Expected loss in the worst (1 - level) tail, per row; 0 if the tail is a gain."""
    k = max(int(pnl.shape[-1] * (1 - level)), 1)
    tail = np.sort(pnl, axis=-1)[..., :k]
    return np.maximum(-tail.mean(axis=-1), 0.0)


@dataclass
class Candidate:
    symbol: str
    right: str
    K: float
    expiry: pd.Timestamp
    days: int
    T: float
    iv: float
    mid: float
    fill: float
    fair: float
    edge: float              # per share, after commission and expected hedge cost
    cvar: float              # per share, hedged
    p_itm: float
    score: float
    delta: float
    greeks: dict = field(default_factory=dict)


def wheel_candidates(u: Underlying, t: pd.Timestamp, right: str, lam: float, hedged: bool, min_strike: float = 0.0,
                     n_paths: int = 1500, seed: int = 0, slip: float = COSTS.slippage_frac_of_half_spread,
                     model: str = "har_iv") -> list[Candidate]:
    row = u.df.loc[t]
    S, r, q = float(row["close"]), float(row["r"]), float(row["q"])
    fhs = u.fhs_at(t)
    if fhs is None or not np.isfinite(row["iv"]):
        return []
    out: list[Candidate] = []
    lam_cost = stock_cost_frac(S)
    for expiry in monthly_expiries(t, 3):
        cal_days = (expiry - t).days
        if not any(abs(cal_days - d) <= 8 for d in PARAMS.dte_grid):
            continue
        if u.earnings_between(t, expiry):
            continue                                            # guardrail: never sell premium across earnings
        T = year_frac(t, expiry)
        days = trading_days(t, expiry)
        fv = forecast_var(u, t, T, expiry, model)
        if not np.isfinite(fv):
            continue
        dist = distribution(fhs, S, days, r, q, n=n_paths, seed=seed, target_var_annual=fv, keep_paths=True)
        dist.T = T
        strikes = set()
        atm = u.atm_vol(t, T)
        for dlt in PARAMS.put_delta_grid:
            target = -dlt if right == "P" else dlt
            sig = atm
            for _ in range(6):
                k = float(pricing.strike_for_delta(S, T, r, q, sig, target, right))
                sig = float(u.vol(t, [k], T)[0])
            step = strike_step(S, u.etf)
            kk = np.floor(k / step) * step if right == "P" else np.ceil(k / step) * step
            if kk > 0 and kk >= min_strike:
                strikes.add(round(kk, 2))
        if not strikes:
            continue
        Ks = np.array(sorted(strikes))
        qt = u.quote(t, Ks, expiry, right)
        half = qt["ask"] - qt["mid"]
        fill = np.maximum(qt["mid"] - slip * half, 0.0)
        fair = dist.fair_price(Ks, right)
        if hedged:
            pnl, turn = hedged_paths_pnl(dist.paths, Ks, T, r, q, qt["iv"], right, fill)
            hedge_cost = turn.mean(axis=1) * S * lam_cost
        else:
            pay = np.maximum(Ks[:, None] - dist.ST[None, :], 0) if right == "P" else np.maximum(dist.ST[None, :] - Ks[:, None], 0)
            pnl, hedge_cost = fill[:, None] - pay, np.zeros(len(Ks))
        cvar = _cvar(pnl)
        pb = dist.prob_below(Ks)
        for i, K in enumerate(Ks):
            if fill[i] < 0.05:
                continue                                        # nothing worth selling below 5 cents
            edge = float(fill[i] - fair[i] - COSTS.option_commission / 100 - hedge_cost[i])
            score = (edge - lam * cvar[i]) / K * 365 / max(cal_days, 1)
            p_itm = float(pb[i]) if right == "P" else float(1 - pb[i])
            out.append(Candidate(u.symbol, right, float(K), expiry, days, T, float(qt["iv"][i]), float(qt["mid"][i]),
                                 float(fill[i]), float(fair[i]), edge, float(cvar[i]), p_itm, float(score),
                                 float(qt["greeks"].delta[i])))
    return out


def best_candidate(cands: list[Candidate]) -> Optional[Candidate]:
    good = [c for c in cands if c.edge > 0 and c.score > 0]
    return max(good, key=lambda c: c.score) if good else None


def baseline_candidate(u: Underlying, t: pd.Timestamp, right: str, min_strike: float = 0.0,
                       slip: float = COSTS.slippage_frac_of_half_spread, target_delta: float = 0.30,
                       target_days: int = 35) -> Optional[Candidate]:
    """Fixed-rule wheel (the comparison baseline): |delta| 0.30, the monthly expiry nearest 35 days, no model."""
    row = u.df.loc[t]
    if not np.isfinite(row["iv"]):
        return None
    S, r, q = float(row["close"]), float(row["r"]), float(row["q"])
    exps = [e for e in monthly_expiries(t, 3) if not u.earnings_between(t, e) and 14 <= (e - t).days <= 60]
    if not exps:
        return None
    expiry = min(exps, key=lambda e: abs((e - t).days - target_days))
    T = year_frac(t, expiry)
    Ks = u.strikes_near(t, 0.6, 1.4)
    Ks = Ks[Ks >= min_strike]
    if not len(Ks):
        return None
    qt = u.quote(t, Ks, expiry, right)
    i = int(np.argmin(np.abs(np.abs(qt["greeks"].delta) - target_delta)))
    half = qt["ask"][i] - qt["mid"][i]
    fill = max(qt["mid"][i] - slip * half, 0.0)
    if fill < 0.05:
        return None
    return Candidate(u.symbol, right, float(Ks[i]), expiry, trading_days(t, expiry), T, float(qt["iv"][i]),
                     float(qt["mid"][i]), float(fill), np.nan, np.nan, np.nan, np.nan, np.nan, float(qt["greeks"].delta[i]))


def hold_or_close(u: Underlying, t: pd.Timestamp, right: str, K: float, expiry: pd.Timestamp, lam: float,
                  hedged: bool, slip: float = COSTS.slippage_frac_of_half_spread, n_paths: int = 1000,
                  model: str = "har_iv") -> tuple[bool, str]:
    """(close?, reason) for an open short option: close when buying back costs less than the expected cost of holding."""
    if u.earnings_between(t, expiry):
        return True, "earnings now inside the option's life"
    fhs = u.fhs_at(t)
    row = u.df.loc[t]
    if fhs is None or not np.isfinite(row["iv"]):
        return False, ""
    S, r, q = float(row["close"]), float(row["r"]), float(row["q"])
    T = year_frac(t, expiry)
    days = trading_days(t, expiry)
    if days <= 1:
        return False, ""
    fv = forecast_var(u, t, T, expiry, model)
    if not np.isfinite(fv):
        return False, ""
    qt = u.quote(t, [K], expiry, right)
    mid = float(qt["mid"][0])
    ask = mid + slip * float(qt["ask"][0] - mid)
    bid = mid - slip * float(qt["ask"][0] - mid)
    dist = distribution(fhs, S, days, r, q, n=n_paths, seed=1, target_var_annual=fv, keep_paths=hedged)
    dist.T = T
    fair = float(dist.fair_price([K], right)[0])
    if hedged:
        pnl, turn = hedged_paths_pnl(dist.paths, [K], T, r, q, qt["iv"], right, [ask])
        hedge_cost = float(turn.mean()) * S * stock_cost_frac(S)
    else:
        pay = np.maximum(K - dist.ST, 0) if right == "P" else np.maximum(dist.ST - K, 0)
        pnl, hedge_cost = (ask - pay)[None, :], 0.0
    # the same score the entry used, as if the position were sold again at today's price: hold while it is not
    # clearly negative (a full spread of hysteresis stops churning)
    hold_value = (ask - fair - hedge_cost) - lam * float(_cvar(pnl)[0])
    if hold_value < -(ask - bid) - COSTS.option_commission / 100:
        return True, f"hold value {hold_value:.2f}/sh < 0: buy back {ask:.2f}, fair {fair:.2f}"
    return False, ""


@dataclass
class StraddleIdea:
    symbol: str
    K: float
    expiry: pd.Timestamp
    T: float
    iv: float
    ask: float               # straddle ask per share (call + put)
    fill: float
    model_value: float
    edge_vol: float          # sigma_f - IV (vol units)
    edge: float              # model value - fill - costs, per share
    event: bool


def straddle_idea(u: Underlying, t: pd.Timestamp, threshold: float, slip: float = COSTS.slippage_frac_of_half_spread,
                  model: str = "har_iv") -> Optional[StraddleIdea]:
    row = u.df.loc[t]
    if not np.isfinite(row["iv"]):
        return None
    S, r, q = float(row["close"]), float(row["r"]), float(row["q"])
    exps = [e for e in monthly_expiries(t, 3) if 21 <= (e - t).days <= 45]
    if not exps:
        return None
    expiry = exps[0]
    T = year_frac(t, expiry)
    F = S * np.exp((r - q) * T)
    Ks = u.strikes_near(t, 0.95, 1.05)
    K = float(Ks[np.argmin(np.abs(Ks - F))])
    c, p = u.quote(t, [K], expiry, "C"), u.quote(t, [K], expiry, "P")
    mid = float(c["mid"][0] + p["mid"][0])
    ask_full = float(c["ask"][0] + p["ask"][0])
    iv = float(0.5 * (c["iv"][0] + p["iv"][0]))
    fv = forecast_var(u, t, T, expiry, model)
    if not np.isfinite(fv) or fv <= 0:
        return None
    sf = float(np.sqrt(fv))
    ec = straddle_economics(S, K, T, r, q, mid, ask_full, iv, sf, trading_days(t, expiry), slip)
    idea = StraddleIdea(u.symbol, K, expiry, T, iv, ask_full, ec["fill"], ec["model_value"], ec["edge_vol"], ec["edge"],
                        bool(u.earnings_between(t, expiry)))
    if idea.edge_vol >= threshold and idea.edge > 0:
        return idea
    return None


def straddle_economics(S: float, K: float, T: float, r: float, q: float, mid: float, ask: float, iv: float,
                       sigma_f: float, days: int, slip: float = COSTS.slippage_frac_of_half_spread) -> dict:
    """Value of buying the K straddle at mid + slip x (ask - mid) and delta-hedging it to expiry, if realized vol is
    sigma_f: model value (BSM at sigma_f), the fill, costs (2 option commissions + expected gamma-scalping stock
    costs), edge per share, and edge in vol points (sigma_f minus the IV paid)."""
    fill = mid + slip * (ask - mid)
    model_value = float(pricing.price(S, [K], T, r, q, [sigma_f], "C")[0] + pricing.price(S, [K], T, r, q, [sigma_f], "P")[0])
    # expected hedge cost of gamma scalping ~ turnover x cost; approximated from gamma and the forecast move
    g = pricing.greeks(S, [K], T, r, q, [iv], "C")
    turnover = 2 * float(g.gamma[0]) * S * sigma_f / np.sqrt(252) * np.sqrt(2 / np.pi) * days
    costs = 2 * COSTS.option_commission / 100 + turnover * S * stock_cost_frac(S)
    iv_fill = float(pricing.implied_vol([fill / 2], S, [K], T, r, q, "C")[0]) if fill > 0 else iv
    iv_fill = iv_fill if np.isfinite(iv_fill) else iv
    return {"fill": float(fill), "model_value": model_value, "costs": float(costs), "edge": float(model_value - fill - costs),
            "iv_fill": iv_fill, "edge_vol": float(sigma_f - iv_fill), "hedge_turnover_shares_per_share": float(turnover)}


def straddle_exit(u: Underlying, t: pd.Timestamp, K: float, expiry: pd.Timestamp, entry_iv: float, had_event: bool,
                  model: str = "har_iv") -> tuple[bool, str]:
    row = u.df.loc[t]
    days_left = (expiry - t).days
    if days_left <= 7:
        return True, "time stop: 7 days to expiry"
    event_left = bool(u.earnings_between(t, expiry))
    if had_event and not event_left:
        return True, "event passed"
    if not np.isfinite(row["iv"]):
        return False, ""
    T = year_frac(t, expiry)
    fv = forecast_var(u, t, T, expiry, model)
    qc, qp = u.quote(t, [K], expiry, "C"), u.quote(t, [K], expiry, "P")
    iv_bid = float(pricing.implied_vol((qc["bid"][0] + qp["bid"][0]) / 2, row["close"], K, T, row["r"], row["q"], "C"))
    if np.isfinite(fv) and np.isfinite(iv_bid) and np.sqrt(fv) < iv_bid:
        return True, "remaining edge < 0"
    iv_now = float(0.5 * (qc["iv"][0] + qp["iv"][0]))
    if not event_left and iv_now < entry_iv - 0.05:
        return True, "vega stop"
    return False, ""
