"""
premia/screener.py — phase 2: point-in-time universe, hard filters, liquidity filters and the model layer.

Pipeline at each quarterly evaluation date t (all inputs known at t):
  1. universe      S&P 500 members on t (membership intervals; survivorship control) with price data, minus exclusions
  2. hard filters  premia.fundamentals (net margin, FCF increasing, FCF growth by cap bucket, cap >= 2 $bn)
  3. liquidity     20-day average dollar volume >= $50 M and price >= $10. Option open interest and spreads have no free
                   history, so they are enforced at trade time on live chains only (premia.risk); in backtests the
                   stock's dollar volume is the proxy
  4. model layer   a *safety score* ranks the passing names (higher = safer for selling premium):
                     gap risk     mean |earnings reaction| over the last 8 releases, in units of the stock's median
                                  absolute daily move (a large ratio = surprises the vol forecast misses)
                     vol stability  standard deviation of log 21-day realized vol over the last year (vol of vol)
                   safety = -(rank(gap risk) + rank(vol of vol)) / 2, cross-sectional ranks in [0, 1]
                   Kept only if it predicts next-quarter standardised tail loss out of sample (SPEC.md section 6).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import numpy as np
import pandas as pd

from . import data, store
from .config import LIMITS, load_exclusions
from .fundamentals import evaluate

MIN_PRICE = 10.0


def quarter_dates(start: str = "2010-01-01", end: Optional[str] = None) -> list[pd.Timestamp]:
    return list(pd.date_range(start, end or pd.Timestamp.today(), freq="QE"))


def _eval_symbol(args) -> list[dict]:
    sym, dates = args
    f = store.load_facts(sym)
    px = store.load_prices(sym)
    if not f or px.empty:
        return []
    acts = store.load_actions(sym)
    un = data.unadjusted_close(px, acts)
    splits = [(d.strftime("%Y-%m-%d"), v) for d, k, v in acts.itertuples(index=False) if k == "split"]
    out = []
    for d in dates:
        ds = d.strftime("%Y-%m-%d")
        upto = un[:ds]
        if upto.empty or (d - upto.index[-1]).days > 7:
            continue
        r = evaluate(sym, f["facts"], ds, float(upto.iloc[-1]), splits=splits)
        out.append(r.to_dict())
    return out


def run_pit_screen(symbols: list[str], dates: list[pd.Timestamp], workers: int = 8) -> pd.DataFrame:
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for chunk in ex.map(_eval_symbol, [(s, dates) for s in symbols], chunksize=4) for r in chunk]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# model layer
# ---------------------------------------------------------------------------

def earnings_reactions(close: pd.Series, earnings: list[pd.Timestamp]) -> pd.Series:
    """|log return| on the larger of the release day and the next day (the filing time is not precise), per release."""
    r = np.log(close).diff().abs()
    out = {}
    for e in earnings:
        i = close.index.searchsorted(e)
        if 0 < i < len(close) - 1:
            out[close.index[i]] = max(r.iloc[i], r.iloc[i + 1])
    return pd.Series(out, dtype=float).sort_index()


def risk_features(px: pd.DataFrame, earnings: list[pd.Timestamp], t: pd.Timestamp) -> dict:
    close = px["close"][:t]
    if len(close) < 300:
        return {"gap_ratio": np.nan, "volvol": np.nan}
    r = np.log(close).diff().dropna()
    med = r.iloc[-504:].abs().median()
    reac = earnings_reactions(close, [e for e in earnings if e <= t])
    gap = reac.iloc[-8:].mean() / med if len(reac) >= 4 and med > 0 else np.nan
    rv21 = np.sqrt((r ** 2).rolling(21).mean() * 252)
    volvol = np.log(rv21.iloc[-252:]).std()
    return {"gap_ratio": float(gap), "volvol": float(volvol)}


def forward_tail(px: pd.DataFrame, t: pd.Timestamp, horizon: int = 63, window: int = 21) -> float:
    """Standardised worst 21-day log return in the next quarter: min over windows of ret / (trailing vol x sqrt(21/252)).
    This is what loses money on a short put struck a fixed number of standard deviations out."""
    close = px["close"]
    past = close[:t]
    fut = close[t:].iloc[1:horizon + 1]
    if len(past) < 64 or len(fut) < horizon * 0.8:
        return np.nan
    sig = np.log(past).diff().iloc[-63:].std() * np.sqrt(window)
    path = np.log(pd.concat([past.iloc[-1:], fut]))
    worst = min((path.iloc[i + window] - path.iloc[i]) for i in range(0, len(path) - window))
    return float(-worst / sig) if sig > 0 else np.nan          # positive = loss in standard deviations


def add_scores(df: pd.DataFrame, panels: dict) -> pd.DataFrame:
    """Per (symbol, asof): risk features, safety score (cross-sectional within date), forward tail (for validation)."""
    feats = []
    for r in df.itertuples():
        p = panels.get(r.symbol)
        if p is None:
            feats.append({"gap_ratio": np.nan, "volvol": np.nan, "fwd_tail": np.nan})
            continue
        t = pd.Timestamp(r.asof)
        f = risk_features(p["px"], p["earn"], t)
        f["fwd_tail"] = forward_tail(p["px"], t)
        feats.append(f)
    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(feats)], axis=1)
    g = out.groupby("asof")
    out["safety"] = -(g["gap_ratio"].rank(pct=True) + g["volvol"].rank(pct=True)) / 2
    return out


def liquidity_flags(df: pd.DataFrame, panels: dict) -> pd.DataFrame:
    dv, px_ok = [], []
    for r in df.itertuples():
        p = panels.get(r.symbol)
        if p is None:
            dv.append(np.nan)
            px_ok.append(False)
            continue
        sub = p["px"][: pd.Timestamp(r.asof)].iloc[-20:]
        dv.append(float((sub["close"] * sub["volume"]).mean()) if len(sub) else np.nan)
        px_ok.append(bool(len(sub) and sub["close"].iloc[-1] >= MIN_PRICE))
    df = df.copy()
    df["dollar_vol"] = dv
    df["pass_liquidity"] = (df["dollar_vol"] >= LIMITS.min_stock_dollar_volume) & np.array(px_ok)
    return df


def member_flags(df: pd.DataFrame, membership: pd.DataFrame) -> pd.Series:
    cache: dict = {}
    out = []
    for r in df.itertuples():
        if r.asof not in cache:
            cache[r.asof] = data.members_on(r.asof, membership)
        out.append(r.symbol in cache[r.asof])
    return pd.Series(out, index=df.index)


def pass_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Per quarter: how many index members were evaluated and survived each filter (cumulative and individually)."""
    rows = []
    for asof, g in df[df["member"]].groupby("asof"):
        stage = g
        row = {"asof": asof, "members_with_data": len(g)}
        for name, col in [("margin", "pass_margin"), ("fcf_increasing", "pass_fcf_increasing"),
                          ("fcf_growth", "pass_fcf_growth"), ("cap", "pass_cap"), ("liquidity", "pass_liquidity")]:
            row[f"pass_{name}"] = int((g[col] == True).sum())                     # noqa: E712  (None-safe)
            stage = stage[stage[col] == True]                                     # noqa: E712
            row[f"cum_{name}"] = len(stage)
        row["insufficient"] = int((g["status"] == "insufficient").sum())
        row["stale"] = int((g["status"] == "stale").sum())
        rows.append(row)
    return pd.DataFrame(rows)


def validate_score(df: pd.DataFrame, min_names: int = 15) -> dict:
    """Out-of-sample test of the safety score: per quarter Spearman rank correlation between safety at t and the next
    quarter's standardised tail loss, among names passing the hard + liquidity filters. The score is never fitted
    (no parameters), so every quarter is out of sample."""
    from . import stats
    ics = []
    for asof, g in df.groupby("asof"):
        g = g.dropna(subset=["safety", "fwd_tail"])
        if len(g) < min_names:
            continue
        ics.append({"asof": asof, "ic": float(g["safety"].corr(g["fwd_tail"], method="spearman")), "n": len(g)})
    ic = pd.DataFrame(ics)
    if ic.empty:
        return {"n_quarters": 0}
    mt = stats.mean_test(ic["ic"].to_numpy(), lags=2)
    boots = stats.bootstrap(np.mean, ic["ic"].to_numpy(), reps=4000, mean_block=4, seed=1)
    p_boot = float((boots >= 0).mean())                    # one-sided: H1 mean IC < 0
    return {"n_quarters": len(ic), "mean_ic": float(ic["ic"].mean()), "t_hac": mt["t"], "p_boot_one_sided": p_boot,
            "share_negative": float((ic["ic"] < 0).mean()), "ic": ic}
