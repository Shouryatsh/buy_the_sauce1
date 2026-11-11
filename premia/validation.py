"""
premia/validation.py — phase 5 (direction independence) and the phase 6 performance report.

  performance     CAGR, vol, Sharpe, Sortino, max drawdown, worst day/week/month, daily CVaR 95/99, skew, kurtosis
  beta            OLS of daily book returns on SPY returns with Newey-West t-stat; correlation
  underlyings     per name: correlation of that name's daily P&L (options + stock, after costs) with its own return,
                  over the days it was held; summarised as the held-days-weighted mean and the median |corr|
  regimes         month classified by SPY return (up > +2 %, down < -2 %, flat otherwise) and by VIX above / below its
                  median; P&L in the 2020 crash (2020-02-19 .. 2020-04-30), 2020 and 2022
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import stats, store


def returns(nav: pd.Series) -> pd.Series:
    return nav.pct_change().dropna()


def performance(nav: pd.Series, periods: int = 252) -> dict:
    r = returns(nav)
    if len(r) < 20:
        return {}
    years = len(r) / periods
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1
    dd = nav / nav.cummax() - 1
    wk = nav.resample("W").last().pct_change().dropna()
    mo = nav.resample("ME").last().pct_change().dropna()
    q95 = np.quantile(r, 0.05)
    q99 = np.quantile(r, 0.01)
    return {"years": years, "cagr": cagr, "ann_vol": r.std() * np.sqrt(periods), "sharpe": stats.sharpe(r, periods),
            "sortino": stats.sortino(r, periods), "max_drawdown": dd.min(), "worst_day": r.min(), "worst_week": wk.min(),
            "worst_month": mo.min(), "cvar95_daily": r[r <= q95].mean(), "cvar99_daily": r[r <= q99].mean(),
            "skew": float(r.skew()), "kurtosis": float(r.kurt()), "hit_rate_months": float((mo > 0).mean()),
            "mean_t_hac": stats.mean_test(r.to_numpy())["t"]}


def market_returns(index: pd.DatetimeIndex) -> pd.Series:
    px = store.load_prices("SPY")
    tr = px["adj_close"].reindex(index.union(px.index)).ffill().reindex(index)
    return tr.pct_change()


def beta_test(r: pd.Series, m: pd.Series) -> dict:
    d = pd.concat([r.rename("r"), m.rename("m")], axis=1).dropna()
    res = stats.ols_hac(d["r"].to_numpy(), d["m"].to_numpy(), ["spy"])
    return {"beta": float(res.coef[1]), "t_hac": float(res.t[1]), "p": float(res.p[1]),
            "corr": float(d["r"].corr(d["m"])), "alpha_daily": float(res.coef[0]), "alpha_t_hac": float(res.t[0]),
            "n": res.n}


def underlying_correlation(sym_rows: pd.DataFrame, closes: dict[str, pd.Series], min_days: int = 60) -> dict:
    out = []
    for sym, g in sym_rows.groupby("symbol"):
        if sym not in closes:
            continue
        pnl = g.groupby("date")["pnl"].sum()
        ret = closes[sym].pct_change().reindex(pnl.index)
        d = pd.concat([pnl, ret], axis=1).dropna()
        d = d[d.iloc[:, 0] != 0]
        if len(d) < min_days:
            continue
        out.append({"symbol": sym, "days": len(d), "corr": float(d.iloc[:, 0].corr(d.iloc[:, 1]))})
    t = pd.DataFrame(out)
    if t.empty:
        return {"names": 0}
    w = t["days"] / t["days"].sum()
    return {"names": len(t), "weighted_mean_corr": float((w * t["corr"]).sum()), "median_abs_corr": float(t["corr"].abs().median()),
            "share_abs_corr_over_0.2": float((t["corr"].abs() > 0.2).mean()), "table": t}


def regimes(nav: pd.Series) -> dict:
    r = returns(nav)
    m = market_returns(nav.index)
    pnl = nav.diff()
    spy_m = (1 + m.fillna(0)).resample("ME").prod() - 1
    pnl_m = pnl.resample("ME").sum()
    nav_m = nav.resample("ME").first()
    ret_m = (pnl_m / nav_m).dropna()
    lab = pd.cut(spy_m, [-np.inf, -0.02, 0.02, np.inf], labels=["down", "flat", "up"])
    out = {}
    for k in ("down", "flat", "up"):
        x = ret_m[lab.reindex(ret_m.index) == k]
        out[f"market_{k}"] = {"months": int(len(x)), "mean_monthly_return": float(x.mean()) if len(x) else np.nan,
                              "share_positive": float((x > 0).mean()) if len(x) else np.nan}
    vix = store.load_series("^VIX").reindex(r.index).ffill()
    med = vix.median()
    for k, mask in (("vix_high", vix > med), ("vix_low", vix <= med)):
        x = r[mask]
        out[k] = {"days": int(len(x)), "ann_return": float(x.mean() * 252), "sharpe": stats.sharpe(x)}
    for name, a, b in (("crash_2020", "2020-02-19", "2020-04-30"), ("year_2020", "2020-01-01", "2020-12-31"),
                       ("year_2022", "2022-01-01", "2022-12-31"), ("volmageddon_2018", "2018-01-26", "2018-02-28")):
        sub = nav[(nav.index >= a) & (nav.index <= b)]
        spy = (1 + m[(m.index >= a) & (m.index <= b)].fillna(0)).prod() - 1
        out[name] = {"return": float(sub.iloc[-1] / sub.iloc[0] - 1) if len(sub) > 1 else np.nan,
                     "max_drawdown": float((sub / sub.cummax() - 1).min()) if len(sub) > 1 else np.nan,
                     "spy_return": float(spy)}
    return out


def delta_profile(daily: pd.DataFrame) -> dict:
    x = daily["net_delta"] / daily["nav"]
    return {"mean_net_delta_pct_nav": float(x.mean()), "mean_abs_net_delta_pct_nav": float(x.abs().mean()),
            "p95_abs_net_delta_pct_nav": float(x.abs().quantile(0.95)), "max_abs_net_delta_pct_nav": float(x.abs().max())}


def hedge_cost(daily: pd.DataFrame) -> dict:
    yrs = len(daily) / 252
    nav = daily["nav"].mean()
    stock = daily["cost_stock"].sum()
    borrow = daily["cost_borrow"].sum()
    return {"stock_trading_per_year_pct_nav": stock / yrs / nav, "borrow_per_year_pct_nav": borrow / yrs / nav,
            "total_hedge_cost_per_year_pct_nav": (stock + borrow) / yrs / nav,
            "option_costs_per_year_pct_nav": (daily["cost_opt_comm"].sum() + daily["cost_opt_spread"].sum()) / yrs / nav,
            "hedge_shares_per_day": float(daily["hedge_shares"].mean())}
