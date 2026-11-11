"""
premia/market.py — the modelled option market the backtester trades in (and the same pricing the live edge model uses).

Per underlying and day (all known at that day's close):
  * spot S (split-adjusted close), rate r (^IRX), dividend yield q (trailing 12 months)
  * ATM implied vol: IBKR 30-day IV (real). Term structure: flat for single stocks; for SPX-tracking ETFs scaled by
    the VIX9D / VIX / VIX3M / VIX6M ratios (real, interpolated in total variance)
  * smile: ATM vol x normalised SVI shape (premia.vol.SmileShape), shape fitted on current real chains (assumption)
  * quotes: mid from BSM at the smile vol; bid/ask from a spread model (half spread = max(tick floor, fraction x mid))
  * listed expiries: monthly third Fridays; strikes on the exchange grid for the price level

Forecasts (premia.rv, walk-forward) and the FHS state (premia.physical) are precomputed per underlying with
refits that only use data up to each refit date.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from . import data, pricing, rv, store
from .config import COSTS, RESULTS_DIR, TRADING_DAYS
from .physical import FHS, distribution
from .vol import SmileShape

INDEX_LIKE = {"SPY", "IVV", "VOO", "DIA", "QQQ", "IWM"}
SPX_TRACKERS = {"SPY", "IVV", "VOO"}
CACHE_DIR = RESULTS_DIR / "cache"


def is_etf(symbol: str) -> bool:
    from .config import ETFS
    return symbol in INDEX_LIKE or symbol in ETFS


def strike_step(S: float, etf: bool = False) -> float:
    if etf:
        return 1.0 if S < 500 else 5.0
    return 0.5 if S < 25 else 1.0 if S < 100 else 2.5 if S < 200 else 5.0


def third_friday(y: int, m: int) -> pd.Timestamp:
    d = pd.Timestamp(y, m, 15)
    return d + pd.Timedelta(days=(4 - d.weekday()) % 7)


def monthly_expiries(asof: pd.Timestamp, n: int = 4) -> list[pd.Timestamp]:
    out = []
    y, m = asof.year, asof.month
    while len(out) < n:
        f = third_friday(y, m)
        if f > asof:
            out.append(f)
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def year_frac(asof: pd.Timestamp, expiry: pd.Timestamp) -> float:
    return max((expiry - asof).days, 0) / 365.0


def add_index_term(u: "Underlying") -> "Underlying":
    """Price a single name's 3-6 month options off its 30-day IV times that day's SPX term-structure ratios."""
    if "ts_93" not in u.df.columns:
        u.df = u.df.join(_term_ratios(u.df.index))
    u.index_term = True
    return u


@dataclass
class SpreadModel:
    min_half: float
    frac: float
    max_frac: float = 0.25          # half spread never above 25 % of mid (beyond that, the quote is untradeable)

    def half(self, mid):
        mid = np.asarray(mid, float)
        return np.minimum(np.maximum(self.min_half, self.frac * mid), np.maximum(self.max_frac * mid, 0.01))


ETF_SPREAD = SpreadModel(min_half=0.01, frac=0.015)
STOCK_SPREAD = SpreadModel(min_half=COSTS.option_min_half_spread, frac=COSTS.option_half_spread_frac)


@dataclass
class Underlying:
    symbol: str
    df: pd.DataFrame                   # panel + forecast columns
    earnings: list
    shape: SmileShape
    etf: bool
    fhs_fits: dict = field(default_factory=dict)      # refit date -> (GarchFit, z)
    skew_mult: float = 1.0

    @property
    def spread(self) -> SpreadModel:
        return ETF_SPREAD if self.etf else STOCK_SPREAD

    def row(self, t: pd.Timestamp) -> Optional[pd.Series]:
        try:
            return self.df.loc[t]
        except KeyError:
            return None

    def atm_vol(self, t: pd.Timestamp, T: float) -> float:
        r = self.df.loc[t]
        base = float(r["iv"])
        if self.symbol in SPX_TRACKERS and np.isfinite(r.get("ts_9", np.nan)):
            days = np.array([9, 30, 93, 183]) / 365.0
            vols = np.array([r["ts_9"], 1.0, r["ts_93"], r["ts_183"]]) * base
            w = vols ** 2 * days
            ok = np.isfinite(w)
            wT = np.interp(T, days[ok], w[ok]) if T <= days[ok][-1] else w[ok][-1] * T / days[ok][-1]
            return float(np.sqrt(max(wT, 1e-8) / max(T, 1e-6)))
        if getattr(self, "index_term", False) and np.isfinite(r.get("ts_93", np.nan)) and np.isfinite(r.get("ts_183", np.nan)):
            # single name, longer-dated options: borrow that day's SPX term-structure ratios (VIX3M/VIX, VIX6M/VIX)
            days = np.array([30, 93, 183]) / 365.0
            w = (np.array([1.0, r["ts_93"], r["ts_183"]]) * base) ** 2 * days
            wT = np.interp(T, days, w) if T <= days[-1] else w[-1] * T / days[-1]
            return float(np.sqrt(max(wT, 1e-8) / max(T, 1e-6)))
        return base

    def vol(self, t: pd.Timestamp, K, T: float) -> np.ndarray:
        r = self.df.loc[t]
        F = r["close"] * np.exp((r["r"] - r["q"]) * T)
        atm = self.atm_vol(t, T)
        return self.shape.iv(np.log(np.asarray(K, float) / F), atm, T, self.skew_mult)

    def quote(self, t: pd.Timestamp, K, expiry: pd.Timestamp, right) -> dict:
        """Model quote: mid, bid, ask, vol and Greeks for strikes K (array) at one expiry."""
        r = self.df.loc[t]
        T = year_frac(t, expiry)
        K = np.atleast_1d(np.asarray(K, float))
        sig = self.vol(t, K, max(T, 1e-4))
        g = pricing.greeks(r["close"], K, max(T, 1e-4), r["r"], r["q"], sig, right)
        mid = np.maximum(g.price, 0.0)
        h = self.spread.half(mid)
        return {"K": K, "T": T, "iv": sig, "mid": mid, "bid": np.maximum(mid - h, 0.0), "ask": mid + h, "greeks": g}

    def strikes_near(self, t: pd.Timestamp, lo: float, hi: float) -> np.ndarray:
        S = float(self.df.at[t, "close"])
        step = strike_step(S, self.etf)
        return np.arange(np.floor(S * lo / step) * step, np.ceil(S * hi / step) * step + step / 2, step)

    def earnings_between(self, a: pd.Timestamp, b: pd.Timestamp) -> list:
        return [e for e in self.earnings if a < e <= b + pd.Timedelta(days=1)]

    def fhs_at(self, t: pd.Timestamp) -> Optional[FHS]:
        """GJR fit from the latest refit on or before t, filtered to t with that fit (no look-ahead)."""
        keys = [k for k in self.fhs_fits if k <= t]
        if not keys:
            return None
        k = max(keys)
        fit, z, mu = self.fhs_fits[k]
        r = np.log(self.df["close"]).diff()
        seg = r[(r.index > k - pd.Timedelta(days=400)) & (r.index <= t)].dropna().to_numpy() - mu
        from .rv import garch_filter
        s2 = garch_filter(seg, fit, np.var(seg) if len(seg) > 5 else fit.uncond)
        return FHS(fit, z, float(s2[-1]))


def _term_ratios(index: pd.DatetimeIndex) -> pd.DataFrame:
    vix = store.load_series("^VIX")
    out = pd.DataFrame(index=index)
    for col, name in (("ts_9", "^VIX9D"), ("ts_93", "^VIX3M"), ("ts_183", "^VIX6M")):
        s = store.load_series(name)
        ratio = (s / vix).reindex(index.union(s.index)).ffill(limit=3).reindex(index)
        out[col] = ratio
    return out


def prepare(symbol: str, df: pd.DataFrame, earnings: list, shape: SmileShape, h: int = 21,
            refit_fhs_days: int = 252, min_train: int = 750) -> Underlying:
    """Add walk-forward RV forecasts and yearly GJR refits to a raw panel (close, open, high, low, iv, r, q, div_cash).
    Every forecast and fit at t uses data up to t only."""
    df = df.copy()
    fc = rv.forecasts(df, h=h, iv=df["iv"], models=("naive", "har", "har_iv", "gjr"), min_train=min_train)
    for c in ("naive", "har", "har_iv", "gjr"):
        df[f"fc_{c}"] = fc[c]
    df["rv_target"] = fc["target"]
    if symbol in SPX_TRACKERS and "ts_9" not in df.columns:
        df = df.join(_term_ratios(df.index))
    u = Underlying(symbol, df, earnings, shape, etf=is_etf(symbol))
    r = np.log(df["close"]).diff()
    idx = df.index
    for i in range(min_train, len(idx), refit_fhs_days):
        t = idx[i]
        hist = r[:t].dropna().to_numpy()[-2520:]
        mu = float(hist.mean())
        fit = rv.fit_garch(hist, leverage=True)
        s2 = rv.garch_filter(hist - mu, fit, np.var(hist))
        z = (hist - mu) / np.sqrt(s2[:-1])
        z = (z - z.mean()) / z.std()
        u.fhs_fits[t] = (fit, z.astype(np.float32), mu)
    return u


def build_underlying(symbol: str, shape: SmileShape, start: str = "2007-01-01", use_cache: bool = True) -> Optional[Underlying]:
    """Panel from the research store + forecasts + GJR refits. Cached on disk (inputs are raw data only)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"u_{symbol}.pkl"
    if use_cache and cache.exists():
        u = pickle.loads(cache.read_bytes())
        u.shape = shape
        return u
    p = data.load_panel(symbol)
    if p is None or len(p.df) < 800:
        return None
    df = p.df[p.df.index >= pd.Timestamp(start)].copy()
    if df["iv"].notna().sum() < 500:
        return None
    u = prepare(symbol, df, p.earnings, shape)
    cache.write_bytes(pickle.dumps(u))
    return u
