"""
premia/backtest.py — phase 6: the event-driven daily backtester (one "sleeve" per run: wheel or straddle).

Timing (no look-ahead; tested):
  close of day t   mark every position with day-t data; settle expiries and assignments; pay dividends and borrow;
                   run the hedge engine (stock trades at the day-t close, which is observable); run the strategy
                   models on data up to t; check the risk layer; queue option orders
  close of t+1     queued option orders fill at the day-(t+1) model quotes: sell at mid - s x half spread, buy at
                   mid + s x half spread (s = 0.5 default, 1.0 in the cost-sensitivity run), plus commission

Accounting: cash + option marks (at mid) + stock x close = equity, every day (tested to reconcile). Costs are kept
in their own lines (option commission, option spread, stock commission + spread, borrow fee).

Assignment: at expiry every in-the-money option is exercised (physical delivery). Early assignment of a short American
option is assumed when it has almost no time value left (time value < max(0.02, K (1 - e^{-r tau})) for puts, the
standard early-exercise boundary approximation) or, for a short call, when tomorrow's dividend exceeds its time value
(McMillan; Hull).

P&L attribution (phase 3B): every option held over a day is decomposed into delta, gamma, vega, theta, vanna, volga
and residual; stock P&L (hedges and assigned shares) is its own "stock" line.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import pricing, risk, store
from .attribution import attribute_option
from .config import COSTS, LIMITS, PARAMS
from .hedging import HedgePolicy, borrow_cost, hedge_trade, stock_trade_cost, tail_hedge_quantity
from .market import Underlying, monthly_expiries, strike_step, year_frac
from . import strategies as st

MULT = 100


@dataclass
class OptPos:
    pid: int
    symbol: str
    right: str
    K: float
    expiry: pd.Timestamp
    qty: int                      # contracts, negative = short
    kind: str                     # csp | cc | straddle | tail
    group: int
    entry_date: pd.Timestamp
    entry_px: float
    entry_iv: float
    mark: float = 0.0
    iv: float = 0.0
    S: float = 0.0
    T: float = 0.0
    r: float = 0.0
    q: float = 0.0
    meta: dict = field(default_factory=dict)


@dataclass
class Order:
    symbol: str
    right: str
    K: float
    expiry: pd.Timestamp
    qty: int                      # signed contracts
    kind: str
    group: int
    close_pid: Optional[int] = None
    reason: str = ""
    meta: dict = field(default_factory=dict)


@dataclass
class RunConfig:
    sleeve: str = "wheel"                     # wheel | straddle
    policy: HedgePolicy = field(default_factory=HedgePolicy)
    lam: float = 0.1                          # wheel: CVaR aversion
    straddle_threshold: float = 0.02          # straddle: min sigma_f - IV
    use_model: bool = True                    # False = fixed-rule baseline
    slip: float = COSTS.slippage_frac_of_half_spread
    skew_mult: float = 1.0
    nav0: float = 1_000_000.0
    n_names: int = 15
    spy_weight: float = 0.15                  # wheel: cash-secured weight on SPY
    stock_weight: float = 0.025               # wheel: per name
    straddle_budget: float = 0.005            # straddle: premium per trade as a fraction of NAV
    straddle_max_total: float = 0.05
    tail_hedge: bool = True
    stress_gap: float = 0.20
    stress_vol_up: float = 0.10
    stress_limit: float = 0.10                # stress loss limit as a fraction of NAV
    manage_every: int = 5
    risk_premium_size_mult: float = 1.0       # phase 3B: size multiplier for trades classified as risk premium
    forecast_model: str = "har_iv"
    # McMillan straddle sleeve (premia/mcmillan.py holds the book's rules; these are the knobs varied in sensitivity)
    mm_budget: float = 0.005                  # premium per straddle as a fraction of NAV
    mm_one_contract_max: float = 0.015        # still buy one straddle if it costs up to this share of NAV
    mm_max_total: float = 0.10                # all long straddle premium together
    mm_hist_freq: float = 0.50                # criterion 3 threshold (the book says "frequently")
    mm_target_days: int = 182                 # expiry nearest this many days, at least 90 (criterion 1)
    mm_follow_up: bool = True                 # False: only the 60 % stop and the time exit
    label: str = ""


class Backtest:
    def __init__(self, cfg: RunConfig, underlyings: dict[str, Underlying], schedule: Callable[[pd.Timestamp], list[str]],
                 dates: pd.DatetimeIndex, sectors: Optional[dict] = None):
        self.cfg = cfg
        self.u = underlyings
        for u in self.u.values():
            u.skew_mult = cfg.skew_mult
        self.schedule = schedule
        self.dates = dates
        self.sectors = sectors or {}
        self.cash = cfg.nav0
        self.opts: dict[int, OptPos] = {}
        self.stock: dict[str, float] = {}
        self.basis: dict[str, float] = {}        # wheel: cost basis per share of assigned lots
        self.lots: dict[str, int] = {}           # wheel: assigned shares not yet called away
        self.pending: list[Order] = []
        self.kill = risk.KillSwitch(hwm=cfg.nav0)
        self._pid = 0
        self._grp = 0
        self.daily: list[dict] = []
        self.trades: list[dict] = []
        self.attrib: list[dict] = []
        self._last_eval: dict[tuple, pd.Timestamp] = {}
        self.blocked: list[dict] = []
        self.sym_rows: list[dict] = []           # (date, symbol, pnl) after costs: per-underlying P&L for correlation tests
        self._sym: dict[str, float] = {}
        self._mm: dict[int, dict] = {}           # McMillan sleeve: follow-up state per straddle group
        self.mm_signals: list[dict] = []         # McMillan sleeve: every name that passed criterion 1, with the rest

    def _book(self, sym: str, amount: float) -> None:
        self._sym[sym] = self._sym.get(sym, 0.0) + amount

    # ------------------------------------------------------------------ helpers
    def _row(self, sym: str, t: pd.Timestamp):
        u = self.u.get(sym)
        if u is None:
            return None
        try:
            return u.df.loc[t]
        except KeyError:
            return None

    def _price_opts(self, t: pd.Timestamp, positions: list[OptPos]):
        """Mid marks and Greeks for the given positions at t (vectorised per call)."""
        if not positions:
            return None
        S, K, T, r, q, sig, right = [], [], [], [], [], [], []
        for p in positions:
            row = self._row(p.symbol, t)
            u = self.u[p.symbol]
            if row is None or not np.isfinite(row["iv"]):
                S.append(p.S); r.append(p.r); q.append(p.q); sig.append(p.iv)        # stale: keep last inputs
            else:
                S.append(float(row["close"])); r.append(float(row["r"])); q.append(float(row["q"]))
                tau = year_frac(t, p.expiry)
                # on the last day vol no longer matters for the price; freezing it keeps the Taylor attribution valid
                sig.append(p.iv if tau < 1.5 / 365 and p.iv > 0 else float(u.vol(t, [p.K], max(tau, 1 / 365))[0]))
            K.append(p.K); T.append(max(year_frac(t, p.expiry), 1e-4)); right.append(p.right)
        g = pricing.greeks(np.array(S), np.array(K), np.array(T), np.array(r), np.array(q), np.array(sig), np.array(right))
        return {"S": np.array(S), "T": np.array(T), "r": np.array(r), "q": np.array(q), "iv": np.array(sig), "g": g}

    def equity(self, t: pd.Timestamp) -> float:
        val = self.cash
        for p in self.opts.values():
            val += p.mark * p.qty * MULT
        for sym, sh in self.stock.items():
            row = self._row(sym, t)
            if row is not None and sh:
                val += sh * float(row["close"])
        return val

    def _fill(self, o: Order, t: pd.Timestamp) -> Optional[tuple[float, float, float]]:
        """(fill price per share, mid, iv) for an order at t's model quote; None if the name cannot be quoted."""
        row = self._row(o.symbol, t)
        if row is None or not np.isfinite(row["iv"]):
            return None
        u = self.u[o.symbol]
        if o.expiry <= t:
            return None
        qt = u.quote(t, [o.K], o.expiry, o.right)
        mid, ask = float(qt["mid"][0]), float(qt["ask"][0])
        half = ask - mid
        px = mid + self.cfg.slip * half if o.qty > 0 else max(mid - self.cfg.slip * half, 0.0)
        return px, mid, float(qt["iv"][0])

    # ------------------------------------------------------------------ day steps
    def _execute_pending(self, t: pd.Timestamp, rec: dict) -> None:
        orders, self.pending = self.pending, []
        for o in orders:
            f = self._fill(o, t)
            if f is None:
                if o.close_pid is not None:
                    self.pending.append(o)                       # closing orders retry next day
                continue
            px, mid, iv = f
            n = abs(o.qty)
            comm = COSTS.option_commission * n
            spread_cost = abs(px - mid) * n * MULT
            self.cash -= px * o.qty * MULT + comm
            rec["cost_opt_comm"] += comm
            rec["cost_opt_spread"] += spread_cost
            rec[f"cost_{o.kind}"] = rec.get(f"cost_{o.kind}", 0.0) + comm + spread_cost
            self._book(o.symbol, -(comm + spread_cost))
            if o.close_pid is not None:
                p = self.opts.get(o.close_pid)
                if p is not None and abs(o.qty) < abs(p.qty):          # partial close: the rest stays on the book
                    p.qty += o.qty
                    self.trades.append({"date": t, "action": "close", "pid": p.pid, "symbol": p.symbol, "kind": p.kind,
                                        "right": p.right, "K": p.K, "expiry": p.expiry, "qty": o.qty, "px": px,
                                        "entry_px": p.entry_px, "entry_date": p.entry_date, "reason": o.reason,
                                        "pnl": (px - p.entry_px) * o.qty * MULT, "partial": True})
                    continue
                p = self.opts.pop(o.close_pid, None)
                if p is not None:
                    self.trades.append({"date": t, "action": "close", "pid": p.pid, "symbol": p.symbol, "kind": p.kind,
                                        "right": p.right, "K": p.K, "expiry": p.expiry, "qty": o.qty, "px": px,
                                        "entry_px": p.entry_px, "entry_date": p.entry_date, "reason": o.reason,
                                        "pnl": (px - p.entry_px) * o.qty * MULT})
                continue
            self._pid += 1
            p = OptPos(self._pid, o.symbol, o.right, o.K, o.expiry, o.qty, o.kind, o.group, t, px, iv, mark=mid, iv=iv,
                       meta=dict(o.meta))
            row = self._row(o.symbol, t)
            p.S, p.r, p.q, p.T = float(row["close"]), float(row["r"]), float(row["q"]), year_frac(t, o.expiry)
            self.opts[p.pid] = p
            self.trades.append({"date": t, "action": "open", "pid": p.pid, "symbol": p.symbol, "kind": p.kind,
                                "right": p.right, "K": p.K, "expiry": p.expiry, "qty": o.qty, "px": px, "iv": iv,
                                "reason": o.reason, **{f"m_{k}": v for k, v in o.meta.items()}})

    def _mark_and_attribute(self, t: pd.Timestamp, prev_t: Optional[pd.Timestamp], rec: dict) -> None:
        book = list(self.opts.values())
        pr = self._price_opts(t, book)
        if pr is None:
            return
        for i, p in enumerate(book):
            S1, iv1, T1, r1, q1 = pr["S"][i], pr["iv"][i], pr["T"][i], pr["r"][i], pr["q"][i]
            new_mark = float(max(pr["g"].price[i], 0.0))
            if p.entry_date < t and prev_t is not None and p.S > 0:
                dt = (t - prev_t).days / 365.0
                a = attribute_option(p.S, S1, p.iv, iv1, p.K, max(p.T, 1e-4), dt, p.r, p.q, p.right, p.qty)
                for k in ("delta", "gamma", "vega", "theta", "vanna", "volga", "residual"):
                    rec[f"a_{k}"] += a[k]
                rec[f"res_{p.kind}"] = rec.get(f"res_{p.kind}", 0.0) + a["residual"]
                chg = (new_mark - p.mark) * p.qty * MULT
                rec["a_total_opt"] += chg
                self._book(p.symbol, chg)
                g = pr["g"]
                self.attrib.append({"date": t, "pid": p.pid, "kind": p.kind, "symbol": p.symbol, **a,
                                    "moneyness": p.K / p.S, "dte": (p.expiry - prev_t).days,
                                    "vanna_usd": float(g.vanna[i]) * p.qty * MULT / 100 * S1,
                                    "volga_usd": float(g.volga[i]) * p.qty * MULT / 1e4,
                                    "vega_usd": float(g.vega[i]) * p.qty * MULT / 100})
            p.mark, p.iv, p.S, p.T, p.r, p.q = new_mark, iv1, S1, T1, r1, q1

    def _stock_pnl_and_carry(self, t: pd.Timestamp, prev_t: Optional[pd.Timestamp], rec: dict) -> None:
        for sym, sh in list(self.stock.items()):
            if not sh:
                continue
            row = self._row(sym, t)
            prow = self._row(sym, prev_t) if prev_t is not None else None
            if row is None:
                continue
            if prow is not None:
                rec["a_stock"] += sh * (float(row["close"]) - float(prow["close"]))
                self._book(sym, sh * (float(row["close"]) - float(prow["close"])))
            div = float(row.get("div_cash", 0.0) or 0.0)
            if div:
                self.cash += sh * div
                rec["dividends"] += sh * div
                self._book(sym, sh * div)
            if sh < 0 and prev_t is not None:
                c = borrow_cost(sh, float(row["close"]), (t - prev_t).days)
                self.cash -= c
                rec["cost_borrow"] += c
                self._book(sym, -c)

    def _settle(self, t: pd.Timestamp, rec: dict) -> None:
        for p in list(self.opts.values()):
            row = self._row(p.symbol, t)
            if row is None:
                if p.expiry < t - pd.Timedelta(days=5):              # name stopped trading: settle at last mark
                    self.cash += p.mark * p.qty * MULT
                    self.opts.pop(p.pid)
                continue
            S = float(row["close"])
            intrinsic = max(p.K - S, 0.0) if p.right == "P" else max(S - p.K, 0.0)
            expired = p.expiry <= t
            early = False
            if not expired and p.qty < 0 and intrinsic > 0:
                tau = year_frac(t, p.expiry)
                tv = p.mark - intrinsic
                if p.right == "P":
                    early = tv < max(0.02, p.K * (1 - math.exp(-float(row["r"]) * tau)))
                else:
                    nxt = self._next_div(p.symbol, t)
                    early = nxt > 0 and nxt > tv
            if not (expired or early):
                continue
            self.opts.pop(p.pid)
            if intrinsic > 0:
                sh = MULT * p.qty if p.right == "C" else -MULT * p.qty
                # long call / short put -> receive shares; short call / long put -> deliver shares
                self.stock[p.symbol] = self.stock.get(p.symbol, 0.0) + sh
                self.cash -= sh * p.K
                rec["assignments"] += 1 if p.qty < 0 else 0
                rec["early_assignments"] += 1 if early else 0
                if p.kind == "csp":
                    lots = self.lots.get(p.symbol, 0)
                    self.basis[p.symbol] = ((self.basis.get(p.symbol, 0.0) * lots + (p.K - p.entry_px) * sh) /
                                            max(lots + sh, 1))
                    self.lots[p.symbol] = lots + int(sh)
                elif p.kind == "cc":
                    self.lots[p.symbol] = max(self.lots.get(p.symbol, 0) + int(sh), 0)
            # the option leaves the book at intrinsic: its mark becomes the stock delivered (cash moved above)
            self.trades.append({"date": t, "action": "expire" if expired else "early_assign", "pid": p.pid,
                                "symbol": p.symbol, "kind": p.kind, "right": p.right, "K": p.K, "expiry": p.expiry,
                                "qty": -p.qty, "px": intrinsic, "entry_px": p.entry_px, "entry_date": p.entry_date,
                                "pnl": (intrinsic - p.entry_px) * p.qty * MULT})
            rec["a_total_opt"] += (intrinsic - p.mark) * p.qty * MULT
            self._book(p.symbol, (intrinsic - p.mark) * p.qty * MULT)
            p.mark = intrinsic

    def _next_div(self, sym: str, t: pd.Timestamp) -> float:
        df = self.u[sym].df
        i = df.index.searchsorted(t)
        return float(df["div_cash"].iloc[i + 1]) if i + 1 < len(df) else 0.0

    def _book_greeks(self, t: pd.Timestamp, sym: str) -> tuple[float, float, float]:
        """(net share delta, net share gamma, shortest time to expiry) for one underlying."""
        ps = [p for p in self.opts.values() if p.symbol == sym]
        d = self.stock.get(sym, 0.0)
        gam, tau = 0.0, 1.0
        if ps:
            pr = self._price_opts(t, ps)
            d += float(np.sum(pr["g"].delta * np.array([p.qty for p in ps]) * MULT))
            gam = float(np.sum(pr["g"].gamma * np.array([p.qty for p in ps]) * MULT))
            tau = float(np.min(pr["T"]))
        return d, gam, tau

    def _hedge(self, t: pd.Timestamp, i: int, nav: float, rec: dict) -> None:
        syms = set(self.stock) | {p.symbol for p in self.opts.values()}
        for sym in syms:
            row = self._row(sym, t)
            if row is None:
                continue
            S = float(row["close"])
            d, gam, tau = self._book_greeks(t, sym)
            has_opts = any(p.symbol == sym for p in self.opts.values())
            if not has_opts:
                # wheel shares without an option are a short-put equivalent only with a call; flat otherwise
                target_trade = -d if (self.cfg.policy.kind != "none" or self.cfg.sleeve == "straddle") else 0.0
                lots = self.lots.get(sym, 0) if self.cfg.sleeve == "wheel" else 0
                if self.cfg.policy.kind == "none":
                    target_trade = 0.0 if lots else -d
                trade = float(np.round(target_trade))
            else:
                max_sh = LIMITS.max_net_delta_pct_nav * nav / S / 2 if self.cfg.policy.kind != "none" else np.inf
                trade = hedge_trade(self.cfg.policy, d, i, S, gam, tau, float(row["r"]), nav, max_sh)
            if trade:
                c = stock_trade_cost(trade, S)
                self.cash -= trade * S + c
                self.stock[sym] = self.stock.get(sym, 0.0) + trade
                rec["cost_stock"] += c
                rec["hedge_shares"] += abs(trade)
                self._book(sym, -c)

    # ------------------------------------------------------------------ exposures and risk
    def exposures(self, t: pd.Timestamp, betas: dict) -> risk.Exposures:
        e = risk.Exposures()
        ps = list(self.opts.values())
        pr = self._price_opts(t, ps) if ps else None
        for i, p in enumerate(ps):
            g = pr["g"]
            n = p.qty * MULT
            S = pr["S"][i]
            b = betas.get(p.symbol, 1.0)
            e.net_delta += g.delta[i] * n * S * b
            e.net_vega += g.vega[i] * n / 100
            e.net_vanna += g.vanna[i] * n / 100 * S
            e.net_volga += g.volga[i] * n / 1e4
            if p.qty < 0 and p.right == "P":
                e.margin += p.K * -n
                e.by_name[p.symbol] = e.by_name.get(p.symbol, 0.0) + p.K * -n
                sec = self.sectors.get(p.symbol, "other")
                e.by_sector[sec] = e.by_sector.get(sec, 0.0) + p.K * -n
        for sym, sh in self.stock.items():
            row = self._row(sym, t)
            if row is None or not sh:
                continue
            S = float(row["close"])
            e.net_delta += sh * S * betas.get(sym, 1.0)
            e.margin += 0.3 * abs(sh) * S
        return e

    def _impact(self, t: pd.Timestamp, c_sym: str, right: str, K: float, expiry: pd.Timestamp, qty: int, hedged: bool,
                betas: dict) -> risk.Exposures:
        u = self.u[c_sym]
        row = u.df.loc[t]
        T = year_frac(t, expiry)
        sig = float(u.vol(t, [K], T)[0])
        g = pricing.greeks(float(row["close"]), [K], T, float(row["r"]), float(row["q"]), [sig], right)
        n = qty * MULT
        S = float(row["close"])
        e = risk.Exposures(net_delta=0.0 if hedged else float(g.delta[0]) * n * S * betas.get(c_sym, 1.0),
                           net_vega=float(g.vega[0]) * n / 100, net_vanna=float(g.vanna[0]) * n / 100 * S,
                           net_volga=float(g.volga[0]) * n / 1e4)
        if qty < 0 and right == "P":
            e.margin = K * -n
            e.by_name = {c_sym: K * -n}
            e.by_sector = {self.sectors.get(c_sym, "other"): K * -n}
        return e

    # ------------------------------------------------------------------ strategies
    def _wheel_decisions(self, t: pd.Timestamp, i: int, nav: float, betas: dict) -> None:
        cfg = self.cfg
        hedged = cfg.policy.kind != "none"
        active = self.schedule(t)
        names = (["SPY"] if "SPY" in self.u else []) + [s for s in active if s in self.u and s != "SPY"][: cfg.n_names]
        open_by = {}
        for p in self.opts.values():
            if p.kind in ("csp", "cc"):
                open_by.setdefault(p.symbol, []).append(p)
        pending_syms = {o.symbol for o in self.pending}
        exp = self.exposures(t, betas)
        for sym in set(names) | set(open_by) | {s for s, l in self.lots.items() if l > 0}:
            if sym in pending_syms:
                continue
            u = self.u.get(sym)
            row = self._row(sym, t)
            if u is None or row is None or not np.isfinite(row["iv"]):
                continue
            # manage open short options
            for p in open_by.get(sym, []):
                loss = (p.mark - p.entry_px) * -p.qty * MULT
                if loss > LIMITS.max_loss_per_position_pct_nav * nav:
                    self.pending.append(Order(sym, p.right, p.K, p.expiry, -p.qty, p.kind, p.group, p.pid, "max loss rule"))
                    continue
                if cfg.use_model and (i % cfg.manage_every == 0 or u.earnings_between(t, p.expiry)):
                    close, why = st.hold_or_close(u, t, p.right, p.K, p.expiry, cfg.lam, hedged, cfg.slip,
                                                  model=cfg.forecast_model)
                    if close:
                        self.pending.append(Order(sym, p.right, p.K, p.expiry, -p.qty, p.kind, p.group, p.pid, why))
                elif not cfg.use_model and u.earnings_between(t, p.expiry):
                    self.pending.append(Order(sym, p.right, p.K, p.expiry, -p.qty, p.kind, p.group, p.pid, "earnings"))
            if open_by.get(sym) or self.kill.engaged:
                continue
            lots = self.lots.get(sym, 0) if self.cfg.sleeve == "wheel" else 0
            if lots > 0 and self.stock.get(sym, 0) == 0 and not hedged:
                self.lots[sym] = 0
                lots = 0
            if sym not in names and lots <= 0:
                continue
            key = (sym, "eval")
            last = self._last_eval.get(key)
            if last is not None and (t - last).days < 5:
                continue
            self._last_eval[key] = t
            right = "C" if lots > 0 else "P"
            min_k = self.basis.get(sym, 0.0) if right == "C" else 0.0
            if cfg.use_model:
                cands = st.wheel_candidates(u, t, right, cfg.lam, hedged, min_k, slip=cfg.slip, model=cfg.forecast_model,
                                            seed=i)
                c = st.best_candidate(cands)
            else:
                c = st.baseline_candidate(u, t, right, min_k, slip=cfg.slip)
            if c is None:
                continue
            if right == "P":
                weight = (cfg.spy_weight if sym == "SPY" else cfg.stock_weight) * cfg.risk_premium_size_mult
                n = int(round(weight * nav / (c.K * MULT)))
                if n == 0 and c.K * MULT <= LIMITS.max_name_weight * nav * 0.5:
                    n = 1                                        # one contract when the name is pricier than its slot
            else:
                n = int(lots // MULT)
            if n <= 0:
                continue
            imp = self._impact(t, sym, right, c.K, c.expiry, -n, hedged, betas)
            ok, why = risk.check_open(exp, imp, nav) if right == "P" else (True, [])
            if not ok:
                self.blocked.append({"date": t, "symbol": sym, "reasons": why})
                continue
            exp = exp.plus(imp)
            self._grp += 1
            self.pending.append(Order(sym, right, c.K, c.expiry, -n, "csp" if right == "P" else "cc", self._grp,
                                      reason="model" if cfg.use_model else "baseline",
                                      meta={"edge": c.edge, "cvar": c.cvar, "fair": c.fair, "p_itm": c.p_itm,
                                            "score": c.score, "delta": c.delta}))

    def _straddle_decisions(self, t: pd.Timestamp, i: int, nav: float, betas: dict) -> None:
        cfg = self.cfg
        groups: dict[int, list[OptPos]] = {}
        for p in self.opts.values():
            if p.kind == "straddle":
                groups.setdefault(p.group, []).append(p)
        pending_syms = {o.symbol for o in self.pending}
        for g, legs in groups.items():
            p0 = legs[0]
            u = self.u[p0.symbol]
            if self._row(p0.symbol, t) is None:
                continue
            close, why = st.straddle_exit(u, t, p0.K, p0.expiry, p0.entry_iv, p0.meta.get("event", False),
                                          model=cfg.forecast_model) if cfg.use_model else (
                (p0.expiry - t).days <= 7, "time stop")
            if close and p0.symbol not in pending_syms:
                for p in legs:
                    self.pending.append(Order(p.symbol, p.right, p.K, p.expiry, -p.qty, "straddle", g, p.pid, why))
        if self.kill.engaged or i % 5 != 0 and not cfg.use_model:
            return
        held = {p.symbol for p in self.opts.values() if p.kind == "straddle"}
        prem = sum(p.mark * p.qty * MULT for p in self.opts.values() if p.kind == "straddle")
        names = [s for s in (["SPY", "QQQ", "IWM"] + self.schedule(t)) if s in self.u][: cfg.n_names + 3]
        exp = self.exposures(t, betas)
        for sym in names:
            if sym in held or sym in pending_syms or prem > cfg.straddle_max_total * nav:
                continue
            u = self.u[sym]
            row = self._row(sym, t)
            if row is None or not np.isfinite(row["iv"]):
                continue
            key = (sym, "straddle")
            last = self._last_eval.get(key)
            if last is not None and (t - last).days < 5:
                continue
            self._last_eval[key] = t
            if cfg.use_model:
                idea = st.straddle_idea(u, t, cfg.straddle_threshold, cfg.slip, model=cfg.forecast_model)
            else:
                idea = self._baseline_straddle(u, t)
            if idea is None:
                continue
            n = int(cfg.straddle_budget * nav // (idea.fill * MULT)) if idea.fill > 0 else 0
            if n <= 0:
                continue
            imp = self._impact(t, sym, "C", idea.K, idea.expiry, n, True, betas).plus(
                self._impact(t, sym, "P", idea.K, idea.expiry, n, True, betas))
            ok, why = risk.check_open(exp, imp, nav)
            if not ok:
                self.blocked.append({"date": t, "symbol": sym, "reasons": why})
                continue
            exp = exp.plus(imp)
            self._grp += 1
            meta = {"event": idea.event, "edge_vol": idea.edge_vol, "edge": idea.edge}
            for right in ("C", "P"):
                self.pending.append(Order(sym, right, idea.K, idea.expiry, n, "straddle", self._grp, reason="model" if cfg.use_model else "baseline", meta=meta))
            prem += idea.fill * n * MULT

    def _baseline_straddle(self, u: Underlying, t: pd.Timestamp) -> Optional[st.StraddleIdea]:
        """Fixed-rule baseline for the model straddle sleeve: buy a 21-45 day straddle when IV is in the bottom 30 % of
        its own 1-year history. An IV-rank rule, NOT McMillan's method (that is sleeve="mcmillan", premia/mcmillan.py)."""
        iv = u.df["iv"][:t].dropna().iloc[-252:]
        if len(iv) < 200 or (iv < iv.iloc[-1]).mean() > 0.30:
            return None
        idea = st.straddle_idea(u, t, threshold=-1.0, slip=self.cfg.slip)
        if idea is None:
            row = u.df.loc[t]
            exps = [e for e in monthly_expiries(t, 3) if 21 <= (e - t).days <= 45]
            if not exps:
                return None
            S = float(row["close"])
            K = float(u.strikes_near(t, 0.95, 1.05)[np.argmin(np.abs(u.strikes_near(t, 0.95, 1.05) - S))])
            c, p = u.quote(t, [K], exps[0], "C"), u.quote(t, [K], exps[0], "P")
            mid = float(c["mid"][0] + p["mid"][0])
            fill = mid + self.cfg.slip * (float(c["ask"][0] + p["ask"][0]) - mid)
            return st.StraddleIdea(u.symbol, K, exps[0], year_frac(t, exps[0]), float(c["iv"][0]), fill, fill, np.nan,
                                   np.nan, np.nan, bool(u.earnings_between(t, exps[0])))
        return idea

    def _mcmillan_decisions(self, t: pd.Timestamp, i: int, nav: float, betas: dict) -> None:
        """McMillan's straddle buying (premia/mcmillan.py): follow-up on open straddles, then new entries where all four
        criteria hold. Decided on today's close; orders fill on the next bar. No delta hedging (the book rejects it)."""
        from . import mcmillan as mm
        cfg = self.cfg
        pending_syms = {o.symbol for o in self.pending}
        groups: dict[int, list[OptPos]] = {}
        for p in self.opts.values():
            if p.kind == "mcm":
                groups.setdefault(p.group, []).append(p)
        for g, legs in groups.items():
            sym = legs[0].symbol
            row = self._row(sym, t)
            if sym in pending_syms or row is None:
                continue
            state = self._mm.setdefault(g, {"mode": "both"})
            calls = [p for p in legs if p.right == "C"]
            puts = [p for p in legs if p.right == "P"]
            if "cost" not in state:
                state["cost"] = sum(p.entry_px for p in legs)              # one call + one put, filled the same day
            value = (calls[0].mark if calls else 0.0) + (puts[0].mark if puts else 0.0)
            expiry, K = legs[0].expiry, legs[0].K
            close = float(row["close"])
            days_left = (expiry - t).days
            tdl = int(np.busday_count(t.date(), expiry.date()))
            n_c, n_p = (calls[0].qty if calls else 0), (puts[0].qty if puts else 0)
            if cfg.mm_follow_up:
                f = mm.follow_up(state["mode"], n_c, n_p, close, float(row.get("mm_sma", np.nan)), value, state["cost"],
                                 K, days_left, tdl)
            elif value <= (1 - mm.RISK_FRAC) * state["cost"] or days_left <= mm.TIME_EXIT_DAYS:
                f = mm.FollowUp(n_c, n_p, "closed", "60 % stop" if value <= (1 - mm.RISK_FRAC) * state["cost"] else "time")
            else:
                f = mm.FollowUp(0, 0, state["mode"])
            for legs_, n in ((calls, f.sell_calls), (puts, f.sell_puts)):
                if n > 0 and legs_:
                    p = legs_[0]
                    self.pending.append(Order(sym, p.right, p.K, p.expiry, -n, "mcm", g, p.pid, f.reason))
            state["mode"] = f.mode
        if self.kill.engaged:
            return

        held = {p.symbol for p in self.opts.values() if p.kind == "mcm"} | {o.symbol for o in self.pending}
        prem = sum(p.mark * p.qty * MULT for p in self.opts.values() if p.kind == "mcm")
        cands = []
        for sym in self.schedule(t):
            u = self.u.get(sym)
            row = self._row(sym, t)
            if sym in held or u is None or row is None or not np.isfinite(row["iv"]):
                continue
            pct = float(row.get("mm_ivpct", np.nan))
            if not np.isfinite(pct) or pct > mm.IV_PCT_MAX:
                continue
            exps = [e for e in monthly_expiries(t, 10) if (e - t).days >= mm.MIN_LIFE_DAYS]
            if not exps:
                continue
            expiry = min(exps, key=lambda e: abs((e - t).days - cfg.mm_target_days))
            S = float(row["close"])
            Ks = u.strikes_near(t, 0.9, 1.1)
            K = float(Ks[np.argmin(np.abs(Ks - S))])
            qs = [u.quote(t, [K], expiry, r) for r in ("C", "P")]
            fill = sum(float(q["mid"][0]) + cfg.slip * float(q["ask"][0] - q["mid"][0]) for q in qs)
            if fill <= 0:
                continue
            tdays = int(np.busday_count(t.date(), expiry.date()))
            crit = mm.evaluate(S, K, fill, (expiry - t).days, tdays, pct, float(row.get("mm_minhv", np.nan)), u.df,
                               u.df.index.get_loc(t), hist_freq_min=cfg.mm_hist_freq)
            # the dashboard's liquidity limit on the straddle (open interest has no history here, so it is not tested)
            spread = mm.straddle_spread_frac(sum(float(q["bid"][0]) for q in qs), sum(float(q["ask"][0]) for q in qs))
            if spread > LIMITS.max_option_spread_frac:
                crit.reasons.append(f"limit: straddle spread {spread:.0%} of mid > {LIMITS.max_option_spread_frac:.0%}")
            sig = {"date": t, "symbol": sym, "S": S, "K": K, "expiry": expiry, "cost": fill, "iv": float(row["iv"]),
                   "iv_pct": pct, "min_hv": crit.min_hv, "ever_prob": crit.ever_prob, "hist_freq": crit.hist_freq,
                   "pinned": crit.pinned, "spread": spread, "passes": crit.passes, "reasons": "; ".join(crit.reasons),
                   "taken": False}
            self.mm_signals.append(sig)
            if crit.passes:
                cands.append((crit.ever_prob, sym, K, expiry, fill, sig))
        if not cands:
            return
        exp = self.exposures(t, betas)
        for ever, sym, K, expiry, fill, sig in sorted(cands, key=lambda c: -c[0]):     # best "ever" first, as on the page
            n, why = mm.size(fill, nav, prem, cfg.mm_budget, cfg.mm_one_contract_max, cfg.mm_max_total)
            if n <= 0:
                sig["reasons"] = why
                continue
            imp = self._impact(t, sym, "C", K, expiry, n, False, betas).plus(
                self._impact(t, sym, "P", K, expiry, n, False, betas))
            ok, why = risk.check_open(exp, imp, nav)
            if not ok:
                self.blocked.append({"date": t, "symbol": sym, "reasons": why})
                continue
            exp = exp.plus(imp)
            self._grp += 1
            meta = {"ever_prob": ever, "hist_freq": sig["hist_freq"], "iv_pct": sig["iv_pct"]}
            for right in ("C", "P"):
                self.pending.append(Order(sym, right, K, expiry, n, "mcm", self._grp, reason="mcmillan", meta=meta))
            prem += fill * n * MULT
            sig["taken"] = True

    def _tail_hedge(self, t: pd.Timestamp, nav: float, betas: dict) -> None:
        if "SPY" not in self.u or self._row("SPY", t) is None or self.kill.engaged:
            return
        if any(o.kind == "tail" for o in self.pending):
            return
        loss = self.stress_loss(t, betas)
        e = self.exposures(t, betas)
        spy = self.u["SPY"]
        row = spy.df.loc[t]
        S = float(row["close"])
        exps = [x for x in monthly_expiries(t, 4) if 40 <= (x - t).days <= 80]
        if not exps:
            return
        expiry = exps[0]
        T = year_frac(t, expiry)
        K = float(np.floor(S * 0.90 / strike_step(S, True)) * strike_step(S, True))
        sig0 = float(spy.vol(t, [K], T)[0])
        g = pricing.greeks(S, [K], T, float(row["r"]), float(row["q"]), [sig0], "P")
        S1 = S * (1 - self.cfg.stress_gap)
        sig1 = float(spy.shape.iv(np.log(K / (S1 * np.exp((row["r"] - row["q"]) * T))), spy.atm_vol(t, T) + self.cfg.stress_vol_up * 2, T, spy.skew_mult))
        gain = (float(pricing.price(S1, K, T, row["r"], row["q"], sig1, "P")) - float(g.price[0])) * MULT
        qty = tail_hedge_quantity(loss, self.cfg.stress_limit * nav, gain, e.net_volga,
                                  -LIMITS.max_net_volga_pct_nav * nav, float(g.volga[0]) * MULT / 1e4)
        if qty > 0:
            self._grp += 1
            self.pending.append(Order("SPY", "P", K, expiry, qty, "tail", self._grp, reason=f"stress loss {loss:,.0f}"))

    def stress_loss(self, t: pd.Timestamp, betas: dict) -> float:
        """Loss of the whole book if every name gaps down beta x stress_gap and ATM vols rise (full revaluation)."""
        loss = 0.0
        for p in self.opts.values():
            row = self._row(p.symbol, t)
            if row is None:
                continue
            u = self.u[p.symbol]
            S0 = float(row["close"])
            S1 = S0 * (1 - self.cfg.stress_gap * betas.get(p.symbol, 1.0))
            T = max(year_frac(t, p.expiry), 1e-4)
            F1 = S1 * np.exp((row["r"] - row["q"]) * T)
            atm1 = u.atm_vol(t, T) + self.cfg.stress_vol_up * (2.0 if u.etf else 1.5)
            sig1 = float(u.shape.iv(np.log(p.K / F1), atm1, T, u.skew_mult))
            v1 = float(pricing.price(S1, p.K, T, row["r"], row["q"], sig1, p.right))
            loss -= (v1 - p.mark) * p.qty * MULT
        for sym, sh in self.stock.items():
            row = self._row(sym, t)
            if row is None or not sh:
                continue
            loss -= sh * float(row["close"]) * (-self.cfg.stress_gap * betas.get(sym, 1.0))
        return loss

    def _flatten(self, t: pd.Timestamp) -> None:
        for p in self.opts.values():
            if not any(o.close_pid == p.pid for o in self.pending):
                self.pending.append(Order(p.symbol, p.right, p.K, p.expiry, -p.qty, p.kind, p.group, p.pid, "kill switch"))

    # ------------------------------------------------------------------ main loop
    def run(self, betas_fn: Callable[[pd.Timestamp], dict]) -> pd.DataFrame:
        prev_t, prev_nav = None, self.cfg.nav0
        for i, t in enumerate(self.dates):
            rec = {k: 0.0 for k in ("cost_opt_comm", "cost_opt_spread", "cost_stock", "cost_borrow", "dividends",
                                    "a_delta", "a_gamma", "a_vega", "a_theta", "a_vanna", "a_volga", "a_residual",
                                    "a_total_opt", "a_stock", "hedge_shares", "assignments", "early_assignments")}
            rec["date"] = t
            self._mark_and_attribute(t, prev_t, rec)
            self._execute_pending(t, rec)
            self._stock_pnl_and_carry(t, prev_t, rec)
            self._settle(t, rec)
            nav = self.equity(t)
            betas = betas_fn(t)
            self._hedge(t, i, nav, rec)
            nav = self.equity(t)
            tripped = self.kill.update(nav, prev_nav, str(t.date()))
            if tripped:
                self._flatten(t)
                rec["kill"] = self.kill.reason
            elif self.kill.engaged:
                if not self.opts and not self.pending:
                    for sym, sh in list(self.stock.items()):         # flat in stock as well while the switch is on
                        row = self._row(sym, t)
                        if sh and row is not None:
                            c = stock_trade_cost(-sh, float(row["close"]))
                            self.cash += sh * float(row["close"]) - c
                            rec["cost_stock"] += c
                            self._book(sym, -c)
                            self.stock[sym] = 0.0
                            self.lots[sym] = 0
                self.kill.maybe_release_after_cooldown(nav)
            else:
                if self.cfg.sleeve == "wheel":
                    self._wheel_decisions(t, i, nav, betas)
                    if self.cfg.tail_hedge and i % 5 == 0:
                        self._tail_hedge(t, nav, betas)
                elif self.cfg.sleeve == "mcmillan":
                    self._mcmillan_decisions(t, i, nav, betas)
                else:
                    self._straddle_decisions(t, i, nav, betas)
            e = self.exposures(t, betas)
            rec.update({"nav": nav, "cash": self.cash, "net_delta": e.net_delta, "net_vega": e.net_vega,
                        "net_vanna": e.net_vanna, "net_volga": e.net_volga, "margin": e.margin,
                        "n_opts": len(self.opts), "kill_on": self.kill.engaged})
            self.daily.append(rec)
            for sym, v in self._sym.items():
                self.sym_rows.append({"date": t, "symbol": sym, "pnl": v})
            self._sym = {}
            prev_t, prev_nav = t, nav
        return pd.DataFrame(self.daily).set_index("date")


# ---------------------------------------------------------------------------
# inputs: betas and the universe schedule
# ---------------------------------------------------------------------------

def rolling_betas(underlyings: dict[str, Underlying], window: int = 252) -> pd.DataFrame:
    """Trailing beta of each name to SPY (known at t), clipped to [0, 3]."""
    spy = np.log(underlyings["SPY"].df["close"]).diff()
    out = {}
    for s, u in underlyings.items():
        r = np.log(u.df["close"]).diff().reindex(spy.index)
        cov = r.rolling(window, min_periods=120).cov(spy)
        var = spy.rolling(window, min_periods=120).var()
        out[s] = (cov / var).clip(0, 3)
    return pd.DataFrame(out).ffill()


def schedule_from_screen(screen: pd.DataFrame, use_score: bool) -> Callable[[pd.Timestamp], list[str]]:
    """Names passing the PIT screen at the latest quarter-end strictly before t, ranked by safety (if the score was
    kept) or by dollar volume."""
    s = screen[screen["pass_all"]].copy()
    s["asof"] = pd.to_datetime(s["asof"])
    by_q = {}
    for q, g in s.groupby("asof"):
        g = g.sort_values("safety" if use_score else "dollar_vol", ascending=False)
        by_q[q] = list(g["symbol"])
    qs = sorted(by_q)

    def f(t: pd.Timestamp) -> list[str]:
        i = np.searchsorted(np.array(qs, dtype="datetime64[ns]"), np.datetime64(t)) - 1
        return by_q[qs[i]] if i >= 0 else []
    return f
