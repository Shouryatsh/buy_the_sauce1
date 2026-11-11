"""
premia/risk.py — phase 7: the hard limits and the kill switch. Models propose; this module disposes.

Every proposed opening trade is checked against the portfolio *after* the trade; any breach blocks it. Closing and
risk-reducing trades are always allowed. The kill switch trips on the daily-loss or drawdown limit and then blocks
every opening trade until it is released (by a person in live/paper trading; after a fixed cooldown in backtests,
which is a modelling assumption documented in RESULTS.md).

Limits (config.Limits, all relative to NAV):
  net dollar delta, net vega, net vanna, net volga, margin use, per-name and per-sector weight,
  max loss per position, max daily loss (-> kill), max drawdown (-> kill), option liquidity (open interest, spread)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .config import LIMITS, Limits


@dataclass
class Exposures:
    """Portfolio exposures in dollars (vega/vanna/volga per 1 vol point)."""
    net_delta: float = 0.0            # $ delta, beta-weighted to the market proxy
    net_vega: float = 0.0
    net_vanna: float = 0.0            # $ delta change per vol point
    net_volga: float = 0.0            # $ vega change per vol point
    margin: float = 0.0               # cash-secured / margin requirement
    by_name: dict = field(default_factory=dict)       # name -> short-premium notional
    by_sector: dict = field(default_factory=dict)

    def plus(self, o: "Exposures") -> "Exposures":
        names = dict(self.by_name)
        for k, v in o.by_name.items():
            names[k] = names.get(k, 0.0) + v
        secs = dict(self.by_sector)
        for k, v in o.by_sector.items():
            secs[k] = secs.get(k, 0.0) + v
        return Exposures(self.net_delta + o.net_delta, self.net_vega + o.net_vega, self.net_vanna + o.net_vanna,
                         self.net_volga + o.net_volga, self.margin + o.margin, names, secs)


def breaches(e: Exposures, nav: float, lim: Limits = LIMITS, check_greeks: bool = True) -> dict[str, str]:
    """{limit key: message} for every limit the exposures breach."""
    out: dict[str, str] = {}
    if check_greeks:
        for key, val, cap, fmt in (("net_delta", e.net_delta, lim.max_net_delta_pct_nav, ".0%"),
                                   ("net_vega", e.net_vega, lim.max_net_vega_pct_nav, ".2%"),
                                   ("net_vanna", e.net_vanna, lim.max_net_vanna_pct_nav, ".2%"),
                                   ("net_volga", e.net_volga, lim.max_net_volga_pct_nav, ".2%")):
            if abs(val) > cap * nav:
                out[key] = f"{key} {val:,.0f} > {cap:{fmt}} NAV"
    if e.margin > lim.max_margin_use * nav:
        out["margin"] = f"margin {e.margin:,.0f} > {lim.max_margin_use:.0%} NAV"
    for k, v in e.by_name.items():
        if abs(v) > lim.max_name_weight * nav:
            out[f"name:{k}"] = f"{k} notional {v:,.0f} > {lim.max_name_weight:.0%} NAV"
    gross = sum(abs(v) for v in e.by_name.values())
    for k, v in e.by_sector.items():
        if gross > 0 and abs(v) / gross > lim.max_sector_weight and abs(v) > lim.max_name_weight * nav:
            out[f"sector:{k}"] = f"sector {k} {abs(v) / gross:.0%} of short-premium notional > {lim.max_sector_weight:.0%}"
    return out


def _size(e: Exposures, key: str) -> float:
    if key.startswith("name:"):
        return abs(e.by_name.get(key[5:], 0.0))
    if key.startswith("sector:"):
        return abs(e.by_sector.get(key[7:], 0.0))
    return abs(getattr(e, key))


def check_open(before: Exposures, impact: Exposures, nav: float, lim: Limits = LIMITS) -> tuple[bool, list[str]]:
    """An opening trade is allowed only if it creates no new breach and does not worsen an existing one.
    The trade's impact must include its delta hedge (the hedge engine runs before the limit check)."""
    after = before.plus(impact)
    b, a = breaches(before, nav, lim), breaches(after, nav, lim)
    reasons = [msg for key, msg in a.items() if key not in b or _size(after, key) > _size(before, key) + 1e-9]
    return (not reasons), reasons


def liquidity_ok(open_interest: Optional[float], bid: Optional[float], ask: Optional[float],
                 lim: Limits = LIMITS) -> tuple[bool, str]:
    """Live/paper only: a quote must exist on both sides, with enough open interest and a tight enough spread."""
    if bid is None or ask is None or not (bid > 0 and ask >= bid):
        return False, "no two-sided quote"
    mid = 0.5 * (bid + ask)
    if (ask - bid) / mid > lim.max_option_spread_frac:
        return False, f"spread {(ask - bid) / mid:.0%} of mid > {lim.max_option_spread_frac:.0%}"
    if open_interest is None:
        return False, "open interest unknown"
    if open_interest < lim.min_option_open_interest:
        return False, f"open interest {open_interest:.0f} < {lim.min_option_open_interest}"
    return True, ""


@dataclass
class KillSwitch:
    engaged: bool = False
    reason: str = ""
    since: Optional[str] = None
    hwm: float = 0.0
    cooldown_days: int = 21            # backtests only: days flat before trading resumes
    _days_engaged: int = 0

    def update(self, nav: float, prev_nav: float, date: str, lim: Limits = LIMITS) -> bool:
        """Call once per day after marking. Returns True when the switch trips today."""
        self.hwm = max(self.hwm, nav)
        if self.engaged:
            self._days_engaged += 1
            return False
        day_loss = (prev_nav - nav) / prev_nav if prev_nav > 0 else 0.0
        dd = (self.hwm - nav) / self.hwm if self.hwm > 0 else 0.0
        if day_loss > lim.max_daily_loss_pct_nav:
            self._trip(f"daily loss {day_loss:.1%} > {lim.max_daily_loss_pct_nav:.0%}", date)
            return True
        if dd > lim.max_drawdown_kill_pct:
            self._trip(f"drawdown {dd:.1%} > {lim.max_drawdown_kill_pct:.0%}", date)
            return True
        return False

    def _trip(self, reason: str, date: str) -> None:
        self.engaged, self.reason, self.since, self._days_engaged = True, reason, date, 0

    def maybe_release_after_cooldown(self, nav: float) -> bool:
        """Backtest stand-in for a human reset: release after the cooldown, with the high-water mark reset to NAV."""
        if self.engaged and self._days_engaged >= self.cooldown_days:
            self.engaged, self.reason, self.since = False, "", None
            self.hwm = nav
            return True
        return False

    def release(self, nav: float) -> None:
        self.engaged, self.reason, self.since = False, "", None
        self.hwm = nav
