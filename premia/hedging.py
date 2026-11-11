"""
premia/hedging.py — phase 4: the delta-hedging engine and tail protection (Taleb, *Dynamic Hedging*).

Rebalancing policies (per underlying, on the net share delta of options + stock):
  none          never hedge (reference: the plain, directional wheel / straddle)
  time:n        restore delta to zero every n trading days
  band          Whalley & Wilmott (1997) no-trade band: do nothing while |delta| <= H, otherwise trade back to the
                edge of the band (not to zero; trading to the edge is optimal under proportional costs). Half-width

                    H = m * ( 3/2 * exp(-r tau) * lam * S * Gamma^2 / gamma_ra )^(1/3)

                lam = proportional cost of trading the stock (half spread + commission, as a fraction of price),
                Gamma = net share gamma of the book, gamma_ra = risk aversion (per $, fixed at 1000 / NAV, which puts
                the m = 1 band at roughly a quarter of a typical single position's delta), m = band
                multiplier [walk-forward]. Taleb's point that hedge frequency trades cost against variance is the
                frontier that phase 4 measures; the band's cube-root form is Whalley & Wilmott's.

Hedge costs are charged as commissions + half spread on every share traded + borrow fee on short stock.

Tail protection: buy index puts until the portfolio's stress loss (spot gap x beta, vol up) is within the limit and
net volga is not below its floor (a short-premium book is short volga; protection buys it back).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import COSTS


@dataclass(frozen=True)
class HedgePolicy:
    kind: str = "band"           # none | time | band
    every: int = 1               # for kind == "time"
    band_mult: float = 1.0       # for kind == "band"

    @property
    def label(self) -> str:
        return {"none": "no hedge", "time": f"every {self.every}d", "band": f"WW band x{self.band_mult:g}"}[self.kind]


def stock_cost_frac(S: float) -> float:
    """Proportional cost of trading one share: half spread + commission per share."""
    return COSTS.stock_half_spread_bps / 1e4 + COSTS.stock_commission_per_share / max(S, 1e-6)


def ww_half_width(S: float, gamma_shares: float, tau: float, r: float, nav: float, mult: float) -> float:
    """Whalley-Wilmott band half-width in shares. gamma_shares = d(share delta)/dS of the whole book."""
    lam = stock_cost_frac(S)
    gamma_ra = 1000.0 / max(nav, 1.0)
    core = 1.5 * np.exp(-r * tau) * lam * S * gamma_shares ** 2 / gamma_ra
    return float(mult * np.cbrt(max(core, 0.0)))


def hedge_trade(policy: HedgePolicy, net_delta_shares: float, day_index: int, S: float, gamma_shares: float,
                tau: float, r: float, nav: float, max_delta_shares: float = np.inf) -> float:
    """Shares to trade (signed) today. The hard limit max_delta_shares always applies, whatever the policy."""
    d = net_delta_shares
    if policy.kind == "none":
        trade = 0.0
    elif policy.kind == "time":
        trade = -d if day_index % max(policy.every, 1) == 0 else 0.0
    else:
        H = ww_half_width(S, gamma_shares, tau, r, nav, policy.band_mult)
        trade = 0.0 if abs(d) <= H else -(d - np.sign(d) * H)
    after = d + trade
    if abs(after) > max_delta_shares:                       # rule layer: bring the residual inside the limit
        trade += -(after - np.sign(after) * max_delta_shares)
    return float(np.round(trade))


def stock_trade_cost(shares: float, S: float) -> float:
    if shares == 0:
        return 0.0
    commission = max(COSTS.stock_min_commission, COSTS.stock_commission_per_share * abs(shares))
    return float(commission + abs(shares) * S * COSTS.stock_half_spread_bps / 1e4)


def borrow_cost(short_shares: float, S: float, days: float) -> float:
    return float(max(-short_shares, 0.0) * S * COSTS.borrow_fee_annual * days / 365.0)


def tail_hedge_quantity(stress_loss: float, loss_limit: float, put_stress_gain: float, net_volga: float,
                        volga_floor: float, put_volga: float, max_contracts: int = 10_000) -> int:
    """Index puts to buy: the smallest whole number that brings the stress loss within the limit and net volga up to
    its floor (both are per-contract linear to first order). Zero if already within both."""
    need = 0.0
    if stress_loss > loss_limit and put_stress_gain > 0:
        need = max(need, (stress_loss - loss_limit) / put_stress_gain)
    if net_volga < volga_floor and put_volga > 0:
        need = max(need, (volga_floor - net_volga) / put_volga)
    return int(min(np.ceil(need), max_contracts))
