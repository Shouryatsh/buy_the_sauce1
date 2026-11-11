# McMillan straddle buying: specification and pre-registered test

Written 2026-11-10, before any backtest of this strategy was run. Source: Lawrence G. McMillan, *McMillan on
Options*, ch. 7 "Buy Low and Sell High — Volatility, That Is", "My Favorite Strategy" (pp. 221-235). Code:
`premia/mcmillan.py` (rules), `premia/backtest.py` (`sleeve="mcmillan"`), `premia/research/mcmillan_bt.py` (runs).

## Rules, book → code

| Book | Code |
|---|---|
| Criterion 1: IV in the 10th percentile or lower of past readings; at least three months of life (pp. 221-223) | `iv_percentile_all` (all readings to date, ≥ 504 of them) ≤ 10; monthly expiry nearest 182 days, at least 90 |
| Criterion 2: "ever" probability ≥ 80 % of touching a breakeven (strike ± straddle price) before expiry; lognormal; buyer uses the minimum of the 10/20/50/100-day historical vols (pp. 224-228) | `ever_probability` (closed form, continuous monitoring, zero log drift; tested against a Monte Carlo of the same model); `min_historical_vol` |
| Criterion 3: the stock has frequently made the required move in that time (p. 228) | `move_frequency`: share of past 5 years of same-length windows whose high/low reached the breakeven distance, ≥ 50 % (my reading of "frequently") |
| Criterion 4: no fundamental reason the options are cheap, e.g. a cash takeover (p. 229) | live: a news check by you; backtest: `deal_pinned` proxy (a ≥ 15 % one-day jump, then 20-day vol < 35 % of its pre-jump 100-day vol) |
| Risk 60 % of the initial price (p. 232) | close both legs when the straddle is worth ≤ 40 % of its cost |
| Hold until about one month of life remains (p. 232) | close at ≤ 30 calendar days to expiry |
| Breakeven crossed: sell the losing side, sell ~⅓ of the winning side, trail the rest with the 20-day moving average (p. 232) | `follow_up`; decided on the close, filled the next morning (p. 233) |
| No "trading against the straddle" (p. 232) | no delta hedging |

Size (not in the book): 0.5 % of NAV of premium per straddle (one contract allowed up to 1.5 %), 10 % of NAV for all
straddles together, and the premia hard limits (vega 0.4 % NAV per vol point etc.) on every opening.

Added 2026-11-11, still before any strategy run, so the backtest opens exactly what the dashboard page would: sizing is
one function, `mcmillan.size`, used by both (near the 10 % cap the order is cut down to fit, not skipped), and the
backtest applies the page's liquidity limit, straddle spread <= 10 % of mid (`LIMITS.max_option_spread_frac`). The
page's other limit, open interest >= 500 per leg, cannot be tested: there is no open-interest history. Candidates
are taken best "ever" probability first, as the page ranks them.

## Pricing

Options are priced like every other premia backtest: real 30-day ATM IV (IBKR history), a modelled smile and spread,
next-bar fills at mid + ½ the half-spread, $0.65/contract. New here: 3-6 month options are priced off each name's
30-day IV times that day's SPX term-structure ratios (VIX3M/VIX, VIX6M/VIX; `market.add_index_term`). Without that, a
six-month straddle bought when 30-day IV is at its 10th percentile would be priced too cheaply: in that regime SPX
six-month IV averaged 4.4 vol pts (×1.36) above 30-day IV. The bias is shown by the `mcm_flat_term` run.
A fitted mean-reversion model (κ, long-run level, calibrated 2008-2012) was tried first and rejected before any
strategy run: out of sample it still priced six-month SPX vol ~1.9 vol pts too low in low-IV regimes. Limitation:
single names are assumed to share the index's term-structure slope.

## Universe

The dashboard's wheel watchlist (`WHEEL_WATCHLIST`) minus the manual exclusions, every name with at least 500 days of
IV history, 2009-2026. That is today's list applied to the past, so every name survived to today (look-ahead and
survivorship bias, in the strategy's favour for a buyer only if survivors trend; stated, not corrected).

## Pre-registered pass criteria (base run, 2013-01 → 2026-11)

The strategy replaces the current straddle model on the dashboard as a **recommended** strategy only if all hold:

1. net annualised return after costs > 0;
2. the mean daily return is significant (Newey-West t ≥ 2);
3. it beats the earlier model straddle sleeve's walk-forward Sharpe (−0.77);
4. it still makes money with fills at the full half-spread (`mcm_fullspread`).

Reported regardless: β and correlation to SPY, up/down/flat months, every exit type, and how often the stock really
touched a breakeven against criterion 2's predicted probability (the book claims real markets touch more often than
the lognormal model says, p. 227). Sensitivity runs (not selection): criterion 3 at 80 %, three-month options, full
spread, no follow-up management, flat term structure.

Whatever the verdict, the dashboard's straddle page applies these rules (the user asked for McMillan's method); the
verdict decides what the page's banner says about it.
