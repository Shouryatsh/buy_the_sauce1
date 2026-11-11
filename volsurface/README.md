# volsurface — Gatheral's volatility surface, on real option quotes

Powers the **Vol surface** page (Research group). It follows *The Volatility Surface* (Gatheral) and Gatheral & Jacquier,
"Arbitrage-free SVI volatility surfaces" (2014), one numbered section per idea. SPX by default; any optionable ticker works.

> Analysis only. The dashboard is read-only: it never places, changes or cancels an order. Model values describe how the market
> prices options now; they are not forecasts, and a quote off the smooth surface is not a guaranteed edge.

## The seven sections

| # | What it shows |
|---|---|
| 1 | **The smile** of one expiry (default nearest 30 days): OTM puts left of the forward, OTM calls right, bid-ask range as error bars, the SVI fit, forward, ATM vol, 90 %-strike skew. |
| 2 | **The surface** across expiries and strikes: 3-D surface (or heat map) of the fitted slices, interpolated in total variance, with the market quotes as dots; a term-structure table. |
| 3 | **Total variance** w = σ²T against log-moneyness k = ln(K/F) (or K/F, or k/√w_ATM): the curves nest in maturity order; ATM variance must rise with T. |
| 4 | **SVI lab**: raw SVI `w(k) = a + b(ρ(k−m) + √((k−m)²+σ²))`. Auto-fit, then move a, b, ρ, m, σ live and watch the curve, fit error, % inside the bid-ask band, Durrleman's g(k), Lee wing slopes and the risk-neutral density (negative density shown in red). Runs as a Streamlit fragment, so a slider only reruns this block. |
| 5 | **Arbitrage**: butterfly (g(k) ≥ 0, Lee b(1±ρ) ≤ 2), calendar (w(k,T₂) ≥ w(k,T₁)), ATM-variance monotonicity, and a model-free check on the quotes. Violations *where the market quotes* are separated from violations in *extrapolated* wings. |
| 6 | **Dynamics**: compare with any saved earlier surface (Δ-IV heat map, ATM term structure overlay), history of ATM vol / skew across snapshots, and the VIX family (VIX, VIX/VIX3M, SKEW, VVIX, SPX 20-day realized) each against its own two-year history (McMillan: percentiles, never one series against another's level). |
| 7 | **Decisions**: a plain-language scorecard, a cash-secured-put scan on the fitted smile, and where your open short options sit on their own smiles. |

## Method

* **Time to expiry** from the settlement moment (SPX standard monthlies settle at the open, SPXW/ETF/equity at the close), ACT/365.
  Where an expiry lists both SPX and SPXW, the PM-settled SPXW is used.
* **Forward and discount factor from put-call parity**: for each strike `C − P = D(F − K)`; a weighted line over the ~14 strikes
  where C ≈ P gives `D = −slope`, `F = intercept/D` (outliers beyond 3 MAD dropped, then refit). No rate or dividend assumption. If
  parity cannot be estimated the slice is flagged and a rate-based fallback forward is used.
* **Implied vols** by Black-76 with a vectorised bisection (cannot diverge); from the mid, and from bid and ask for the error bars.
  Only OTM options are used (puts below the forward, calls above). A quote is dropped, never repaired, if it has no bid (< $0.05),
  a spread over 50 % of the mid, no implied vol inside the no-arbitrage band, or k outside the window. Slices with too few quotes are
  dropped with the reason shown.
* **SVI fit**: weighted least squares in implied-vol space, multi-start. Weights: `spread` (1/half bid-ask in vol terms, default),
  `vega`, or `equal`. Lee's wing bound and non-negative variance always apply. **Arbitrage-aware** (default on) adds soft penalties
  for negative density (target g ≥ 0.004, on a grid reaching well past the quoted strikes) and for crossing the previous expiry.
  On the live SPX chain this keeps all slices butterfly-clean at a cost of ~0.06 vol points of fit error and gives economically sensible
  skew (ρ ≈ −0.1…−0.4) instead of a degenerate fit with ρ > 0 and a wing that goes arbitrageable beyond the quoted strikes.
* **Between expiries**: total variance interpolated linearly in T at fixed k (preserves calendar consistency); shorter than the first
  expiry w scales with T; beyond the last, implied vol is held flat.
* **Risk-neutral probability from the smile**: `P(F_T < K) = N(−d₂) + φ(d₁)e^{−k} w′(k) / (2√w)`. Checked against the numerical
  derivative of the put price. For an equity skew it is *lower* than a flat-vol number near the money (the median sits above the
  forward) and far *higher* on deep OTM puts.
* **RICH / CHEAP / FAIR**: a put is RICH only if even its *bid* implied vol exceeds the fit by ≥ 0.25 vol points, CHEAP only if its
  *ask* is ≥ 0.25 points below; otherwise FAIR (inside, or within a quarter point of, the bid-ask band).
* **Model-free quote checks** test calls as calls and puts as puts (no discount factor), only where each is out of the money, on
  bid/ask, and only count violations larger than one price tick.

## Data sources

| Source | Notes |
|---|---|
| **Yahoo (delayed)** | Free, ~15 min delayed, no TWS needed; whole chain in ~1.5 s. Bid/ask exist only while the market is open. Delayed quotes can give a few spurious model-free flags. |
| **IBKR (live)** | Via TWS, read-only (contract details + market-data requests only). One contract-list request per expiry (so only strikes that exist are quoted), OTM half plus a few ATM pairs, thinned to the budget, quoted by *streaming in batches of 80 lines* then cancelled. Market-data type 2 (live, or frozen after the close), falling back to delayed if no two-sided quote arrives. Measured on SPX: **~105 s for 6 expiries / 1,282 contracts, 1,265 two-sided**; ATM vols and forwards agree with Yahoo to a few hundredths of a vol point. Expiries IBKR does not list are skipped in one note. |
| **Saved snapshot** | Every load is stored in `wheel.db` (`volsurface_snapshot`), so you can study a surface when the market is closed and compare over time. |

`python -m volsurface.record --symbol SPX [--source yahoo|ibkr]` saves a snapshot (with its summary metrics) from the command line —
schedule it (e.g. every 30–60 minutes in market hours) to build the dynamics history while the dashboard is closed.

## Decisions section

* **Scorecard**: VIX level, VIX/VIX3M term structure, this surface's 30- vs 90-day ATM vol, skew (and CBOE SKEW percentile), and
  implied vs realized — each only against its own history. Wording is conditional, never predictive.
* **Put scan**: every OTM put of one expiry with market vs fitted IV, signal, |delta|, risk-neutral P(below) from the smile vs flat vol,
  yield, annualised yield, break-even cushion; filters default to your wheel rules (|delta| ≤ 0.15, spread ≤ 40 %, bid ≥ $0.10); sort by
  annualised yield (default), richness, or strike. A RICH flag is a pricing comment, not a reason to ignore assignment or gap risk.
* **Account fit**: shows the cash needed per contract (strike × 100) against your buying power and warns when the underlying is
  too large (one SPX contract ≈ $670k; SPY or XSP are the practical analogues).
* **Check my short options**: each open short option located on the fitted smile of its own expiry (Yahoo chains): market vs fit IV,
  signal, risk-neutral P(ITM) from the smile vs flat, model price vs bid/ask.

## Limitations

* An SVI slice is a smooth parametric curve: it will not pass through every quote (SPX spreads are often 0.1 vol point), so only a
  fraction of quotes land inside their bid-ask band. Treat fit residuals under ~0.25 vol points as noise.
* Beyond the quoted strikes the fitted form is extrapolation (the page labels violations there as such). Long-dated calls above the
  money are thinly quoted, so right wings are the least reliable.
* Implied rates and dividends from parity are only meaningful for longer expiries; short ones are dominated by quote timing.
* Yahoo prices are delayed and sometimes stale; IBKR live avoids that.
* Calendar time (ACT/365) is used for T; trading-time variants would change short-dated smiles around weekends and events.
* One surface per underlying; no cross-asset (e.g. SPX–VIX) modelling and no SSVI/eSSVI fit (the natural next step).
