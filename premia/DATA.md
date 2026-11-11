# premia — data (phase 1)

## What is used now (free) and how it is handled

| Data | Source | Coverage in `premia_data.db` | Handling |
|---|---|---|---|
| Daily OHLCV, dividends, splits | Yahoo Finance (`yfinance`) | 552 of 773 tickers that were S&P 500 members since 2009 (+ ETFs), 2006-11 → today | `close` is split-adjusted (it sets option strikes and returns, since OCC adjusts listed options on splits). Market caps use the unadjusted close (split-adjusted × later split ratios) × the share count filed at the time. Dividends: a trailing-12-month yield for pricing, and cash on the ex-date for stock positions. |
| 30-day ATM implied vol | IBKR TWS `OPTION_IMPLIED_VOLATILITY`, read-only historical-data requests | up to 20 years per name (SPY from 2006-11) | Bad prints are removed when a jump of more than 42 % reverses by more than 75 % the next day, never smoothed. Values are carried forward at most 3 trading days; a longer gap leaves the day untradeable. |
| SPX implied vol by tenor, SKEW, VVIX, 13-week T-bill | Yahoo index series (^VIX9D, ^VIX, ^VIX3M, ^VIX6M, ^SKEW, ^VVIX, ^IRX) | 2006 → today | Term-structure ratios for SPY; the risk-free rate as continuously compounded ^IRX. |
| Fundamentals | SEC EDGAR XBRL company facts, kept with **every filing date** | 519 companies | Point in time: at date t only facts filed on or before t are used, so a restatement is invisible until it is filed. XBRL starts in 2009–2011, so the screen is thin before 2012 (see RESULTS.md). |
| Earnings dates | SEC 8-K item 2.02 filing dates | 511 companies, from 2004 | The reaction is the larger of the filing day and the next day, because the release time is not precise. |
| Index membership | fja05680/sp500 (GitHub) | intervals since 1996 | The universe at t is the set of members at t. |
| Real option chains (today only) | IBKR frozen/live quotes; saved SPX surfaces in `wheel.db` | SPX (9 snapshots), 22 large caps, IWM | SVI fit-error evidence and the normalised smile shapes. |

### Survivorship
Membership is point in time, so the backtest never selects a name because it survived. Names that later disappeared
(acquired or bankrupt) have no Yahoo or IBKR data, though, so they drop out of the tradeable set. That is residual
survivorship bias, and it flatters a short-put strategy, because blow-ups are under-represented. `data.coverage_report`
measures it per quarter, and RESULTS.md reports it. Fixing it needs a vendor with delisted securities (below).

### Missing quotes
Nothing is ever imputed beyond the 3-day carry-forward. A name without a current IV cannot be opened, closed (the order
retries the next day) or quoted. An open position on such a day keeps its last mark, and the day is counted as stale.

### The gap that matters
There are **no historical option chains** (bid/ask, IV by strike, open interest) in any free source. Historical OTM vols
are therefore modelled as ATM IV × a normalised smile from today's real chains. The **skew premium, real historical
fills and the open-interest filter therefore cannot be validated historically** (SPEC.md section 3).

## Recommended paid sources
Prices below are what public pages and third-party comparisons showed in November 2026. They disagree with each other
and change often, so **check them on the vendor's own pricing page before buying**.

| Need | Recommendation | Indicative cost | Why |
|---|---|---|---|
| Historical chains, end-of-day, incl. delisted names, with smoothed IV surfaces | **ORATS** (data API / historical EOD back to 2007) | ~$99–$299 / month reported; one ORATS page shows $199 / month delayed data with EOD history from 2007; full history may be priced separately | Ready-made SMV/ IV surfaces and earnings-adjusted IV fit this design directly (replace `SmileShape` with real per-date smiles); one subscription covers the backtest and the daily paper-trading marks. |
| Intraday / tick option quotes (fill realism, intraday hedging) | **ThetaData** | tiers reported at ~$40 / $80 / $160 per month (Value / Standard / Pro; 4–8+ years of history); a free tier has 1 year of EOD | Cheapest route to real bid/ask at fill times; Standard adds tick-level NBBO. |
| Exchange-official EOD option data (audit-grade) | **Cboe DataShop** "Option EOD Summary" / "Option Quotes EOD with Calcs" | quote on request, priced per symbol-history; academic discount 50 % (min $500) | Two snapshots a day (15:45 and the close) with IV and Greeks; the reference source for SPX. |
| Academic research standard | **OptionMetrics IvyDB** (through WRDS) | institutional licence only | Best documented and used in most VRP papers. Only realistic with a university affiliation. |
| Point-in-time fundamentals incl. delisted companies | **Sharadar Core US Fundamentals (SF1)** via Nasdaq Data Link | subscription, price on the product page (individual plans have been in the low hundreds of $ per year; verify) | As-reported plus restated values with dates, delisted companies included; this closes the survivorship gap in phase 2. |

**Minimum useful upgrade:** ORATS (chains, including delisted names) plus Sharadar SF1 (point-in-time fundamentals,
including delisted). That removes the two largest caveats: the modelled skew and residual survivorship. ThetaData is
only needed for intraday fill and hedge realism.

Sources: [ORATS AI-agents / plans](https://orats.com/ai-agents), [ORATS for Tradier](https://orats.com/tradier),
[ORATS 2-minute data](https://orats.com/two-minute-data),
[options data API comparison (dev.to)](https://dev.to/tomasz_dobrowolski_35d32c/best-options-data-apis-2026-7-compared-pricing-free-tiers-1lf5),
[ThetaData subscriptions](https://docs.thetadata.us/Articles/Getting-Started/Subscriptions.html),
[options data pricing comparison (FlashAlpha, a competitor)](https://flashalpha.com/articles/options-data-pricing-comparison-flashalpha-thetadata-polygon-spotgamma-squeezemetrics),
[Cboe DataShop end-of-day options summary](https://datashop.cboe.com/end-of-day-options-summary),
[Cboe DataShop academic discount](https://datashop.cboe.com/academic-discount),
[Sharadar via QuantRocket docs](https://www.quantrocket.com/docs/data/fundamental/sharadar/),
[FMP vs Sharadar (dev.to)](https://dev.to/pickuma/financial-modeling-prep-vs-sharadar-fundamental-data-apis-for-quant-backtests-la2).
