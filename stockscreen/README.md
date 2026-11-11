# stockscreen — quality stocks near attractive entry points

Powers the **Fundamentals** page of the dashboard. Philosophy:

> high-quality business + strong fundamentals + attractive FCF yield + favourable price percentile + technical confirmation

All outputs are **research signals, not investment advice or return predictions**. The scores and opportunity classes are
**heuristic rankings that have not been backtested**. A "Potential buy zone" label describes price location and
trend/fundamental context — it is not proof that a stock is undervalued.

## Quick start

* Open **Fundamentals** → choose a universe → **Run scan** (runs in the background; keep using other pages).
* Headless / scheduled: `python -m stockscreen.scan --universe wheel [--force]` (e.g. Windows Task Scheduler after the close),
  or tick *Auto-scan* in the screening profile (runs on weekdays after the configured time **while the dashboard is open**).
* Thresholds, weights and alert preferences: the **Screening profile** expander (saved to `stockscreen_config.json`).
  The scan stores *threshold-free raw data*, so changing a threshold re-classifies every stock instantly — no rescan.

## Universe and coverage

| Option | What it is |
|---|---|
| Wheel universe (shortlist) | The app's existing universe: `wheel_shortlist.json`, built by `scan_universe.py` from the largest ~1,200 SEC-registered tickers, pre-filtered to net margin ≥ 10 % and positive FCF, minus your manual exclusions (`wheel_exclusions.json`). |
| Wheel universe + holdings | The above plus your current stock positions. |
| SEC top *N* by size | The first *N* tickers in SEC's company list (≈ market-cap order), common-stock-looking symbols only. |
| Custom list | Tickers you type. |

**This is never a whole-market scan**, and the page says so. Every scan records its universe, size, and status:
`complete` only if every ticker was processed with no errors; anything cancelled, failed or with errors is `partial` and is
labelled that way. ETFs, funds and foreign filers without XBRL cash-flow statements are not silently dropped — they are counted as
"without SEC statements" and appear as *Insufficient data* with the reason.

## Data sources

| Data | Source | Notes |
|---|---|---|
| Cash flow, income statement | SEC EDGAR XBRL company facts | Shared 8 req/s limiter, retry/backoff on 429/5xx. Parsed observations cached `fundamentals_ttl_days` (3). |
| Share count | SEC cover page → balance sheet → weighted-average (approx.) → Yahoo Finance share count (labelled fallback) | A share count older than `max_data_age_days` is never used. |
| Prices | Yahoo Finance daily history via `yfinance`, batched | **Split-adjusted, not dividend-adjusted** closes. Cached `price_ttl_hours` (6); a failed download falls back to the stale cache, never to invented data. |

### How quarters are built
Cash-flow statements are filed *year-to-date*. A discrete fiscal quarter is derived by differencing within a fiscal year:
`Q1 = YTD3, Q2 = YTD6 − YTD3, Q3 = YTD9 − YTD6, Q4 = FY − YTD9`. A directly reported 3-month value is preferred. Periods are matched by
their start/end dates (never the `fy`/`fp` labels); 52/53-week and 12-12-12-16-week calendars are handled. Each observation shows its basis
("reported 3M", "derived YTD", "derived Q4 = FY − 9M"), first-publication date, and XBRL tag. Nothing is imputed: a quarter that can't be
derived is simply absent, and a gap makes the dependent TTM figure *unavailable*.

`FCF = Cash flow from operations − capital expenditures`. Capex tags accepted: purchases of PP&E, productive assets, capital
improvements, oil & gas development, real-estate development. **Not** treated as capex: asset *acquisitions*, non-cash accruals, and
software-only lines. **Missing capex is never treated as zero** (opt-in `allow_missing_capex_as_zero` exists and is flagged when used).
Banks, insurers and many asset managers report no capex, so they end as *Insufficient data*, not as passes.

### Financial companies (FCF can be meaningless)
For banks, lenders, brokers/asset managers, investment holdings and insurers (SEC SIC codes 60-64 and 67, excluding REITs) operating cash flow can
contain investment, lending or trading flows, so every such company is flagged "FCF may not be comparable". One distortion is detected and handled:
when the **reported investing total equals capex** (nothing else flows through investing) the company's investment purchases and sales must sit
*inside operating cash flow*, so `OCF - capex` is not free cash flow. The FCF criteria then become `na` ("not meaningful"), FCF is left out of the
scores, and the company can never pass. Worked example, Blackstone FY2025 (10-K): operating cash flow +$4,663M, investing -$116M (= PP&E purchases),
financing -$3,879M. Inside operating activities sit investments purchased -$3,807M and sale proceeds +$6,679M (net +$2.9B); data vendors reclassify those
to investing (and NCI contributions of +$2,525M out of financing), which gives operating +$1.86B, investing +$5.28B, financing -$6.41B and a negative
"FCF" once the investment purchases are counted as capex. Both views reconcile to the same +$678M change in cash; the screen's FCF is "as filed", and for
Blackstone that is not a usable quality signal. Policy (`financial_fcf_policy`): `na` (default), `flag` (warn only) or `off`. SIC codes come from SEC's
submissions API and are cached permanently.

## Fundamental screen (defaults)

| # | Criterion | Rule |
|---|---|---|
| A | FCF consistency | FCF **> 0** in each of the latest **4 consecutive reported quarters** and the latest **2 fiscal years** (6 observations, all shown in the detail view). Zero fails. |
| B | Net margin | TTM net income / TTM revenue **> 10 %** (strict). |
| C | Revenue growth | **> 10 %** (strict): TTM vs prior TTM when 8 consecutive quarters exist, otherwise last fiscal year vs prior (labelled). |
| D | FCF yield | TTM FCF / market cap **≥ 5 %** (inclusive). |
| E | Size | Market cap **≥ $1 B** (inclusive); market cap = share count × latest close (both dates shown). |
| F | Trend | No *critical* metric (FCF, net margin, revenue) classified **Declining**. |

Per-criterion result: `pass`, `fail`, or `na` (could not be computed — **never a pass**). Overall status: `ERROR` → `FAILED` (any criterion failed) →
`STALE_DATA` (latest quarter older than `max_data_age_days`, 200) → `INSUFFICIENT_DATA` → `PASSED`. Boundary comparisons are float-safe
(`110/100 − 1` is not "above 10 %").

**Trend classification** (TTM series over the last 5 quarter ends, so seasonality cancels): *Improving* / *Stable* / *Declining* /
*Insufficient data*. Level metrics use relative change (tolerance 5 %), margins and growth use percentage points (1 pp / 3 pp). A metric is
**Declining** if it fell by more than the tolerance **and** at least 75 % of the sequential moves were down, or fell by more than 3× the
tolerance regardless of path; **Improving** is the mirror image. One noisy quarter can therefore never make a trend Improving. Slower
revenue *growth* is not a revenue decline (growth trend is informational by default). A falling FCF *yield* caused by a rising price is
reported as Stable, not as deterioration. One-offs are flagged (discontinued operations > 10 % of net income, a quarter > 60 % of TTM net
income, margin jumping > 5 pp in a quarter, FCF down > 50 % YoY, capex/income-statement lag, approximated share counts).

## Price position and percentiles

Using split-adjusted closes over calendar lookbacks of 1, 2 and 3 years (a lookback is *unavailable* unless history covers it):

* **Range position** = (price − low) / (high − low) × 100 — 0 at the low, 100 at the high; undefined when high = low.
* **Closing-price percentile** = share of the window's *previous* closes below today's close (ties count half) — the current close is never
  counted against itself. This is a different question from range position (a stock can sit at the bottom of its range yet mid-distribution).
* Also shown: price, high/low, % below high, % above low.

**Price zone** (a lookback is "low" if its range position **or** its percentile is ≤ 20): *Deep value-zone candidate* (≥ 2 lookbacks low),
*Low-price candidate* (1), *Near highs* (52-week position ≥ 90), *Mid-range*, *Insufficient history*. The windows are independent signals —
they are not required to agree.

## Technicals

SMA 20/50/200, 14-day Wilder RSI, MACD 12/26/9, ATR % of price, ADX, relative volume, SMA slopes (20 days), distance from SMA50/200.
Trend states (explicit, configurable rules; evaluated in this order):

1. **Insufficient data** — fewer than 200 + slope-window closes.
2. **Sideways/consolidating** — price and SMA50 both within 3 % of SMA200 (no directional structure).
3. **Strong uptrend** — price > SMA50 and SMA200, SMA50 > SMA200, both slopes positive.
4. **Healthy pullback** — price > SMA200, SMA50 > SMA200, SMA200 not falling (long-term intact, short-term weak).
5. **Trend reversal candidate** — price and SMA50 below SMA200 (or price recovering through SMA200) **with ≥ 1 recovery signal**; signals: price reclaimed SMA20, RSI > 40 and rising, MACD bullish (histogram > 0 or recent bullish crossover), SMA20 rising. ≥ 2 signals counts as *confirmed*.
6. **Downtrend** — price and SMA50 below SMA200 with no recovery signal.
7. Otherwise **Sideways/consolidating** (mixed structure).

A low RSI or a price near its 52-week low is never, by itself, an uptrend.

## Opportunity classes

| Class | Meaning |
|---|---|
| A. Quality stock – monitor | Passes the fundamental screen; price not in the low zone. |
| B. Potential buy zone – strong fundamentals | Passes; low-price zone; technical confirmation incomplete. |
| C. Potential buy zone – healthy pullback | Passes; low-price zone; constructive long-term trend. |
| D. Potential buy zone – reversal confirmation | Passes; low-price zone; ≥ 2 recovery signals. |
| E. Fundamental or technical warning | In the low-price zone but in a persistent downtrend, **or** failing only because a critical fundamental metric is declining. Shown, never dropped. |
| Not eligible | Does not pass the fundamental screen. |

## Scores (0–100, heuristic, not backtested)

Weighted averages over the components that could be computed; missing components are excluded, weights renormalised, and the
**coverage** reported (a score with coverage < 60 % is blank). Defaults (all configurable):

* **Fundamental quality** — FCF consistency 20 %, FCF trend 20 %, net margin 15 % (10 % → 50 pts, 30 % → 100), revenue growth 15 % (same scale), stability 10 %, FCF yield 20 % (5 % → 50, 15 % → 100).
* **Price opportunity** — 52w range 30 %, 2y 20 %, 3y 15 % (each 100 − position), percentiles 20 %, distance below the 52-week high 15 % (40 % below → 100).
* **Technical trend** — trend class 35 %, price vs SMAs 20 %, SMA50 > SMA200 10 %, slopes 10 %, RSI 10 %, MACD 15 %.
* **Rank score** — fundamental 45 %, price 35 %, technical 20 %.

## Alerts

Kinds: low-price, healthy-pullback, reversal-confirmation, fundamental-deterioration. Per (stock, kind) state machine, advanced once per scan:

| Condition | State | Result |
|---|---|---|
| true | inactive, cooldown over | **fire** |
| true | inactive, cooling down | suppressed |
| true | active | nothing (deduplicated) |
| false | active | cleared (re-entry may fire again) |
| unknown (stale / insufficient / error) | any | nothing; state unchanged; **never fires** |

Deterioration only fires for stocks that previously qualified. Alerts are in-app (Fundamentals page + Overview alerts); the project has no
email/push infrastructure, so none was added. Each alert carries ticker, company, price, trigger, lowest percentile, FCF yield,
fundamental and technical status, reason, and data/price dates.

## Files

`config.py` thresholds · `sec_data.py` XBRL parsing · `fundamentals.py` screen & trends · `technicals.py` · `scoring.py` · `evaluate.py` ·
`prices.py` · `store.py` (SQLite tables `screener_*` in `wheel.db`) · `alerts.py` · `scan.py` (background scan, CLI) · `ui.py`.
Tests: `tests/test_stockscreen_*.py` (synthetic data, no network).

## Known limitations

* SEC `companyfacts` omits class-specific (dimensioned) share counts, so several dual-class companies fall back to an approximate
  share count or the labelled Yahoo count; a few companies use custom XBRL tags for capex or revenue and end up *Insufficient data*.
* REITs report acquisitions rather than capex; those acquisition lines are deliberately not treated as capex.
* Yahoo Finance prices are unofficial and can lag or occasionally be wrong; the price date is always shown.
* Day counts and "age" use calendar days; exchange holidays are not modelled.
* Acquisition-heavy oil & gas producers (e.g. PR, DVN) show higher FCF than vendors that count acquisitions as capex; acquisitions are deliberately not capex here.
* No survivorship-bias handling: the screen is point-in-time on today's universe and is not a historical backtest.
