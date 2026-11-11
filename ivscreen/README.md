# ivscreen — IV percentile / realized-vol percentile screener

Powers the **IV screener** page (Research group). It answers, for every stock in your universe:

* which stocks have exceptionally **low** implied-volatility percentile, and which exceptionally **high**;
* is **realized** volatility also low/high against *its own* history;
* is IV **rising or falling**, and is realized vol **following**;
* are options pricing unusually high/low volatility compared with the stock's historical behaviour.

It follows McMillan (*Profits with Options*): implied volatility is **never compared directly with realized volatility**. Each
is ranked only against **its own history**, and the two *percentiles* are then set side by side. The page shows percentiles,
not raw IV / RV levels (those are on **Volatility & events**).

> Research signals, not investment advice. A high IV percentile means implied vol is high compared with that stock's own past —
> it is **not proof the options are overpriced** or that selling them will be profitable (earnings and other events lift IV for a
> reason). A low percentile is not proof the options are cheap.

## Quick start

1. Connect to IBKR in the left rail (TWS running; the connection is read-only).
2. **IV screener → Run a scan.** Pick the universe lists (Wheel universe, Holdings, ETFs, Custom) and press **Run scan**. It runs
   in the background; keep using other pages. IBKR sets the pace: observed 0.15–0.4 symbols/s, so a first scan of ~300 symbols takes
   roughly 15–40 minutes. Timeouts under load are retried once automatically at the end of the scan; a symbol fetched within the
   last 6 hours is skipped, so a cancelled or partial scan **resumes** where it stopped when you press Run scan again (tick
   *Re-download full histories* to force everything).
3. Headless / scheduled (TWS must be up): `python -m ivscreen.scan --universe wheel [--etf] [--tickers "NVDA SAP"] [--force]`
   (uses client id 78, or `WHEEL_IBKR_CLIENT_ID`).
4. Everything under **Settings** (bands, lookback, RV window, trend, gap, freshness, ETF list) re-classifies instantly — scans store
   only raw history, so no rescan is needed. Saved to `ivscreen_config.json`.

## Definitions (all configurable defaults)

| Term | Definition |
|---|---|
| **Percentile** | Share of the lookback window's **prior** daily readings that were **strictly below** today's, 0–100. Today is not in its own window; ties do not count as below. Rounded to 6 decimals so a value exactly on a band edge is never moved by float noise. Same definition as the Volatility page. |
| **Lookback** | 504 trading days (2Y); selectable 252 / 504 / 756. |
| **Minimum history** | 126 prior readings. Fewer ⇒ **no percentile** (never a guess). A window shorter than the lookback but ≥ 126 is ranked against what exists and flagged (`History` column, `*`). |
| **Implied vol** | IBKR's 30-day implied volatility for the underlying (`OPTION_IMPLIED_VOLATILITY`, daily, 3 years). Works without an OPRA subscription. |
| **Realized vol** | 20-day close-to-close volatility, annualized, from daily closes (shared price cache, Yahoo split-adjusted closes); window selectable 10 / 20 / 50 / 100. Ranked against its own history only. |
| **Bands** | Very low ≤ 10 · Low > 10 and ≤ 25 · Normal > 25 and < 75 · High ≥ 75 and < 90 · Very high ≥ 90. |
| **Trend** | Change in the *percentile* over 10 sessions: Rising ≥ +5 pts, Falling ≤ −5 pts, else Flat. |
| **Relationship** | Precedence: both ≤ Low edge ⇒ *Both subdued*; IV pct − RV pct ≥ 25 ⇒ *IV elevated vs RV*; ≤ −25 ⇒ *RV elevated vs IV*; both ≥ High edge ⇒ *Both elevated*; else *In line*. It compares where each sits in its own range, nothing more. |
| **RV following IV?** | Plain-language read of the two trends ("Both rising — RV is following IV up", "IV rising, RV not following", …). |
| **Event note** | Earnings inside the 30-day IV window (Yahoo Finance dates) — part of an elevated IV is then a known event. |

## The three sections

* **A · Low IV watchlist** — ranked symbols at or below the *Low* edge, lowest first; *Very low* is highlighted.
* **B · High IV watchlist** — ranked symbols at or above the *High* edge, highest first; *Very high* (≥ 90) is highlighted; earnings
  inside the window are annotated.
* **C · IV percentile vs RV percentile** — scatter (IV pct × RV pct, coloured by relationship, shaded corners where one is elevated
  relative to the other), table with gap, both trends and the relationship, and a symbol detail with the **percentile history** of
  both series (each day ranked against the window before it). Click a table row or a scatter point to open the detail.

## Universe

| List | Content |
|---|---|
| Wheel universe | Your existing screener shortlist (`wheel_shortlist.json`) minus manual exclusions. |
| Holdings | Your current stock positions (from the connected account). |
| ETFs | **Opt-in category.** The editable list in Settings (default ~33 optionable index/sector/commodity/bond ETFs). |
| Custom | Tickers you type. |

* **Manual exclusions (`wheel_exclusions.json`) are absolute** — in every list, every spelling (`BRK.B`/`BRK-B`), at scan time
  (never requested), at storage (purged, also when added mid-scan) and at view time (never evaluated or shown).
* **Duplicates**: spellings are normalised and merged; two tickers resolving to the same IBKR contract are marked `duplicate`.
* **Instrument type** comes from IBKR (`Common stock`, `ETF`, `ADR`, `REIT / trust`, `Other`). The page's *Instrument type* filter
  defaults to stocks only, so **ETFs are never silently mixed in**. Preferred shares, warrants, rights and non-stock contracts are
  `unsupported`.
* Filters: instrument type, watchlist (source list, incl. live Holdings), ticker text, exchange (IBKR primary exchange), sector
  (IBKR industry), earnings inside / outside the window.
* It is a shortlist you control, **not a whole-market scan**, and the coverage banner says so.

## Data availability — unknown is never low

Every stored symbol carries a status; only `Ranked` symbols appear in A/B/C. Missing data is **never treated as zero volatility**.

| Status | Meaning |
|---|---|
| Ranked | Fresh IV history with enough prior readings. |
| Too little implied-vol history | IBKR has data but < minimum history (e.g. a recent IPO). The reading count is shown. |
| Implied-vol data is stale | Latest IV bar older than 7 days. |
| No options data from IBKR | IBKR answered with no implied-vol history (no listed options or no data). |
| Fetch failed — retry | Timeout, pacing limit, lost connection or an exception. A **refresh that returns nothing never overwrites stored data** — it is recorded as a failure and the history is kept. |
| Ticker not found / Unsupported / Duplicate | Resolution outcomes. |

If realized vol is unavailable (no or stale prices, short history) the IV percentile still shows and the relationship reads
*RV unavailable*; those symbols are listed separately under section C. The page shows scan coverage (counts per status, latest data
point, last checked), flags partial scans, and the **Data availability** expander lists every unranked symbol with the reason.

## Storage

SQLite in `wheel.db`: `ivscreen_run` (one row per scan; `complete` only if every symbol processed with no failures),
`ivscreen_symbol` (contract facts, sources, fetch status/detail, next earnings), `ivscreen_iv` (daily IV per symbol; top-ups overwrite
overlap). Prices reuse `stockscreen_cache/prices` (4 years, 6 h TTL). The first download per symbol is 3 years; later scans fetch
2 months, with a full re-download every 30 days (or when the stored data is > 35 days old, or on *Re-download full histories*).

## Read-only

A scan only issues contract-details and historical-data requests. No orders, no market-data subscriptions, no account calls.

## Limitations

* IBKR's 30-day IV is a constant-maturity measure for the underlying, not one expiry; it is not the VIX-style term structure.
* Realized vol uses Yahoo closes, implied vol uses IBKR's feed: levels may differ, which is why only percentiles of each own series
  are used.
* A percentile is relative to the chosen lookback; a regime change (e.g. after a merger) can make the past a poor guide.
* IV history longer than IBKR provides cannot be ranked (≈ 3 years requested; younger listings are flagged).
* Earnings dates come from Yahoo Finance and may be missing for some symbols; confirm before trading around them.
* Throughput is bounded by IBKR (0.15–0.4 symbols/s observed), so a full scan takes tens of minutes, not seconds.
