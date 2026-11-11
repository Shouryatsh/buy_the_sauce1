# premia: results, verdicts and recommendation

Run date 2026-11-08. All numbers are reproducible with the commands in `README.md`; the raw outputs are in
`results/premia/` (one `verdict_phaseN.json` per phase). Pass criteria were fixed in `SPEC.md` §6 before any result
was seen and are applied as written. Where I deviated from "stop on a fail", that is said explicitly below.

> Backtests are not predictions. The options in every backtest are priced from **real** 30-day ATM implied vol
> (IBKR history) but a **modelled** smile, spread and expiry grid (SPEC.md §3). Fills are next-bar at mid minus
> half the half-spread, plus $0.65/contract; stock hedges pay 2 bps + $0.005/share; borrow 0.5 %/yr.

## 1. Verdicts at a glance

| Phase | Verdict | One-line reason |
|---|---|---|
| 0 Spec | done | `SPEC.md`: book → model → module → test, 10 conflicts resolved, pre-registered criteria |
| 1 Data | **PASS** | data-rule tests 9/9; IV coverage of index members 79.4 % (≥ 70 %) |
| 2 Screener | **PASS** | point-in-time filters, pass counts per quarter; model-layer safety score **dropped** (failed OOS) |
| 3 Vol models | **FAIL as pre-registered** | SVI, local vol and RV forecasts pass; FHS probability calibration 3.70 pts (limit 3.0) |
| 3B Attribution | **FAIL as pre-registered** | the "detect Sharpe 0.5 at 80 % power" criterion is statistically impossible on ≤ 14 years; everything else passes |
| 4 Strategy models | **PASS** | both models beat their fixed-rule baselines OOS; band hedge is on the time-hedge frontier |
| 5 Direction independence | **FAIL** | direction independence achieved (β 0.005, corr 0.07) but net return after costs is **negative** (CAGR −0.86 %, HAC t −2.96) |
| 6 Backtester | **PASS** | accounting, no-lookahead, fills, (early) assignment, hedge costs: all unit-tested |
| 7 Risk & execution | **PASS (tests)** | every limit blocks, kill switch, live port/account refused, paper flow vs fake broker, daily report. **Paper trading not run** |

**Bottom line: neither strategy, as built, earns a positive return after costs while staying direction-neutral.
Phase 5 fails, so the system is not ready for paper trading as a profit strategy.** Section 9 says what is worth
keeping and the one cheap next step that could change this conclusion (real historical option chains).

## 2. Phase 1: data

* Sources: Yahoo (prices, dividends, splits), IBKR historical 30-day IV (read-only requests, port 7496, no orders),
  SEC XBRL company facts and 8-K dates, Wikipedia S&P 500 membership history, Cboe VIX family and ^IRX. Detail and
  paid alternatives with costs: `DATA.md`.
* 773 tickers ever in the index since 2007 (after the 152 manual exclusions). IV status: 528 ok, 227 unknown to IBKR
  (delisted or renamed), 18 no data. 2,858 bad IV prints removed; IV carried forward at most 3 days.
* Coverage of index member-quarters since 2010: IV 79.4 % (62 % in 2010, 99 % in 2026), prices 81.5 %.
  **The ~20 % gap is the residual survivorship bias**: missing names are mostly companies that were acquired or
  failed, so results are tilted toward survivors (which flatters short puts).

## 3. Phase 2: fundamental screener

* Hard filters, point-in-time (facts used only once filed): net margin > 10 %; TTM FCF higher than 4 and 8 quarters
  earlier; TTM FCF YoY growth > 10 % (mid cap $2–10 B) / > 5 % (large ≥ $10 B); stock and option liquidity.
  The "> x %" reading of the FCF rule is SPEC conflict #1.
* 31,584 symbol-quarters evaluated; 22–68 names pass per quarter (25 in 2012, 69 in 2026). Pass counts per filter
  per quarter: `phase2_pass_counts.csv`. XBRL is thin before 2012, so early quarters pass few names.
* Model layer (a "safety" score meant to predict next-quarter short-put tail loss): mean rank IC **+0.048** over
  62 quarters (one-sided bootstrap p = 0.98: the wrong sign). **Dropped**; ranking falls back to dollar volume.

## 4. Phase 3: volatility models

| Test (pre-registered) | Result | Pass |
|---|---|---|
| SVI median IV RMSE ≤ 0.5 vol pt (real chains) | 0.366 over 136 slices | ✔ |
| ≥ 90 % slices butterfly- and calendar-clean (Durrleman g, Lee, calendar) | 99.3 % | ✔ |
| Local vol (Gatheral eq. 1.10 → Dupire PDE) reprices 80–120 % vanillas within 0.5 vol pt | max 0.129 over 1,805 quotes (772 quotes on calendar-arbitraged slices excluded) | ✔ |
| HAR/GARCH beats trailing RV on QLIKE for ≥ 60 % of names | HAR-IV better for 99.5 % of 201 names | ✔ |
| Pooled Diebold-Mariano significant | HAR-IV t = −13.2 (GJR −7.5, GARCH −7.1, HAR −8.1) | ✔ |
| FHS P(S_T < K) calibration error ≤ 3 pts and ≤ BS at IV | **3.70 pts** (BS at IV: 6.19) over 106,020 forecasts | ✘ |

**Variance risk premium** (IV² against the next period's realized variance):

| | VRP (vol pts) | IV > RV |
|---|---|---|
| SPX 9-day (VIX9D) | 3.6 | 82 % |
| SPX 30-day (VIX) | 3.7 | 85 % |
| SPX 3-month | 5.1 | 86 % |
| SPX 6-month | 6.1 | 86 % |
| Single names, 30-day (mean of 201) | 2.0 | 64.5 % |

Real SPX quotes on 2026-11-08 against the model's physical fair value: every put is RICH (ATM 14.5 % vs fair 13.1 %;
80 % strike 38 % vs fair 32 %). `phase3_option_signals_real_spx.csv`.

**Why FHS failed, and what I did.** The diagnostic (`phase3_diagnostics.json`) re-scores the same forecasts with a
point-in-time drift (trailing 5-year mean return, known at t). On all 418 names (223,626 forecasts) the error falls
from 3.69 to **1.91 pts**. The shape of
the distribution (vol level, tails) is calibrated. The miss comes from the deliberate drift-neutral mean
(E[S_T] = forward): stocks drifted up over 2010–2026, so puts finished in the money less often than a martingale says.

SPEC §6 says a model that fails is **dropped** and the phase fails "only if a later phase cannot work without it".
I applied that narrowly. FHS **probabilities** are not used for any decision. FHS **paths** are still used to estimate
the P&L distribution of a *delta-hedged* option, whose expected value depends on realized variance, not drift
(P&L ≈ ½ΓS²(σ²_imp − σ²_real) dt; Taleb, *Dynamic Hedging*; Ahmad & Wilmott 2005). **This is my judgment call. Under a
strict reading of "do not proceed on a fail", work would have stopped here.** Phase 3 stays FAIL in the record.

## 5. Phase 3B: Greeks under smile dynamics, attribution, alpha vs premium

* **Greeks under three smile regimes** on the real SPX surface (Derman; σ_new(k) = σ_old(k + c·ln F′/F), c = 0 / −1 / +1).
  For a 90 % put, delta is −0.072 sticky-strike, −0.027 sticky-delta, −0.118 local vol. The hedge ratio of an OTM put
  can differ by a factor of 4 depending on the regime. The book hedges sticky-strike (SPEC conflict #6) and reports
  the spread.
* **Vanna-volga** (Castagna-Mercurio first order) from three SVI pivots: RMSE 0.16 vol pt between the pivots
  (SVI 0.10), but 9.4 across all strikes (SVI 0.29). **Dropped as a signal** (SPEC conflict #8).
* **Attribution** (Taylor: delta, gamma, vega, theta, vanna, volga, residual), walk-forward OOS 2013–2026, in $ on $1 M:

| Trade type | delta | gamma | vega | theta | vanna | volga | residual | option total | option costs |
|---|---|---|---|---|---|---|---|---|---|
| short put (wheel) | +142,437 | −265,185 | −106,966 | +362,907 | +25,550 | −15,717 | +18,616 | +161,641 | 12,443 |
| covered call | −36,341 | −124,048 | −33,392 | +135,746 | +18,324 | −6,301 | +8,147 | −37,865 | 4,806 |
| long straddle | +41,087 | +675,950 | +17,080 | −625,745 | −32,393 | +15,251 | −1,397 | +89,832 | 129,985 |
| SPY tail put | −392 | +856 | +1,060 | −3,235 | −4 | +201 | +50 | −1,464 | 23 |

  The delta column is offset by the stock hedges (separate line, not shown). The straddle's gross gamma − theta is
  +$50 k over 14 years. Option spreads and commissions are $130 k: **costs, not the model, sink the straddle.**
* **Classification** (residual after costs; HAC t, stationary bootstrap CI, Benjamini-Hochberg):
  short put: risk premium (t 1.77); straddle: risk premium (residual −$41/day, t −10.7); tail: too few days (86).
  Covered calls show a significant residual (+$1.50/day, t 2.17). That is **not alpha**: on model-priced history the
  residual is just the higher-order terms of our own pricing model (SPEC §3). The label is capped at "risk premium"
  in code, and the raw statistic is kept.
* **Residual regression** on SPY return, ΔVIX, ΔVIX² (HAC + bootstrap CIs): `analysis.json → residual_regression`.
  The residual loads negatively on SPY (t −2.8) and ΔVIX (t −1.9), not on ΔVIX² (t 0.5): it is market-linked, which
  is one more reason not to call it alpha.
* **Why "FAIL as pre-registered":** the criterion "detects an injected Sharpe 0.5 in ≥ 80 % of simulations" cannot be
  met. The minimum detectable Sharpe at 80 % power is about 2.8/√years, so detecting 0.5 needs ~31 years of daily data.
  The test that *can* be run (size ≤ 10 %, power ≥ 80 % at the minimum detectable effect) passes.
  Exposure and premium by strike × DTE: `phase3b_exposure_by_strike_dte.csv`, `phase3b_premium_by_strike_dte.csv`.

## 6. Phase 4: strategy models

Walk-forward: 3-year training window, 6-month test, from 2013-01, pick the best trailing-Sharpe configuration, stitch
the test periods. 24 full-history runs (9 wheel grid, 5 wheel comparisons, 9 straddle grid, 1 straddle baseline).

| | Model (walk-forward OOS) | Fixed-rule baseline (OOS) | Decision |
|---|---|---|---|
| Wheel (expected-edge − λ·CVaR optimizer, hedged) | Sharpe −0.06 | 0.30-delta / 35-DTE, hedged: Sharpe −0.50 | keep (beats baseline) |
| Straddle (forecast-RV vs IV edge, event-aware) | Sharpe −0.77 | always-on ATM straddle: Sharpe −0.83 | keep (beats baseline) |

"Beats the baseline" only means less negative. Deflated Sharpe ratio (9 trials each): wheel 0.03, straddle 0.0001.

**Hedging frontier** (wheel, λ = 0.1, OOS), Whalley-Wilmott band vs fixed-interval hedging:

| Policy | Hedge cost (% NAV/yr) | Mean \|delta\| (% NAV) | Sharpe |
|---|---|---|---|
| band ×0.5 | 0.14 | 0.23 | −0.30 |
| band ×1 | 0.11 | 0.41 | −0.13 |
| band ×2 | 0.09 | 0.70 | +0.09 |
| daily | 0.19 | 0.01 | −0.47 |
| every 5 days | 0.08 | 0.85 | +0.21 |
| none | 0 | 10.3 | +0.41 |

No band point is dominated by time hedging: **band kept** (selected: ×2). Wider hedging scores better because every
hedge trade costs money and the short-put P&L is mildly positive in delta.

## 7. Phase 5: direction independence (FAIL)

| Walk-forward OOS 2013-01 → 2026-11 | Wheel | Straddle | Combined |
|---|---|---|---|
| CAGR | −0.03 % | −0.83 % | **−0.86 %** |
| Annual vol | 0.57 % | 1.08 % | 1.12 % |
| Sharpe | −0.06 | −0.77 | −0.77 |
| Max drawdown | −2.1 % | −13.4 % | −13.6 % |
| β to SPY (NW t) | 0.012 (4.3) | −0.007 (−4.7) | **0.005 (1.5)** ✔ |
| Correlation to SPY | 0.34 | −0.11 | **0.07** ✔ |
| Per-name P&L corr. to its stock (held-day weighted) | 0.09 | 0.06 | ✔ |

P&L by regime (combined, mean monthly return): up markets −0.06 %, flat −0.09 %, down −0.06 %. High VIX −0.78 %/yr,
low VIX −0.94 %/yr. COVID crash (Feb–Mar 2020) −0.98 % (SPY −13.2 %); 2022 −1.15 % (SPY −18.2 %).
**The book is direction-independent: it loses about the same in every market.**

Calendar years (%):

| | 2013 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 2026 YTD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wheel | 0.56 | −0.24 | 0.03 | 0.00 | 0.47 | 0.06 | 0.14 | −0.53 | 0.30 | 0.24 | 0.17 | −0.44 | −0.37 | −0.85 |
| straddle | 0.69 | −0.34 | −0.55 | −2.16 | −2.44 | 1.75 | −1.55 | −0.73 | −1.01 | −1.30 | −1.47 | −1.17 | −1.22 | 0.10 |

**What the hedge costs.** Wheel stock hedging: 0.105 % NAV/yr in commissions and spread, plus 0.013 % borrow, about
64 shares/day. The bigger cost is what hedging gives up. The unhedged 0.30-delta wheel earned **+1.13 %/yr, Sharpe
0.64**, but with β 0.075 (t 13.9) and correlation 0.70 to SPY, and its alpha t-stat is **0.08**. Hedged, the same
rules earn −0.40 %/yr. **The unhedged wheel's profit is the equity risk premium collected through short puts, not an
option premium.** That matters for a live wheel run the McMillan way: remove the direction and, at these costs,
little is left.

P&L by underlying (wheel, OOS): SPY +$5,903, 136 single names −$10,677 (worst MU −$8.2 k). The index has the largest
VRP (3.7 vs 2.0 vol pts) and the tightest spreads. This is an observation from the test period, **not** a validated
result: choosing "index only" now would be selection on out-of-sample data.

## 8. Phases 6–7: engine, limits, execution

* Engine: daily, event-driven. Order of each day: mark and attribute → fill yesterday's orders → stock, dividends,
  borrow → expiry and early assignment → hedge → kill switch → decisions. Accounting reconciles to the cent every day,
  per underlying too. Changing future data leaves past trades and NAV identical. Early assignment: a short put with
  time value below max(0.02, K(1 − e^{−rτ})), or a short call before a dividend larger than its time value (McMillan; Hull).
* Hard limits (% NAV): net delta 5, vega 0.4, vanna/volga 0.2, margin 50, sector 30, name 15, per-position max loss 2;
  liquidity: OI ≥ 500, spread ≤ 10 %, stock $50 M/day. Kill switch at −3 % day or −15 % drawdown, which
  stays tripped until released (21-day cooldown in backtests). It never fired on the stitched walk-forward path or in
  any wheel run. It did fire (15 % drawdown) in 7 of the 10 full-history straddle runs, e.g. the always-on baseline
  in mid-2014.
* Executor: `PaperExecutor` refuses port 7496/4001 before connecting and any account not starting `DU`. Every order is
  re-checked against the kill switch, liquidity and limits, then sent as a limit order. Audit log, daily marks and
  daily report are in `results/premia/paper/`.
* **Not done: the 2–3 month paper-trading period.** No paper TWS session was available, and it needs calendar time.
  Stock hedge orders are computed and logged by `premia.paper` but not sent (option orders are).

## 9. Final recommendation

**Strategies**
* **Long straddle: drop.** It buys the premium that Phase 3 measures (IV above later realized vol 82–86 % of the time on
  SPX, 64.5 % on single names) and pays a 3 % half-spread on two legs. The model's edge threshold cut losses
  monotonically (full-period NAV 757 k → 814 k → 875 k for thresholds 0 / 2 / 4 vol pts), which says "trade less",
  not "trade". Keep it only as a conditional hedge for a short-vol book, not as a profit sleeve.
* **Hedged wheel: do not deploy; keep as a research line.** OOS Sharpe −0.06, DSR 0.03: no evidence of edge after costs.
  It is the one sleeve whose gross P&L has the right sign (short-put option P&L +$162 k over 14 years before hedges).
* **Unhedged wheel: outside the brief.** Positive (Sharpe 0.64) but it is equity beta (corr 0.70, alpha t 0.08).

**Models and filters to keep:** SVI with arbitrage checks; local vol (for regime Greeks); HAR-IV RV forecast (the
strongest result here, DM t −13); FHS for hedged-P&L distributions only; the Taylor attribution and classifier;
Whalley-Wilmott band hedging (×2); hard limits and kill switch; the liquidity filter (costs decide everything here).
**Drop:** the fundamental safety score, FHS probabilities, vanna-volga as a signal. The fundamental hard filters stay as
the universe definition the brief asked for, but they were never shown to add value to a hedged book (untested).

**The one next step that could change the verdict:** buy historical option chains (ORATS ~$99–299/mo, or ThetaData from
~$40/mo; `DATA.md`) and re-run phases 3B–5 on real quotes. The modelled smile holds skew constant in normalized terms,
so the backtest cannot see whether the skew premium on OTM puts (where a short-put book earns most) varies over time.
Real spreads may also be tighter than the flat 3 % model for liquid names. Until then, the honest use of the paper
account is a 2–3 month **index-only hedged put book** to collect real marks for the alpha/premium classification.
That is a data-collection exercise, not a deployment decision.

**Sensitivity of the selected parameters** (λ 0.1 / band ×2 wheel; 4-pt / band ×2 straddle):

| Run (OOS 2013→) | Wheel Sharpe | Wheel HAC t | Straddle Sharpe | Straddle HAC t |
|---|---|---|---|---|
| as selected | +0.09 | 0.37 | −0.64 | −2.33 |
| smile skew ×0.5 | −0.22 | −0.94 | −0.51 | −1.88 |
| smile skew ×1.5 | **+0.83** | **2.84** | −0.66 | −2.47 |
| fills at the full half-spread (slippage ×2) | +0.06 | 0.24 | −1.28 | −4.77 |

The straddle loses in every variant. **The wheel's result depends on the skew assumption.** With 50 % more skew in
the modelled smile it is significantly positive; with half the skew it is negative. The hedged short-put book's
edge is the skew premium, and the skew is exactly what the backtest models instead of observing. That is why real
historical chains are the deciding next step. The ×1.5 run is a sensitivity check, not a result to select.

## 10. What was NOT implemented, and why

| Source | Not implemented | Why |
|---|---|---|
| Gatheral | Heston / Heston-Nandi calibration, jumps (ch. 2–5), default (ch. 6), exotics (ch. 9–10) | no historical chains to calibrate dynamics against; outside two-strategy scope. SVI + local vol cover the static surface |
| Gatheral ch. 11 | variance-swap trading | used only as the VRP lens (VIX family) |
| Derman | Derman-Kani implied trees | replaced by continuous Dupire/Gatheral local vol (same object, verified by PDE repricing) |
| Derman | stochastic-vol and jump smile models | regime Greeks (sticky strike/delta/local) give the hedge-ratio spread without them |
| Taleb | liquidity holes, multi-asset correlation books, exotic hedging, trader heuristics | need intraday and cross-asset data; heuristics are not quantitative rules |
| McMillan | other structures (ratio writes, calendars, backspreads), intraday follow-up tactics | outside the two-strategy scope |
| McMillan | own-history IV/RV percentiles as an entry *signal* | used as regime labels only; the edge model uses forecast RV vs IV (SPEC conflict #4) |
| Castagna / Wystup | vanna-volga as a trading signal | failed its test (RMSE 9.4 vs SVI 0.29 off the pivots); FX method |
| Spec phase 3B | power to detect Sharpe 0.5 | statistically impossible on available history (§5) |
| Spec phase 3B | real alpha classification | needs real quotes; every trade type defaults to risk premium (SPEC §3) |
| Spec phase 4 | weekly expiries, intraday hedging | modelled monthly third-Friday grid; daily close data only |
| Spec phase 6 | historical fills, open-interest history | need paid chain data |
| Spec phase 7 | 2–3 months of paper trading; automated stock hedge orders | needs a paper TWS session and calendar time; hedges are logged, not sent |
| Data | 227 delisted/renamed tickers' IV | IBKR does not serve them, which leaves the ~20 % survivorship gap |

## 11. Limitations, stated plainly

* Option prices in backtests are modelled (real ATM IV × fixed normalized smile × spread model), not historical quotes.
* Monthly expiries only; one 30-day IV per name per day; term structure from the VIX family for SPY, flat elsewhere.
* Sizing is conservative: average margin use about 16 % of NAV, short vega about $100 per vol point per $1 M, so
  returns are small in absolute terms. Sharpe and t-stats, not CAGR, are the meaningful numbers.
* Survivorship: about 20 % of member-quarters lack IV (delisted names), which flatters short puts.
* Phase 3 was run on the 201 names with IV at the time; the walk-forward used all 418 cached names.
