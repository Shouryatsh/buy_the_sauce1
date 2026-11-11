# premia — specification (phase 0)

A model-driven options system for two strategies, a **delta-hedged wheel** (short premium) and a **long straddle** (long
volatility). It has a rules-based risk layer, a walk-forward backtester and an IBKR **paper** executor.
The goal is profit from volatility and time-decay premia (and genuine mispricing), with **no view on direction**.

This document was written *before* any result was computed. The pass/fail criteria in section 6 are fixed here and are
not changed after seeing results.

## 1. Architecture: models decide, rules limit

```
data (phase 1) ──> screener (2) ──> vol models (3) ──> greeks & attribution (3B) ──> strategy models (4)
                                                                                        │
                       hard limits + kill switch (7)  <──────────── proposed trades ────┘
                                    │
                    backtester (6) / paper executor (7) ──> daily report, P&L attribution ──> (3B) evidence
```

| Layer | Decides | Can be overridden by |
|---|---|---|
| Models (`premia.vol`, `premia.physical`, `premia.strategies`, `premia.hedging`) | fair value, expected edge, strike/DTE, size, hedge ratio, rebalancing | the rules only |
| Rules (`premia.risk`) | max loss (position, day, drawdown), net delta/vega/vanna/volga, margin, name/sector concentration, liquidity, kill switch | nothing (a breach blocks the order or trips the kill switch) |

### The key structural fact (connects McMillan to Taleb and Gatheral)
A plain wheel is short a put, then long stock plus short a call. Put-call parity says *long stock + short call = short
put + cash*, so **both legs of the wheel are economically a short put**. Delta-hedging the whole wheel therefore turns it into a
**delta-hedged short put book**. Its expected P&L is the volatility risk premium, not the stock's drift. For a
delta-hedged option the P&L over a short interval is approximately

  dΠ ≈ ½ Γ S² (σ_imp² − σ_real²) dt + (vanna, volga and jump terms)

(Taleb, *Dynamic Hedging*, gamma/theta "bleed" discussion; Ahmad & Wilmott 2005; Bakshi & Kapadia 2003 for the empirical
sign). The long straddle hedged the same way earns the mirror image. **The two strategies are opposite bets on the same
quantity (σ_imp − σ_real).** The system is direction-neutral only once both are hedged, and the risk that remains is a
volatility risk: it is not hedged away and must be sized and capped.

## 2. Book → model → module map

Notation: S spot, F forward, K strike, k = ln(K/F), T years, w = σ²T total implied variance.

### Gatheral — *The Volatility Surface* (2006) and Gatheral & Jacquier (2014)
Chapter numbers below are from memory of the 2006 edition. Check them against your copy; the papers cited alongside
are exact.

| Model | Source | Inputs | Outputs | Module | Validation test |
|---|---|---|---|---|---|
| Raw SVI per expiry: w(k)=a+b(ρ(k−m)+√((k−m)²+σ²)) | Gatheral ch. 3; Gatheral & Jacquier (2014) §3 | OTM bid/ask quotes, F, T | 5 params/slice, fit error | `volsurface.svi` (reused) | synthetic chain from known SVI recovered; real SPX snapshots: IV RMSE, % inside bid-ask |
| Forward/discount from put-call parity | standard; Gatheral ch. 3 practice | C, P pairs near ATM | F, D | `volsurface.surface` | round-trip on synthetic chain |
| Butterfly no-arbitrage g(k) ≥ 0 | Durrleman (2005); Gatheral & Jacquier (2014) Thm 2.1 | SVI params | g(k), density, violations in/out of quoted range | `volsurface.arbitrage` | known-arbitrageable SVI (G&J's Axel Vogt example) flagged |
| Calendar no-arbitrage ∂_T w(k,T) ≥ 0 | Gatheral & Jacquier (2014) Lemma 2.1; Gatheral ch. 3 | adjacent slices | crossings | `volsurface.arbitrage` | crossing slices flagged |
| Wing slope bound b(1+\|ρ\|) ≤ 2 | Lee (2004); Gatheral ch. 7 | SVI params | bound check | `volsurface.svi` | |
| Local volatility from implied total variance | Gatheral ch. 1 eq. (1.10); Dupire (1994) | fitted surface w(k,T) | σ_loc(k,T) | `premia.vol.local_vol` | flat surface gives flat local vol; PDE under σ_loc reprices the surface's vanillas (≤ 0.5 vol pt) |
| Surface dynamics, skew-stickiness | Gatheral ch. 8 | surface history | regime sensitivity | `premia.greeks` | see Derman regimes |
| Black-76 pricing on the forward | Black (1976) | F, K, T, σ, D | price, Greeks | `premia.pricing` | put-call parity, finite-difference Greeks |

### Derman — *The Volatility Smile* (Derman & Miller, 2016) and Derman's papers

| Model | Source | Inputs | Outputs | Module | Validation |
|---|---|---|---|---|---|
| Smile regimes: sticky-strike, sticky-delta (moneyness), sticky-implied-tree (local vol) | Derman (1999) "Regimes of Volatility"; Derman & Miller (2016), smile-dynamics chapters | surface, spot shift | σ(K) after a spot move: σ_new(k₀)=σ_old(k₀+c·ln(F′/F)), c = 0, −1, +1 | `premia.greeks` | sticky-strike Greeks equal Black-Scholes Greeks; regimes ordered as Derman describes on a negative skew |
| Local vol (implied tree) | Derman & Kani (1994); Dupire (1994) | surface | σ_loc | `premia.vol` | as above |
| Static replication / variance-swap view of VRP | Derman & Miller (2016) variance-swap chapters; Carr & Wu (2009) | IV, RV | VRP | `premia.vrp` | VRP sign and size match the literature on SPX |

### Taleb — *Dynamic Hedging* (1997)
Taleb is cited by topic. His text predates the papers that give exact formulas, so the formulas cite those papers.

| Model | Source | Inputs | Outputs | Module | Validation |
|---|---|---|---|---|---|
| Delta-hedging engine: time-based vs band rebalancing | Taleb (rebalancing frequency, transaction costs); band width: Whalley & Wilmott (1997) H = (3/2 · e^{−r(T−t)} λ S Γ² / γ)^{1/3}; Leland (1985) cost-adjusted vol | position Greeks, cost λ, risk aversion γ | hedge trades | `premia.hedging` | on simulated GBM, the hedge error shrinks with frequency and costs grow, and the band dominates the time grid on the cost/variance frontier (out of sample) |
| Second-order Greeks: vanna (∂Δ/∂σ, Taleb's "DdeltaDvol"), volga (∂vega/∂σ, "DvegaDvol") | Taleb; definitions as in Castagna (2010) ch. 1 | surface, regime | vanna, volga per contract | `premia.greeks` | finite differences vs closed form under sticky-strike |
| Tail / stress scenarios, "shadow" Greeks under large moves | Taleb (stress testing, shadow gamma) | portfolio | P&L grid over ±spot × ±vol, gap scenarios | `premia.risk` | scenario P&L equals full revaluation |
| Tail protection sized from volga and stress loss | Taleb (convexity, buying wings) | stress loss, volga | long-wing quantity | `premia.hedging.tail_hedge` | worst stress loss after protection ≤ limit |

### McMillan — *Profit with Options* (2002), with *Options as a Strategic Investment* for detail
The rules are cited by topic; chapter numbers were not verified.

| Rule/model | Source | Inputs | Outputs | Module | Validation |
|---|---|---|---|---|---|
| IV and RV each judged against **their own** history (percentiles), never one against the other's level | McMillan, volatility-trading chapters | IV history, RV history | percentiles, regime labels | `ivscreen.metrics` (reused), `premia.vrp` | |
| Naked-put / covered-write structure, strike and expiry choice, rolling ("roll down and out" when tested, take profits early) | McMillan, covered writing and naked put selling | position, price, DTE | roll/close/hold | `premia.strategies.wheel` (guardrails) | state-machine tests |
| Straddle buying: buy when IV is low against its own history; judge the trade by the probability of the stock moving past the break-even (Monte Carlo) | McMillan, straddle-buying chapter | IV percentile, simulated paths | entry, exit | `premia.strategies.straddle` | |
| Earnings avoidance for sold premium; earnings as a catalyst for bought premium | McMillan | earnings dates | allowed expiries | `premia.strategies` | no short expiry spans an earnings date |
| Early assignment risk (deep ITM, little time value; calls before ex-dividend) | McMillan; Hull (American options) | option, dividend | assignment event | `premia.backtest` | |

### Statistics, data and other sources

| Model | Source | Module |
|---|---|---|
| Realized-vol estimators (close-to-close, Parkinson, Garman-Klass, Yang-Zhang) | Parkinson (1980); Garman & Klass (1980); Yang & Zhang (2000) | `premia.rv` |
| HAR-RV forecast | Corsi (2009) | `premia.rv` |
| GARCH(1,1), GJR-GARCH | Bollerslev (1986); Glosten, Jagannathan & Runkle (1993) | `premia.rv` |
| Forecast loss (QLIKE) and comparison test | Patton (2011); Diebold & Mariano (1995) | `premia.rv` |
| Physical distribution by filtered historical simulation | Barone-Adesi, Engle & Mancini (2008) | `premia.physical` |
| Expected P&L of an option hedged at a chosen vol | Ahmad & Wilmott (2005); Carr ("FAQs in option pricing theory") | `premia.physical` |
| Volatility risk premium | Carr & Wu (2009); Bakshi & Kapadia (2003) | `premia.vrp` |
| Vanna-volga price | Castagna & Mercurio (2007); Castagna (2010); Wystup (2006). This is an FX method, not from the four books | `premia.greeks.vanna_volga` |
| Stationary bootstrap | Politis & Romano (1994) | `premia.stats` |
| HAC standard errors | Newey & West (1987) | `premia.stats` |
| Deflated Sharpe ratio | Bailey & López de Prado (2014) | `premia.stats` |
| False-discovery control | Benjamini & Hochberg (1995) | `premia.stats` |

## 3. Data reality and what it means for validation
No free source has historical option *chains*. The data used here (detail and paid alternatives in DATA.md):

* **Real**: daily underlying OHLCV and dividends (Yahoo), and **30-day ATM implied volatility since 2006** (IBKR, about 770
  index members and ETFs). For SPX there is also the VIX family by tenor, CBOE SKEW and the T-bill rate. Fundamentals are
  point in time (SEC filing dates), and earnings dates come from SEC 8-K item 2.02.
* **Modelled**: the smile away from the money and the term structure away from 30 days.
  * Historical OTM vols are ATM IV × a fixed normalized SVI shape fitted on current real chains. A sensitivity run scales
    the skew by 0.5× and 1.5×.
  * Bid/ask comes from a spread model.
  * Single-name term structure is flat; for SPY it is scaled by VIX9D/VIX/VIX3M.

Consequences, stated now so they are not discovered later:

1. **The ATM volatility risk premium is measured on real data** (real IV against the real later path).
2. **The skew premium is not measured.** Results for 10–20-delta puts depend on the assumed skew shape. The sensitivity
   run shows by how much.
3. **"Alpha" (phase 3B residual) cannot be established on model-priced history.** By construction the model explains
   the model's own prices. The attribution and statistics machinery is validated on simulated data with known injected
   alpha. The real classification needs real quotes, which the paper-trading period collects
   (`premia.report` stores daily marks). Until then every trade type is classified **risk premium**, the conservative
   default in phase 3B step 6.

## 4. Conflicts, ambiguities and decisions

| # | Issue | Decision |
|---|---|---|
| 1 | "FCF increasing and > 10 % (mid-cap) or > 5 % (large-cap)": the base of the % is not stated (growth, margin or yield?) | Read as **year-over-year growth of trailing-12-month FCF** (> 10 % mid, > 5 % large). "Increasing" means TTM FCF is higher than 4 *and* 8 quarters earlier (2 years). FCF = operating cash flow − capex, as filed. Margin and yield variants are configurable in `config.Screen` and not used. |
| 2 | Cap cutoffs | Point-in-time market cap = shares outstanding (latest filing on or before the date) × unadjusted close. Mid $2–10 B, large ≥ $10 B, small (< $2 B) excluded. |
| 3 | Wheel vs direction independence (the prompt's own note) | Hedge the whole wheel's delta with the underlying stock (short stock against short puts; assigned stock is mostly sold down by the hedge). The result is a delta-hedged short put book. Hedge cost (commissions, spread, borrow fee) is reported separately. |
| 4 | McMillan's own-history percentiles vs Gatheral/Taleb's IV − RV spread | Both are used, for different jobs. Percentiles (McMillan) are **regime labels** for the guardrails. The edge model uses **forecast** RV against IV, the quantity that drives the P&L of a hedged option (Taleb, Ahmad & Wilmott). IV is never compared with *trailing* RV as a signal; that mixes a forecast with a backward-looking number. |
| 5 | Phase 3B step 7 ("trade only where expected residual edge > 0") vs step 6 ("risk premium: keep, hedge and size-limit") | Alpha trades need positive expected residual. Risk-premium trades need positive expected premium net of costs **and** carry tighter size limits (`risk_premium_size_mult`). Both are subject to the portfolio vanna/volga caps. |
| 6 | Local vol (Derman/Dupire) vs sticky-delta trading practice for Greeks | Neither is assumed. Greeks are computed in all three regimes and the spread is reported (phase 3B step 1). Hedging uses sticky-strike by default (the conservative choice for a short-skew book, which under-hedges least in a sell-off). |
| 7 | Taleb: models are fragile, use stress tests; Gatheral: arbitrage-free models | Models give the edge, and stress scenarios (not model VaR) size the tail and set the limits. |
| 8 | Vanna-volga is an FX method (Castagna/Wystup) | Implemented as an optional signal only. It is kept only if it passes phase 3B step 5, which it cannot do without historical chains. |
| 9 | Phases 5 (validation) and 6 (backtester) are in the wrong dependency order | The backtester engine is built first. Phase 6 adds the realism features and phase 5's validation is reported on top. Verdicts stay per phase. |
| 10 | "Paper trade 2–3 months before live", against the user's standing rule that the live account is read-only | The executor refuses any non-paper account (DU…) and the live port. Going live is out of scope for this code. |

## 5. Not codable / not implemented (preliminary; final list in RESULTS.md)
* **Gatheral**: Heston and Heston-Nandi calibration (ch. 2, 4) and jump models (ch. 5) are not needed for the static
  surface and have no historical chains to calibrate dynamics against. Default modelling (ch. 6) is out of scope.
  Exotics (barriers, cliquets, ch. 9–10) are not traded. Volatility derivatives (ch. 11) are used only as the VRP lens.
* **Derman**: full implied-tree construction (Derman-Kani binomial trees) is replaced by the Dupire/Gatheral continuous
  local vol. Stochastic-vol and jump smile models are not implemented.
* **Taleb**: liquidity holes, multi-asset correlation books and exotic hedging cannot be coded meaningfully without
  intraday and cross-asset data. His trader heuristics are not quantitative rules and are not coded.
* **McMillan**: the wide catalogue of other structures (ratio writes, calendars, backspreads, combinations) is outside
  the two-strategy scope. Intraday "follow-up" tactics are not coded.
* **Data**: the skew premium, intraday hedging, real historical fills and open-interest history all need paid chain data
  (DATA.md).

## 6. Phase plan and pre-registered pass criteria
"Fail" means: stop, report, do not build the next phase on it. A model that fails its out-of-sample test is **dropped**
(recorded as such). That is a valid outcome and fails the phase only if a later phase cannot work without it.

| Phase | PASS requires |
|---|---|
| 1 Data | All data-rule tests pass (splits, dividends, bad prints, no forward-fill beyond 3 days, point-in-time fundamentals). IV coverage of index members (member-quarters since 2010, after exclusions) ≥ 70 %, with the shortfall reported as survivorship bias. |
| 2 Screener | Hard filters use only data filed on or before each date (test). Pass counts per filter per quarter reported. Model-layer score kept only if its out-of-sample rank correlation with the next quarter's realized short-put tail loss is negative and significant (p < 0.05, block bootstrap); otherwise dropped and ranking falls back to liquidity. |
| 3 Vol models | SVI on real snapshots: median IV RMSE ≤ 0.5 vol pt and ≥ 90 % of slices butterfly- and calendar-clean inside the quoted range. Local vol: PDE reprices the surface's vanillas within 0.5 vol pt (strikes 80–120 %). RV: HAR or GARCH beats the trailing-RV forecast out of sample on QLIKE for ≥ 60 % of names, with a significant pooled Diebold-Mariano test. Physical model: FHS probabilities of finishing below K calibrated out of sample (mean absolute calibration error ≤ 3 pts and no worse than Black-Scholes at IV). |
| 3B Attribution | On simulated paths with known dynamics, the Taylor attribution leaves a residual ≤ 10 % of gross P&L variance. The alpha test detects an injected alpha of 0.5 Sharpe in ≥ 80 % of simulations and has a false-positive rate ≤ 10 % without one. Every backtested trade type is classified (expected result: risk premium, see section 3). |
| 4 Strategies | The engines run to specification (tests). The wheel optimizer and the straddle entry model are kept only if they beat their fixed-rule baselines out of sample (net Sharpe). The band hedge is kept only if it is on or inside the time-hedge cost/variance frontier out of sample. |
| 5 Direction independence | Walk-forward test periods: \|β\| to SPX < 0.10 and not significant (Newey-West \|t\| < 2), \|correlation\| to SPX and to underlyings < 0.20, **and** net annualized return after costs > 0. P&L reported by up/down/flat market, high/low vol, 2020 and 2022. |
| 6 Backtester | Accounting reconciles (cash + marked positions = equity, every day). No-lookahead test (changing future data leaves past decisions unchanged). Fills, costs, early assignment and hedge costs are each unit-tested. |
| 7 Risk & execution | Every hard limit blocks in tests. The kill switch trips on daily loss and drawdown. The executor refuses live accounts and ports. End-to-end paper flow works against a fake broker. The daily report is generated. |
