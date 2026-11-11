# premia — direction-neutral options research system (wheel + long straddle)

Models decide: fair value, expected edge, strike/expiry, size, hedge ratio. Hard rules in `risk.py` limit them.
Profit is meant to come from volatility and time-decay premia, not from direction. The wheel is delta-hedged,
which makes it a delta-hedged short-put book (SPEC.md §1).

* `SPEC.md`: phase 0 (book → model → module → validation test, conflicts, what is not codable, pre-registered pass/fail criteria)
* `DATA.md`: phase 1 (sources, handling, survivorship, paid recommendations with costs)
* `RESULTS.md`: per-phase results and verdicts, final recommendation, what was not implemented

> Research and **paper** trading only. `execution.py` refuses the live port and any non-paper (non-DU) account.
> Backtests price options from real 30-day ATM implied vol but a *modelled* smile (SPEC.md §3): they are not predictions.

## Modules

| Module | Phase | What |
|---|---|---|
| `data.py`, `store.py`, `fetch.py` | 1 | Yahoo prices/dividends/splits, IBKR 20-year IV history (read-only), SEC XBRL and 8-K dates, S&P 500 membership; cleaning and point-in-time rules |
| `fundamentals.py`, `screener.py` | 2 | point-in-time hard filters, liquidity, model-layer score and its out-of-sample test |
| `vol.py`, `rv.py`, `physical.py`, `pricing.py` | 3 | local vol from SVI (Gatheral eq. 1.10) + Dupire PDE check; HAR / GARCH / GJR forecasts; filtered historical simulation; BSM Greeks incl. vanna/volga |
| `greeks.py`, `attribution.py`, `stats.py` | 3B | surface Greeks under sticky-strike / sticky-delta / local-vol; vanna-volga; Taylor P&L attribution; HAC, bootstrap, deflated Sharpe, BH; alpha vs risk-premium classification |
| `strategies.py`, `hedging.py`, `market.py` | 4 | wheel optimizer, straddle entry/exit, Whalley-Wilmott band / time hedging, tail protection; the modelled option market |
| `validation.py` | 5 | beta, correlations, regimes, net-delta profile, hedge cost |
| `backtest.py` | 6 | daily event-driven engine: next-bar fills, costs, early assignment, accounting, attribution |
| `risk.py`, `execution.py`, `report.py`, `paper.py` | 7 | hard limits, kill switch, paper executor, daily marks and report, daily paper loop |

## Reproduce

```powershell
python -m premia.fetch all                        # ~2-4 h, IBKR TWS on 7496 for read-only IV history (pacing-limited)
python -m premia.research.phase2                  # point-in-time screen (~1.5 h on 8 cores)
python -m premia.research.shapes                  # real chains -> SVI -> smile shapes (TWS, read-only)
python -m premia.research.build                   # per-underlying forecasts + GJR fits (cache)
python -m premia.research.phase3                  # SVI / local vol / RV / VRP / calibration / option signals
python -m premia.research.phase3b                 # regime Greeks and vanna-volga on the real SPX surface
python -m premia.research.walkforward run         # 25 backtests (8 processes)
python -m premia.research.analysis                # walk-forward selection, phases 3B-6, verdicts
python -m premia.research.walkforward sensitivity # skew x0.5 / x1.5, full-spread fills
python -m pytest tests/premia -q                  # all tests are offline; no order can be sent
```

Paper trading (TWS logged in to a **paper** account, API port 7497):

```powershell
python -m premia.paper --names SPY,MSFT           # dry run: proposals, rule checks, hedges, marks, report
python -m premia.paper --names SPY,MSFT --execute # send limit orders to the paper account
```

Outputs: `results/premia/` (CSV/JSON per phase, `runs/*.pkl` backtests, `paper/` audit log, marks, kill switch, reports).
Data: `premia_data.db` (research store, separate from the dashboard's `wheel.db`).
