"""
premia — a model-driven, direction-neutral options research system (long straddle + delta-hedged wheel).

Profit is meant to come from the volatility and time-decay premia (and genuine mispricing), never from forecasting the
direction of the underlying. Models make the decisions (fair value, expected edge, size, hedge ratio); hard rules in
`premia.risk` cap what any model can do. See SPEC.md (phase 0), DATA.md (phase 1) and RESULTS.md (per-phase verdicts).

Research and paper trading only: `premia.execution` refuses any account that is not an IBKR paper account (DU...).
"""
