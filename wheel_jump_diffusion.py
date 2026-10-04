"""
wheel_jump_diffusion.py — [Phase 3 stub] Merton jump-diffusion fair-value IV
for the richness score, replacing the Phase 1 Black-Scholes-vs-realized-vol
proxy in wheel_options.py.

Planned functionality
----------------------
- calibrate_merton(atm_options, underlying_history): least-squares fit of the
  Merton jump-diffusion parameters (diffusion vol sigma, jump intensity
  lambda, mean jump size mu_j, jump vol sigma_j) to the underlying's own ATM
  option prices across expiries. QuantLib's AnalyticHestonEngine / a custom
  characteristic-function pricer are both reasonable implementations; a pure
  numpy/scipy least_squares fit against the Merton closed-form series is
  enough for Phase 3 and avoids adding QuantLib as a hard dependency.
- fair_value_iv(calibrated_params, strike, dte): backs out the IV a
  Black-Scholes price would need to match the Merton price at that
  strike/expiry, for direct comparison against market IV (replaces
  realized_vol as the richness score's denominator in wheel_options.py).

Optionally extend to Heston (stochastic vol) if jump-diffusion alone doesn't
explain the observed skew well.

Not implemented yet -- wheel_options.py keeps using the Phase 1 proxy until
this lands.
"""

from __future__ import annotations


def calibrate_merton(*args, **kwargs):
    raise NotImplementedError(
        "wheel_jump_diffusion.calibrate_merton is a Phase 3 feature — not yet implemented"
    )


def fair_value_iv(*args, **kwargs):
    raise NotImplementedError(
        "wheel_jump_diffusion.fair_value_iv is a Phase 3 feature — not yet implemented"
    )
