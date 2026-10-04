"""
wheel_skew.py — [Phase 2 stub] Full IV smile/skew visualization + skew
diagnostics for the options wheel "hidden gems" engine.

Planned functionality
----------------------
- build_smile(chain): IV vs strike and IV vs delta curves per expiry, from
  the same reqTickers()/modelGreeks pull wheel_options.fetch_put_candidates
  already does (extend it to walk the full chain, not just the wheel's
  target-delta band).
- skew_diagnostics(smile): 25-delta risk reversal (IV(25d put) - IV(25d call)),
  skew slope (linear fit of IV vs delta), and curvature (quadratic term) --
  standard vol-surface diagnostics used to gauge how "bid" downside puts are
  relative to calls.

Not implemented yet -- wired in as a dashboard placeholder for Phase 1.
"""

from __future__ import annotations


def build_smile(*args, **kwargs):
    raise NotImplementedError("wheel_skew.build_smile is a Phase 2 feature — not yet implemented")


def skew_diagnostics(*args, **kwargs):
    raise NotImplementedError("wheel_skew.skew_diagnostics is a Phase 2 feature — not yet implemented")
