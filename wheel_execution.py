"""
wheel_execution.py — [Phase 5 stub] Manual-confirm trade execution for the
options wheel dashboard.

This dashboard is decision-support, not an auto-trading bot: nothing in this
module (or anywhere else in the wheel_* codebase) may submit an order
without an explicit human click in the dashboard "Execute" button.

Planned functionality
----------------------
- stage_limit_order(recommendation): build an ib_insync Option contract +
  LimitOrder (SELL, quantity=recommendation.contracts_suggested, limit price
  defaulted to the recommendation's premium) and return it *unsubmitted* for
  the dashboard to display as a confirmation dialog.
- submit_staged_order(ib, contract, order): the only function in this whole
  project allowed to call ib.placeOrder(); only reachable from the dashboard
  after the user reviews the staged order and clicks confirm a second time.

Not implemented yet -- the Phase 1 dashboard's "Execute" button is disabled
and calls into this module purely to show the "coming in Phase 5" message.
"""

from __future__ import annotations


def stage_limit_order(*args, **kwargs):
    raise NotImplementedError(
        "wheel_execution.stage_limit_order is a Phase 5 feature — not yet implemented. "
        "No orders can be placed from this dashboard yet."
    )


def submit_staged_order(*args, **kwargs):
    raise NotImplementedError(
        "wheel_execution.submit_staged_order is a Phase 5 feature — not yet implemented. "
        "No orders can be placed from this dashboard yet."
    )
