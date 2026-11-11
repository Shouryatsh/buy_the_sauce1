"""
Phases 6 (backtester) and 7 (risk & execution) verdicts: each pre-registered criterion maps to the unit tests that
check it (SPEC.md section 6). Paper trading itself (2-3 months on a paper account) is reported separately: it needs
a TWS paper login and calendar time, which no test can stand in for.

    python -m premia.research.phase67
"""

from __future__ import annotations

import subprocess
import sys

from .common import verdict

E, X = "tests/premia/test_engine.py", "tests/premia/test_execution.py"

PHASE6 = [
    ("accounting reconciles (cash + marked positions = equity, every day, also per underlying)",
     [f"{E}::test_accounting_reconciles_every_day", f"{E}::test_straddle_sleeve_runs_and_reconciles"]),
    ("no lookahead (changing future data leaves past decisions unchanged)",
     [f"{E}::test_no_lookahead_changing_the_future_leaves_the_past_unchanged"]),
    ("fills at the next bar with spread, slippage and commission",
     [f"{E}::test_orders_fill_on_the_next_bar_with_spread_and_commission"]),
    ("assignment at expiry and early assignment",
     [f"{E}::test_assignment_delivers_shares_at_the_strike", f"{E}::test_early_assignment_only_when_time_value_is_gone"]),
    ("hedge costs and hedge policies",
     [f"{E}::test_hedge_policies_and_the_hard_delta_cap", f"{E}::test_hedge_frequency_trades_cost_against_variance_on_gbm",
      f"{E}::test_ww_band_grows_with_cost_and_gamma_and_shrinks_with_risk_aversion"]),
]

PHASE7 = [
    ("every hard limit blocks", [f"{E}::test_every_greek_and_margin_limit_blocks",
                                 f"{E}::test_name_and_sector_concentration_block_and_risk_reducing_trades_pass",
                                 f"{E}::test_liquidity_rules", f"{X}::test_rules_block_illiquid_and_limit_breaching_orders"]),
    ("kill switch trips on daily loss and drawdown; blocks openings, not closings",
     [f"{E}::test_kill_switch_trips_on_daily_loss_and_drawdown_and_stays_until_released",
      f"{X}::test_kill_switch_blocks_openings_but_not_closings"]),
    ("executor refuses live ports and live accounts",
     [f"{X}::test_live_port_is_refused_before_any_connection",
      f"{X}::test_a_live_account_on_the_paper_port_is_refused_and_disconnected"]),
    ("end-to-end paper flow against a fake broker",
     [f"{X}::test_orders_pass_every_rule_then_are_limit_priced",
      f"{X}::test_paper_loop_proposals_need_a_live_quote_that_confirms_the_edge",
      f"{X}::test_paper_hedge_orders_respect_band_and_cap"]),
    ("daily report generated", [f"{X}::test_daily_report_attributes_pnl_from_stored_marks"]),
]


def _run(nodes: list[str]) -> tuple[bool, str]:
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", *nodes], capture_output=True, text=True)
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-200:]
    return r.returncode == 0, tail


def main() -> dict:
    out = {}
    for phase, spec in (("phase6", PHASE6), ("phase7", PHASE7)):
        checks = []
        for crit, nodes in spec:
            ok, tail = _run(nodes)
            checks.append((crit, ok, f"{tail} ({len(nodes)} test{'s' if len(nodes) > 1 else ''})"))
        out[phase] = verdict(phase, checks)
        print(phase, out[phase]["verdict"], [(c["criterion"][:40], c["pass"]) for c in out[phase]["checks"]])
    return out


if __name__ == "__main__":
    main()
