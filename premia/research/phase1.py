"""
Phase 1 runner: data-rule tests and the coverage / survivorship measurement.

    python -m premia.research.phase1
"""

from __future__ import annotations

import subprocess
import sys

from .. import data, store
from .common import save_csv, save_json, verdict


def main() -> dict:
    tests = subprocess.run([sys.executable, "-m", "pytest", "tests/premia/test_data.py", "-q"], capture_output=True, text=True)
    tests_ok = tests.returncode == 0
    m = store.load_membership()
    cov = data.coverage_report(m, "2010-01-01")
    save_csv("phase1_coverage", cov)
    st = store.fetch_status()
    counts = st.groupby(["kind", "status"]).size().rename("n").reset_index()
    save_csv("phase1_fetch_status", counts)
    mean_cov = float(cov["iv_coverage"].mean())
    px_cov = float((cov["with_prices"] / cov["members"]).mean())
    bad = 0
    for s in store.symbols_with("iv")[:]:
        _, b = data.clean_iv(store.load_iv(s))
        bad += int(b.sum())
    summary = {"iv_coverage_mean": mean_cov, "price_coverage_mean": px_cov, "bad_iv_prints_removed": bad,
               "coverage_first": cov.iloc[0].to_dict(), "coverage_last": cov.iloc[-1].to_dict(),
               "tests_tail": tests.stdout.strip().splitlines()[-1] if tests.stdout else tests.stderr[-300:]}
    save_json("phase1_summary", summary)
    v = verdict("phase1", [
        ("data-rule tests pass (splits, dividends, bad prints, carry-forward, membership, point-in-time facts)", tests_ok,
         summary["tests_tail"]),
        ("IV coverage of index members (member-quarters since 2010, after exclusions) >= 70 %", mean_cov >= 0.70,
         f"{mean_cov:.1%} mean (prices {px_cov:.1%}); the shortfall is the residual survivorship bias"),
    ])
    print(v)
    return v


if __name__ == "__main__":
    main()
