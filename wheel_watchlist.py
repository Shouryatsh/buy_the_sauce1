"""
wheel_watchlist.py — Universe for the options wheel screener.

Loads wheel_config.SHORTLIST_PATH (wheel_shortlist.json) if it exists --
that's the output of a one-time broad scan (scan_universe.py) across SEC
EDGAR's whole ticker universe, filtered to net margin >= 10% and positive
FCF (see wheel_universe.py). Falls back to the small curated default list
below until you've run that scan at least once.

Run `python scan_universe.py` to (re)generate the shortlist; nothing else
in the dashboard needs to change afterward -- every tab already reads
WHEEL_WATCHLIST from here.
"""

import wheel_config as wc
from wheel_universe import load_shortlist

_DEFAULT_WATCHLIST: list[str] = [
    # Technology
    "AAPL", "MSFT", "CRM", "ADBE", "ORCL", "INTC", "CSCO", "AMD", "QCOM", "IBM",
    # Consumer Discretionary / Staples
    "COST", "WMT", "HD", "NKE", "SBUX", "MCD", "TGT", "LOW",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "ABT", "HIMS",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "AXP", "V", "MA",
    # Energy
    "XOM", "CVX", "COP", "SLB",
    # Industrials
    "CAT", "DE", "HON", "GE", "UPS", "FDX",
    # Communication Services
    "DIS", "CMCSA", "TMUS",
    # Materials & Utilities
    "LIN", "NEE",
]

_shortlist = load_shortlist(wc.SHORTLIST_PATH)
WHEEL_WATCHLIST: list[str] = _shortlist if _shortlist else _DEFAULT_WATCHLIST


# ---------------------------------------------------------------------------
# Manual exclusions — tickers you've removed by hand. Kept in their own file
# so they survive a re-run of scan_universe.py (which rewrites the shortlist).
# Managed from the dashboard's Fundamentals page.
# ---------------------------------------------------------------------------
import json
from pathlib import Path

EXCLUSIONS_PATH = Path(__file__).with_name("wheel_exclusions.json")


def load_exclusions() -> set[str]:
    try:
        return {t.upper() for t in json.loads(EXCLUSIONS_PATH.read_text())}
    except (FileNotFoundError, ValueError):
        return set()


def save_exclusions(tickers) -> None:
    EXCLUSIONS_PATH.write_text(json.dumps(sorted({t.upper() for t in tickers}), indent=2))


def active_watchlist() -> list[str]:
    """The screener universe minus manual exclusions (read fresh each call)."""
    excluded = load_exclusions()
    return [t for t in WHEEL_WATCHLIST if t.upper() not in excluded]
