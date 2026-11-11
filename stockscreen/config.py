"""
stockscreen/config.py — every threshold, weight and preference of the quality
stock screener, in one dataclass with JSON persistence.

Units: ratios (margins, growth, yield) are fractions (0.10 = 10%); price
positions and percentiles are 0-100.

Metrics are stored threshold-free by the scan; these settings are applied when
results are *evaluated*, so changing a threshold in the UI re-classifies
instantly without a rescan.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "stockscreen_config.json"
CACHE_DIR = ROOT / "stockscreen_cache"


@dataclass
class ScreenConfig:
    # ---- A. FCF consistency -------------------------------------------------
    require_fcf_quarters: int = 4            # latest N reported quarters must all be > 0
    require_fcf_years: int = 2               # latest N fiscal years must all be > 0
    allow_missing_capex_as_zero: bool = False  # False: no capex tag => insufficient data, never "positive"
    # Financial companies whose investment purchases/sales sit inside operating cash flow (e.g. Blackstone: reported
    # investing == capex exactly): "na" = FCF is marked not meaningful (default), "flag" = only warn, "off" = ignore.
    financial_fcf_policy: str = "na"

    # ---- B/C/D/E thresholds -------------------------------------------------
    min_net_margin: float = 0.10             # TTM net margin strictly greater than
    min_revenue_growth: float = 0.10         # strictly greater than
    growth_basis: str = "auto"               # "ttm" | "fy" | "auto" (TTM, else fiscal year — labelled)
    min_fcf_yield: float = 0.05              # TTM FCF / market cap, at least (>=)
    min_market_cap: float = 1_000_000_000.0  # at least (>=)

    # ---- freshness ----------------------------------------------------------
    max_data_age_days: int = 200             # latest quarter end older than this => stale

    # ---- F. fundamental trend ----------------------------------------------
    critical_trend_metrics: tuple = ("fcf", "net_margin", "revenue")  # "Declining" on any => not eligible
    trend_tol_level: float = 0.05            # relative change tolerance for level metrics (FCF, revenue)
    trend_tol_margin_pp: float = 0.01        # absolute tolerance for margin (1 percentage point)
    trend_tol_growth_pp: float = 0.03        # absolute tolerance for revenue growth (3 pp)
    trend_severe_mult: float = 3.0           # a change beyond tolerance x this is "Declining" regardless of path
    trend_min_points: int = 4                # fewer TTM observations than this => Insufficient data

    # ---- price zones (0-100) -----------------------------------------------
    low_range_pos_52w: float = 20.0
    low_range_pos_2y: float = 20.0
    low_range_pos_3y: float = 20.0
    low_percentile: float = 20.0
    near_high_range_pos: float = 90.0        # 52-week range position at/above => "Near highs"
    deep_zone_min_lookbacks: int = 2         # lookbacks in the low zone for "Deep value-zone candidate"
    history_grace_days: int = 7              # a window is "available" if data starts within this of its start

    # ---- technicals ---------------------------------------------------------
    slope_window: int = 20                   # trading days for SMA slope
    pullback_from_high_pct: float = 5.0      # % below 60-day high that counts as short-term weakness
    reversal_min_signals: int = 1            # recovery signals => "Trend reversal candidate"
    reversal_confirm_signals: int = 2        # recovery signals => confirmed (opportunity class D)
    sideways_band_pct: float = 3.0           # |SMA50/SMA200 - 1| below this counts as flat structure
    rsi_oversold: float = 30.0
    rsi_weak: float = 40.0
    rsi_positive: float = 60.0
    rsi_overbought: float = 70.0

    # ---- scoring weights (renormalised over the components available) -------
    weights_fundamental: dict = field(default_factory=lambda: {
        "fcf_consistency": 0.20, "fcf_trend": 0.20, "net_margin": 0.15,
        "revenue_growth": 0.15, "stability": 0.10, "fcf_yield": 0.20})
    weights_price: dict = field(default_factory=lambda: {
        "range_52w": 0.30, "range_2y": 0.20, "range_3y": 0.15, "percentiles": 0.20, "from_high": 0.15})
    weights_technical: dict = field(default_factory=lambda: {
        "trend": 0.35, "price_vs_sma": 0.20, "sma_cross": 0.10, "slopes": 0.10, "rsi": 0.10, "macd": 0.15})
    weights_rank: dict = field(default_factory=lambda: {
        "fundamental": 0.45, "price": 0.35, "technical": 0.20})
    min_score_coverage: float = 0.6          # share of a score's weight that must be computable

    # ---- alerts -------------------------------------------------------------
    alert_kinds: tuple = ("low_price", "healthy_pullback", "reversal_confirmation", "fundamental_deterioration")
    alert_cooldown_days: int = 7             # after an alert fires, don't re-fire for this long
    alert_max_data_age_days: int = 200       # never alert on data staler than this

    # ---- scanning -----------------------------------------------------------
    fundamentals_ttl_days: int = 3           # re-download SEC statements after this long
    price_ttl_hours: float = 6.0             # re-download price history after this long
    price_history_years: int = 4
    scan_workers: int = 8
    auto_scan_enabled: bool = False
    auto_scan_time: str = "17:30"            # local time (after the US close)

    # -------------------------------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "ScreenConfig":
        known = {f.name: f for f in fields(cls)}
        base = cls()
        for k, v in (data or {}).items():
            if k not in known:
                continue
            cur = getattr(base, k)
            if isinstance(cur, tuple):
                v = tuple(v)
            elif isinstance(cur, dict):
                v = {**cur, **v}
            elif isinstance(cur, bool):
                v = bool(v)
            elif isinstance(cur, int) and not isinstance(cur, bool):
                v = int(v)
            elif isinstance(cur, float):
                v = float(v)
            setattr(base, k, v)
        return base


def load_config(path: Optional[Path] = None) -> ScreenConfig:
    path = Path(path or CONFIG_PATH)
    try:
        return ScreenConfig.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (FileNotFoundError, ValueError):
        return ScreenConfig()


def save_config(cfg: ScreenConfig, path: Optional[Path] = None) -> None:
    Path(path or CONFIG_PATH).write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
