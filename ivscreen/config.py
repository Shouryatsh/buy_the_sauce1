"""
ivscreen/config.py — every threshold of the IV / RV percentile screener, in one dataclass with
JSON persistence (ivscreen_config.json). Percentiles are 0-100.

Scans store raw history only; these settings are applied when results are *evaluated*, so
changing a threshold, lookback or window re-classifies instantly without a rescan.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "ivscreen_config.json"

# Optionable ETFs offered as an opt-in universe category. Never mixed into the stock lists
# unless the "ETF" instrument type is switched on in the filters.
DEFAULT_ETFS = (
    "SPY QQQ IWM DIA XLF XLE XLK XLV XLY XLP XLI XLU XLB XLRE XLC SMH XBI KRE XOP GDX GLD SLV USO "
    "TLT HYG LQD EEM EFA FXI EWZ VNQ ARKK"
).split()

INSTRUMENT_TYPES = ("Common stock", "ETF", "ADR", "REIT / trust", "Other")


@dataclass
class IVConfig:
    # ---- classification bands (percentile of the symbol's own history) ----------------------
    very_low_max: float = 10.0       # Very low   : pct <= 10
    low_max: float = 25.0            # Low        : 10 < pct <= 25
    high_min: float = 75.0           # High       : 75 <= pct < 90      (Normal: 25 < pct < 75)
    very_high_min: float = 90.0      # Very high  : pct >= 90

    # ---- percentile definition --------------------------------------------------------------
    lookback_days: int = 504         # trading days of own history (252 = 1Y, 504 = 2Y, 756 = 3Y)
    rv_window: int = 20              # realized-vol window in trading days (10 / 20 / 50 / 100)
    min_history_days: int = 126      # fewer prior observations than this => no percentile (never a guess)

    # ---- trend (change in PERCENTILE points over N sessions) -------------------------------
    trend_days: int = 10
    trend_tolerance: float = 5.0     # |change| below this is "Flat"

    # ---- IV vs RV relationship (difference of the two percentiles) --------------------------
    gap_threshold: float = 25.0      # IV pct - RV pct >= this => "IV elevated relative to RV" (and vice versa)

    # ---- data freshness ----------------------------------------------------------------------
    max_iv_age_days: int = 7         # latest IV bar older than this => stale, kept out of the watchlists
    max_price_age_days: int = 7      # same for the price bar behind realized vol
    earnings_flag_days: int = 30     # earnings inside the 30-day IV window explains part of elevated IV

    # ---- universe ----------------------------------------------------------------------------
    etf_symbols: list = field(default_factory=lambda: list(DEFAULT_ETFS))

    # ---- scanning ----------------------------------------------------------------------------
    iv_duration: str = "3 Y"         # first download; later scans only fetch the missing days
    refresh_duration: str = "2 M"    # incremental top-up (covers ~42 sessions)
    full_refresh_days: int = 30      # re-download the whole history this often (heals revisions)
    skip_fresh_hours: float = 6.0    # a symbol fetched this recently is not requested again (resume after a cancelled scan)
    concurrency: int = 3             # simultaneous IBKR historical-data requests
    request_timeout: float = 60.0
    price_years: int = 4
    price_ttl_hours: float = 6.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "IVConfig":
        known = {f.name for f in fields(cls)}
        base = cls()
        for k, v in (data or {}).items():
            if k not in known:
                continue
            cur = getattr(base, k)
            try:
                if isinstance(cur, bool):
                    v = bool(v)
                elif isinstance(cur, int):
                    v = int(v)
                elif isinstance(cur, float):
                    v = float(v)
                elif isinstance(cur, list):
                    v = [str(x).strip().upper() for x in v if str(x).strip()]
            except (TypeError, ValueError):
                continue
            setattr(base, k, v)
        return base

    def problems(self) -> list[str]:
        """Inconsistent settings (the UI refuses to save these)."""
        out = []
        if not (0 <= self.very_low_max < self.low_max < self.high_min < self.very_high_min <= 100):
            out.append("Bands must satisfy 0 ≤ very low < low < high < very high ≤ 100.")
        if self.lookback_days < 20 or self.min_history_days < 20 or self.min_history_days > self.lookback_days:
            out.append("History: lookback ≥ 20 days and 20 ≤ minimum history ≤ lookback.")
        if self.rv_window < 2:
            out.append("Realized-vol window must be at least 2 days.")
        return out


def load_config(path: Optional[Path] = None) -> IVConfig:
    try:
        return IVConfig.from_dict(json.loads(Path(path or CONFIG_PATH).read_text(encoding="utf-8")))
    except (FileNotFoundError, ValueError):
        return IVConfig()


def save_config(cfg: IVConfig, path: Optional[Path] = None) -> None:
    Path(path or CONFIG_PATH).write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
