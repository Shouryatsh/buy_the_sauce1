"""
ivscreen/evaluate.py — stored data + an IVConfig -> one classified row per symbol.

Data availability is explicit and never confused with low volatility. A symbol is only ranked
(status OK) when it has a fresh implied-vol history with enough prior observations; everything
else carries a status saying why it has no percentile:

    OK              ranked
    SHORT_HISTORY   IBKR has implied-vol data, but fewer than min_history_days prior observations
    STALE           the latest implied-vol bar is older than max_iv_age_days
    NO_OPTIONS_DATA IBKR returned no implied-vol history (no listed options / no data for the underlying)
    UNRESOLVED      IBKR does not know the ticker
    UNSUPPORTED     not a stock/ETF type this screener ranks (e.g. preferred shares, non-stock contracts)
    FETCH_ERROR     the request failed or timed out — retry; says nothing about volatility
    DUPLICATE       resolves to the same IBKR contract as another symbol in the universe
    NOT_SCANNED     in the universe but not scanned yet

Realized vol is ranked the same way from daily closes; when it is unavailable the IV percentile still
shows and the relationship says "RV unavailable".
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Callable, Optional

import pandas as pd

from . import metrics as mx

OK, SHORT_HISTORY, STALE = "OK", "SHORT_HISTORY", "STALE"
NO_OPTIONS_DATA, UNRESOLVED, UNSUPPORTED = "NO_OPTIONS_DATA", "UNRESOLVED", "UNSUPPORTED"
FETCH_ERROR, DUPLICATE, NOT_SCANNED = "FETCH_ERROR", "DUPLICATE", "NOT_SCANNED"

STATUS_TEXT = {
    OK: "Ranked",
    SHORT_HISTORY: "Too little implied-vol history to rank",
    STALE: "Implied-vol data is stale",
    NO_OPTIONS_DATA: "No options data from IBKR (not low IV — unknown)",
    UNRESOLVED: "Ticker not found at IBKR",
    UNSUPPORTED: "Instrument type not supported",
    FETCH_ERROR: "Fetch failed — retry (unknown, not low IV)",
    DUPLICATE: "Duplicate of another symbol",
    NOT_SCANNED: "Not scanned yet",
}
UNAVAILABLE = (NO_OPTIONS_DATA, UNRESOLVED, UNSUPPORTED, FETCH_ERROR, DUPLICATE, NOT_SCANNED)

_FETCH_TO_STATUS = {"no_data": NO_OPTIONS_DATA, "unresolved": UNRESOLVED, "unsupported": UNSUPPORTED,
                    "error": FETCH_ERROR, "duplicate": DUPLICATE}


def _to_date(s) -> Optional[date]:
    if not s:
        return None
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _blank(meta: dict, status: str, detail: str = "") -> dict:
    return {
        "symbol": meta["symbol"], "name": meta.get("name") or "", "instrument": meta.get("category") or "",
        "exchange": meta.get("exchange") or "", "sector": meta.get("sector") or "", "industry": meta.get("industry") or "",
        "sources": meta.get("sources") or "", "status": status, "status_text": STATUS_TEXT[status],
        "detail": detail or meta.get("fetch_detail") or "",
        "iv_pct": None, "iv_band": mx.NA, "iv_chg": None, "iv_trend": mx.NA,
        "rv_pct": None, "rv_band": mx.NA, "rv_chg": None, "rv_trend": mx.NA,
        "gap": None, "relationship": mx.REL_NA, "trend_summary": "n/a",
        "iv_obs": 0, "rv_obs": 0, "window_days": 0, "partial_window": False,
        "iv_asof": None, "rv_asof": None, "iv_age": None, "rv_status": "none",
        "next_earnings": _to_date(meta.get("next_earnings")), "days_to_earnings": None, "earnings_in_window": False,
        "fetched_at": meta.get("iv_checked_at"),
    }


def evaluate_symbol(meta: dict, iv: Optional[pd.Series], close: Optional[pd.Series], cfg, today: Optional[date] = None) -> dict:
    today = today or date.today()
    fs = meta.get("fetch_status")
    if fs in _FETCH_TO_STATUS:
        return _blank(meta, _FETCH_TO_STATUS[fs])
    if not fs:
        return _blank(meta, NOT_SCANNED)
    row = _blank(meta, OK)
    ne = row["next_earnings"]
    if ne is not None:
        row["days_to_earnings"] = (ne - today).days
        row["earnings_in_window"] = 0 <= row["days_to_earnings"] <= cfg.earnings_flag_days
        if row["days_to_earnings"] < 0:                       # a past date is not "next"
            row["next_earnings"], row["days_to_earnings"] = None, None

    if iv is None or iv.dropna().empty:
        return _blank(meta, NO_OPTIONS_DATA)
    iv = iv.dropna()
    row["iv_asof"] = iv.index[-1].date()
    row["iv_age"] = (today - row["iv_asof"]).days

    pct, n = mx.rank(iv, cfg.lookback_days, cfg.min_history_days)
    row["iv_obs"], row["window_days"] = n, cfg.lookback_days
    row["partial_window"] = n < cfg.lookback_days
    if pct is None:
        row["status"], row["status_text"] = SHORT_HISTORY, STATUS_TEXT[SHORT_HISTORY]
        row["detail"] = f"{n} prior daily readings; {cfg.min_history_days} needed"
    else:
        row["iv_pct"], row["iv_band"] = pct, mx.classify(pct, cfg.very_low_max, cfg.low_max, cfg.high_min, cfg.very_high_min)
        prev, _ = mx.rank(iv, cfg.lookback_days, cfg.min_history_days, offset=cfg.trend_days)
        row["iv_trend"], row["iv_chg"] = mx.trend(pct, prev, cfg.trend_tolerance)
        if row["iv_age"] > cfg.max_iv_age_days:
            row["status"], row["status_text"] = STALE, STATUS_TEXT[STALE]
            row["detail"] = f"latest implied-vol reading is from {row['iv_asof']}"

    # --- realized vol, ranked against its own history ------------------------------------
    if close is not None and len(close.dropna()) > cfg.rv_window + cfg.min_history_days:
        rv = mx.realized_vol(close, cfg.rv_window)
        row["rv_asof"] = rv.index[-1].date() if len(rv) else None
        rv_age = (today - row["rv_asof"]).days if row["rv_asof"] else None
        rpct, rn = mx.rank(rv, cfg.lookback_days, cfg.min_history_days)
        row["rv_obs"] = rn
        if rpct is None:
            row["rv_status"] = "short"
        elif rv_age is None or rv_age > cfg.max_price_age_days:
            row["rv_status"] = "stale"
        else:
            row["rv_status"], row["rv_pct"] = "ok", rpct
            row["rv_band"] = mx.classify(rpct, cfg.very_low_max, cfg.low_max, cfg.high_min, cfg.very_high_min)
            rprev, _ = mx.rank(rv, cfg.lookback_days, cfg.min_history_days, offset=cfg.trend_days)
            row["rv_trend"], row["rv_chg"] = mx.trend(rpct, rprev, cfg.trend_tolerance)

    if row["iv_pct"] is not None and row["rv_pct"] is not None:
        row["gap"] = round(row["iv_pct"] - row["rv_pct"], mx.DECIMALS)
        row["relationship"] = mx.relationship(row["iv_pct"], row["rv_pct"], cfg.low_max, cfg.high_min, cfg.gap_threshold)
        row["trend_summary"] = mx.trend_relationship(row["iv_trend"], row["rv_trend"])
    return row


def evaluate_all(metas: dict[str, dict], ivs: dict[str, pd.Series], close_loader: Callable[[str], Optional[pd.Series]],
                 cfg, today: Optional[date] = None, exclude: Optional[set] = None) -> pd.DataFrame:
    """One row per symbol; excluded tickers are not evaluated at all."""
    ex = {s.upper() for s in (exclude or ())}
    rows = []
    for sym, meta in metas.items():
        if sym.upper() in ex:
            continue
        meta = {**meta, "symbol": sym}
        try:
            close = close_loader(sym) if meta.get("fetch_status") == "ok" else None
            rows.append(evaluate_symbol(meta, ivs.get(sym), close, cfg, today))
        except Exception as exc:                              # one bad symbol must never hide the rest
            row = _blank(meta, FETCH_ERROR, f"evaluation error: {exc}")
            rows.append(row)
    return pd.DataFrame(rows)


def percentile_history(iv: pd.Series, close: Optional[pd.Series], cfg) -> pd.DataFrame:
    """Daily IV percentile and RV percentile (each against the lookback window before that day), for the
    detail chart. Columns iv_pct / rv_pct; either may be empty when its history is too short."""
    out = {}
    if iv is not None and len(iv.dropna()) > cfg.min_history_days:
        out["iv_pct"] = mx.rolling_rank(iv, cfg.lookback_days, cfg.min_history_days)
    if close is not None and len(close.dropna()) > cfg.rv_window + cfg.min_history_days:
        rv = mx.realized_vol(close, cfg.rv_window)
        out["rv_pct"] = mx.rolling_rank(rv, cfg.lookback_days, cfg.min_history_days)
    return pd.DataFrame(out).dropna(how="all")
