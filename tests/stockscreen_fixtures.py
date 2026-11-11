"""Shared builders for the stockscreen tests: synthetic scan records with dates relative to today."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

M = 1_000_000


def quarter_ends(n: int = 12, latest_age_days: int = 100) -> list[str]:
    latest = date.today() - timedelta(days=latest_age_days)
    return [(latest - timedelta(days=91 * k)).isoformat() for k in range(n - 1, -1, -1)]


def _q(values, ends):
    return [{"end": e, "value": v * M, "basis": "reported 3M", "filed": ends[-1], "first_filed": ends[-1]}
            for e, v in zip(ends, values)]


def good_fund(name="Good Co", ocf=30, capex=10, rev=(25,) * 8 + (30,) * 4, ni=(4,) * 8 + (5.4,) * 4, shares=150 * M,
              latest_age_days=100) -> dict:
    ends = quarter_ends(12, latest_age_days)
    fy = [ends[-5], ends[-1]]
    year = lambda v: [{"start": "", "end": e, "value": v * M, "basis": "reported FY", "filed": e, "first_filed": e} for e in fy]  # noqa: E731
    return {
        "fetch": "ok", "parser_version": 3, "name": name, "latest_filed": ends[-1],
        "quarters": {"ocf": _q([ocf] * 12, ends), "capex": _q([capex] * 12, ends), "revenue": _q(rev, ends),
                     "net_income": _q(ni, ends), "disc_ops": []},
        "years": {"ocf": year(ocf * 4), "capex": year(capex * 4), "revenue": year(sum(rev[-4:])), "net_income": year(sum(ni[-4:]))},
        "shares": [{"value": float(shares), "end": ends[-1], "source": "SEC cover page (dei)", "approx": False, "classes": 1}],
    }


def positions(rp52=50.0, rp2=50.0, rp3=50.0, pc=50.0):
    def one(rp):
        return {"available": True, "range_pos": rp, "percentile": pc, "low": 80.0, "high": 160.0, "price": 100.0,
                "pct_below_high": 37.5, "pct_above_low": 25.0, "n": 250, "window_start": "2025-10-20"}
    return {"52w": one(rp52), "2y": one(rp2), "3y": one(rp3)}


TECH = {
    "uptrend": dict(price=130.0, sma20=125.0, sma50=115.0, sma200=100.0, sma20_slope=.02, sma50_slope=.03, sma200_slope=.02,
                    rsi=60.0, rsi_5d_ago=55.0, macd_hist=1.0, macd_hist_prev=.5, macd_cross=None, macd_cross_days_ago=None,
                    dev_sma200_pct=30.0, sma50_vs_sma200_pct=15.0, n_bars=900),
    "pullback": dict(price=90.0, sma20=95.0, sma50=100.0, sma200=80.0, sma20_slope=-.02, sma50_slope=.01, sma200_slope=.02,
                     rsi=42.0, rsi_5d_ago=44.0, macd_hist=-.5, macd_hist_prev=-.3, macd_cross=None, macd_cross_days_ago=None,
                     dev_sma200_pct=12.5, sma50_vs_sma200_pct=25.0, n_bars=900),
    "downtrend": dict(price=60.0, sma20=65.0, sma50=75.0, sma200=100.0, sma20_slope=-.03, sma50_slope=-.04, sma200_slope=-.02,
                      rsi=28.0, rsi_5d_ago=30.0, macd_hist=-.8, macd_hist_prev=-.6, macd_cross=None, macd_cross_days_ago=None,
                      dev_sma200_pct=-40.0, sma50_vs_sma200_pct=-25.0, n_bars=900),
    "reversal": dict(price=72.0, sma20=70.0, sma50=76.0, sma200=100.0, sma20_slope=.01, sma50_slope=-.02, sma200_slope=-.02,
                     rsi=48.0, rsi_5d_ago=38.0, macd_hist=.4, macd_hist_prev=-.1, macd_cross="bullish", macd_cross_days_ago=3,
                     dev_sma200_pct=-28.0, sma50_vs_sma200_pct=-24.0, n_bars=900),
}


def record(symbol: str, *, fund: dict | None = None, pos: dict | None = None, tech: str = "uptrend", name: str | None = None) -> dict:
    fund = fund or good_fund(name or f"{symbol} Corp")
    ends = [o["end"] for o in fund["quarters"]["ocf"]] if fund.get("fetch") == "ok" else []
    return {
        "symbol": symbol, "name": fund.get("name", name or symbol), "scanned_at": date.today().isoformat() + "T09:00:00",
        "fund": fund,
        "price": {"last": 10.0, "asof": date.today().isoformat(), "positions": pos or positions(),
                  "qe_close": {e: 10.0 for e in ends}, "n_bars": 900},
        "tech": dict(TECH[tech], last_date=date.today().isoformat()),
    }


def price_frame(values, end=None) -> pd.DataFrame:
    end = end or pd.Timestamp(date.today())
    idx = pd.bdate_range(end=end, periods=len(values))
    c = pd.Series(values, index=idx, dtype=float)
    return pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * .99, "Close": c, "Volume": 1000.0})
