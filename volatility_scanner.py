"""
volatility_scanner.py — Find stocks oscillating ±20% in 6 months with good fundamentals.

Identifies volatile-but-quality stocks: those whose price has swung between
-20% and +20% from their 6-month starting price, AND that pass the
fundamental quality screen.  Also computes relative position vs 200-day MA.

Data source: Stooq via pandas_datareader (same as dip_detector.py).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import pandas_datareader.data as web

import config
import edgar
from screener import screen_fundamental, FundamentalProfile

logger = logging.getLogger(__name__)

LOOKBACK_MONTHS = 6
SWING_THRESHOLD = 0.20        # ±20%
MA_LONG_PERIOD = 200           # long-term moving average


@dataclass
class VolatileStock:
    symbol: str
    price: float
    max_drawdown_pct: float       # worst drawdown from start (negative)
    max_rally_pct: float          # best rally from start (positive)
    swing_range_pct: float        # max_rally - max_drawdown (total oscillation)
    ma200: Optional[float]        # 200-day moving average
    pct_from_ma200: Optional[float]  # (price - ma200) / ma200
    ma50: Optional[float]
    pct_from_ma50: Optional[float]
    fundamental_pass: bool
    pe_ratio: Optional[float]
    profit_margin: Optional[float]
    roe: Optional[float]
    fcf_yield: Optional[float]
    debt_to_equity: Optional[float]


def _fetch_price_history(symbol: str, months: int = 12) -> Optional[pd.DataFrame]:
    """Fetch daily price history from Stooq (enough for 200-day MA + 6-month lookback)."""
    try:
        end = pd.Timestamp.now()
        start = end - pd.DateOffset(months=months)
        df = web.DataReader(symbol, "stooq", start, end)
        if df is None or df.empty:
            return None
        df = df.sort_index()
        return df
    except Exception as e:
        logger.warning("%s: price fetch failed — %s", symbol, e)
        return None


def scan_volatile_stocks(
    tickers: list[str],
    swing_threshold: float = SWING_THRESHOLD,
    require_fundamental_pass: bool = True,
) -> list[VolatileStock]:
    """Scan tickers for ±swing_threshold oscillation in last 6 months with good fundamentals."""

    results: list[VolatileStock] = []

    for symbol in tickers:
        try:
            # 1. Fetch price history (12 months for 200-day MA)
            df = _fetch_price_history(symbol, months=12)
            if df is None or len(df) < 30:
                logger.info("%s: insufficient price history", symbol)
                continue

            close = df["Close"]
            current_price = float(close.iloc[-1])

            # 2. Compute 6-month window
            six_months_ago = pd.Timestamp.now() - pd.DateOffset(months=LOOKBACK_MONTHS)
            recent = close[close.index >= six_months_ago]
            if len(recent) < 20:
                continue

            start_price = float(recent.iloc[0])
            returns_from_start = (recent - start_price) / start_price

            max_rally = float(returns_from_start.max())
            max_drawdown = float(returns_from_start.min())
            swing_range = max_rally - max_drawdown

            # 3. Check if stock oscillated within ±threshold
            if max_rally < swing_threshold * 0.5 and abs(max_drawdown) < swing_threshold * 0.5:
                # Not volatile enough — skip
                continue

            # We want stocks that have BOTH dipped and rallied meaningfully
            if swing_range < swing_threshold:
                continue

            # 4. Moving averages
            ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
            ma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None

            pct_from_ma200 = ((current_price - ma200) / ma200) if ma200 else None
            pct_from_ma50 = ((current_price - ma50) / ma50) if ma50 else None

            # 5. Fundamental screen
            profile = screen_fundamental(symbol)

            if require_fundamental_pass and not profile.passes:
                continue

            results.append(VolatileStock(
                symbol=symbol,
                price=current_price,
                max_drawdown_pct=max_drawdown,
                max_rally_pct=max_rally,
                swing_range_pct=swing_range,
                ma200=ma200,
                pct_from_ma200=pct_from_ma200,
                ma50=ma50,
                pct_from_ma50=pct_from_ma50,
                fundamental_pass=profile.passes,
                pe_ratio=profile.pe_ratio,
                profit_margin=profile.profit_margin,
                roe=profile.return_on_equity,
                fcf_yield=profile.fcf_yield,
                debt_to_equity=profile.debt_to_equity,
            ))

        except Exception as e:
            logger.warning("%s: volatility scan error — %s", symbol, e)
            continue

    # Sort by swing range (most volatile first)
    results.sort(key=lambda x: x.swing_range_pct, reverse=True)
    return results


def volatile_stocks_to_df(stocks: list[VolatileStock]) -> pd.DataFrame:
    """Convert scan results to a display DataFrame."""
    if not stocks:
        return pd.DataFrame()

    rows = []
    for s in stocks:
        rows.append({
            "Symbol": s.symbol,
            "Price": f"${s.price:.2f}",
            "6M Low %": f"{s.max_drawdown_pct:+.1%}",
            "6M High %": f"{s.max_rally_pct:+.1%}",
            "Swing Range": f"{s.swing_range_pct:.1%}",
            "MA200": f"${s.ma200:.2f}" if s.ma200 else "n/a",
            "vs MA200": f"{s.pct_from_ma200:+.1%}" if s.pct_from_ma200 is not None else "n/a",
            "MA50": f"${s.ma50:.2f}" if s.ma50 else "n/a",
            "vs MA50": f"{s.pct_from_ma50:+.1%}" if s.pct_from_ma50 is not None else "n/a",
            "MA Level": _ma_level(s.pct_from_ma200),
            "Fund Pass": "✅" if s.fundamental_pass else "❌",
            "P/E": f"{s.pe_ratio:.1f}" if s.pe_ratio else "n/a",
            "Margin": f"{s.profit_margin:.1%}" if s.profit_margin is not None else "n/a",
            "ROE": f"{s.roe:.1%}" if s.roe is not None else "n/a",
            "FCF Yield": f"{s.fcf_yield:.2%}" if s.fcf_yield is not None else "n/a",
            "D/E": f"{s.debt_to_equity:.2f}" if s.debt_to_equity is not None else "n/a",
        })

    return pd.DataFrame(rows)


def _ma_level(pct_from_ma200: Optional[float]) -> str:
    """Classify stock's position relative to 200-day MA."""
    if pct_from_ma200 is None:
        return "n/a"
    if pct_from_ma200 > 0.10:
        return "🟢 Well Above"
    elif pct_from_ma200 > 0.02:
        return "🟢 Above"
    elif pct_from_ma200 > -0.02:
        return "🟡 At MA200"
    elif pct_from_ma200 > -0.10:
        return "🔴 Below"
    else:
        return "🔴 Well Below"
