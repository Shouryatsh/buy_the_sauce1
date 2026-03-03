"""
edgar.py — Free fundamental data from SEC EDGAR + Stooq price history.

Data sources
------------
- Fundamentals  : SEC EDGAR XBRL API  (https://data.sec.gov)
                  No API key required. Rate-limit: ~10 req/s, we stay well under.
- Price history : Stooq via pandas_datareader
                  No API key required.
- Market cap    : Computed from Stooq price × shares outstanding (EDGAR)

Public API
----------
    get_fundamentals(symbol)  -> dict   (same keys screener.py expects)
    get_price_history(symbol) -> pd.DataFrame  (columns: Open High Low Close Volume)
"""

from __future__ import annotations

import logging
import math
import time
from functools import lru_cache
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EDGAR_HEADERS = {
    "User-Agent": "buy-the-sauce trading-bot contact@example.com",  # SEC requires this
    "Accept-Encoding": "gzip, deflate",
}
EDGAR_CIK_URL  = "https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt=2020-01-01&enddt=2099-01-01&forms=10-K"
EDGAR_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
EDGAR_FACTS_URL   = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

_RATE_LIMIT_SLEEP = 0.12   # seconds between EDGAR requests (~8 req/s, under the 10/s limit)


# ---------------------------------------------------------------------------
# CIK lookup (cached for the process lifetime)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_ticker_map() -> dict:
    """Download SEC's full ticker→CIK map once and cache it."""
    try:
        resp = requests.get(EDGAR_TICKERS_URL, headers=EDGAR_HEADERS, timeout=15)
        resp.raise_for_status()
        raw = resp.json()
        # raw = { "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ... }
        return {v["ticker"].upper(): int(v["cik_str"]) for v in raw.values()}
    except Exception as exc:
        logger.error("Could not load SEC ticker map: %s", exc)
        return {}


def _get_cik(symbol: str) -> Optional[int]:
    ticker_map = _load_ticker_map()
    cik = ticker_map.get(symbol.upper())
    if cik is None:
        logger.warning("%s: CIK not found in SEC ticker map", symbol)
    return cik


# ---------------------------------------------------------------------------
# EDGAR XBRL concept fetcher
# ---------------------------------------------------------------------------

def _fetch_company_facts(cik: int) -> dict:
    """Fetch the full XBRL company facts JSON from EDGAR."""
    url = EDGAR_FACTS_URL.format(cik=cik)
    time.sleep(_RATE_LIMIT_SLEEP)
    resp = requests.get(url, headers=EDGAR_HEADERS, timeout=20)
    resp.raise_for_status()
    return resp.json()


def _get_annual_values(facts: dict, concept: str, unit: str = "USD") -> list[float]:
    """Extract the most-recent annual (10-K) filed values for a given XBRL concept.

    Returns a list ordered newest → oldest, up to 5 years.
    """
    try:
        entries = (
            facts["facts"]["us-gaap"][concept]["units"][unit]
        )
    except KeyError:
        return []

    # Keep only 10-K filings, deduplicate by fiscal year (use latest amendment)
    annual: dict[str, float] = {}
    for e in entries:
        if e.get("form") in ("10-K", "10-K/A") and e.get("val") is not None:
            fy = str(e.get("fy") or e.get("end", "")[:4])
            if fy:
                # Keep the entry with the latest filing date per fiscal year
                if fy not in annual or e["filed"] > annual[fy][1]:
                    annual[fy] = (float(e["val"]), e["filed"])

    # Sort newest → oldest
    sorted_years = sorted(annual.keys(), reverse=True)[:5]
    return [annual[y][0] for y in sorted_years]


def _safe(value) -> Optional[float]:
    try:
        v = float(value)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Price history via Stooq
# ---------------------------------------------------------------------------

def get_price_history(symbol: str, period_years: int = 2) -> Optional[pd.DataFrame]:
    """Fetch daily OHLCV price history from Stooq (free, no API key).

    Returns a DataFrame with columns [Open, High, Low, Close, Volume]
    indexed by date, or None on failure.
    """
    try:
        from pandas_datareader import data as pdr
        end   = pd.Timestamp.today()
        start = end - pd.DateOffset(years=period_years)
        df = pdr.DataReader(symbol.upper(), "stooq", start=start, end=end)
        if df.empty:
            raise ValueError("empty response")
        # Stooq returns newest-first; flip to oldest-first to match yfinance convention
        df = df.sort_index()
        logger.info("%s: fetched %d price bars from Stooq", symbol, len(df))
        return df
    except Exception as exc:
        logger.warning("%s: Stooq price fetch failed — %s", symbol, exc)
        return None


# ---------------------------------------------------------------------------
# Main fundamentals builder
# ---------------------------------------------------------------------------

def get_fundamentals(symbol: str) -> dict:
    """Return a fundamentals dict for *symbol* sourced entirely from SEC EDGAR.

    Keys mirror the ones screener.py reads so it works as a drop-in
    replacement for ``yf.Ticker(symbol).info``.

    Keys returned
    -------------
    forwardPE, trailingPE   – EPS-based P/E (uses reported EPS + Stooq price)
    profitMargins           – Net income / Revenue (trailing 12 months)
    debtToEquity            – (Total liabilities / Equity) × 100  ← screener divides by 100
    freeCashflow            – Operating CF − CapEx (most-recent fiscal year)
    marketCap               – Shares outstanding × latest close price
    returnOnEquity          – Net income / Stockholders equity
    capitalExpenditures     – CapEx (negative per convention)
    _fcf_history            – list of annual FCF values newest→oldest (for trend check)
    """
    result: dict = {}
    cik = _get_cik(symbol)
    if cik is None:
        logger.warning("%s: no CIK found — returning empty fundamentals", symbol)
        return result

    try:
        facts = _fetch_company_facts(cik)
    except Exception as exc:
        logger.error("%s: EDGAR facts fetch failed — %s", symbol, exc)
        return result

    # --- Operating cash flow ---
    op_cf_vals = (
        _get_annual_values(facts, "NetCashProvidedByUsedInOperatingActivities")
        or _get_annual_values(facts, "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations")
    )

    # --- Capital expenditures ---
    capex_vals = (
        _get_annual_values(facts, "PaymentsToAcquirePropertyPlantAndEquipment")
        or _get_annual_values(facts, "CapitalExpenditureDiscontinuedOperations")
    )

    # --- Free Cash Flow (annual series) ---
    fcf_history: list[float] = []
    if op_cf_vals and capex_vals:
        pairs = min(len(op_cf_vals), len(capex_vals))
        fcf_history = [op_cf_vals[i] - capex_vals[i] for i in range(pairs)]
    result["_fcf_history"] = fcf_history
    if fcf_history:
        result["freeCashflow"] = fcf_history[0]            # most recent year
        if capex_vals:
            result["capitalExpenditures"] = -abs(capex_vals[0])  # negative by convention

    # --- Net income ---
    net_income_vals = (
        _get_annual_values(facts, "NetIncomeLoss")
        or _get_annual_values(facts, "ProfitLoss")
    )

    # --- Revenue ---
    revenue_vals = (
        _get_annual_values(facts, "RevenueFromContractWithCustomerExcludingAssessedTax")
        or _get_annual_values(facts, "Revenues")
        or _get_annual_values(facts, "SalesRevenueNet")
    )

    # --- Profit margin ---
    if net_income_vals and revenue_vals and revenue_vals[0]:
        result["profitMargins"] = net_income_vals[0] / revenue_vals[0]

    # --- Debt / equity (×100 so screener divides back by 100) ---
    liabilities_vals = _get_annual_values(facts, "Liabilities")
    equity_vals      = (
        _get_annual_values(facts, "StockholdersEquity")
        or _get_annual_values(facts, "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")
    )
    if liabilities_vals and equity_vals and equity_vals[0] != 0:
        result["debtToEquity"] = (liabilities_vals[0] / equity_vals[0]) * 100.0

    # --- Return on equity ---
    if net_income_vals and equity_vals and equity_vals[0] != 0:
        result["returnOnEquity"] = net_income_vals[0] / equity_vals[0]

    # --- Shares outstanding (for market cap + P/E) ---
    shares_vals = (
        _get_annual_values(facts, "CommonStockSharesOutstanding", unit="shares")
        or _get_annual_values(facts, "EntityCommonStockSharesOutstanding", unit="shares")
    )

    # --- Latest price from Stooq ---
    price_hist = get_price_history(symbol, period_years=2)
    latest_price: Optional[float] = None
    if price_hist is not None and not price_hist.empty:
        latest_price = _safe(price_hist["Close"].iloc[-1])

    # --- Market cap ---
    if shares_vals and latest_price:
        result["marketCap"] = shares_vals[0] * latest_price

    # --- P/E (trailing) ---
    if net_income_vals and shares_vals and shares_vals[0] > 0 and latest_price:
        eps = net_income_vals[0] / shares_vals[0]
        if eps > 0:
            result["trailingPE"] = latest_price / eps

    logger.info(
        "%s: EDGAR fundamentals fetched — FCF=$%s, margin=%s, D/E=%s, ROE=%s",
        symbol,
        f"{result.get('freeCashflow', 'N/A'):,.0f}" if isinstance(result.get("freeCashflow"), float) else "N/A",
        f"{result['profitMargins']:.1%}" if "profitMargins" in result else "N/A",
        f"{result['debtToEquity']/100:.2f}x" if "debtToEquity" in result else "N/A",
        f"{result['returnOnEquity']:.1%}" if "returnOnEquity" in result else "N/A",
    )
    return result
