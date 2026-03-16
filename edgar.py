"""
edgar.py — Free fundamental data from SEC EDGAR + Stooq price history.

Data sources
------------
- Fundamentals  : SEC EDGAR XBRL API  (https://data.sec.gov)
                  No API key required. Rate-limit: ~10 req/s, we stay well under.
- Price history : IBKR TWS/Gateway (live, if running) with Stooq fallback
                  No API key required for Stooq.
- Market cap    : Computed from Stooq price × shares outstanding (EDGAR)

Public API
----------
    get_fundamentals(symbol)  -> dict   (same keys screener.py expects)
    get_price_history(symbol) -> (pd.DataFrame, str)  ("IBKR", "Stooq", or "unavailable")
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
EDGAR_TICKERS_URL    = "https://www.sec.gov/files/company_tickers.json"
EDGAR_FACTS_URL      = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

_RATE_LIMIT_SLEEP = 0.12   # ~8 req/s — well under SEC's 10/s limit


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
        return {v["ticker"].upper(): int(v["cik_str"]) for v in raw.values()}
    except Exception as exc:
        logger.error("Could not load SEC ticker map: %s", exc)
        return {}


def _get_cik(symbol: str) -> Optional[int]:
    cik = _load_ticker_map().get(symbol.upper())
    if cik is None:
        logger.warning("%s: CIK not found in SEC ticker map", symbol)
    return cik


# ---------------------------------------------------------------------------
# SIC code lookup — used for sector-aware D/E check
# ---------------------------------------------------------------------------

def _get_sic(cik: int) -> Optional[str]:
    """Return the 4-digit SIC code string for a company, or None on failure."""
    try:
        time.sleep(_RATE_LIMIT_SLEEP)
        url = EDGAR_SUBMISSIONS_URL.format(cik=cik)
        resp = requests.get(url, headers=EDGAR_HEADERS, timeout=15)
        resp.raise_for_status()
        return str(resp.json().get("sic", "") or "")
    except Exception as exc:
        logger.debug("SIC fetch failed for CIK %d: %s", cik, exc)
        return None


# ---------------------------------------------------------------------------
# EDGAR XBRL concept fetcher
# ---------------------------------------------------------------------------

def _fetch_company_facts(cik: int) -> dict:
    url = EDGAR_FACTS_URL.format(cik=cik)
    time.sleep(_RATE_LIMIT_SLEEP)
    resp = requests.get(url, headers=EDGAR_HEADERS, timeout=20)
    resp.raise_for_status()
    return resp.json()


def _get_annual_values(facts: dict, concept: str, unit: str = "USD") -> list:
    """Extract most-recent annual (10-K) filed values for an XBRL concept.

    Returns a list ordered newest → oldest, up to 5 years.
    """
    try:
        entries = facts["facts"]["us-gaap"][concept]["units"][unit]
    except KeyError:
        return []

    annual: dict = {}
    for e in entries:
        if e.get("form") in ("10-K", "10-K/A") and e.get("val") is not None:
            fy = str(e.get("fy") or e.get("end", "")[:4])
            if fy and (fy not in annual or e["filed"] > annual[fy][1]):
                annual[fy] = (float(e["val"]), e["filed"])

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

def _fetch_price_from_ibkr(symbol: str, period_years: int = 2) -> Optional[pd.DataFrame]:
    """Try to fetch price history from a running IBKR TWS/Gateway instance.

    ib_insync uses asyncio internally.  When called from a Dash callback
    (which runs in a worker thread with no event loop), we must create one
    explicitly, run the fetch inside it, then close it.

    Returns a DataFrame (oldest-first, OHLCV columns) or None on any failure.
    """
    import asyncio, random, time as _time

    async def _fetch_async():
        from ib_insync import IB, Stock, util
        import config as _cfg
        ib = IB()
        for attempt in range(3):
            cid = random.randint(100, 199)
            try:
                await ib.connectAsync(_cfg.IBKR_HOST, _cfg.IBKR_PORT,
                                      clientId=cid, readonly=True, timeout=4)
                break
            except Exception as conn_exc:
                if "already in use" in str(conn_exc).lower() and attempt < 2:
                    await asyncio.sleep(0.3)
                    continue
                raise

        contract = Stock(symbol.upper(), "SMART", "USD")
        duration = f"{period_years} Y"
        bars = await ib.reqHistoricalDataAsync(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=True,
        )
        ib.disconnect()
        return bars

    try:
        # Create a fresh event loop for this thread (safe in Dash worker threads)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            bars = loop.run_until_complete(_fetch_async())
        finally:
            loop.close()
            asyncio.set_event_loop(None)

        if not bars:
            return None

        from ib_insync import util
        df = util.df(bars)[["date", "open", "high", "low", "close", "volume"]].copy()
        df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
        df = df.set_index("Date").sort_index()
        # ib_insync util.df returns datetime.date objects — coerce to DatetimeIndex
        # so downstream feature engineering (ml_predictor.build_calendar_features)
        # can call .dayofweek / .month / etc. without AttributeError.
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        logger.info("%s: fetched %d price bars from IBKR", symbol, len(df))
        return df
    except Exception as exc:
        print(f"[edgar] {symbol}: IBKR skipped — {type(exc).__name__}: {exc}")
        logger.debug("%s: IBKR price fetch skipped — %s", symbol, exc)
        return None


def _fetch_price_from_yfinance(symbol: str, period_years: int = 2) -> Optional[pd.DataFrame]:
    """Fetch price history via yfinance (Yahoo Finance).  Always available, no key needed."""
    try:
        import yfinance as yf
        period_str = f"{period_years}y"
        t  = yf.Ticker(symbol.upper())
        df = t.history(period=period_str, auto_adjust=True)
        if df is None or df.empty:
            return None
        # Normalise: strip timezone, keep only OHLCV columns, sort oldest-first
        df.index = df.index.tz_localize(None) if df.index.tzinfo else df.index
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df = df.sort_index()
        logger.info("%s: fetched %d price bars from yfinance", symbol, len(df))
        return df
    except Exception as exc:
        logger.warning("%s: yfinance price fetch failed — %s", symbol, exc)
        return None


def _fetch_price_from_stooq(symbol: str, period_years: int = 2) -> Optional[pd.DataFrame]:
    """Fetch price history from Stooq via pandas_datareader (secondary free fallback)."""
    try:
        from pandas_datareader import data as pdr
        end   = pd.Timestamp.today()
        start = end - pd.DateOffset(years=period_years)
        df = pdr.DataReader(symbol.upper(), "stooq", start=start, end=end)
        if df is None or df.empty:
            return None
        df = df.sort_index()   # Stooq returns newest-first; flip to oldest-first
        logger.info("%s: fetched %d price bars from Stooq", symbol, len(df))
        return df
    except Exception as exc:
        logger.warning("%s: Stooq price fetch failed — %s", symbol, exc)
        return None


def get_price_history(symbol: str, period_years: int = 2) -> tuple[Optional[pd.DataFrame], str]:
    """Fetch daily OHLCV price history.

    Priority
    --------
    1. IBKR TWS/Gateway  (live, only if running)
    2. yfinance / Yahoo Finance  (always available, free)
    3. Stooq via pandas_datareader  (fallback; subject to daily rate limits)

    Returns
    -------
    (DataFrame, source_label)
        DataFrame has columns [Open, High, Low, Close, Volume], indexed by date
        (oldest first).  source_label is "IBKR", "yfinance", "Stooq", or "unavailable".
        On complete failure returns (None, "unavailable").
    """
    # 1 — IBKR (live, best quality)
    df = _fetch_price_from_ibkr(symbol, period_years=period_years)
    if df is not None and not df.empty:
        return df, "IBKR"

    # 2 — yfinance (Yahoo Finance — always available, no rate limit issues)
    df = _fetch_price_from_yfinance(symbol, period_years=period_years)
    if df is not None and not df.empty:
        return df, "yfinance"

    # 3 — Stooq (secondary fallback — may hit daily hits limit)
    df = _fetch_price_from_stooq(symbol, period_years=period_years)
    if df is not None and not df.empty:
        return df, "Stooq"

    logger.warning("%s: all price sources failed — returning unavailable", symbol)
    return None, "unavailable"


# ---------------------------------------------------------------------------
# Main fundamentals builder
# ---------------------------------------------------------------------------

def get_fundamentals(symbol: str) -> dict:
    """Return a fundamentals dict for *symbol* sourced from SEC EDGAR.

    Keys returned (mirror what screener.py reads)
    ---------------------------------------------
    trailingPE          – latest-price / (net-income / shares)
    profitMargins       – net income / revenue
    debtToEquity        – (total liabilities / equity) × 100  [screener divides by 100]
    freeCashflow        – operating CF − capex (most recent fiscal year)
    marketCap           – shares × latest price
    returnOnEquity      – net income / stockholders equity
    capitalExpenditures – capex as a negative number (convention)
    _fcf_history        – [FCF_yr0, FCF_yr1, ...] newest→oldest (for trend check)
    _sic                – 4-digit SIC string (for sector-aware D/E in screener)

    New quality metrics (YoY comparisons)
    --------------------------------------
    roic                – net income / (equity + long-term debt)  [most recent year]
    roic_prior          – same, prior year
    fcf_margin          – FCF / revenue  [most recent year]
    fcf_margin_prior    – same, prior year
    cash_conversion     – FCF / net income  (>1 = converting earnings to cash)
    cash_conversion_prior
    accruals_ratio      – (net_income − FCF) / avg_assets  (lower = higher quality)
    accruals_ratio_prior
    receivables_growth  – YoY growth in accounts receivable
    revenue_growth      – YoY growth in revenue  (compare to receivables_growth)
    revenue_prior       – prior-year revenue  (for display)
    revenue_current     – current-year revenue (for display)
    """
    result: dict = {}
    cik = _get_cik(symbol)
    if cik is None:
        return result

    # SIC code for sector detection
    sic = _get_sic(cik)
    if sic:
        result["_sic"] = sic

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

    # --- Capital expenditures (multiple common XBRL tags) ---
    capex_vals = (
        _get_annual_values(facts, "PaymentsToAcquirePropertyPlantAndEquipment")
        or _get_annual_values(facts, "PaymentsForCapitalImprovements")
        or _get_annual_values(facts, "CapitalExpendituresIncurredButNotYetPaid")
        or _get_annual_values(facts, "PaymentsToAcquireProductiveAssets")
    )

    # --- Free Cash Flow series ---
    fcf_history: list = []
    if op_cf_vals and capex_vals:
        pairs = min(len(op_cf_vals), len(capex_vals))
        fcf_history = [op_cf_vals[i] - capex_vals[i] for i in range(pairs)]
    elif op_cf_vals:
        fcf_history = op_cf_vals[:]
        logger.debug("%s: no capex tag found — using operating CF as FCF proxy", symbol)

    result["_fcf_history"] = fcf_history
    if fcf_history:
        result["freeCashflow"] = fcf_history[0]
    if capex_vals:
        result["capitalExpenditures"] = -abs(capex_vals[0])

    # --- Net income ---
    net_income_vals = (
        _get_annual_values(facts, "NetIncomeLoss")
        or _get_annual_values(facts, "ProfitLoss")
    )

    # --- Revenue (multiple tags for different reporting styles) ---
    revenue_vals = (
        _get_annual_values(facts, "RevenueFromContractWithCustomerExcludingAssessedTax")
        or _get_annual_values(facts, "Revenues")
        or _get_annual_values(facts, "SalesRevenueNet")
        or _get_annual_values(facts, "RevenueFromContractWithCustomerIncludingAssessedTax")
        or _get_annual_values(facts, "SalesRevenueGoodsNet")
    )

    # --- Profit margin ---
    if net_income_vals and revenue_vals and revenue_vals[0]:
        result["profitMargins"] = net_income_vals[0] / revenue_vals[0]

    # --- Debt / equity (×100 — screener.py divides by 100) ---
    liabilities_vals = _get_annual_values(facts, "Liabilities")
    equity_vals = (
        _get_annual_values(facts, "StockholdersEquity")
        or _get_annual_values(facts, "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")
    )
    if liabilities_vals and equity_vals and equity_vals[0] != 0:
        result["debtToEquity"] = (liabilities_vals[0] / equity_vals[0]) * 100.0

    # --- Return on equity ---
    if net_income_vals and equity_vals and equity_vals[0] != 0:
        result["returnOnEquity"] = net_income_vals[0] / equity_vals[0]

    # --- Long-term debt (for ROIC) ---
    ltd_vals = (
        _get_annual_values(facts, "LongTermDebt")
        or _get_annual_values(facts, "LongTermDebtNoncurrent")
        or _get_annual_values(facts, "LongTermNotesPayable")
    )

    # --- ROIC: net income / (equity + long-term debt) ---
    # Current year
    if net_income_vals and equity_vals and equity_vals[0] != 0:
        ltd_0 = ltd_vals[0] if ltd_vals else 0.0
        invested_capital_0 = equity_vals[0] + ltd_0
        if invested_capital_0 > 0:
            result["roic"] = net_income_vals[0] / invested_capital_0
    # Prior year
    if (len(net_income_vals) > 1 and len(equity_vals) > 1):
        ltd_1 = ltd_vals[1] if ltd_vals and len(ltd_vals) > 1 else 0.0
        invested_capital_1 = equity_vals[1] + ltd_1
        if invested_capital_1 > 0:
            result["roic_prior"] = net_income_vals[1] / invested_capital_1

    # --- FCF margin: FCF / revenue ---
    if fcf_history and revenue_vals and revenue_vals[0]:
        result["fcf_margin"] = fcf_history[0] / revenue_vals[0]
    if len(fcf_history) > 1 and revenue_vals and len(revenue_vals) > 1 and revenue_vals[1]:
        result["fcf_margin_prior"] = fcf_history[1] / revenue_vals[1]

    # --- Cash conversion: FCF / net income (>1 = cash compounder) ---
    if fcf_history and net_income_vals and net_income_vals[0] and net_income_vals[0] != 0:
        result["cash_conversion"] = fcf_history[0] / net_income_vals[0]
    if (len(fcf_history) > 1 and len(net_income_vals) > 1
            and net_income_vals[1] and net_income_vals[1] != 0):
        result["cash_conversion_prior"] = fcf_history[1] / net_income_vals[1]

    # --- Accruals ratio: (net_income − FCF) / avg_assets ---
    # Lower (or negative) accruals ratio = higher earnings quality
    total_assets_vals = _get_annual_values(facts, "Assets")
    if (net_income_vals and fcf_history and total_assets_vals
            and len(total_assets_vals) >= 2 and total_assets_vals[0] > 0):
        avg_assets = (total_assets_vals[0] + total_assets_vals[1]) / 2
        result["accruals_ratio"] = (net_income_vals[0] - fcf_history[0]) / avg_assets
    if (len(net_income_vals) > 1 and len(fcf_history) > 1
            and len(total_assets_vals) >= 3 and total_assets_vals[1] > 0):
        avg_assets_prior = (total_assets_vals[1] + total_assets_vals[2]) / 2
        result["accruals_ratio_prior"] = (net_income_vals[1] - fcf_history[1]) / avg_assets_prior

    # --- Receivables growth vs revenue growth ---
    receivables_vals = (
        _get_annual_values(facts, "AccountsReceivableNetCurrent")
        or _get_annual_values(facts, "ReceivablesNetCurrent")
        or _get_annual_values(facts, "AccountsReceivableNet")
    )
    if receivables_vals and len(receivables_vals) >= 2 and receivables_vals[1] > 0:
        result["receivables_growth"] = (receivables_vals[0] / receivables_vals[1]) - 1
    if revenue_vals and len(revenue_vals) >= 2 and revenue_vals[1] > 0:
        result["revenue_growth"] = (revenue_vals[0] / revenue_vals[1]) - 1
        result["revenue_current"] = revenue_vals[0]
        result["revenue_prior"]   = revenue_vals[1]

    # --- Shares outstanding ---
    shares_vals = (
        _get_annual_values(facts, "CommonStockSharesOutstanding", unit="shares")
        or _get_annual_values(facts, "EntityCommonStockSharesOutstanding", unit="shares")
    )

    # --- Latest price from IBKR / yfinance ---
    price_hist, _price_src = get_price_history(symbol, period_years=2)
    latest_price: Optional[float] = None
    if price_hist is not None and not price_hist.empty:
        latest_price = _safe(price_hist["Close"].iloc[-1])

    # --- Market cap ---
    if shares_vals and latest_price:
        result["marketCap"] = shares_vals[0] * latest_price

    # --- Trailing P/E ---
    if net_income_vals and shares_vals and shares_vals[0] > 0 and latest_price:
        eps = net_income_vals[0] / shares_vals[0]
        if eps > 0:
            result["trailingPE"] = latest_price / eps

    fcf_display = f"${result['freeCashflow']:,.0f}" if isinstance(result.get("freeCashflow"), float) else "N/A"
    de_display  = f"{result['debtToEquity']/100:.2f}x" if "debtToEquity" in result else "N/A"
    logger.info(
        "%s: EDGAR fundamentals fetched — FCF=%s, margin=%s, D/E=%s, ROE=%s, "
        "ROIC=%s, FCFmargin=%s, CashConv=%s, AccrualsRatio=%s, SIC=%s",
        symbol, fcf_display,
        f"{result['profitMargins']:.1%}" if "profitMargins" in result else "N/A",
        de_display,
        f"{result['returnOnEquity']:.1%}" if "returnOnEquity" in result else "N/A",
        f"{result['roic']:.1%}" if "roic" in result else "N/A",
        f"{result['fcf_margin']:.1%}" if "fcf_margin" in result else "N/A",
        f"{result['cash_conversion']:.2f}x" if "cash_conversion" in result else "N/A",
        f"{result['accruals_ratio']:.3f}" if "accruals_ratio" in result else "N/A",
        sic or "N/A",
    )
    return result
