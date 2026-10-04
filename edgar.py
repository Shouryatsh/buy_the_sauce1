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
import threading
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

_RATE_LIMIT_SLEEP = 0.12   # ~8 req/s — well under SEC's 10/s limit, single-threaded callers


class _RateLimiter:
    """Thread-safe token-pacing limiter shared across every caller.

    A plain `time.sleep(x)` before each request only limits *one* thread's
    own request rate; running screen_wheel_candidate for hundreds of
    symbols across a thread pool (wheel_universe.py) would otherwise let
    each worker sleep independently and the *combined* rate across all of
    them blow well past SEC's ~10 req/s guidance. This limiter tracks one
    shared "next allowed time" so the aggregate rate stays capped no matter
    how many threads are calling in concurrently.
    """

    def __init__(self, requests_per_second: float) -> None:
        self._interval = 1.0 / requests_per_second
        self._lock = threading.Lock()
        self._next_time = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_time - now
            if wait > 0:
                time.sleep(wait)
                now = self._next_time
            self._next_time = max(now, self._next_time) + self._interval


_sec_rate_limiter = _RateLimiter(8.0)   # ~8 req/s system-wide, however many threads are calling


# ---------------------------------------------------------------------------
# CIK lookup (cached for the process lifetime)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_ticker_map() -> dict:
    """Download SEC's full ticker→CIK map once and cache it.

    The returned dict preserves SEC's own ordering, which is roughly
    market-cap descending (NVDA, AAPL, GOOGL, MSFT, AMZN, ... in practice) --
    wheel_universe.py relies on this to build a "top N by size" universe
    without needing a separate paid index-constituent data source.
    """
    try:
        resp = requests.get(EDGAR_TICKERS_URL, headers=EDGAR_HEADERS, timeout=15)
        resp.raise_for_status()
        raw = resp.json()
        return {v["ticker"].upper(): int(v["cik_str"]) for v in raw.values()}
    except Exception as exc:
        logger.error("Could not load SEC ticker map: %s", exc)
        return {}


def get_ticker_universe(n: Optional[int] = None) -> list[str]:
    """Return SEC-registered tickers in their roughly market-cap-descending
    order, optionally truncated to the first *n*."""
    tickers = list(_load_ticker_map().keys())
    return tickers[:n] if n else tickers


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
        _sec_rate_limiter.acquire()
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
    _sec_rate_limiter.acquire()
    resp = requests.get(url, headers=EDGAR_HEADERS, timeout=20)
    resp.raise_for_status()
    return resp.json()


def _get_annual_values(
    facts: dict,
    concept: str,
    unit: str = "USD",
    taxonomy: str = "us-gaap",
    forms: tuple = ("10-K", "10-K/A"),
) -> list:
    """Extract the most-recent filed value per fiscal year for an XBRL
    concept, from the given *taxonomy* and *forms*.

    Returns a list ordered newest -> oldest, up to 5 fiscal years.

    Almost every income-statement/cash-flow/balance-sheet concept lives
    under the "us-gaap" taxonomy and is only meaningful annually (10-K),
    hence those defaults. Shares outstanding is the notable exception: many
    large caps (confirmed: JNJ, XOM, META, V) report
    EntityCommonStockSharesOutstanding under the "dei" (Document Entity
    Information) taxonomy instead, updated every quarter on the 10-Q cover
    page -- restricting to us-gaap/10-K silently drops market cap (and
    everything downstream of it: FCF yield, trailing P/E) for those names.
    """
    try:
        entries = facts["facts"][taxonomy][concept]["units"][unit]
    except KeyError:
        return []

    annual: dict = {}
    for e in entries:
        if e.get("form") in forms and e.get("val") is not None:
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

    Returns a DataFrame (oldest-first, columns matching Stooq output) or None
    if TWS is not running or the request fails.
    """
    try:
        # ib_insync (and eventkit, which it imports) needs a thread-local
        # asyncio event loop -- eventkit even calls get_event_loop() at
        # *import* time, so this must run before `from ib_insync import
        # ...`, not after. A loop only exists automatically on the main
        # thread; callers like the wheel dashboard (via Streamlit) run this
        # on a worker thread that never gets one. This connection is fully
        # self-contained within this function call (connect, fetch,
        # disconnect, all here, no reuse across calls), so a plain
        # per-thread loop is enough -- unlike wheel_ibkr.py's persistent
        # connection, there's no "which thread owns the live socket"
        # problem to worry about here.
        import asyncio
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

        from ib_insync import IB, Stock, util
        import config as _cfg
        ib = IB()
        ib.connect(_cfg.IBKR_HOST, _cfg.IBKR_PORT, clientId=_cfg.IBKR_CLIENT_ID + 99, readonly=True, timeout=4)
        contract = Stock(symbol.upper(), "SMART", "USD")
        duration = f"{period_years} Y"
        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting="1 day",
            whatToShow="ADJUSTED_LAST",
            useRTH=True,
        )
        ib.disconnect()
        if not bars:
            return None
        df = util.df(bars)[["date", "open", "high", "low", "close", "volume"]].copy()
        df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
        df = df.set_index("Date").sort_index()
        logger.info("%s: fetched %d price bars from IBKR", symbol, len(df))
        return df
    except Exception as exc:
        logger.debug("%s: IBKR price fetch skipped — %s", symbol, exc)
        return None


def get_price_history(symbol: str, period_years: int = 2) -> tuple[Optional[pd.DataFrame], str]:
    """Fetch daily OHLCV price history.  Tries IBKR first, falls back to Stooq.

    Returns
    -------
    (DataFrame, source_label)
        DataFrame has columns [Open, High, Low, Close, Volume], indexed by date
        (oldest first).  source_label is "IBKR" or "Stooq".
        On complete failure returns (None, "unavailable").
    """
    # --- Try IBKR first (live data, only works when TWS/Gateway is running) ---
    df = _fetch_price_from_ibkr(symbol, period_years=period_years)
    if df is not None and not df.empty:
        return df, "IBKR"

    # --- Fall back to Stooq (free, always available) ---
    try:
        from pandas_datareader import data as pdr
        end   = pd.Timestamp.today()
        start = end - pd.DateOffset(years=period_years)
        df = pdr.DataReader(symbol.upper(), "stooq", start=start, end=end)
        if df.empty:
            raise ValueError("empty response")
        df = df.sort_index()   # Stooq returns newest-first; flip to oldest-first
        logger.info("%s: fetched %d price bars from Stooq", symbol, len(df))
        return df, "Stooq"
    except Exception as exc:
        logger.warning("%s: Stooq price fetch failed — %s", symbol, exc)
        return None, "unavailable"


# ---------------------------------------------------------------------------
# Main fundamentals builder
# ---------------------------------------------------------------------------

def get_fundamentals(symbol: str, latest_price: Optional[float] = None) -> dict:
    """Return a fundamentals dict for *symbol* sourced from SEC EDGAR.

    Keys returned (mirror what screener.py reads)
    ---------------------------------------------
    trailingPE          – latest-price / (net-income / shares)
    profitMargins       – net income / revenue
    debtToEquity        – (total liabilities / equity) × 100  [screener divides by 100]
    freeCashflow        – operating CF − capex (most recent fiscal year)
    marketCap           – shares × latest Stooq close
    returnOnEquity      – net income / stockholders equity
    capitalExpenditures – capex as a negative number (convention)
    _fcf_history        – [FCF_yr0, FCF_yr1, ...] newest→oldest (for trend check)
    _sic                – 4-digit SIC string (for sector-aware D/E in screener)

    Parameters
    ----------
    latest_price:
        If given, used directly for market cap instead of calling
        get_price_history() internally. Bulk callers screening hundreds of
        symbols (wheel_universe.py) fetch prices in a handful of batched
        yfinance calls up front and pass them in here -- IBKR's
        historical-data pacing limits and Stooq's now-blocked CSV export
        make a fresh per-symbol price lookup impractical at that scale.
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
        # Some asset-light companies (e.g. Visa) have near-zero capex not tagged separately
        # Use operating CF as a conservative FCF proxy
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

    # --- History arrays (newest -> oldest) for trend scoring, e.g. wheel_fundamentals.py ---
    result["_net_income_history"] = net_income_vals
    result["_revenue_history"] = revenue_vals

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

    # --- Shares outstanding ---
    # CommonStockSharesOutstanding (us-gaap, 10-K only) covers some
    # companies; EntityCommonStockSharesOutstanding (dei taxonomy, updated
    # every 10-Q) is where many large caps actually report it -- see
    # _get_annual_values' docstring.
    shares_vals = (
        _get_annual_values(facts, "CommonStockSharesOutstanding", unit="shares")
        or _get_annual_values(
            facts, "EntityCommonStockSharesOutstanding", unit="shares",
            taxonomy="dei", forms=("10-K", "10-K/A", "10-Q"),
        )
    )

    # --- Latest price: caller-supplied (bulk scans), else Stooq / IBKR ---
    if latest_price is None:
        price_hist, _price_src = get_price_history(symbol, period_years=2)
        if price_hist is not None and not price_hist.empty:
            latest_price = _safe(price_hist["Close"].iloc[-1])
    result["_latest_price"] = latest_price   # avoids re-fetching price for callers that already need it

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
        "%s: EDGAR fundamentals fetched — FCF=%s, margin=%s, D/E=%s, ROE=%s, SIC=%s",
        symbol, fcf_display,
        f"{result['profitMargins']:.1%}" if "profitMargins" in result else "N/A",
        de_display,
        f"{result['returnOnEquity']:.1%}" if "returnOnEquity" in result else "N/A",
        sic or "N/A",
    )
    return result
