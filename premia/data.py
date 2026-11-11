"""
premia/data.py — phase 1: downloads, corporate actions, data-quality rules, point-in-time access.

Sources (free; the paid alternatives and their costs are in DATA.md):
  * Yahoo Finance (yfinance): daily OHLCV, split-adjusted close, dividend-adjusted close, dividends, splits, index series.
  * IBKR TWS (read-only historical-data requests): 30-day ATM implied volatility per stock/ETF, daily since 2006.
  * SEC EDGAR: XBRL company facts (with filing dates, so fundamentals are point in time) and 8-K item 2.02 filing
    dates (earnings releases).
  * fja05680/sp500 (GitHub): S&P 500 membership intervals since 1996, used to stop survivorship bias at the universe level.

Rules applied to every series (and unit-tested):
  * Missing quotes are never invented. Implied vol may be carried forward at most `MAX_FFILL` trading days; beyond that
    the day is unusable and the strategies do not trade it.
  * Isolated bad prints in implied vol (a large jump that fully reverses the next day) are removed, not smoothed.
  * Prices are split-adjusted for returns and option strikes (OCC adjusts listed options on splits); market caps use the
    *unadjusted* close (split-adjusted close x later split ratios) times the share count filed at the time.
  * Dividends enter option pricing as a trailing-12-month continuous yield known at the date, and stock P&L as cash on
    the ex-date.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from . import store
from .config import ETFS, HISTORY_START, INDEX_SERIES, load_exclusions

log = logging.getLogger(__name__)

MAX_FFILL = 3                      # trading days an implied vol may be carried forward
BAD_PRINT_LOG_JUMP = 0.35          # |log IV change| that counts as a jump (~ +42 % / -30 %)
BAD_PRINT_REVERT = 0.75            # ...and the next day reverses at least this fraction of it
MEMBERSHIP_URL = "https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv"
FACT_TAGS_EXTRA = ["CommonStockSharesOutstanding", "WeightedAverageNumberOfSharesOutstandingBasic"]


# ---------------------------------------------------------------------------
# symbols
# ---------------------------------------------------------------------------

def norm(sym: str) -> str:
    """Canonical (Yahoo-style) ticker: BRK.B -> BRK-B."""
    return sym.strip().upper().replace(".", "-").replace(" ", "-")


def ib_symbol(sym: str) -> str:
    return norm(sym).replace("-", " ")


# ---------------------------------------------------------------------------
# survivorship: index membership
# ---------------------------------------------------------------------------

def fetch_membership() -> pd.DataFrame:
    import requests
    r = requests.get(MEMBERSHIP_URL, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df["ticker"] = df["ticker"].map(norm)
    store.put_membership(df)
    return df


def members_on(day, membership: pd.DataFrame) -> set[str]:
    """Tickers in the S&P 500 on *day* according to the membership intervals (end date exclusive)."""
    d = pd.Timestamp(day).strftime("%Y-%m-%d")
    m = membership
    live = (m["start"] <= d) & (m["end"].isna() | (m["end"] > d))
    return set(m.loc[live, "ticker"])


def candidate_universe(membership: pd.DataFrame, since: str = "2009-01-01") -> list[str]:
    """Every ticker that was an index member at any time since *since*, plus the ETFs, minus manual exclusions."""
    m = membership
    hit = m[(m["end"].isna()) | (m["end"] >= since)]
    names = sorted(set(hit["ticker"]) | set(ETFS))
    excl = load_exclusions()
    return [s for s in names if s not in excl]


# ---------------------------------------------------------------------------
# Yahoo: prices, actions, index series
# ---------------------------------------------------------------------------

def _yahoo_frame(raw: pd.DataFrame) -> pd.DataFrame:
    cols = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Adj Close": "adj_close", "Volume": "volume"}
    df = raw.rename(columns=cols)[[c for c in cols.values() if c in raw.rename(columns=cols).columns]]
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df.dropna(subset=["close"])


def fetch_yahoo_prices(symbols: list[str], start: str = HISTORY_START, batch: int = 40) -> dict[str, str]:
    """Daily bars + dividends + splits for *symbols*; returns {symbol: status}."""
    import yfinance as yf
    out: dict[str, str] = {}
    for i in range(0, len(symbols), batch):
        chunk = symbols[i:i + batch]
        try:
            raw = yf.download(chunk, start=start, auto_adjust=False, actions=True, group_by="ticker",
                              threads=True, progress=False)
        except Exception as exc:                                       # network: log and move on, never invent data
            for s in chunk:
                out[s] = "error"
                store.log_fetch(s, "price", "error", str(exc))
            continue
        for s in chunk:
            try:
                sub = raw[s] if isinstance(raw.columns, pd.MultiIndex) else raw
            except KeyError:
                sub = pd.DataFrame()
            sub = sub.dropna(how="all") if not sub.empty else sub
            if sub.empty or "Close" not in sub or sub["Close"].dropna().empty:
                out[s] = "no_data"
                store.log_fetch(s, "price", "no_data", "Yahoo returned no rows")
                continue
            df = _yahoo_frame(sub)
            n = store.put_prices(s, df)
            acts = []
            if "Dividends" in sub:
                acts += [(d.strftime("%Y-%m-%d"), "div", v) for d, v in sub["Dividends"].items() if v and v > 0]
            if "Stock Splits" in sub:
                acts += [(d.strftime("%Y-%m-%d"), "split", v) for d, v in sub["Stock Splits"].items() if v and v > 0]
            store.put_actions(s, [(d[:10], k, v) for d, k, v in acts])
            out[s] = "ok"
            store.log_fetch(s, "price", "ok", "", n, df.index[0].strftime("%Y-%m-%d"), df.index[-1].strftime("%Y-%m-%d"))
    return out


def fetch_index_series(names: Iterable[str] = INDEX_SERIES, start: str = HISTORY_START) -> dict[str, int]:
    import yfinance as yf
    out = {}
    for name in names:
        raw = yf.download(name, start=start, auto_adjust=False, progress=False)
        if raw.empty:
            out[name] = 0
            store.log_fetch(name, "series", "no_data")
            continue
        close = raw["Close"]
        close = close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close
        close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
        out[name] = store.put_series(name, close.dropna())
        store.log_fetch(name, "series", "ok", "", out[name])
    return out


# ---------------------------------------------------------------------------
# IBKR: 30-day implied volatility history (read-only)
# ---------------------------------------------------------------------------

async def _iv_one(ib, sym: str, duration: str, timeout: float) -> tuple[str, pd.Series, str]:
    from ib_insync import Stock
    c = Stock(ib_symbol(sym), "SMART", "USD")
    try:
        q = await asyncio.wait_for(ib.qualifyContractsAsync(c), 30)
    except Exception as exc:
        return "error", pd.Series(dtype=float), f"qualify: {exc}"
    if not q or not getattr(q[0], "conId", 0):
        return "unknown", pd.Series(dtype=float), "IBKR does not know this ticker (delisted or renamed)"
    errors: list = []

    def on_error(req_id, code, msg, contract):
        if contract is not None and getattr(contract, "conId", None) == c.conId and code not in (2104, 2106, 2158):
            errors.append(f"{code}: {msg}")

    ib.errorEvent += on_error
    t0 = time.monotonic()
    try:
        bars = await ib.reqHistoricalDataAsync(c, "", duration, "1 day", "OPTION_IMPLIED_VOLATILITY", True,
                                               formatDate=1, timeout=timeout)
    except Exception as exc:
        return "error", pd.Series(dtype=float), str(exc)
    finally:
        ib.errorEvent -= on_error
    if not bars:
        if time.monotonic() - t0 >= timeout - 1 or any("pacing" in e.lower() for e in errors):
            return "error", pd.Series(dtype=float), "; ".join(errors) or "timeout"
        return "no_data", pd.Series(dtype=float), "; ".join(errors) or "no implied-vol history"
    s = pd.Series([b.close for b in bars], index=pd.to_datetime([b.date for b in bars]), name=sym)
    return "ok", s[s > 0], ""


async def _iv_many(ib, symbols: list[str], duration: str, concurrency: int, timeout: float, on_done) -> None:
    sem = asyncio.Semaphore(concurrency)

    async def one(sym: str) -> None:
        async with sem:
            status, s, detail = await _iv_one(ib, sym, duration, timeout)
            on_done(sym, status, s, detail)

    await asyncio.gather(*(one(s) for s in symbols))


def fetch_ibkr_iv(symbols: list[str], port: int = 7496, client_id: int = 84, duration: str = "20 Y",
                  concurrency: int = 3, timeout: float = 240.0, retries: int = 1) -> dict[str, str]:
    """IBKR OPTION_IMPLIED_VOLATILITY daily history. Connects read-only; only historical-data requests are sent."""
    from ib_insync import IB
    result: dict[str, str] = {}

    def done(sym: str, status: str, s: pd.Series, detail: str) -> None:
        result[sym] = status
        n = store.put_iv(sym, s) if status == "ok" else 0
        store.log_fetch(sym, "iv", status, detail, n,
                        s.index[0].strftime("%Y-%m-%d") if n else None, s.index[-1].strftime("%Y-%m-%d") if n else None)
        log.info("iv %-6s %-8s %5d %s", sym, status, n, detail[:80])

    async def main() -> None:
        ib = IB()
        await ib.connectAsync("127.0.0.1", port, clientId=client_id, timeout=15, readonly=True)
        try:
            todo = list(symbols)
            for attempt in range(retries + 1):
                await _iv_many(ib, todo, duration, concurrency, timeout, done)
                todo = [s for s in todo if result.get(s) == "error"]
                if not todo:
                    break
                await asyncio.sleep(60)                                   # let pacing windows clear before the retry pass
        finally:
            ib.disconnect()

    asyncio.run(main())
    return result


# ---------------------------------------------------------------------------
# SEC: company facts (point in time) and earnings-release dates
# ---------------------------------------------------------------------------

def reduce_facts(facts: dict) -> dict:
    """Keep only the tags the screener uses, every filing of each (the filing dates make it point in time)."""
    from stockscreen.sec_data import TAGS
    keep = {t for tags in TAGS.values() for t in tags} | set(FACT_TAGS_EXTRA)
    out = {"cik": facts.get("cik"), "entityName": facts.get("entityName"), "facts": {"us-gaap": {}, "dei": {}}}
    gaap = facts.get("facts", {}).get("us-gaap", {})
    for tag in keep:
        if tag in gaap:
            out["facts"]["us-gaap"][tag] = {"units": gaap[tag].get("units", {})}
    dei = facts.get("facts", {}).get("dei", {})
    if "EntityCommonStockSharesOutstanding" in dei:
        out["facts"]["dei"]["EntityCommonStockSharesOutstanding"] = {"units": dei["EntityCommonStockSharesOutstanding"]["units"]}
    return out


def earnings_dates_from_submissions(sub: dict, older: Iterable[dict] = ()) -> list[str]:
    """Filing dates of 8-K reports carrying item 2.02 (Results of Operations and Financial Condition)."""
    out: list[str] = []
    for block in [sub.get("filings", {}).get("recent", {}), *older]:
        forms, dates, items = block.get("form", []), block.get("filingDate", []), block.get("items", [])
        for f, d, it in zip(forms, dates, items):
            if f in ("8-K", "8-K/A") and "2.02" in str(it):
                out.append(d)
    return sorted(set(out))


def _sec_get(url: str):
    import requests
    import edgar
    for attempt in range(4):
        edgar._sec_rate_limiter.acquire()
        try:
            r = requests.get(url, headers=edgar.EDGAR_HEADERS, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except Exception:
            time.sleep(2.0 * (attempt + 1))
    return None


def fetch_sec(symbols: list[str], workers: int = 4) -> dict[str, str]:
    """Reduced company facts + SIC + earnings dates per symbol (current SEC ticker map: delisted tickers have no CIK)."""
    from concurrent.futures import ThreadPoolExecutor
    import edgar
    from stockscreen.sec_data import get_cik
    out: dict[str, str] = {}

    def one(sym: str) -> None:
        cik = get_cik(sym)
        if not cik:
            out[sym] = "no_cik"
            store.log_fetch(sym, "sec", "no_cik", "not in SEC's current ticker map (ETF, delisted or renamed)")
            return
        facts = _sec_get(edgar.EDGAR_FACTS_URL.format(cik=cik))
        sub = _sec_get(edgar.EDGAR_SUBMISSIONS_URL.format(cik=cik)) or {}
        older = []
        for f in sub.get("filings", {}).get("files", []):
            j = _sec_get(f"https://data.sec.gov/submissions/{f['name']}")
            if j:
                older.append(j)
        if facts:
            store.put_facts(sym, cik, facts.get("entityName", ""), str(sub.get("sic", "") or ""), reduce_facts(facts))
        ed = earnings_dates_from_submissions(sub, older)
        store.put_earnings(sym, ed, "sec-8k-2.02")
        out[sym] = "ok" if facts else "no_facts"
        store.log_fetch(sym, "sec", out[sym], f"{len(ed)} earnings dates", len(ed), ed[0] if ed else None, ed[-1] if ed else None)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, symbols))
    return out


# ---------------------------------------------------------------------------
# cleaning and point-in-time helpers
# ---------------------------------------------------------------------------

def clean_iv(iv: pd.Series) -> tuple[pd.Series, pd.Series]:
    """(cleaned IV, boolean mask of removed bad prints). Removes isolated spikes that reverse the next day."""
    s = iv.dropna()
    s = s[(s > 0.01) & (s < 5.0)]
    lg = np.log(s)
    d = lg.diff()
    nxt = d.shift(-1)
    bad = (d.abs() > BAD_PRINT_LOG_JUMP) & (np.sign(nxt) == -np.sign(d)) & (nxt.abs() >= BAD_PRINT_REVERT * d.abs())
    bad = bad.fillna(False)
    return s[~bad], bad.reindex(iv.index, fill_value=False)


def align(prices: pd.DataFrame, iv: pd.Series) -> pd.DataFrame:
    """Price calendar is the master. IV carried forward at most MAX_FFILL days; `iv_age` says how stale it is."""
    out = prices.copy()
    s = iv.reindex(out.index)
    have = s.notna()
    grp = have.cumsum()
    age = have.groupby(grp).cumcount().where(grp > 0)
    out["iv"] = s.ffill(limit=MAX_FFILL)
    out["iv_age"] = age.where(out["iv"].notna())
    return out


def split_factor_after(actions: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Product of split ratios strictly after each date: unadjusted price = split-adjusted price x this."""
    sp = actions[actions["kind"] == "split"].set_index("date")["value"] if not actions.empty else pd.Series(dtype=float)
    f = pd.Series(1.0, index=index)
    for d, ratio in sp.items():
        f[index < d] *= ratio
    return f


def unadjusted_close(prices: pd.DataFrame, actions: pd.DataFrame) -> pd.Series:
    return prices["close"] * split_factor_after(actions, prices.index)


def dividend_yield_ttm(prices: pd.DataFrame, actions: pd.DataFrame) -> pd.Series:
    """Trailing-12-month dividends / close, known at each date (no look-ahead)."""
    divs = actions[actions["kind"] == "div"].set_index("date")["value"] if not actions.empty else pd.Series(dtype=float)
    daily = divs.groupby(level=0).sum().reindex(prices.index, fill_value=0.0)
    ttm = daily.rolling("365D").sum()
    return (ttm / prices["close"]).fillna(0.0)


def dividends_cash(prices: pd.DataFrame, actions: pd.DataFrame) -> pd.Series:
    divs = actions[actions["kind"] == "div"].set_index("date")["value"] if not actions.empty else pd.Series(dtype=float)
    return divs.groupby(level=0).sum().reindex(prices.index, fill_value=0.0)


def risk_free(index: pd.DatetimeIndex) -> pd.Series:
    """13-week T-bill yield (^IRX, percent) as a continuously-compounded decimal, carried forward."""
    irx = store.load_series("^IRX")
    if irx.empty:
        return pd.Series(0.02, index=index)
    r = np.log1p(irx / 100.0).reindex(index.union(irx.index)).ffill().reindex(index)
    return r.fillna(0.02)


@dataclass
class Panel:
    """Everything one underlying needs, aligned on its trading calendar."""
    symbol: str
    df: pd.DataFrame          # open high low close adj_close volume iv iv_age q r div_cash dollar_vol
    earnings: list
    bad_prints: int


def load_panel(symbol: str, start: Optional[str] = None) -> Optional[Panel]:
    px = store.load_prices(symbol)
    iv = store.load_iv(symbol)
    if px.empty or iv.empty:
        return None
    acts = store.load_actions(symbol)
    iv_clean, bad = clean_iv(iv)
    df = align(px, iv_clean)
    df["q"] = dividend_yield_ttm(px, acts)
    df["div_cash"] = dividends_cash(px, acts)
    df["r"] = risk_free(df.index)
    df["dollar_vol"] = (px["close"] * px["volume"]).rolling(20, min_periods=10).mean()
    df["unadj_close"] = unadjusted_close(px, acts)
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    return Panel(symbol, df, store.load_earnings(symbol), int(bad.sum()))


def coverage_report(membership: pd.DataFrame, start: str = "2010-01-01", step: str = "QS") -> pd.DataFrame:
    """Per quarter: index members, members with price data, with IV data. The shortfall is the survivorship gap."""
    have_px = set(store.symbols_with("price"))
    have_iv = set(store.symbols_with("iv"))
    excl = load_exclusions()
    rows = []
    for d in pd.date_range(start, pd.Timestamp.today(), freq=step):
        mem = members_on(d, membership) - excl
        rows.append({"date": d.date(), "members": len(mem), "with_prices": len(mem & have_px),
                     "with_iv": len(mem & have_iv)})
    out = pd.DataFrame(rows)
    out["iv_coverage"] = out["with_iv"] / out["members"]
    return out
