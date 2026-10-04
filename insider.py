"""
insider.py — Insider transaction analysis from SEC EDGAR Form 4 filings.

Data source: SEC EDGAR ownership filings (Form 3/4/5)
    https://data.sec.gov/submissions/CIK{cik}.json → recent filings
    https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=4&dateb=&owner=include&count=40&search_text=&action=getcompany

This module fetches ONLY from official, public SEC EDGAR endpoints.
No scraping, no paid APIs, no non-public data. Fully legal.

Taxonomy of insider transaction categories (strength-ranked)
------------------------------------------------------------
  1. CEO_OPEN_MARKET_BUY     — CEO buying shares on the open market (strongest signal)
  2. CLUSTER_BUY             — 3+ different insiders buying in the same 7-day window
  3. DIRECTOR_OPEN_MARKET_BUY — Director buying shares on the open market
  4. OFFICER_OPEN_MARKET_BUY — Other officer (CFO, COO, etc.) buying on the open market
  5. 10B5_1_PLAN_SALE        — Pre-planned automatic sale under Rule 10b5-1
  6. OPTION_EXERCISE_SALE    — Option exercise + immediate sale (often routine compensation)
  7. LARGE_SHAREHOLDER_TRIM  — 10%+ owner reducing stake

Each transaction record includes:
  - filer name, relationship (CEO/Director/Officer/10% Owner)
  - transaction type and category
  - transaction date, price, shares, total value
  - shares as % of outstanding
  - whether it's a buy or sell
"""

from __future__ import annotations

import datetime
import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import requests

import edgar as _edgar_module

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EDGAR_HEADERS = _edgar_module.EDGAR_HEADERS
_RATE_LIMIT_SLEEP = _edgar_module._RATE_LIMIT_SLEEP

# How far back to look for insider trades (days)
INSIDER_LOOKBACK_DAYS = 90

# Cluster buy detection: min distinct insiders buying within N days
CLUSTER_BUY_MIN_INSIDERS = 3
CLUSTER_BUY_WINDOW_DAYS = 7


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class InsiderTransaction:
    """A single insider transaction parsed from a Form 4 filing."""
    symbol: str
    filer_name: str
    filer_title: str               # e.g. "CEO", "Director", "CFO", "10% Owner"
    relationship: str              # "officer", "director", "tenPercentOwner", "other"
    is_officer: bool
    is_director: bool
    is_ten_pct_owner: bool
    transaction_date: Optional[str]  # YYYY-MM-DD
    transaction_code: str          # "P"=purchase, "S"=sale, "A"=award, "M"=exercise, etc
    shares: float
    price_per_share: Optional[float]
    total_value: Optional[float]
    shares_owned_after: Optional[float]
    acquired_disposed: str         # "A" (acquired) or "D" (disposed)
    is_10b5_1: bool                # transaction under Rule 10b5-1 plan
    filing_date: Optional[str]
    # Computed
    category: str = ""             # set by categorize()
    pct_of_outstanding: Optional[float] = None  # shares / shares_outstanding * 100


@dataclass
class InsiderSummary:
    """Aggregated insider activity summary for a single ticker."""
    symbol: str
    transactions: list[InsiderTransaction] = field(default_factory=list)
    lookback_days: int = INSIDER_LOOKBACK_DAYS

    # Aggregated signals (populated by analyze())
    total_insider_buys: int = 0
    total_insider_sells: int = 0
    net_insider_shares: float = 0.0        # positive = net buying
    net_insider_value: float = 0.0         # positive = net buying ($)
    cluster_buy_detected: bool = False     # 3+ insiders bought in same week
    ceo_bought: bool = False
    director_bought: bool = False
    officer_bought: bool = False
    has_10b5_1_sales: bool = False
    has_option_exercise_sales: bool = False
    large_shareholder_trimming: bool = False

    # Strength signal (computed)
    insider_signal: str = "NEUTRAL"        # STRONG_BUY / BUY / NEUTRAL / CAUTION / SELL
    insider_score: int = 0                 # 0–100 (higher = more bullish insider activity)

    # Display fields
    top_category: str = "—"               # highest-ranked category detected
    notable_transactions: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# SEC EDGAR Form 4 fetcher
# ---------------------------------------------------------------------------

def _get_recent_form4_urls(cik: int, count: int = 40) -> list[dict]:
    """Fetch recent Form 4 filing metadata from EDGAR submissions endpoint.

    Returns list of dicts with keys: accessionNumber, filingDate, primaryDocument.
    """
    time.sleep(_RATE_LIMIT_SLEEP)
    url = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
    try:
        resp = requests.get(url, headers=EDGAR_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("Failed to fetch submissions for CIK %d: %s", cik, exc)
        return []

    recent = data.get("filings", {}).get("recent", {})
    if not recent:
        return []

    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])

    results = []
    for i, form_type in enumerate(forms):
        if form_type in ("4", "4/A"):
            results.append({
                "accessionNumber": accessions[i],
                "filingDate": filing_dates[i],
                "primaryDocument": primary_docs[i],
            })
        if len(results) >= count:
            break
    return results


def _parse_form4_xml(cik: int, accession: str, primary_doc: str,
                     filing_date: str, symbol: str) -> list[InsiderTransaction]:
    """Fetch and parse a single Form 4 XML filing into InsiderTransaction objects."""
    # Build the URL — accession without dashes for the directory
    acc_nodash = accession.replace("-", "")
    xml_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{primary_doc}"

    time.sleep(_RATE_LIMIT_SLEEP)
    try:
        resp = requests.get(xml_url, headers=EDGAR_HEADERS, timeout=15)
        resp.raise_for_status()
        content = resp.text
    except Exception as exc:
        logger.debug("Failed to fetch Form 4 XML %s: %s", xml_url, exc)
        return []

    # Parse XML
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        logger.debug("Failed to parse Form 4 XML: %s", xml_url)
        return []

    # Namespace handling — Form 4 XML may or may not use a namespace
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    # ── Filer info ────────────────────────────────────────────────────────────
    def _text(parent, tag, default=""):
        el = parent.find(f"{ns}{tag}")
        if el is None:
            # Try without namespace
            el = parent.find(tag)
        return (el.text or "").strip() if el is not None else default

    def _find(parent, tag):
        el = parent.find(f"{ns}{tag}")
        if el is None:
            el = parent.find(tag)
        return el

    # Reporting owner info
    owner_el = _find(root, "reportingOwner")
    if owner_el is None:
        # Try finding under ownershipDocument
        doc_el = _find(root, "ownershipDocument")
        if doc_el is not None:
            owner_el = _find(doc_el, "reportingOwner")

    filer_name = ""
    filer_title = ""
    is_officer = False
    is_director = False
    is_ten_pct = False

    if owner_el is not None:
        id_el = _find(owner_el, "reportingOwnerId")
        if id_el is not None:
            filer_name = _text(id_el, "rptOwnerName")

        rel_el = _find(owner_el, "reportingOwnerRelationship")
        if rel_el is not None:
            is_director = _text(rel_el, "isDirector") == "1" or _text(rel_el, "isDirector").lower() == "true"
            is_officer = _text(rel_el, "isOfficer") == "1" or _text(rel_el, "isOfficer").lower() == "true"
            is_ten_pct = _text(rel_el, "isTenPercentOwner") == "1" or _text(rel_el, "isTenPercentOwner").lower() == "true"
            filer_title = _text(rel_el, "officerTitle")

    # Determine relationship string
    relationship = "other"
    if is_officer:
        relationship = "officer"
    elif is_director:
        relationship = "director"
    elif is_ten_pct:
        relationship = "tenPercentOwner"

    # ── Parse non-derivative transactions ─────────────────────────────────────
    transactions: list[InsiderTransaction] = []

    # Try both with and without wrapping ownershipDocument
    for search_root in [root, _find(root, "ownershipDocument") or root]:
        nd_table = _find(search_root, "nonDerivativeTable")
        if nd_table is not None:
            for txn_el in nd_table.findall(f"{ns}nonDerivativeTransaction") + nd_table.findall("nonDerivativeTransaction"):
                txn = _parse_single_transaction(
                    txn_el, ns, symbol, filer_name, filer_title,
                    relationship, is_officer, is_director, is_ten_pct,
                    filing_date,
                )
                if txn is not None:
                    transactions.append(txn)

        # Also check derivative transactions (option exercises)
        d_table = _find(search_root, "derivativeTable")
        if d_table is not None:
            for txn_el in d_table.findall(f"{ns}derivativeTransaction") + d_table.findall("derivativeTransaction"):
                txn = _parse_derivative_transaction(
                    txn_el, ns, symbol, filer_name, filer_title,
                    relationship, is_officer, is_director, is_ten_pct,
                    filing_date,
                )
                if txn is not None:
                    transactions.append(txn)

    return transactions


def _parse_single_transaction(
    txn_el, ns: str, symbol: str,
    filer_name: str, filer_title: str, relationship: str,
    is_officer: bool, is_director: bool, is_ten_pct: bool,
    filing_date: str,
) -> Optional[InsiderTransaction]:
    """Parse a single <nonDerivativeTransaction> element."""

    def _text(parent, tag, default=""):
        el = parent.find(f"{ns}{tag}")
        if el is None:
            el = parent.find(tag)
        return (el.text or "").strip() if el is not None else default

    def _find(parent, tag):
        el = parent.find(f"{ns}{tag}")
        if el is None:
            el = parent.find(tag)
        return el

    def _val(parent, tag):
        """Extract a <value> sub-element's text."""
        el = _find(parent, tag)
        if el is not None:
            v_el = _find(el, "value")
            if v_el is not None and v_el.text:
                try:
                    return float(v_el.text.strip())
                except ValueError:
                    pass
            # Sometimes value is directly in the element
            if el.text and el.text.strip():
                try:
                    return float(el.text.strip())
                except ValueError:
                    pass
        return None

    # Transaction coding
    coding_el = _find(txn_el, "transactionCoding")
    txn_code = ""
    is_10b5_1 = False
    if coding_el is not None:
        txn_code = _text(coding_el, "transactionCode")
        is_10b5_1_text = _text(coding_el, "equitySwapInvolved")
        # 10b5-1 flag can also be in transactionTimeliness or footnotes
        # The standard field is transactionFormType, but the plan indicator
        # is often in the coding element
        plan_text = _text(coding_el, "transactionCode")
        # Actually, the 10b5-1 indicator is in a footnote or the
        # "equitySwapInvolved" field or a separate element
        # Let's check the footnote references as well
        is_10b5_1 = False

    # Check for 10b5-1 in footnotes (common pattern)
    footnote_els = txn_el.findall(f".//{ns}footnoteId") + txn_el.findall(".//footnoteId")

    # Transaction amounts
    amounts_el = _find(txn_el, "transactionAmounts")
    shares = None
    price = None
    acq_disp = "A"
    if amounts_el is not None:
        shares = _val(amounts_el, "transactionShares")
        price = _val(amounts_el, "transactionPricePerShare")
        ad_el = _find(amounts_el, "transactionAcquiredDisposedCode")
        if ad_el is not None:
            acq_disp = _text(ad_el, "value") or "A"

    # Transaction date
    date_el = _find(txn_el, "transactionDate")
    txn_date = None
    if date_el is not None:
        txn_date = _text(date_el, "value")

    # Post-transaction holdings
    post_el = _find(txn_el, "postTransactionAmounts")
    shares_after = None
    if post_el is not None:
        shares_after = _val(post_el, "sharesOwnedFollowingTransaction")

    if shares is None or shares == 0:
        return None

    total_value = round(shares * price, 2) if price and shares else None

    return InsiderTransaction(
        symbol=symbol,
        filer_name=filer_name,
        filer_title=filer_title,
        relationship=relationship,
        is_officer=is_officer,
        is_director=is_director,
        is_ten_pct_owner=is_ten_pct,
        transaction_date=txn_date,
        transaction_code=txn_code,
        shares=abs(shares),
        price_per_share=price,
        total_value=total_value,
        shares_owned_after=shares_after,
        acquired_disposed=acq_disp,
        is_10b5_1=is_10b5_1,
        filing_date=filing_date,
    )


def _parse_derivative_transaction(
    txn_el, ns: str, symbol: str,
    filer_name: str, filer_title: str, relationship: str,
    is_officer: bool, is_director: bool, is_ten_pct: bool,
    filing_date: str,
) -> Optional[InsiderTransaction]:
    """Parse a <derivativeTransaction> element (option exercises, etc.)."""

    def _text(parent, tag, default=""):
        el = parent.find(f"{ns}{tag}")
        if el is None:
            el = parent.find(tag)
        return (el.text or "").strip() if el is not None else default

    def _find(parent, tag):
        el = parent.find(f"{ns}{tag}")
        if el is None:
            el = parent.find(tag)
        return el

    def _val(parent, tag):
        el = _find(parent, tag)
        if el is not None:
            v_el = _find(el, "value")
            if v_el is not None and v_el.text:
                try:
                    return float(v_el.text.strip())
                except ValueError:
                    pass
        return None

    coding_el = _find(txn_el, "transactionCoding")
    txn_code = ""
    if coding_el is not None:
        txn_code = _text(coding_el, "transactionCode")

    # Only care about exercises (M/C) and dispositions
    if txn_code not in ("M", "C", "S", "P"):
        return None

    amounts_el = _find(txn_el, "transactionAmounts")
    shares = None
    price = None
    acq_disp = "A"
    if amounts_el is not None:
        shares = _val(amounts_el, "transactionShares")
        price = _val(amounts_el, "transactionPricePerShare")
        ad_el = _find(amounts_el, "transactionAcquiredDisposedCode")
        if ad_el is not None:
            acq_disp = _text(ad_el, "value") or "A"

    date_el = _find(txn_el, "transactionDate")
    txn_date = None
    if date_el is not None:
        txn_date = _text(date_el, "value")

    post_el = _find(txn_el, "postTransactionAmounts")
    shares_after = None
    if post_el is not None:
        shares_after = _val(post_el, "sharesOwnedFollowingTransaction")

    if shares is None or shares == 0:
        return None

    total_value = round(shares * price, 2) if price and shares else None

    return InsiderTransaction(
        symbol=symbol,
        filer_name=filer_name,
        filer_title=filer_title,
        relationship=relationship,
        is_officer=is_officer,
        is_director=is_director,
        is_ten_pct_owner=is_ten_pct,
        transaction_date=txn_date,
        transaction_code=txn_code,
        shares=abs(shares),
        price_per_share=price,
        total_value=total_value,
        shares_owned_after=shares_after,
        acquired_disposed=acq_disp,
        is_10b5_1=False,
        filing_date=filing_date,
    )


# ---------------------------------------------------------------------------
# 10b5-1 detection (heuristic — SEC XML doesn't always have a clean flag)
# ---------------------------------------------------------------------------

def _detect_10b5_1_from_filing(cik: int, accession: str) -> bool:
    """Check the filing index page for Rule 10b5-1 references.

    Many Form 4 filings include a footnote or cover page mentioning
    the 10b5-1 plan. We check the filing index for these keywords.
    """
    acc_nodash = accession.replace("-", "")
    index_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/"
    time.sleep(_RATE_LIMIT_SLEEP)
    try:
        resp = requests.get(index_url, headers=EDGAR_HEADERS, timeout=10)
        text = resp.text.lower()
        return "10b5-1" in text or "rule 10b5" in text or "10b-5-1" in text
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Categorisation logic
# ---------------------------------------------------------------------------

_CEO_TITLES = re.compile(
    r"(chief\s+executive|ceo|president\s+and\s+ceo|president\s*&\s*ceo"
    r"|chief\s+exec|ceo\s*[,&])",
    re.IGNORECASE,
)

_CFO_TITLES = re.compile(
    r"(chief\s+financial|cfo|treasurer|chief\s+accounting)",
    re.IGNORECASE,
)


def _is_ceo(txn: InsiderTransaction) -> bool:
    """Detect if the filer is the CEO (or equivalent)."""
    title = txn.filer_title or ""
    return bool(_CEO_TITLES.search(title))


def _is_cfo(txn: InsiderTransaction) -> bool:
    title = txn.filer_title or ""
    return bool(_CFO_TITLES.search(title))


def categorize(txn: InsiderTransaction) -> str:
    """Assign a category string to a single transaction.

    Categories (in priority order for display):
      CEO_OPEN_MARKET_BUY, DIRECTOR_OPEN_MARKET_BUY, OFFICER_OPEN_MARKET_BUY,
      10B5_1_PLAN_SALE, OPTION_EXERCISE_SALE, LARGE_SHAREHOLDER_TRIM,
      INSIDER_BUY, INSIDER_SELL, OTHER
    """
    is_buy = txn.transaction_code == "P" and txn.acquired_disposed == "A"
    is_sale = txn.transaction_code == "S" and txn.acquired_disposed == "D"
    is_exercise = txn.transaction_code in ("M", "C")

    if is_buy:
        if _is_ceo(txn):
            return "CEO_OPEN_MARKET_BUY"
        if txn.is_director:
            return "DIRECTOR_OPEN_MARKET_BUY"
        if txn.is_officer:
            return "OFFICER_OPEN_MARKET_BUY"
        return "INSIDER_BUY"

    if is_sale:
        if txn.is_10b5_1:
            return "10B5_1_PLAN_SALE"
        if txn.is_ten_pct_owner:
            return "LARGE_SHAREHOLDER_TRIM"
        return "INSIDER_SELL"

    if is_exercise:
        return "OPTION_EXERCISE_SALE"

    return "OTHER"


# Category ranking (lower number = stronger bullish signal)
_CATEGORY_RANK = {
    "CEO_OPEN_MARKET_BUY": 1,
    "CLUSTER_BUY": 2,
    "DIRECTOR_OPEN_MARKET_BUY": 3,
    "OFFICER_OPEN_MARKET_BUY": 4,
    "INSIDER_BUY": 5,
    "OTHER": 50,
    "10B5_1_PLAN_SALE": 60,
    "OPTION_EXERCISE_SALE": 65,
    "INSIDER_SELL": 70,
    "LARGE_SHAREHOLDER_TRIM": 75,
}

# Human-readable labels
CATEGORY_LABELS = {
    "CEO_OPEN_MARKET_BUY": "🟢 CEO Open Market Buy",
    "CLUSTER_BUY": "🟢 Cluster Buy (3+ insiders)",
    "DIRECTOR_OPEN_MARKET_BUY": "🟢 Director Open Market Buy",
    "OFFICER_OPEN_MARKET_BUY": "🟢 Officer Open Market Buy",
    "INSIDER_BUY": "🟢 Insider Buy",
    "10B5_1_PLAN_SALE": "⚪ 10b5-1 Plan Sale",
    "OPTION_EXERCISE_SALE": "⚪ Option Exercise + Sale",
    "INSIDER_SELL": "🟡 Insider Sell",
    "LARGE_SHAREHOLDER_TRIM": "🟡 Large Shareholder Trim",
    "OTHER": "⚪ Other",
    "NEUTRAL": "—",
}


# ---------------------------------------------------------------------------
# Cluster buy detection
# ---------------------------------------------------------------------------

def _detect_cluster_buys(transactions: list[InsiderTransaction]) -> bool:
    """Return True if 3+ distinct insiders bought within a 7-day window."""
    buys = [
        t for t in transactions
        if t.transaction_code == "P"
        and t.acquired_disposed == "A"
        and t.transaction_date
    ]
    if len(buys) < CLUSTER_BUY_MIN_INSIDERS:
        return False

    # Parse dates and group by filer
    dated_buys: list[tuple[datetime.date, str]] = []
    for b in buys:
        try:
            d = datetime.date.fromisoformat(b.transaction_date)
            dated_buys.append((d, b.filer_name))
        except (ValueError, TypeError):
            continue

    if len(dated_buys) < CLUSTER_BUY_MIN_INSIDERS:
        return False

    # Sort by date, then sliding window
    dated_buys.sort(key=lambda x: x[0])
    for i in range(len(dated_buys)):
        window_start = dated_buys[i][0]
        window_end = window_start + datetime.timedelta(days=CLUSTER_BUY_WINDOW_DAYS)
        unique_filers = set()
        for d, filer in dated_buys[i:]:
            if d <= window_end:
                unique_filers.add(filer)
            else:
                break
        if len(unique_filers) >= CLUSTER_BUY_MIN_INSIDERS:
            return True

    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_insider_activity(symbol: str, lookback_days: int = INSIDER_LOOKBACK_DAYS,
                         shares_outstanding: Optional[float] = None) -> InsiderSummary:
    """Fetch and analyze recent insider transactions for a symbol.

    Parameters
    ----------
    symbol : str
        Ticker symbol (e.g. "AAPL").
    lookback_days : int
        How many days back to look for transactions.
    shares_outstanding : float, optional
        Total shares outstanding (for computing % of shares).
        If None, will attempt to fetch from EDGAR.

    Returns
    -------
    InsiderSummary
        Aggregated insider activity with category, score, and signal.
    """
    summary = InsiderSummary(symbol=symbol.upper(), lookback_days=lookback_days)

    cik = _edgar_module._get_cik(symbol)
    if cik is None:
        logger.warning("%s: CIK not found — cannot fetch insider data", symbol)
        return summary

    # ── Fetch Form 4 filings ──────────────────────────────────────────────────
    filings = _get_recent_form4_urls(cik, count=40)
    if not filings:
        logger.info("%s: no recent Form 4 filings found", symbol)
        return summary

    cutoff = datetime.date.today() - datetime.timedelta(days=lookback_days)

    all_transactions: list[InsiderTransaction] = []
    for f in filings:
        # Filter by filing date (quick pre-filter before parsing XML)
        try:
            fdate = datetime.date.fromisoformat(f["filingDate"])
            if fdate < cutoff:
                continue
        except (ValueError, TypeError):
            continue

        txns = _parse_form4_xml(
            cik, f["accessionNumber"], f["primaryDocument"],
            f["filingDate"], symbol.upper(),
        )

        # Post-filter by transaction date
        for t in txns:
            try:
                if t.transaction_date:
                    tdate = datetime.date.fromisoformat(t.transaction_date)
                    if tdate < cutoff:
                        continue
            except (ValueError, TypeError):
                pass
            all_transactions.append(t)

    if not all_transactions:
        return summary

    # ── Categorize each transaction ───────────────────────────────────────────
    for t in all_transactions:
        t.category = categorize(t)

        # Compute % of shares outstanding
        if shares_outstanding and shares_outstanding > 0:
            t.pct_of_outstanding = round(t.shares / shares_outstanding * 100, 4)

    # ── Detect 10b5-1 plans (heuristic — check filing text) ──────────────────
    # Only check a few sell filings to limit API calls
    sell_filings_checked = 0
    for f in filings:
        if sell_filings_checked >= 5:
            break
        try:
            fdate = datetime.date.fromisoformat(f["filingDate"])
            if fdate < cutoff:
                continue
        except (ValueError, TypeError):
            continue

        # Check if this accession has any sales
        has_sales = any(
            t.transaction_code == "S" and t.filing_date == f["filingDate"]
            for t in all_transactions
        )
        if has_sales:
            is_plan = _detect_10b5_1_from_filing(cik, f["accessionNumber"])
            if is_plan:
                for t in all_transactions:
                    if (t.filing_date == f["filingDate"]
                            and t.transaction_code == "S"):
                        t.is_10b5_1 = True
                        t.category = "10B5_1_PLAN_SALE"
            sell_filings_checked += 1

    # ── Re-categorize option exercises followed by immediate sales ────────────
    # Look for exercise (M) + sale (S) by same filer on same date
    exercises_by_filer_date: dict[tuple[str, str], bool] = {}
    for t in all_transactions:
        if t.transaction_code in ("M", "C"):
            key = (t.filer_name, t.transaction_date or "")
            exercises_by_filer_date[key] = True

    for t in all_transactions:
        if t.transaction_code == "S" and not t.is_10b5_1:
            key = (t.filer_name, t.transaction_date or "")
            if key in exercises_by_filer_date:
                t.category = "OPTION_EXERCISE_SALE"

    # ── Re-categorize large shareholder trims ─────────────────────────────────
    for t in all_transactions:
        if t.is_ten_pct_owner and t.transaction_code == "S" and t.category == "INSIDER_SELL":
            t.category = "LARGE_SHAREHOLDER_TRIM"

    # ── Detect cluster buys ───────────────────────────────────────────────────
    cluster = _detect_cluster_buys(all_transactions)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    summary.transactions = all_transactions
    summary.cluster_buy_detected = cluster

    for t in all_transactions:
        is_buy = t.transaction_code == "P" and t.acquired_disposed == "A"
        is_sell = t.acquired_disposed == "D"

        if is_buy:
            summary.total_insider_buys += 1
            summary.net_insider_shares += t.shares
            summary.net_insider_value += (t.total_value or 0)
        elif is_sell:
            summary.total_insider_sells += 1
            summary.net_insider_shares -= t.shares
            summary.net_insider_value -= (t.total_value or 0)

        if is_buy and _is_ceo(t):
            summary.ceo_bought = True
        if is_buy and t.is_director:
            summary.director_bought = True
        if is_buy and t.is_officer:
            summary.officer_bought = True
        if t.is_10b5_1:
            summary.has_10b5_1_sales = True
        if t.category == "OPTION_EXERCISE_SALE":
            summary.has_option_exercise_sales = True
        if t.category == "LARGE_SHAREHOLDER_TRIM":
            summary.large_shareholder_trimming = True

    # ── Score & signal ────────────────────────────────────────────────────────
    score = 50  # neutral baseline

    if summary.ceo_bought:
        score += 25
    if summary.cluster_buy_detected:
        score += 20
    if summary.director_bought:
        score += 10
    if summary.officer_bought:
        score += 5
    if summary.total_insider_buys > 0 and summary.total_insider_sells == 0:
        score += 10
    if summary.net_insider_value > 500_000:
        score += 5
    if summary.net_insider_value > 1_000_000:
        score += 5

    # Penalties for selling activity
    if summary.large_shareholder_trimming:
        score -= 10
    if summary.total_insider_sells > summary.total_insider_buys * 2:
        score -= 15
    if summary.has_option_exercise_sales and summary.total_insider_buys == 0:
        score -= 5

    # 10b5-1 plan sales are less concerning (pre-planned, not based on MNPI)
    if summary.has_10b5_1_sales and not summary.large_shareholder_trimming:
        score += 5  # slight positive: orderly, planned

    score = max(0, min(100, score))
    summary.insider_score = score

    if score >= 80:
        summary.insider_signal = "STRONG_BUY"
    elif score >= 65:
        summary.insider_signal = "BUY"
    elif score >= 40:
        summary.insider_signal = "NEUTRAL"
    elif score >= 25:
        summary.insider_signal = "CAUTION"
    else:
        summary.insider_signal = "SELL"

    # ── Top category ──────────────────────────────────────────────────────────
    if cluster:
        summary.top_category = "CLUSTER_BUY"
    else:
        categories_seen = [t.category for t in all_transactions if t.category != "OTHER"]
        if categories_seen:
            summary.top_category = min(categories_seen,
                                       key=lambda c: _CATEGORY_RANK.get(c, 99))
        else:
            summary.top_category = "NEUTRAL"

    # ── Notable transactions (for display) ────────────────────────────────────
    notable = []
    for t in sorted(all_transactions,
                    key=lambda x: _CATEGORY_RANK.get(x.category, 99)):
        if len(notable) >= 5:
            break
        action = "bought" if t.acquired_disposed == "A" else "sold"
        price_str = f"@ ${t.price_per_share:,.2f}" if t.price_per_share else ""
        val_str = f"(${t.total_value:,.0f})" if t.total_value else ""
        pct_str = f"[{t.pct_of_outstanding:.3f}%]" if t.pct_of_outstanding else ""
        notable.append(
            f"{t.filer_name} ({t.filer_title or t.relationship}) "
            f"{action} {t.shares:,.0f} shares {price_str} {val_str} {pct_str} "
            f"on {t.transaction_date or t.filing_date}"
        )
    summary.notable_transactions = notable

    logger.info(
        "%s: insider activity — %d buys, %d sells, net $%+,.0f, "
        "score=%d, signal=%s, top=%s",
        symbol, summary.total_insider_buys, summary.total_insider_sells,
        summary.net_insider_value, summary.insider_score,
        summary.insider_signal, summary.top_category,
    )

    return summary
