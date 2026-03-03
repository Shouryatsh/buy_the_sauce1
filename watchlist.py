"""
watchlist.py — ~50 fundamentally strong US stocks across key sectors.

Edit this list to add or remove tickers.  The screener will further
filter based on live fundamental data, so broad coverage is fine.
"""

WATCHLIST: list[str] = [
    # Technology
    "AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD", "CRM", "ADBE", "ORCL", "INTC",
    # Consumer Discretionary / Staples
    "AMZN", "COST", "WMT", "HD", "NKE", "SBUX", "MCD", "TGT",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "TMO", "ABT", "HIMS",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "AXP",
    # Energy
    "XOM", "CVX", "COP", "SLB",
    # Industrials
    "CAT", "DE", "HON", "GE", "UPS", "FDX",
    # Communication Services
    "DIS", "NFLX", "CMCSA",
    # Materials & Utilities
    "LIN", "NEE",
]
