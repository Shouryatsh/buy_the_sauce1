"""
ivscreen — IV Percentile / Realized-Volatility Percentile screener.

Follows McMillan ("Profits with Options"): implied volatility is ranked only against its OWN
history, realized volatility only against ITS OWN history; the two percentiles are then set
side by side. Raw IV and RV levels are never compared with each other.

    config.py    thresholds (all configurable) and universe categories
    metrics.py   percentile maths, classification bands, trends, IV-vs-RV relationship
    store.py     SQLite tables (in wheel.db): symbols, IV history, scan runs
    evaluate.py  stored data + config -> one classified row per symbol (data-availability aware)
    scan.py      background IBKR scan (read-only historical-data requests) + CLI
    ui.py        Streamlit rendering for app_pages/iv_screener.py
"""
