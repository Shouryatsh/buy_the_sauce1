"""
volsurface — a working bench for Jim Gatheral's volatility-surface ideas (*The Volatility Surface*, plus
Gatheral & Jacquier, "Arbitrage-free SVI volatility surfaces", 2014), on real option quotes.

    bs.py          Black-76 price / vega / delta and a robust vectorised implied-vol inversion
    svi.py         raw SVI slice  w(k) = a + b (rho (k-m) + sqrt((k-m)^2 + sigma^2)), its derivatives,
                   Durrleman's butterfly condition g(k), and a constrained least-squares fit
    arbitrage.py   butterfly (g(k) >= 0, Lee wing bounds), calendar (w non-decreasing in T) and model-free
                   quote-level checks
    surface.py     raw quotes -> forwards from put-call parity -> OTM implied vols -> total variance w = sigma^2 T
    model.py       a surface of fitted SVI slices, interpolated in total variance across maturities
    data.py        quote sources: Yahoo (free, delayed) and IBKR (live, read-only)
    store.py       saved snapshots (SQLite) so surface dynamics can be studied over time
    analysis.py    constant-maturity metrics, risk-neutral probabilities from the smile, strike scan,
                   position check
    ui.py          Streamlit rendering for app_pages/vol_surface.py

Notation (Gatheral): k = ln(K/F) log-forward-moneyness, T years to expiry, sigma(k,T) Black-Scholes implied
volatility, w(k,T) = sigma(k,T)^2 T total implied variance.
"""
