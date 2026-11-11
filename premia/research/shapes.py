"""
Real option chains -> SVI fits -> normalised smile shapes (the modelled-history assumption, SPEC.md section 3),
plus the phase-3 SVI fit-error evidence on single names.

    python -m premia.research.shapes            # IBKR (TWS 7496, read-only market data; frozen quotes when closed)

Each chain: one or two monthly expiries 25-75 days out, OTM quotes, fitted with volsurface (arbitrage-aware).
Shapes: the median single-stock shape and the SPY shape are saved to results/premia/smile_shapes.json.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pandas as pd

from volsurface import arbitrage as ar
from volsurface import data as vdata
from volsurface import model as vmodel
from volsurface import surface as vsurface

from ..vol import SmileShape, average_shape
from .common import save_json

SAMPLE = ["MSFT", "JPM", "JNJ", "PG", "V", "HD", "UNH", "XOM", "NVDA", "CSCO", "CAT", "AMZN", "WMT", "PEP", "LIN",
          "ADBE", "HON", "IBM", "GS", "MRK", "ABBV", "LOW", "TMO", "QCOM", "COST", "TJX"]
ETF_SAMPLE = ["SPY", "QQQ", "IWM"]


class _Client:
    """Minimal volsurface-compatible client around a read-only ib_insync connection."""

    def __init__(self, ib):
        self.ib = ib

    def run(self, factory, timeout: float = 30.0):
        return self.ib.run(asyncio.wait_for(factory(), timeout))


def pick_expiries(asof: pd.Timestamp, listed: list[str], lo: int = 25, hi: int = 75) -> list[str]:
    out = []
    for e in listed:
        d = (pd.Timestamp(e) - asof.tz_localize(None).normalize()).days
        if lo <= d <= hi and pd.Timestamp(e).weekday() == 4 and 15 <= pd.Timestamp(e).day <= 21:
            out.append(e)
    return out[:2]


def fit_symbol(client, sym: str) -> dict:
    asof = vdata.now_et()
    from ..markets_calendar import monthly_iso
    exps = monthly_iso(asof, 25, 75)
    qs = vdata.fetch_ibkr(client, sym, exps, max_contracts=300)
    surf = vsurface.build_surface(qs.quotes, qs.spot, qs.asof, sym, qs.source)
    fs = vmodel.fit_surface(surf, vmodel.FitSettings(butterfly_strength=1.0, calendar_strength=1.0, nstarts=10))
    rows = []
    for s in fs.fitted:
        u = s.slice.usable
        rng = vmodel.data_range(s.slice)
        bf = ar.butterfly(s.params, rng)
        rows.append({"symbol": sym, "expiry": s.expiry, "T": s.T, "n_quotes": int(len(u)), "rmse_volpts": s.fit.rmse_iv * 100,
                     "inside_band": s.fit.within_band, "butterfly_ok_in_data": bool(bf.ok_in_data),
                     "atm_iv": float(np.sqrt(s.params.w(0.0) / s.T)), "params": s.params.as_tuple(),
                     "k_range": rng, "source": qs.source})
    cal = ar.calendar(fs.labelled(), {s.expiry: vmodel.data_range(s.slice) for s in fs.fitted}) if len(fs.fitted) > 1 else []
    return {"rows": rows, "calendar_ok_in_data": None if not cal else all(c.ok_in_data for c in cal), "notes": qs.notes}


def main(port: int = 7496, client_id: int = 85) -> dict:
    from ib_insync import IB
    from volsurface.svi import SVI
    ib = IB()
    ib.connect("127.0.0.1", port, clientId=client_id, timeout=15, readonly=True)
    client = _Client(ib)
    results, errors = [], {}
    t0 = time.time()
    try:
        for sym in ETF_SAMPLE + SAMPLE:
            try:
                r = fit_symbol(client, sym)
                for row in r["rows"]:
                    row["calendar_ok_in_data"] = r["calendar_ok_in_data"]
                results += r["rows"]
                print(f"{sym}: {[(x['expiry'], round(x['rmse_volpts'], 2)) for x in r['rows']]}  {time.time() - t0:.0f}s")
            except Exception as exc:
                errors[sym] = f"{type(exc).__name__}: {exc}"
                print(sym, "ERROR", errors[sym])
    finally:
        ib.disconnect()
    df = pd.DataFrame(results)
    shapes = {}
    for group, names in (("stock", SAMPLE), ("spy", ["SPY"]), ("etf", ETF_SAMPLE)):
        sub = df[df["symbol"].isin(names)]
        sh = [SmileShape.from_svi(SVI.from_tuple(p), T, f"{s} {e}") for s, e, T, p in
              zip(sub["symbol"], sub["expiry"], sub["T"], sub["params"])]
        if sh:
            shapes[group] = average_shape(sh, f"median of {len(sh)} real IBKR slices ({', '.join(sorted(set(sub['symbol'])))})").to_dict()
    out = {"asof": str(vdata.now_et()), "fits": df.to_dict(orient="records"), "errors": errors, "shapes": shapes}
    save_json("smile_shapes", out)
    return out


if __name__ == "__main__":
    main()
