"""
Smile shapes and the per-underlying cache (panel + walk-forward forecasts + yearly GJR fits) used by phases 3-6.

    python -m premia.research.build            # all symbols with >= 500 days of IV; 8 processes
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .. import market, store
from ..config import RESULTS_DIR, load_exclusions
from ..vol import SmileShape

SHAPES_FILE = RESULTS_DIR / "smile_shapes.json"


def index_shape_from_snapshot(snapshot_id: int = 5) -> SmileShape:
    """SPX smile (30-day slice) from a saved real IBKR snapshot: the shape used for SPY and other index ETFs."""
    from volsurface import model, store as vstore, surface
    qs = vstore.load_snapshot(snapshot_id)
    surf = surface.build_surface(qs.quotes, qs.spot, qs.asof, qs.symbol, qs.source)
    fs = model.fit_surface(surf, model.FitSettings(butterfly_strength=1.0, calendar_strength=1.0, nstarts=10))
    s30 = min(fs.fitted, key=lambda s: abs(s.T - 30 / 365))
    return SmileShape.from_svi(s30.params, s30.T, f"SPX {s30.expiry} slice, IBKR snapshot {qs.asof:%Y-%m-%d}")


def load_shapes() -> dict[str, SmileShape]:
    d = json.loads(SHAPES_FILE.read_text())
    shapes = d.setdefault("shapes", {})
    if "index" not in shapes:
        shapes["index"] = index_shape_from_snapshot().to_dict()
        SHAPES_FILE.write_text(json.dumps(d, indent=1, default=str))
    return {k: SmileShape.from_dict(v) for k, v in shapes.items()}


def shape_for(symbol: str, shapes: dict[str, SmileShape]) -> SmileShape:
    return shapes["index"] if market.is_etf(symbol) else shapes["stock"]


def _build(sym: str) -> tuple[str, str, float]:
    t0 = time.time()
    shapes = load_shapes()
    try:
        u = market.build_underlying(sym, shape_for(sym, shapes))
        return sym, "ok" if u is not None else "skipped (too little data)", time.time() - t0
    except Exception as exc:
        return sym, f"error: {type(exc).__name__}: {exc}", time.time() - t0


def symbols_with_iv(min_days: int = 500) -> list[str]:
    with store.connect() as c:
        rows = c.execute("SELECT symbol, COUNT(*) FROM iv30 GROUP BY symbol").fetchall()
    excl = load_exclusions()
    return sorted(s for s, n in rows if n >= min_days and s not in excl)


def main(workers: int = 8) -> dict:
    load_shapes()
    syms = symbols_with_iv()
    todo = [s for s in syms if not (market.CACHE_DIR / f"u_{s}.pkl").exists()]
    print(f"{len(syms)} symbols with IV, {len(todo)} to build")
    out = {}
    with ProcessPoolExecutor(workers) as ex:
        for sym, status, secs in ex.map(_build, todo):
            out[sym] = status
            if status != "ok":
                print(sym, status)
    print({k: sum(1 for v in out.values() if v.startswith(k)) for k in ("ok", "skipped", "error")})
    return out


if __name__ == "__main__":
    main()
