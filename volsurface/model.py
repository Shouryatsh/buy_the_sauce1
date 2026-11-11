"""
volsurface/model.py — a surface made of one fitted SVI slice per expiry.

Between expiries the total variance is interpolated LINEARLY IN T at fixed log-moneyness k (so if every
slice is calendar-consistent the interpolated surface is too). Before the first expiry w is scaled by T
(constant implied vol); after the last, implied vol is held flat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from . import arbitrage as ar
from .surface import Slice, Surface
from .svi import SVI, FitResult, fit_slice


@dataclass
class FitSettings:
    weights: str = "spread"            # "spread" | "vega" | "equal"
    lee: bool = True
    butterfly_strength: float = 0.0    # 0 = plain least squares; >0 penalises negative density
    calendar_strength: float = 0.0     # >0 penalises w falling below the previous expiry's slice
    nstarts: int = 18


@dataclass
class FittedSlice:
    slice: Slice
    fit: Optional[FitResult]
    error: str = ""

    @property
    def params(self) -> Optional[SVI]:
        return self.fit.params if self.fit else None

    @property
    def expiry(self) -> str:
        return self.slice.expiry

    @property
    def T(self) -> float:
        return self.slice.T


@dataclass
class FittedSurface:
    surface: Surface
    slices: list = field(default_factory=list)
    settings: FitSettings = field(default_factory=FitSettings)

    @property
    def fitted(self) -> list[FittedSlice]:
        return [s for s in self.slices if s.fit is not None]

    def labelled(self) -> list[tuple[str, float, SVI]]:
        return [(s.expiry, s.T, s.params) for s in self.fitted]

    def by_expiry(self, expiry: str) -> FittedSlice:
        return next(s for s in self.slices if s.expiry == expiry)

    # -- the interpolated surface -------------------------------------------------------------
    def w(self, k, T: float) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        fs = self.fitted
        if not fs:
            raise ValueError("no fitted slices")
        ts = np.array([s.T for s in fs])
        if T <= ts[0]:
            return fs[0].params.w(k) * (T / ts[0])
        if T >= ts[-1]:
            return fs[-1].params.w(k) * (T / ts[-1])
        i = int(np.searchsorted(ts, T) - 1)
        w0, w1 = fs[i].params.w(k), fs[i + 1].params.w(k)
        lam = (T - ts[i]) / (ts[i + 1] - ts[i])
        return (1 - lam) * w0 + lam * w1

    def iv(self, k, T: float) -> np.ndarray:
        return np.sqrt(np.maximum(self.w(k, T), 0.0) / T)

    def iv_grid(self, k_grid, T_grid) -> np.ndarray:
        """rows = T, columns = k."""
        return np.vstack([self.iv(k_grid, T) for T in T_grid])


def fit_surface(surface: Surface, settings: Optional[FitSettings] = None,
                progress: Optional[Callable[[int, int, str], None]] = None) -> FittedSurface:
    """Fit every slice, shortest maturity first; each fit warm-starts from the previous slice and (if asked)
    is penalised for crossing it."""
    settings = settings or FitSettings()
    out = FittedSurface(surface=surface, settings=settings)
    prev: Optional[SVI] = None
    for i, sl in enumerate(surface.slices):
        if progress:
            progress(i, len(surface.slices), sl.expiry)
        u = sl.usable
        try:
            fit = fit_slice(u["k"].to_numpy(), u["iv"].to_numpy(), sl.T, u["iv_bid"].to_numpy(), u["iv_ask"].to_numpy(),
                            u["vega"].to_numpy(), weights=settings.weights, lee=settings.lee,
                            butterfly_strength=settings.butterfly_strength, calendar_prev=prev,
                            calendar_strength=settings.calendar_strength, warm=prev, nstarts=settings.nstarts)
            out.slices.append(FittedSlice(sl, fit))
            prev = fit.params
        except Exception as exc:
            out.slices.append(FittedSlice(sl, None, f"{type(exc).__name__}: {exc}"))
    return out


def data_range(sl: Slice) -> tuple[float, float]:
    """(k_min, k_max) of the quotes a slice was fitted to."""
    u = sl.usable
    return float(u["k"].min()), float(u["k"].max())


def arbitrage_summary(fs: FittedSurface) -> dict:
    """Butterfly per slice and calendar per consecutive pair, from the fitted slices."""
    ranges = {s.expiry: data_range(s.slice) for s in fs.fitted}
    bf = {s.expiry: ar.butterfly(s.params, ranges[s.expiry]) for s in fs.fitted}
    cal = ar.calendar(fs.labelled(), ranges)
    atm_ok, atm_bad = ar.atm_variance_monotone(fs.labelled())
    return {"butterfly": bf, "calendar": cal, "atm_monotone": atm_ok, "atm_bad": atm_bad}
