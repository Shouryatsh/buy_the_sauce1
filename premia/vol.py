"""
premia/vol.py — phase 3 surface models on top of `volsurface` (SVI per expiry, static-arbitrage checks).

1. Local volatility from the implied total-variance surface w(k, T), k = ln(K/F_T)   (Gatheral ch. 1, eq. 1.10; Dupire 1994)

       sigma_loc^2(k, T) = (dw/dT) / [ 1 - (k/w) w' + 1/4 (-1/4 - 1/w + k^2/w^2) w'^2 + 1/2 w'' ]

   with w', w'' the k-derivatives of each SVI slice (analytic) and dw/dT from the surface's linear-in-T interpolation.
   Validation: the forward Dupire PDE in log-moneyness (Gatheral ch. 1),
       dC/dT = 1/2 sigma_loc^2(k, T) (d2C/dk2 - dC/dk),   C(k, 0) = (1 - e^k)^+,
   for the forward-normalised call C = E[(F_T/F - K/F)^+], solved by Crank-Nicolson (Rannacher start), must
   reproduce the implied vols of the surface it came from.

2. Smile shape for history. Historical option chains are not available, so the smile away from the money is modelled
   as ATM vol x a normalised shape: sigma(k) = sigma_atm * f(z), z = k / (sigma_atm sqrt T), f taken from a real
   fitted SVI slice and optionally scaled (skew multiplier m: f -> 1 + m (f - 1)) for sensitivity runs.
   This keeps skew slope ~ 1/sqrt(T) (the empirical term-structure of skew, Gatheral ch. 3 and 8) and makes OTM vols
   move with ATM vol. It is an assumption, stated in SPEC.md section 3, not an estimate of the historical skew.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded
from scipy.special import ndtr

from volsurface.svi import SVI


# ---------------------------------------------------------------------------
# local volatility
# ---------------------------------------------------------------------------

def local_var_formula(k, w, dw, d2w, dwdT):
    k, w, dw, d2w, dwdT = (np.asarray(x, float) for x in (k, w, dw, d2w, dwdT))
    w = np.maximum(w, 1e-12)
    den = 1 - k / w * dw + 0.25 * (-0.25 - 1 / w + k * k / (w * w)) * dw * dw + 0.5 * d2w
    return dwdT / den, den


@dataclass
class Slices:
    """A calendar of SVI slices (T ascending) with the same interpolation as volsurface.model.FittedSurface."""
    T: np.ndarray
    svi: list

    @classmethod
    def from_fitted(cls, fs) -> "Slices":
        lab = fs.labelled()
        return cls(np.array([t for _e, t, _p in lab]), [p for _e, _t, p in lab])

    def _bracket(self, T: float):
        ts = self.T
        if T <= ts[0]:
            return 0, 0, T / ts[0]
        if T >= ts[-1]:
            return len(ts) - 1, len(ts) - 1, T / ts[-1]
        i = int(np.searchsorted(ts, T) - 1)
        return i, i + 1, (T - ts[i]) / (ts[i + 1] - ts[i])

    def w(self, k, T: float):
        i, j, lam = self._bracket(T)
        if i == j:
            return self.svi[i].w(k) * lam
        return (1 - lam) * self.svi[i].w(k) + lam * self.svi[j].w(k)

    def derivs(self, k, T: float):
        """(w, w', w'', dw/dT) at fixed k."""
        k = np.asarray(k, float)
        i, j, lam = self._bracket(T)
        if i == j:
            s, scale = self.svi[i], lam
            return s.w(k) * scale, s.dw(k) * scale, s.d2w(k) * scale, s.w(k) / self.T[i]
        a, b = self.svi[i], self.svi[j]
        w = (1 - lam) * a.w(k) + lam * b.w(k)
        dw = (1 - lam) * a.dw(k) + lam * b.dw(k)
        d2w = (1 - lam) * a.d2w(k) + lam * b.d2w(k)
        dT = (b.w(k) - a.w(k)) / (self.T[j] - self.T[i])
        return w, dw, d2w, dT

    def iv(self, k, T: float):
        return np.sqrt(np.maximum(self.w(k, T), 0.0) / T)

    def local_vol(self, k, T: float, floor: float = 1e-4, cap: float = 4.0):
        """sigma_loc(k, T); where the formula is undefined (arbitrage: den <= 0 or dw/dT < 0) the implied vol is used
        instead and the point is flagged."""
        w, dw, d2w, dT = self.derivs(k, T)
        lv2, den = local_var_formula(k, w, dw, d2w, dT)
        bad = (den <= 0) | (dT < 0) | ~np.isfinite(lv2)
        lv2 = np.where(bad, w / max(T, 1e-8), lv2)
        return np.sqrt(np.clip(lv2, floor, cap * cap)), bad


def black_call_norm(k, w):
    """Forward-normalised undiscounted Black call: N(d1) - e^k N(d2)."""
    k, w = np.asarray(k, float), np.maximum(np.asarray(w, float), 1e-14)
    sw = np.sqrt(w)
    d1 = -k / sw + 0.5 * sw
    return ndtr(d1) - np.exp(k) * ndtr(d1 - sw)


def implied_w_from_norm_call(k, c, lo=1e-8, hi=9.0, iters=80):
    k, c = np.asarray(k, float), np.asarray(c, float)
    a, b = np.full_like(k, lo), np.full_like(k, hi)
    for _ in range(iters):
        m = 0.5 * (a + b)
        up = black_call_norm(k, m) > c
        b, a = np.where(up, m, b), np.where(up, a, m)
    intrinsic = np.maximum(1 - np.exp(k), 0)
    return np.where((c > intrinsic + 1e-12) & (c < 1), 0.5 * (a + b), np.nan)


def dupire_pde(lv_fn, T_out: list[float], k_max: float = 2.5, nk: int = 501, steps_per_year: int = 1500,
               rannacher: int = 4) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Solve dC/dT = 1/2 v(k,T) (C_kk - C_k) forward in T. Returns {T: (k grid, C)} for each requested T."""
    k = np.linspace(-k_max, k_max, nk)
    dk = k[1] - k[0]
    C = np.maximum(1 - np.exp(k), 0.0)
    out = {}
    t = 0.0
    targets = sorted(T_out)
    step = 0
    for T_target in targets:
        n = max(int(np.ceil((T_target - t) * steps_per_year)), 4)
        dt = (T_target - t) / n
        for _ in range(n):
            tm = t + 0.5 * dt
            v = lv_fn(k[1:-1], tm) ** 2
            a = 0.5 * v * (1 / dk ** 2 + 0.5 / dk)          # coefficient on C[i-1]
            c = 0.5 * v * (1 / dk ** 2 - 0.5 / dk)          # coefficient on C[i+1]
            b = -v / dk ** 2                                  # on C[i]
            theta = 1.0 if step < rannacher else 0.5         # implicit Euler for the first steps (kills kink oscillation)
            rhs = C[1:-1] + (1 - theta) * dt * (a * C[:-2] + b * C[1:-1] + c * C[2:])
            lo_b, hi_b = 1 - np.exp(k[0]), 0.0                # boundaries: deep ITM ~ intrinsic, deep OTM ~ 0
            rhs[0] += theta * dt * a[0] * lo_b
            ab = np.zeros((3, nk - 2))
            ab[0, 1:] = -theta * dt * c[:-1]
            ab[1, :] = 1 - theta * dt * b
            ab[2, :-1] = -theta * dt * a[1:]
            C[1:-1] = solve_banded((1, 1), ab, rhs)
            C[0], C[-1] = lo_b, hi_b
            t += dt
            step += 1
        out[T_target] = (k.copy(), C.copy())
    return out


def validate_local_vol(sl: Slices, points: dict | None = None, moneyness=(0.8, 1.2), n_strikes: int = 9,
                       min_price: float = 1e-5, **pde_kw) -> list[dict]:
    """Reprice vanillas with the local-vol PDE and compare implied vols with the surface it came from.

    points: {T: array of k} to test (e.g. the quoted strikes of each slice); default a K/F grid per slice.
    Options whose forward-normalised OTM price is below *min_price* (a fraction of a tick) are skipped: their implied
    vol is not defined by anything tradable."""
    lv_fn = lambda kk, tt: sl.local_vol(kk, tt)[0]
    T_max = float(max(sl.T))
    atm_sd = float(np.sqrt(max(sl.w(0.0, T_max), 1e-6)))
    pde_kw.setdefault("k_max", float(np.clip(7 * atm_sd, 1.0, 3.0)))
    pde_kw.setdefault("nk", 1201)
    pde_kw.setdefault("steps_per_year", 4000)
    res = dupire_pde(lv_fn, list(sl.T), **pde_kw)
    rows = []
    for T, (kg, C) in res.items():
        ks = np.asarray(points[T] if points and T in points else np.log(np.linspace(*moneyness, n_strikes)), float)
        c = np.interp(ks, kg, C)
        c_surf = black_call_norm(ks, sl.w(ks, T))
        otm = np.where(ks < 0, c_surf - (1 - np.exp(ks)), c_surf)
        w_pde = implied_w_from_norm_call(ks, c)
        iv_pde = np.sqrt(w_pde / T)
        iv_surf = sl.iv(ks, T)
        for kk, a, b, o in zip(ks, iv_pde, iv_surf, otm):
            if o < min_price:
                continue
            rows.append({"T": float(T), "K/F": float(np.exp(kk)), "iv_surface": float(b), "iv_localvol_pde": float(a),
                         "err_volpts": float((a - b) * 100)})
    return rows


# ---------------------------------------------------------------------------
# normalised smile shape (for modelled historical chains)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SmileShape:
    z: tuple                 # grid of z = k / (sigma_atm sqrt T)
    ratio: tuple             # sigma(k) / sigma_atm on that grid
    source: str = ""

    @classmethod
    def from_svi(cls, p: SVI, T: float, source: str = "", z_range=(-5.0, 4.0), n: int = 181) -> "SmileShape":
        atm = float(np.sqrt(p.w(0.0) / T))
        z = np.linspace(*z_range, n)
        k = z * atm * np.sqrt(T)
        ratio = np.sqrt(np.maximum(p.w(k), 1e-12) / T) / atm
        return cls(tuple(z.round(6)), tuple(ratio.round(6)), source)

    @classmethod
    def flat(cls) -> "SmileShape":
        return cls((-5.0, 4.0), (1.0, 1.0), "flat")

    def iv(self, k, atm, T, skew_mult: float = 1.0):
        k, atm, T = (np.asarray(x, float) for x in (k, atm, T))
        z = k / (atm * np.sqrt(np.maximum(T, 1e-6)))
        r = np.interp(z, self.z, self.ratio)                # flat beyond the grid (bounded wings)
        return atm * (1 + skew_mult * (r - 1))

    def to_dict(self) -> dict:
        return {"z": list(self.z), "ratio": list(self.ratio), "source": self.source}

    @classmethod
    def from_dict(cls, d: dict) -> "SmileShape":
        return cls(tuple(d["z"]), tuple(d["ratio"]), d.get("source", ""))


def average_shape(shapes: list[SmileShape], source: str = "") -> SmileShape:
    z = np.array(shapes[0].z)
    r = np.median(np.vstack([np.interp(z, s.z, s.ratio) for s in shapes]), axis=0)
    return SmileShape(tuple(z), tuple(r.round(6)), source)


def butterfly_ok_shape(shape: SmileShape, atm: float, T: float, skew_mult: float = 1.0, k_range=(-0.6, 0.4)) -> bool:
    """Durrleman check on a modelled slice: the density implied by the shaped smile is non-negative."""
    k = np.linspace(*k_range, 801)
    w = shape.iv(k, atm, T, skew_mult) ** 2 * T
    c = black_call_norm(k, w)
    # density in strike space is d2C/dK2 >= 0  <=>  convexity of C in K = F e^k
    K = np.exp(k)
    d1 = np.gradient(c, K)
    d2 = np.gradient(d1, K)
    return bool(np.all(d2[2:-2] > -1e-6))
