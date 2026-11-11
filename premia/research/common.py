"""Shared helpers for the phase runners: result files and verdict records."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import RESULTS_DIR


def out_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if isinstance(o, pd.DataFrame):
        return o.to_dict(orient="records")
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save_json(name: str, obj) -> Path:
    p = out_dir() / f"{name}.json"
    p.write_text(json.dumps(obj, indent=1, default=_default))
    return p


def save_csv(name: str, df: pd.DataFrame) -> Path:
    p = out_dir() / f"{name}.csv"
    df.to_csv(p, index=False)
    return p


def verdict(phase: str, checks: list[tuple[str, bool, str]]) -> dict:
    """checks: (criterion, passed, evidence). The phase passes only if every required criterion passes."""
    ok = all(c[1] for c in checks)
    v = {"phase": phase, "verdict": "PASS" if ok else "FAIL", "checks": [{"criterion": c, "pass": bool(p), "evidence": e}
                                                                         for c, p, e in checks],
         "at": datetime.now().isoformat(timespec="seconds")}
    save_json(f"verdict_{phase}", v)
    return v
