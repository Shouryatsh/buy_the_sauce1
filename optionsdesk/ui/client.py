"""Thin HTTP client for the optionsdesk API. The dashboard never talks to IBKR directly."""

from __future__ import annotations

import os
from typing import Any, Optional

import httpx

API_URL = os.environ.get("OPTIONSDESK_API_URL", "http://127.0.0.1:8000").rstrip("/")


class ApiError(Exception):
    pass


def _headers() -> dict[str, str]:
    token = os.environ.get("OPTIONSDESK_API_TOKEN")
    return {"X-API-Token": token} if token else {}


def request(method: str, path: str, **kw: Any) -> Any:
    try:
        r = httpx.request(method, f"{API_URL}{path}", headers=_headers(), timeout=kw.pop("timeout", 10.0), **kw)
    except httpx.HTTPError as exc:
        raise ApiError(f"API unreachable at {API_URL}: {exc}") from exc
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise ApiError(f"{r.status_code}: {detail}")
    return r.json()


def get(path: str, **params: Any) -> Any:
    clean = {k: v for k, v in params.items() if v is not None}
    return request("GET", path, params=clean)


def post(path: str, json: Optional[dict] = None) -> Any:
    return request("POST", path, json=json or {})
