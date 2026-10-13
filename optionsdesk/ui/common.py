"""Shared Streamlit helpers."""

from __future__ import annotations

import html
from typing import Any, Callable

import streamlit as st

from . import client, theme

DARK_CSS = """
<style>
.block-container {padding-top: 1.2rem;}
.od-badge {display:inline-block; padding:2px 10px; border-radius:12px; font-size:0.8rem; font-weight:600; color:#0e1117;}
@media (max-width: 640px) {
  .block-container {padding: 0.75rem 0.5rem;}
  [data-testid="stMetricValue"] {font-size: 1.15rem;}
}
</style>
"""

DOT = {"running": "🟢", "starting": "🟡", "degraded": "🟡", "error": "🔴", "stopped": "⚪"}


def fetch(path: str, **params: Any) -> Any:
    """GET from the API; show an error and return None if it fails."""
    try:
        return client.get(path, **params)
    except client.ApiError as exc:
        st.error(str(exc))
        return None


def post(path: str, json: dict | None = None, success: str | None = None) -> Any:
    try:
        out = client.post(path, json)
    except client.ApiError as exc:
        st.error(str(exc))
        return None
    if success:
        st.success(success)
    return out


def badge(text: str, color: str) -> str:
    return f'<span class="od-badge" style="background:{color}">{html.escape(str(text))}</span>'


def live(render: Callable[[], None]) -> None:
    """Run `render` as a fragment that re-runs on the sidebar refresh interval without a full page rerun."""
    secs = st.session_state.get("refresh_s", 5)
    st.fragment(run_every=secs if secs else None)(render)()


def quality_badges(qualities: list[str]) -> str:
    if not qualities:
        return badge("no data yet", theme.MUTED)
    color = {"live": theme.GREEN, "delayed": theme.AMBER, "estimated": theme.AMBER}
    return " ".join(badge(q, color.get(q, theme.MUTED)) for q in qualities)
