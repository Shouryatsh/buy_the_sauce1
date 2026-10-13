"""optionsdesk dashboard (Streamlit). Reads the FastAPI service only; never talks to IBKR.

Run:  python -m optionsdesk dashboard      (API must be running: python -m optionsdesk serve)
"""

from __future__ import annotations

import streamlit as st

from optionsdesk.ui import client
from optionsdesk.ui.common import DARK_CSS
from optionsdesk.ui.live import EventFeed
from optionsdesk.ui.screens import backtest, controls, journal, overview, risk, scanner, straddle, wheel

st.set_page_config(page_title="optionsdesk", page_icon="📈", layout="wide")
st.markdown(DARK_CSS, unsafe_allow_html=True)


@st.cache_resource
def _feed() -> EventFeed:
    f = EventFeed()
    f.start()
    return f


st.session_state["_feed"] = _feed()

REFRESH = {"Off": 0, "2 s": 2, "5 s": 5, "10 s": 10, "30 s": 30, "60 s": 60}
with st.sidebar:
    st.title("optionsdesk")
    choice = st.selectbox("Auto-refresh", list(REFRESH), index=2, key="refresh_choice")
    st.session_state["refresh_s"] = REFRESH[choice]
    try:
        h = client.get("/health")
        st.success(f"API ok ({h['mode']}, {h['data_source']})")
    except client.ApiError as exc:
        st.error(str(exc))
    st.caption("Live event stream: " + ("connected" if st.session_state["_feed"].connected else "connecting..."))
    st.divider()
    st.caption("Legacy wheel dashboard (unchanged): http://localhost:8501")

pages = [
    st.Page(overview.render, title="Overview", icon="🏠", url_path="overview", default=True),
    st.Page(wheel.render, title="Wheel", icon="🎡", url_path="wheel"),
    st.Page(straddle.render, title="Straddle", icon="↕️", url_path="straddle"),
    st.Page(scanner.render, title="Scanner", icon="🔎", url_path="scanner"),
    st.Page(risk.render, title="Risk", icon="🛡️", url_path="risk"),
    st.Page(backtest.render, title="Backtest lab", icon="🧪", url_path="backtest"),
    st.Page(journal.render, title="Journal", icon="📒", url_path="journal"),
    st.Page(controls.render, title="Orders & controls", icon="🕹️", url_path="controls"),
]
st.navigation(pages).run()
