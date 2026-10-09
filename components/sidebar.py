from __future__ import annotations

import streamlit as st

NAV_ITEMS = [
    "Dashboard",
    "Upload Documents",
    "Chat with Documents",
    "Table Explorer",
    "Image Explorer",
    "Search History",
    "Settings",
]


def render_sidebar() -> str:
    with st.sidebar:
        target = st.session_state.pop("navigation_target", None)
        initial_selection = st.session_state.get("nav_selection", "Dashboard")
        if target in NAV_ITEMS:
            initial_selection = target
        st.session_state["nav_selection"] = initial_selection
        st.markdown(
            """
            <div class="brand-lockup">
              <div class="brand-mark">V</div>
              <div><strong>VisionRAG</strong><small>DOCUMENT INTELLIGENCE</small></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        selection = st.radio(
            "Workspace navigation",
            NAV_ITEMS,
            index=NAV_ITEMS.index(initial_selection),
            key="nav_selection",
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.caption("LOCAL WORKSPACE")
        st.markdown(
            '<div class="sidebar-status"><span class="status-dot"></span> Private by default</div>',
            unsafe_allow_html=True,
        )
        return selection
