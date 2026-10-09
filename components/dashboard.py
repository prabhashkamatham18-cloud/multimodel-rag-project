from __future__ import annotations

import streamlit as st

from src.database import list_documents


def render_dashboard() -> None:
    documents = list_documents()
    indexed = [document for document in documents if document["status"] == "Indexed"]
    pages = sum(document["page_count"] for document in indexed)
    tables = sum(document["table_count"] for document in indexed)
    images = sum(document["image_count"] for document in indexed)

    st.markdown(
        """
        <section class="hero-panel">
          <div class="hero-copy">
            <span class="eyebrow">YOUR PRIVATE DOCUMENT WORKSPACE</span>
            <h1>Your Documents.<br><span>Your Intelligence.</span></h1>
            <p>Search across pages, tables, and visual evidence — with answers grounded in your files.</p>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns([1, 1], gap="large")
    with left:
        if st.button("＋  Upload a PDF", type="primary", use_container_width=True):
            st.session_state["navigation_target"] = "Upload Documents"
            st.rerun()
    with right:
        if st.button("Ask your documents  →", use_container_width=True):
            st.session_state["navigation_target"] = "Chat with Documents"
            st.rerun()

    st.markdown('<div class="section-label">WORKSPACE OVERVIEW</div>', unsafe_allow_html=True)
    metric_columns = st.columns(4)
    metrics = [
        ("Indexed documents", len(indexed), "◫"),
        ("Indexed pages", pages, "▤"),
        ("Extracted tables", tables, "▦"),
        ("Extracted images", images, "▧"),
    ]
    for column, (label, value, icon) in zip(metric_columns, metrics, strict=True):
        with column:
            st.markdown(
                f'<div class="metric-card"><div class="metric-icon">{icon}</div>'
                f'<div class="metric-value">{value:,}</div><div class="metric-label">{label}</div></div>',
                unsafe_allow_html=True,
            )

    recent, questions = st.columns([1.2, 0.8], gap="large")
    with recent:
        st.markdown("### Recently uploaded")
        if not documents:
            st.info("Your workspace is ready. Upload a PDF to begin building your private knowledge base.")
        for document in documents[:5]:
            st.markdown(
                f'<div class="list-card"><span class="file-glyph">PDF</span>'
                f'<div class="list-card-main"><strong>{_escape(document["filename"])}</strong>'
                f'<small>{document["page_count"]} pages · {document["status"]}</small></div>'
                f'<span class="doc-status status-{document["status"].lower()}">{document["status"]}</span></div>',
                unsafe_allow_html=True,
            )
            if document["error"]:
                st.caption(document["error"])
            for warning in __import__("json").loads(document["warnings"]):
                st.caption(f"Note: {warning}")
    with questions:
        st.markdown("### Start with a question")
        suggestions = [
            "Summarize this PDF.",
            "What are the main conclusions?",
            "Compare values in the extracted tables.",
            "Which page contains the project methodology?",
        ]
        for index, question in enumerate(suggestions):
            if st.button(f"↗  {question}", key=f"suggestion_{index}", use_container_width=True):
                st.session_state["suggested_question"] = question
                st.session_state["navigation_target"] = "Chat with Documents"
                st.rerun()


def _escape(value: object) -> str:
    import html

    return html.escape(str(value))
