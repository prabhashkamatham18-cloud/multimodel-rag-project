from __future__ import annotations

import uuid
from pathlib import Path

import streamlit as st

from components.document_viewer import render_document_viewer
from src.database import list_documents, save_search
from src.llm_service import OllamaService
from src.rag_pipeline import RAGPipeline

SUGGESTED_QUESTIONS = [
    "Summarize this PDF.",
    "What are the main conclusions?",
    "Compare the values in the extracted tables.",
    "Explain the chart on page 5.",
    "Find the paragraph discussing the project methodology.",
]


def render_chat(pipeline: RAGPipeline, ollama: OllamaService) -> None:
    if "suggested_question" in st.session_state and "pending_question" not in st.session_state:
        st.session_state["pending_question"] = st.session_state.pop("suggested_question")
    documents = [item for item in list_documents() if item["status"] == "Indexed"]
    if "conversation_id" not in st.session_state:
        st.session_state["conversation_id"] = str(uuid.uuid4())
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []
    if st.button("Clear conversation", key="clear_conversation"):
        st.session_state["chat_messages"] = []
        st.session_state["conversation_id"] = str(uuid.uuid4())
        st.rerun()

    options = {"All indexed documents": None}
    options.update({item["filename"]: item["id"] for item in documents})
    chosen = st.selectbox("Filter evidence by document", list(options))
    selected_id = options[chosen]

    chat_col, preview_col = st.columns([1.35, 0.8], gap="large")
    with chat_col:
        st.markdown(
            '<div class="eyebrow">GROUNDED IN YOUR FILES</div><h2>Chat with your documents</h2>',
            unsafe_allow_html=True,
        )
        if not documents:
            st.info("Index a PDF before starting a grounded conversation.")
        if not st.session_state["chat_messages"]:
            st.caption("Ask a question or choose a suggested prompt.")
            for index, suggestion in enumerate(SUGGESTED_QUESTIONS):
                if st.button(suggestion, key=f"chat_suggestion_{index}"):
                    st.session_state["pending_question"] = suggestion
                    st.rerun()

        for message_index, message in enumerate(st.session_state["chat_messages"]):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant":
                    citations = message.get("citations", [])
                    if citations:
                        st.markdown("**Sources**")
                        for citation_index, citation in enumerate(citations):
                            label = f"{citation['document_name']}  ·  page {citation['page_number']}"
                            if st.button(label, key=f"citation_{message_index}_{citation_index}"):
                                st.session_state["preview_document_id"] = citation["document_id"]
                                st.session_state["preview_page"] = citation["page_number"]
                                st.rerun()
                            st.caption(citation["excerpt"])

        question = st.session_state.pop("pending_question", None) or st.chat_input(
            "Ask a question about your indexed PDFs…"
        )
        if question and documents:
            st.session_state["chat_messages"].append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Searching pages, tables, and visual evidence…"):
                    try:
                        answer, citations, _ = pipeline.answer(question, selected_id, ollama)
                    except Exception as exc:
                        answer, citations = f"Unable to answer: {exc}", []
                    st.markdown(answer)
                    if citations:
                        st.markdown("**Sources**")
                        for index, citation in enumerate(citations):
                            if st.button(
                                f"{citation.document_name}  ·  page {citation.page_number}",
                                key=f"new_citation_{index}",
                            ):
                                st.session_state["preview_document_id"] = citation.document_id
                                st.session_state["preview_page"] = citation.page_number
                                st.rerun()
                            st.caption(citation.excerpt)
                citation_data = [citation.__dict__ for citation in citations]
            st.session_state["chat_messages"].append(
                {"role": "assistant", "content": answer, "citations": citation_data}
            )
            save_search(st.session_state["conversation_id"], question, answer, citation_data)
            st.rerun()
        elif question and not documents:
            st.warning("Upload and index a PDF before asking document questions.")

    with preview_col:
        available_ids = {item["id"] for item in documents}
        selected_preview = st.session_state.get("preview_document_id")
        if selected_preview not in available_ids:
            selected_preview = documents[0]["id"] if documents else None
        if documents:
            names = {item["id"]: item["filename"] for item in documents}
            selected_preview = st.selectbox(
                "Preview document",
                list(names),
                format_func=lambda doc_id: names[doc_id],
                index=list(names).index(selected_preview) if selected_preview in names else 0,
            )
            st.session_state["preview_document_id"] = selected_preview
        render_document_viewer(selected_preview, st.session_state.get("preview_page", 1))
