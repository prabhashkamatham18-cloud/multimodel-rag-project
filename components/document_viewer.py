from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.database import get_document, list_images


def render_document_viewer(document_id: str | None, page_number: int = 1) -> None:
    st.markdown("### Document preview")
    if not document_id:
        st.markdown('<div class="empty-preview">Select a document to preview its pages.</div>', unsafe_allow_html=True)
        return
    document = get_document(document_id)
    if not document:
        st.warning("The selected document is no longer available.")
        return
    previews = list_images(document_id, "page_preview")
    if not previews:
        st.info("No page previews are available for this document.")
        return
    page_numbers = [int(item["page_number"]) for item in previews]
    selected_page = page_number if page_number in page_numbers else page_numbers[0]
    image = next(item for item in previews if item["page_number"] == selected_page)
    st.caption(f"{document['filename']}  ·  Page {selected_page} of {document['page_count']}")
    path = Path(image["path"])
    if path.is_file():
        st.image(str(path), use_container_width=True)
    else:
        st.warning("This preview file is missing from local storage.")
    if image["ocr_text"]:
        with st.expander("Extracted page text"):
            st.write(image["ocr_text"][:3000])
