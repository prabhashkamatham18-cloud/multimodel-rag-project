from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from components.chat_interface import render_chat
from components.dashboard import render_dashboard
from components.sidebar import render_sidebar
from src.config import (
    BACKGROUND_PATH,
    UPLOAD_DIR,
    Settings,
    ensure_data_directories,
)
from src.database import (
    add_document,
    clear_search_history,
    find_document_by_hash,
    initialize_database,
    list_documents,
    list_images,
    list_searches,
    list_tables,
    save_search,
    store_processed_document,
    update_document,
)
from src.llm_service import OllamaService
from src.rag_pipeline import RAGPipeline

st.set_page_config(
    page_title="VisionRAG — Document Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_settings() -> Settings:
    return Settings.from_env().with_overrides(st.session_state.get("runtime_settings"))


@st.cache_resource(show_spinner=False)
def get_pipeline(settings: Settings):
    return RAGPipeline(settings)


def apply_styles() -> None:
    css_path = Path(__file__).parent / "styles" / "main.css"
    css = css_path.read_text(encoding="utf-8")
    if BACKGROUND_PATH.is_file():
        css += "\n.hero-panel { background-image: linear-gradient(90deg, rgba(8,12,29,.97) 4%, rgba(8,12,29,.82) 55%, rgba(8,12,29,.45)), url('/app/static/pcb-board.jpg'); }\n"
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_runtime_status(settings: Settings, ollama: OllamaService) -> None:
    try:
        models = ollama.list_models()
    except RuntimeError as exc:
        st.warning(f"Ollama is unavailable. Chat is disabled until it is running. {exc}")
        return
    if settings.text_model not in models:
        st.warning(f"Text model `{settings.text_model}` is missing. Run `ollama pull {settings.text_model}`.")
    if settings.vision_model and settings.vision_model not in models:
        st.info(
            f"Configured vision model `{settings.vision_model}` is not installed. "
            f"Run `ollama pull {settings.vision_model}` to enable visual Q&A."
        )


def render_uploads(pipeline: RAGPipeline, settings: Settings) -> None:
    st.markdown('<div class="eyebrow">BUILD YOUR LOCAL KNOWLEDGE BASE</div>', unsafe_allow_html=True)
    st.title("Upload documents")
    st.write("Add one or more PDFs. Files, extracted content, and vectors stay in this workspace.")
    uploads = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help=f"PDF only · maximum {settings.upload_limit_mb} MB per file",
    )
    if not uploads:
        st.markdown(
            '<div class="upload-note">PDF text, tables, embedded images, page previews, and OCR text '
            'are extracted locally. Scanned pages need Tesseract OCR.</div>',
            unsafe_allow_html=True,
        )
        return
    for upload in uploads:
        st.write(f"**{upload.name}** · {upload.size / (1024 * 1024):.1f} MB")
    if not st.button("Process selected PDFs", type="primary", disabled=not uploads):
        return

    progress = st.progress(0, text="Preparing documents…")
    for index, upload in enumerate(uploads):
        filename = Path(upload.name).name
        content = upload.getvalue()
        if Path(filename).suffix.lower() != ".pdf" or upload.type not in (
            "application/pdf",
            "application/octet-stream",
            "",
            None,
        ):
            st.error(f"{filename}: not a valid PDF upload.")
            continue
        if len(content) > settings.upload_limit_mb * 1024 * 1024:
            st.error(f"{filename}: exceeds the {settings.upload_limit_mb} MB upload limit.")
            continue
        document_id = hashlib.sha256(content).hexdigest()
        duplicate = find_document_by_hash(document_id)
        if duplicate and duplicate["status"] != "Failed":
            st.info(f"{filename}: already indexed as {duplicate['filename']} ({duplicate['status']}).")
            progress.progress((index + 1) / len(uploads), text="Checking uploads…")
            continue
        if duplicate:
            path = Path(duplicate["file_path"])
            if not path.is_file():
                path = UPLOAD_DIR / f"{document_id}.pdf"
                path.write_bytes(content)
            filename = duplicate["filename"]
        else:
            path = UPLOAD_DIR / f"{document_id}.pdf"
            path.write_bytes(content)
            add_document(document_id, filename, document_id, str(path))
        update_document(document_id, status="Processing")
        try:
            with st.spinner(f"Extracting and indexing {filename}…"):
                processed, chunk_count = pipeline.index_document(path, document_id, filename)
                store_processed_document(document_id, processed)
                if chunk_count:
                    update_document(
                        document_id,
                        status="Indexed",
                        page_count=len(processed.pages),
                        chunk_count=chunk_count,
                        warnings=processed.warnings,
                        indexed=True,
                    )
                    st.success(
                        f"{filename}: indexed {len(processed.pages)} pages, "
                        f"{len(processed.tables)} tables, and "
                        f"{sum(image.asset_type == 'embedded' for image in processed.images)} images."
                    )
                else:
                    message = (
                        "No searchable text was extracted. This may be a scanned PDF; "
                        "configure Tesseract OCR and retry."
                    )
                    update_document(
                        document_id,
                        status="Failed",
                        page_count=len(processed.pages),
                        chunk_count=0,
                        error=message,
                        warnings=processed.warnings,
                    )
                    st.error(f"{filename}: {message}")
            for warning in processed.warnings:
                st.warning(warning)
        except Exception as exc:
            update_document(document_id, status="Failed", error=str(exc))
            st.error(f"{filename}: processing failed — {exc}")
        progress.progress((index + 1) / len(uploads), text=f"Processed {index + 1} of {len(uploads)}")
    st.rerun()


def render_table_explorer() -> None:
    st.markdown('<div class="eyebrow">STRUCTURED DATA FROM YOUR PDFs</div>', unsafe_allow_html=True)
    st.title("Table explorer")
    documents = [item for item in list_documents() if item["status"] == "Indexed"]
    if not documents:
        st.info("Indexed documents with extracted tables will appear here.")
        return
    selected = st.selectbox(
        "Filter by document",
        ["All documents"] + [item["filename"] for item in documents],
    )
    document_id = next((item["id"] for item in documents if item["filename"] == selected), None)
    tables = list_tables(document_id)
    if not tables:
        st.info("No tables were detected in the selected documents.")
        return
    labels = [
        f"{table['document_name']} · page {table['page_number']} · table {table['table_index']}"
        for table in tables
    ]
    table = tables[st.selectbox("Extracted table", range(len(labels)), format_func=lambda i: labels[i])]
    dataframe = pd.DataFrame(table["rows"], columns=table["headers"])
    st.dataframe(dataframe, use_container_width=True, hide_index=True)
    numeric = _numeric_columns(dataframe)
    if len(dataframe) and numeric:
        st.markdown("#### Visual comparison")
        value_column = st.selectbox("Numeric measure", numeric)
        category_options = [column for column in dataframe.columns if column != value_column]
        if category_options:
            category = st.selectbox("Group by", category_options)
            chart_df = dataframe[[category, value_column]].copy()
            chart_df[value_column] = _to_numeric(chart_df[value_column])
            chart_df = chart_df.dropna(subset=[value_column])
            if not chart_df.empty:
                st.plotly_chart(
                    px.bar(chart_df, x=category, y=value_column, color=category, template="plotly_dark"),
                    use_container_width=True,
                )
    st.caption(f"Source: {table['document_name']} · page {table['page_number']}")


def _to_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(r"[$,%€£,\s]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def _numeric_columns(dataframe: pd.DataFrame) -> list[str]:
    return [
        str(column)
        for column in dataframe.columns
        if _to_numeric(dataframe[column]).notna().sum() > 0
    ]


def render_image_explorer(settings: Settings, ollama: OllamaService, pipeline: RAGPipeline) -> None:
    st.markdown('<div class="eyebrow">VISUAL EVIDENCE FROM YOUR FILES</div>', unsafe_allow_html=True)
    st.title("Image explorer")
    documents = [
        item for item in list_documents()
        if item["status"] in {"Indexed", "Failed"} and list_images(item["id"])
    ]
    selected_name = st.selectbox("Filter by document", ["All documents"] + [d["filename"] for d in documents])
    document_id = next((d["id"] for d in documents if d["filename"] == selected_name), None)
    assets = list_images(document_id)
    if not assets:
        st.info("Embedded images and scanned-page previews appear after a PDF is indexed.")
        return
    query = st.text_input("Filter image descriptions or OCR text")
    visible = [
        image for image in assets
        if not query or query.casefold() in (
            image["document_name"] + image["description"] + image["ocr_text"]
        ).casefold()
    ]
    if not visible:
        st.info("No images match that filter.")
        return
    columns = st.columns(3)
    for index, image in enumerate(visible):
        with columns[index % 3]:
            st.markdown('<div class="image-card">', unsafe_allow_html=True)
            path = Path(image["path"])
            if path.is_file():
                st.image(str(path), use_container_width=True)
            else:
                st.warning("Image file is missing from local storage.")
            kind = "Scanned page" if image["asset_type"] == "page_preview" else "Embedded image"
            st.markdown(f"**{kind} · {image['document_name']}**")
            st.caption(f"Page {image['page_number']}")
            detail = image["description"] or image["ocr_text"]
            if detail:
                if image["description"]:
                    label = "Vision description"
                elif image["asset_type"] == "page_preview":
                    label = "Extracted page text"
                else:
                    label = "OCR text"
                st.caption(f"{label}: {detail[:400]}")
            else:
                st.caption("No OCR text or image description is available.")
            if path.is_file():
                vision_enabled = bool(settings.vision_model)
                with st.expander("Ask about this image"):
                    question = st.text_input(
                        "Question",
                        key=f"vision_question_{image['id']}",
                        placeholder="What is shown in this image?",
                    )
                    if st.button(
                        "Analyze image",
                        key=f"vision_button_{image['id']}",
                        disabled=not vision_enabled,
                    ):
                        try:
                            ollama.verify_model(settings.vision_model)
                            with st.spinner("Analyzing with the configured vision model…"):
                                answer = pipeline.describe_image(path, ollama, question)
                            st.session_state[f"vision_answer_{image['id']}"] = answer
                            save_search(
                                "image-" + str(image["id"]),
                                question or "Describe this image",
                                answer,
                                [
                                    {
                                        "document_id": image["document_id"],
                                        "document_name": image["document_name"],
                                        "page_number": image["page_number"],
                                        "excerpt": "Vision-model analysis of the selected image.",
                                    }
                                ],
                            )
                        except Exception as exc:
                            st.error(str(exc))
                    if f"vision_answer_{image['id']}" in st.session_state:
                        st.markdown(st.session_state[f"vision_answer_{image['id']}"])
                    if not vision_enabled:
                        st.caption("Configure a vision-capable Ollama model in Settings to enable image Q&A.")
            st.markdown("</div>", unsafe_allow_html=True)


def render_search_history() -> None:
    st.markdown('<div class="eyebrow">YOUR LOCAL CONVERSATION LOG</div>', unsafe_allow_html=True)
    st.title("Search history")
    if st.button("Clear all search history"):
        clear_search_history()
        st.rerun()
    history = list_searches()
    if not history:
        st.info("Your document questions will appear here.")
        return
    for item in history:
        with st.expander(item["question"], expanded=False):
            st.markdown(item["answer"])
            for citation in item["citations"]:
                st.caption(f"{citation['document_name']} · page {citation['page_number']}")
            st.caption(item["created_at"])


def render_settings(settings: Settings, ollama: OllamaService) -> None:
    st.markdown('<div class="eyebrow">LOCAL MODELS AND RETRIEVAL</div>', unsafe_allow_html=True)
    st.title("Settings")
    with st.form("settings_form"):
        url = st.text_input("Ollama base URL", settings.ollama_base_url)
        text_model = st.text_input("Text model", settings.text_model)
        vision_model = st.text_input("Vision model (optional)", settings.vision_model)
        embedding_model = st.text_input("Sentence-transformers embedding model", settings.embedding_model)
        first, second, third = st.columns(3)
        with first:
            chunk_size = st.number_input("Chunk size (characters)", 1, 4000, settings.chunk_size, 100)
        with second:
            chunk_overlap = st.number_input(
                "Chunk overlap", 0, int(chunk_size) - 1, min(settings.chunk_overlap, int(chunk_size) - 1), 20
            )
        with third:
            top_k = st.number_input("Retrieval top-k", 1, 20, settings.retrieval_top_k)
        first, second = st.columns(2)
        with first:
            upload_limit = st.number_input("Upload limit (MB)", 1, 500, settings.upload_limit_mb)
        with second:
            threshold = st.slider("Relevance threshold", 0.0, 1.0, settings.relevance_threshold, 0.05)
        ocr_language = st.text_input("Tesseract OCR language", settings.ocr_language)
        tesseract_cmd = st.text_input("Tesseract executable path (optional)", settings.tesseract_cmd)
        submitted = st.form_submit_button("Save settings", type="primary")
    if submitted:
        if chunk_overlap >= chunk_size:
            st.error("Chunk overlap must be smaller than the chunk size.")
        else:
            st.session_state["runtime_settings"] = {
                "ollama_base_url": url.rstrip("/"),
                "text_model": text_model,
                "vision_model": vision_model,
                "embedding_model": embedding_model,
                "chunk_size": int(chunk_size),
                "chunk_overlap": int(chunk_overlap),
                "retrieval_top_k": int(top_k),
                "upload_limit_mb": int(upload_limit),
                "relevance_threshold": threshold,
                "ocr_language": ocr_language,
                "tesseract_cmd": tesseract_cmd,
            }
            get_pipeline.clear()
            st.success("Settings saved for this browser session.")
            st.rerun()
    st.markdown("#### Model availability")
    try:
        models = ollama.list_models()
        st.success("Ollama is reachable.")
        st.write("Installed models: " + (", ".join(f"`{name}`" for name in models) or "none"))
        if settings.text_model not in models:
            st.warning(f"Text model is not installed. Run `ollama pull {settings.text_model}`.")
        if settings.vision_model and settings.vision_model not in models:
            st.warning(f"Vision model is not installed. Run `ollama pull {settings.vision_model}`.")
    except RuntimeError as exc:
        st.error(str(exc))
    st.caption(
        "Settings are stored in Streamlit session state. For persistent defaults, set environment "
        "variables in your local .env file."
    )


def main() -> None:
    ensure_data_directories()
    initialize_database()
    apply_styles()
    settings = get_settings()
    ollama = OllamaService(settings)
    selection = render_sidebar()
    render_runtime_status(settings, ollama)
    if selection == "Dashboard":
        render_dashboard()
    elif selection == "Upload Documents":
        render_uploads(
            get_pipeline(settings),
            settings,
        )
    elif selection == "Chat with Documents":
        render_chat(
            get_pipeline(settings),
            ollama,
        )
    elif selection == "Table Explorer":
        render_table_explorer()
    elif selection == "Image Explorer":
        render_image_explorer(
            settings,
            ollama,
            get_pipeline(settings),
        )
    elif selection == "Search History":
        render_search_history()
    elif selection == "Settings":
        render_settings(settings, ollama)


if __name__ == "__main__":
    main()
