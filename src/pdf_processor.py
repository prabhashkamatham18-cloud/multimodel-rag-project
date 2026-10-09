from __future__ import annotations

from pathlib import Path

import fitz

from src.config import Settings
from src.image_processor import extract_pdf_images, render_page_for_ocr, render_page_previews
from src.models import PageContent, ProcessedDocument
from src.ocr_engine import OCREngine
from src.table_extractor import extract_tables


def process_pdf(pdf_path: Path, document_id: str, settings: Settings) -> ProcessedDocument:
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        raise ValueError("The selected file is not a readable PDF.")
    try:
        with fitz.open(pdf_path) as document:
            if document.is_encrypted:
                raise ValueError("Password-protected PDFs are not supported.")
            if document.page_count == 0:
                raise ValueError("The PDF contains no pages.")
            pages = [
                (
                    page_number,
                    page.get_text("text").strip(),
                    bool(page.get_images(full=True)) or len(page.get_drawings()) >= 12,
                )
                for page_number, page in enumerate(document, start=1)
            ]
    except (fitz.FileDataError, fitz.EmptyFileError) as exc:
        raise ValueError(f"Unable to read this PDF: {exc}") from exc

    processor = OCREngine(settings)
    warnings: list[str] = []
    previews = render_page_previews(pdf_path, document_id)
    page_contents: list[PageContent] = []
    for (page_number, text, visual_candidate), preview_path in zip(pages, previews, strict=True):
        used_ocr = False
        if len(text) < 40:
            try:
                text = processor.image_to_text(render_page_for_ocr(pdf_path, page_number))
                used_ocr = bool(text)
            except RuntimeError as exc:
                warnings.append(f"Page {page_number} OCR: {exc}")
        page_contents.append(
            PageContent(page_number, text, preview_path, used_ocr, visual_candidate)
        )

    tables = extract_tables(pdf_path)
    images, image_warnings = extract_pdf_images(pdf_path, document_id, processor)
    warnings.extend(image_warnings)
    return ProcessedDocument(page_contents, tables, images, warnings)
