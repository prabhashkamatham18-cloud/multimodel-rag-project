from __future__ import annotations

import hashlib
from pathlib import Path

import fitz

from src.config import IMAGE_DIR, PREVIEW_DIR
from src.models import ExtractedImage
from src.ocr_engine import OCREngine


def extract_pdf_images(
    pdf_path: Path,
    document_id: str,
    ocr: OCREngine,
) -> tuple[list[ExtractedImage], list[str]]:
    images: list[ExtractedImage] = []
    warnings: list[str] = []
    document_image_dir = IMAGE_DIR / document_id
    document_image_dir.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[int, str]] = set()

    with fitz.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            for image_number, image_info in enumerate(page.get_images(full=True), start=1):
                extracted = document.extract_image(image_info[0])
                content = extracted["image"]
                digest = hashlib.sha256(content).hexdigest()
                image_key = (page_number, digest)
                if image_key in seen:
                    continue
                seen.add(image_key)
                extension = extracted.get("ext", "png")
                target = document_image_dir / f"page-{page_number}-image-{image_number}.{extension}"
                target.write_bytes(content)
                try:
                    text = ocr.image_to_text(content)
                except RuntimeError as exc:
                    text = ""
                    warnings.append(f"Page {page_number} image OCR: {exc}")
                images.append(
                    ExtractedImage(page_number, str(target), "embedded", ocr_text=text)
                )
    return images, warnings


def render_page_previews(pdf_path: Path, document_id: str, dpi: int = 110) -> list[str]:
    target_dir = PREVIEW_DIR / document_id
    target_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    with fitz.open(pdf_path) as document:
        scale = dpi / 72
        matrix = fitz.Matrix(scale, scale)
        for page_number, page in enumerate(document, start=1):
            path = target_dir / f"page-{page_number}.png"
            page.get_pixmap(matrix=matrix, alpha=False).save(path)
            paths.append(str(path))
    return paths


def render_page_for_ocr(pdf_path: Path, page_number: int, dpi: int = 220) -> bytes:
    with fitz.open(pdf_path) as document:
        page = document[page_number - 1]
        scale = dpi / 72
        return page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False).tobytes("png")
