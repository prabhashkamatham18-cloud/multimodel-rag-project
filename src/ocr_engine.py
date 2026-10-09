from __future__ import annotations

import io
import shutil
from pathlib import Path

from src.config import Settings


class OCREngine:
    def __init__(self, settings: Settings):
        self.settings = settings

    def available(self) -> bool:
        return bool(self.settings.tesseract_cmd and Path(self.settings.tesseract_cmd).is_file()) or bool(
            shutil.which("tesseract")
        )

    def image_to_text(self, image_bytes: bytes) -> str:
        try:
            import pytesseract
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("OCR dependencies are missing. Install the project requirements.") from exc

        if self.settings.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.settings.tesseract_cmd
        if not self.available():
            raise RuntimeError(
                "Tesseract OCR is not installed or configured. Install Tesseract and set TESSERACT_CMD."
            )
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                return pytesseract.image_to_string(image, lang=self.settings.ocr_language).strip()
        except pytesseract.TesseractNotFoundError as exc:
            raise RuntimeError("Tesseract executable was not found. Configure TESSERACT_CMD.") from exc
        except pytesseract.TesseractError as exc:
            raise RuntimeError(f"Tesseract OCR failed: {exc}") from exc
