from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env", override=False)
DATA_DIR = ROOT_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
IMAGE_DIR = DATA_DIR / "extracted_images"
PREVIEW_DIR = DATA_DIR / "previews"
CHROMA_DIR = ROOT_DIR / "chroma_db"
DATABASE_PATH = DATA_DIR / "visionrag.sqlite3"
BACKGROUND_PATH = ROOT_DIR / "assets" / "backgrounds" / "pcb-board.jpg"


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = "http://localhost:11434"
    text_model: str = "qwen2.5:latest"
    vision_model: str = ""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 900
    chunk_overlap: int = 140
    retrieval_top_k: int = 5
    upload_limit_mb: int = 50
    ocr_language: str = "eng"
    tesseract_cmd: str = ""
    relevance_threshold: float = 0.15

    def __post_init__(self) -> None:
        if not self.ollama_base_url.startswith(("http://", "https://")):
            raise ValueError("OLLAMA_BASE_URL must be an http:// or https:// URL.")
        if not self.text_model.strip() or not self.embedding_model.strip():
            raise ValueError("Text and embedding model names cannot be blank.")
        if self.chunk_size < 1 or not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("Chunk size must be positive and overlap smaller than chunk size.")
        if self.retrieval_top_k < 1 or self.upload_limit_mb < 1:
            raise ValueError("Retrieval top-k and upload limit must be positive.")
        if not 0 <= self.relevance_threshold <= 1:
            raise ValueError("Relevance threshold must be between 0 and 1.")

    @classmethod
    def from_env(cls) -> "Settings":
        defaults = cls()
        return cls(
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", defaults.ollama_base_url).rstrip("/"),
            text_model=os.getenv("OLLAMA_TEXT_MODEL", defaults.text_model),
            vision_model=os.getenv("OLLAMA_VISION_MODEL", defaults.vision_model),
            embedding_model=os.getenv("EMBEDDING_MODEL", defaults.embedding_model),
            chunk_size=int(os.getenv("CHUNK_SIZE", defaults.chunk_size)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", defaults.chunk_overlap)),
            retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", defaults.retrieval_top_k)),
            upload_limit_mb=int(os.getenv("UPLOAD_LIMIT_MB", defaults.upload_limit_mb)),
            ocr_language=os.getenv("OCR_LANGUAGE", defaults.ocr_language),
            tesseract_cmd=os.getenv("TESSERACT_CMD", defaults.tesseract_cmd),
            relevance_threshold=float(os.getenv("RELEVANCE_THRESHOLD", defaults.relevance_threshold)),
        )

    def with_overrides(self, values: dict[str, object] | None) -> "Settings":
        if not values:
            return self
        allowed = set(self.__dataclass_fields__)
        return replace(self, **{key: value for key, value in values.items() if key in allowed})


def ensure_data_directories() -> None:
    for directory in (UPLOAD_DIR, IMAGE_DIR, PREVIEW_DIR, CHROMA_DIR):
        directory.mkdir(parents=True, exist_ok=True)
