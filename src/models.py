from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PageContent:
    page_number: int
    text: str
    preview_path: str = ""
    ocr_used: bool = False
    visual_candidate: bool = False
    description: str = ""


@dataclass
class ExtractedTable:
    page_number: int
    table_index: int
    headers: list[str]
    rows: list[list[str]]

    def as_text(self) -> str:
        lines = [" | ".join(self.headers)]
        lines.extend(" | ".join(row) for row in self.rows)
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "table_index": self.table_index,
            "headers": self.headers,
            "rows": self.rows,
        }


@dataclass
class ExtractedImage:
    page_number: int
    path: str
    asset_type: str
    description: str = ""
    ocr_text: str = ""


@dataclass
class ProcessedDocument:
    pages: list[PageContent] = field(default_factory=list)
    tables: list[ExtractedTable] = field(default_factory=list)
    images: list[ExtractedImage] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class TextChunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]


@dataclass
class RetrievedChunk:
    text: str
    metadata: dict[str, Any]
    distance: float


@dataclass
class Citation:
    document_id: str
    document_name: str
    page_number: int
    excerpt: str
    image_path: str = ""
