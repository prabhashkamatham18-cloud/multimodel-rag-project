from __future__ import annotations

from pathlib import Path

from src.models import ExtractedTable


def extract_tables(pdf_path: Path) -> list[ExtractedTable]:
    import pdfplumber

    extracted: list[ExtractedTable] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            for table_index, raw_table in enumerate(page.extract_tables() or [], start=1):
                cleaned = [
                    [str(cell or "").strip().replace("\n", " ") for cell in row]
                    for row in raw_table
                    if row and any(str(cell or "").strip() for cell in row)
                ]
                if not cleaned:
                    continue
                width = max(len(row) for row in cleaned)
                cleaned = [row + [""] * (width - len(row)) for row in cleaned]
                headers = cleaned[0]
                if not any(headers):
                    headers = [f"Column {index + 1}" for index in range(width)]
                extracted.append(
                    ExtractedTable(page_number, table_index, headers, cleaned[1:])
                )
    return extracted
