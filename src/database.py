from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import DATABASE_PATH, ensure_data_directories
from src.models import ExtractedImage, ExtractedTable, PageContent, ProcessedDocument


def connect(path: Path = DATABASE_PATH) -> sqlite3.Connection:
    ensure_data_directories()
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database() -> None:
    with connect() as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                sha256 TEXT NOT NULL UNIQUE,
                file_path TEXT NOT NULL,
                status TEXT NOT NULL,
                page_count INTEGER NOT NULL DEFAULT 0,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                error TEXT NOT NULL DEFAULT '',
                warnings TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                indexed_at TEXT
            );
            CREATE TABLE IF NOT EXISTS tables (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                page_number INTEGER NOT NULL,
                table_index INTEGER NOT NULL,
                headers TEXT NOT NULL,
                rows TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                page_number INTEGER NOT NULL,
                path TEXT NOT NULL,
                asset_type TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                ocr_text TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                citations TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )


def add_document(document_id: str, filename: str, sha256: str, file_path: str) -> None:
    with connect() as db:
        db.execute(
            """INSERT INTO documents (id, filename, sha256, file_path, status, created_at)
               VALUES (?, ?, ?, ?, 'Uploaded', ?)""",
            (document_id, filename, sha256, file_path, datetime.now(timezone.utc).isoformat()),
        )


def find_document_by_hash(sha256: str) -> dict[str, Any] | None:
    with connect() as db:
        row = db.execute("SELECT * FROM documents WHERE sha256 = ?", (sha256,)).fetchone()
    return dict(row) if row else None


def update_document(
    document_id: str,
    *,
    status: str,
    page_count: int | None = None,
    chunk_count: int | None = None,
    error: str = "",
    warnings: list[str] | None = None,
    indexed: bool = False,
) -> None:
    assignments = ["status = ?", "error = ?", "warnings = ?"]
    values: list[Any] = [status, error, json.dumps(warnings or [], ensure_ascii=False)]
    if page_count is not None:
        assignments.append("page_count = ?")
        values.append(page_count)
    if chunk_count is not None:
        assignments.append("chunk_count = ?")
        values.append(chunk_count)
    if indexed:
        assignments.append("indexed_at = ?")
        values.append(datetime.now(timezone.utc).isoformat())
    values.append(document_id)
    with connect() as db:
        db.execute(f"UPDATE documents SET {', '.join(assignments)} WHERE id = ?", values)


def store_processed_document(document_id: str, result: ProcessedDocument) -> None:
    with connect() as db:
        db.execute("DELETE FROM tables WHERE document_id = ?", (document_id,))
        db.execute("DELETE FROM images WHERE document_id = ?", (document_id,))
        db.executemany(
            """INSERT INTO tables (document_id, page_number, table_index, headers, rows)
               VALUES (?, ?, ?, ?, ?)""",
            [
                (
                    document_id,
                    table.page_number,
                    table.table_index,
                    json.dumps(table.headers, ensure_ascii=False),
                    json.dumps(table.rows, ensure_ascii=False),
                )
                for table in result.tables
            ],
        )
        image_records = [
            (
                document_id,
                image.page_number,
                image.path,
                image.asset_type,
                image.description,
                image.ocr_text,
            )
            for image in result.images
        ]
        image_records.extend(
            (
                document_id,
                page.page_number,
                page.preview_path,
                "page_preview",
                page.description,
                page.text,
            )
            for page in result.pages
            if page.preview_path
        )
        db.executemany(
            """INSERT INTO images
               (document_id, page_number, path, asset_type, description, ocr_text)
               VALUES (?, ?, ?, ?, ?, ?)""",
            image_records,
        )


def list_documents() -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute(
            """SELECT d.*,
                      (SELECT COUNT(*) FROM tables t WHERE t.document_id = d.id) AS table_count,
                      (SELECT COUNT(*) FROM images i
                       WHERE i.document_id = d.id AND i.asset_type = 'embedded') AS image_count
               FROM documents d ORDER BY d.created_at DESC"""
        ).fetchall()
    return [dict(row) for row in rows]


def get_document(document_id: str) -> dict[str, Any] | None:
    with connect() as db:
        row = db.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
    return dict(row) if row else None


def list_tables(document_id: str | None = None) -> list[dict[str, Any]]:
    sql = """SELECT t.*, d.filename AS document_name
             FROM tables t JOIN documents d ON d.id = t.document_id"""
    params: tuple[Any, ...] = ()
    if document_id:
        sql += " WHERE t.document_id = ?"
        params = (document_id,)
    sql += " ORDER BY d.filename, t.page_number, t.table_index"
    with connect() as db:
        rows = db.execute(sql, params).fetchall()
    return [
        {
            **dict(row),
            "headers": json.loads(row["headers"]),
            "rows": json.loads(row["rows"]),
        }
        for row in rows
    ]


def list_images(
    document_id: str | None = None, asset_type: str | None = None
) -> list[dict[str, Any]]:
    sql = """SELECT i.*, d.filename AS document_name
             FROM images i JOIN documents d ON d.id = i.document_id"""
    conditions: list[str] = []
    params: list[Any] = []
    if document_id:
        conditions.append("i.document_id = ?")
        params.append(document_id)
    if asset_type:
        conditions.append("i.asset_type = ?")
        params.append(asset_type)
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY d.filename, i.page_number, i.id"
    with connect() as db:
        rows = db.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def save_search(
    conversation_id: str, question: str, answer: str, citations: list[dict[str, Any]]
) -> None:
    with connect() as db:
        db.execute(
            """INSERT INTO search_history
               (conversation_id, question, answer, citations, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                conversation_id,
                question,
                answer,
                json.dumps(citations, ensure_ascii=False),
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def list_searches(limit: int = 100) -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute(
            "SELECT * FROM search_history ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [
        {**dict(row), "citations": json.loads(row["citations"])}
        for row in rows
    ]


def clear_search_history() -> None:
    with connect() as db:
        db.execute("DELETE FROM search_history")


def store_page_and_image_metadata(
    document_id: str,
    pages: list[PageContent],
    images: list[ExtractedImage],
    tables: list[ExtractedTable],
) -> None:
    processed = ProcessedDocument(pages=pages, images=images, tables=tables)
    store_processed_document(document_id, processed)
