from __future__ import annotations

import tempfile
import unittest
import shutil
from pathlib import Path
from unittest.mock import patch

import fitz

from src.config import PREVIEW_DIR, Settings
from src.llm_service import OllamaService
from src.models import Citation, ExtractedTable, PageContent, ProcessedDocument, RetrievedChunk
from src.pdf_processor import process_pdf
from src.rag_pipeline import NO_EVIDENCE, build_chunks, chunk_text, remove_unverified_citations
from src.retriever import Retriever
from src.table_extractor import extract_tables


class FakeVectorStore:
    def __init__(self, chunks: list[RetrievedChunk]):
        self.chunks = chunks
        self.requested_document_id: str | None = None

    def query(self, question: str, top_k: int, document_id: str | None = None):
        self.requested_document_id = document_id
        return self.chunks[:top_k]


class ChunkingTests(unittest.TestCase):
    def test_invalid_runtime_settings_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlap smaller than chunk size"):
            Settings(chunk_size=10, chunk_overlap=10)

    def test_chunking_preserves_content_and_overlap(self):
        text = "alpha beta gamma delta epsilon zeta eta theta"
        chunks = chunk_text(text, chunk_size=19, overlap=5)
        self.assertGreater(len(chunks), 1)
        joined = " ".join(chunks)
        for word in text.split():
            self.assertIn(word, joined)
        self.assertGreaterEqual(len(set(chunks[0].split()) & set(chunks[1].split())), 1)

    def test_invalid_chunk_settings_are_rejected(self):
        with self.assertRaises(ValueError):
            chunk_text("text", chunk_size=10, overlap=10)

    def test_metadata_keeps_document_and_page(self):
        processed = ProcessedDocument()
        processed.pages.append(PageContent(7, "Evidence on this page."))
        chunks = build_chunks("doc-id", "report.pdf", processed, 100, 10)
        self.assertEqual(chunks[0].metadata["document_id"], "doc-id")
        self.assertEqual(chunks[0].metadata["document_name"], "report.pdf")
        self.assertEqual(chunks[0].metadata["page_number"], 7)

    def test_table_chunks_keep_column_headers_and_complete_rows(self):
        processed = ProcessedDocument(
            tables=[
                ExtractedTable(
                    3,
                    1,
                    ["Name", "Value"],
                    [["Alpha", "100"], ["Beta", "200"], ["Gamma", "300"]],
                )
            ]
        )
        chunks = build_chunks("doc-id", "report.pdf", processed, 34, 5)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.metadata["kind"] == "table" for chunk in chunks))
        self.assertTrue(all(chunk.text.startswith("Name | Value") for chunk in chunks))
        combined = "\n".join(chunk.text for chunk in chunks)
        for row in ("Alpha | 100", "Beta | 200", "Gamma | 300"):
            self.assertIn(row, combined)


class RetrievalAndCitationTests(unittest.TestCase):
    def test_document_filter_and_relevance_threshold(self):
        chunks = [
            RetrievedChunk(
                "Relevant text",
                {"document_id": "doc-a", "document_name": "a.pdf", "page_number": 2, "kind": "text"},
                0.2,
            ),
            RetrievedChunk(
                "Weak match",
                {"document_id": "doc-b", "document_name": "b.pdf", "page_number": 1, "kind": "text"},
                0.95,
            ),
        ]
        store = FakeVectorStore(chunks)
        retriever = Retriever(store, top_k=5, relevance_threshold=0.15)
        found = retriever.retrieve("question", "doc-a")
        self.assertEqual(store.requested_document_id, "doc-a")
        self.assertEqual([chunk.text for chunk in found], ["Relevant text"])

    def test_citations_are_exact_and_deduplicated_by_document_page(self):
        chunks = [
            RetrievedChunk("first excerpt", {"document_id": "a", "document_name": "a.pdf", "page_number": 4}, 0.1),
            RetrievedChunk("second excerpt", {"document_id": "a", "document_name": "a.pdf", "page_number": 4}, 0.2),
            RetrievedChunk("other page", {"document_id": "a", "document_name": "a.pdf", "page_number": 5}, 0.2),
        ]
        citations = Retriever.citations(chunks)
        self.assertEqual([(item.document_name, item.page_number) for item in citations], [
            ("a.pdf", 4), ("a.pdf", 5)
        ])
        self.assertEqual(citations[0].excerpt, "first excerpt")

    def test_no_evidence_is_an_explicit_non_answer(self):
        from src.rag_pipeline import RAGPipeline

        class EmptyRetriever:
            def retrieve(self, question: str, document_id: str | None):
                return []

        pipeline = object.__new__(RAGPipeline)
        pipeline.retriever = EmptyRetriever()
        answer, citations, chunks = pipeline.answer("unknown", None, OllamaService(Settings()))
        self.assertEqual(answer, NO_EVIDENCE)
        self.assertEqual(citations, [])
        self.assertEqual(chunks, [])

    def test_unverified_model_citations_are_removed(self):
        citations = [Citation("doc-id", "report.pdf", 4, "Exact source excerpt")]
        answer = "Evidence [report.pdf, page 4]. Unsupported [other.pdf, page 99]."
        self.assertEqual(
            remove_unverified_citations(answer, citations),
            "Evidence [report.pdf, page 4]. Unsupported .",
        )


class PdfProcessingTests(unittest.TestCase):
    def test_extracts_page_text_and_creates_preview(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.pdf"
            pdf = fitz.open()
            page = pdf.new_page()
            page.insert_text((72, 72), "A generated sample document with enough text for OCR to be skipped.")
            pdf.save(path)
            pdf.close()
            preview_dir = PREVIEW_DIR / "unit-test-document"
            try:
                processed = process_pdf(path, "unit-test-document", Settings())
                self.assertEqual(len(processed.pages), 1)
                self.assertIn("generated sample document", processed.pages[0].text)
                self.assertTrue(Path(processed.pages[0].preview_path).is_file())
            finally:
                shutil.rmtree(preview_dir, ignore_errors=True)

    def test_extracts_table_rows_and_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table.pdf"
            pdf = fitz.open()
            page = pdf.new_page()
            x_positions = [72, 180, 280]
            y_positions = [72, 100, 128, 156]
            for x in x_positions:
                page.draw_line((x, y_positions[0]), (x, y_positions[-1]))
            for y in y_positions:
                page.draw_line((x_positions[0], y), (x_positions[-1], y))
            values = [["Product", "Units"], ["Alpha", "10"], ["Beta", "20"]]
            for row_index, row in enumerate(values):
                for column_index, value in enumerate(row):
                    page.insert_text(
                        (x_positions[column_index] + 4, y_positions[row_index] + 19), value
                    )
            pdf.save(path)
            pdf.close()
            tables = extract_tables(path)
            self.assertEqual(len(tables), 1)
            self.assertEqual(tables[0].headers, ["Product", "Units"])
            self.assertEqual(tables[0].rows, [["Alpha", "10"], ["Beta", "20"]])

    def test_invalid_pdf_path_has_a_clear_error(self):
        with self.assertRaisesRegex(ValueError, "not a readable PDF"):
            process_pdf(Path("missing.pdf"), "missing", Settings())


class OllamaErrorTests(unittest.TestCase):
    def test_missing_model_reports_pull_command(self):
        service = OllamaService(Settings())
        with patch.object(service, "list_models", return_value=["other:latest"]):
            with self.assertRaisesRegex(RuntimeError, "ollama pull qwen2.5:latest"):
                service.verify_model("qwen2.5:latest")

    def test_unavailable_server_has_actionable_message(self):
        service = OllamaService(Settings(ollama_base_url="http://127.0.0.1:1"))
        with self.assertRaisesRegex(RuntimeError, "Cannot connect to Ollama"):
            service.list_models()


if __name__ == "__main__":
    unittest.main()
