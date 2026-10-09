# VisionRAG — Multimodal Document Intelligence

VisionRAG is a local-first Streamlit application for searching PDF text, extracted tables, OCR text, and image descriptions. It uses PyMuPDF and pdfplumber for document processing, sentence-transformers for local embeddings, ChromaDB for persistent vectors, and Ollama for grounded text answers. A separately configured vision-capable Ollama model can optionally answer questions about extracted images.

## Requirements and Python 3.14.8 compatibility

The project targets **Python 3.14.8** and does not silently change the requested interpreter version. The dependency set in `requirements.txt` was resolved on Windows x64 with Python 3.14.8 using pip's dry-run resolver. The resolver selected Python 3.14 wheels for pandas, Pillow, PyTorch, SciPy, scikit-learn, PyMuPDF, ChromaDB, and their compiled dependencies. Python packages do not install the external Tesseract executable or Ollama.

The tested pinned package versions are:

- Streamlit 1.65.0
- PyMuPDF 1.28.2 and pdfplumber 0.11.10
- pandas 3.0.6
- sentence-transformers 6.1.0
- ChromaDB 1.5.9
- pytesseract 0.3.13
- Plotly 7.1.0, Pillow 12.3.0, and torchvision 0.29.1

If a later package release drops Python 3.14 support, keep Python 3.14.8 and select an earlier compatible release; do not switch Python versions implicitly. Installation may download large PyTorch and embedding-model files. The sentence-transformers model is downloaded by Hugging Face on first use, then cached locally.

## Windows setup (PowerShell / VS Code)

1. Install Python 3.14.8 (64-bit), Git if desired, and VS Code with the Python extension.
2. Install [Ollama for Windows](https://ollama.com/download/windows), start Ollama, and pull the text model:

   ```powershell
   ollama pull qwen2.5:latest
   ollama list
   ```

   The application checks Ollama's live `/api/tags` endpoint and reports the model as missing if it is not listed; it does not assume that a configured name is installed.

3. In the project folder, create and activate a virtual environment:

   ```powershell
   py -3.14 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python --version
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

   The version command should print Python 3.14.8. In VS Code, choose `.venv\Scripts\python.exe` with **Python: Select Interpreter**.

4. Copy `.env.example` to `.env`. The defaults use `http://localhost:11434` and `qwen2.5:latest`. Settings can also be changed for the current app session on the Settings page.
5. Start the application:

   ```powershell
   streamlit run app.py
   ```

6. Upload one or more PDFs. The embedding model is downloaded on first indexing; subsequent use is local. Files and Chroma vectors are stored in the project folder.

### Tesseract OCR (optional, required for scanned pages)

Install the Windows Tesseract OCR application separately, for example from the [UB Mannheim Tesseract builds](https://github.com/UB-Mannheim/tesseract/wiki). Install the language data required by your PDFs (English uses `eng`), then either add Tesseract to `PATH` or set its absolute executable path in `.env`:

```dotenv
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
OCR_LANGUAGE=eng
```

VisionRAG attempts OCR for pages with little or no extractable text and for embedded images. If Tesseract is unavailable, it records and displays a clear warning. A document with no searchable text cannot be indexed.

### Optional vision model

The text assistant uses `qwen2.5:latest`; this model is not treated as an image model. For image understanding, install a vision-capable model separately (for example, `qwen2.5vl:latest` if available in your Ollama library), verify it using `ollama list`, then set `OLLAMA_VISION_MODEL=qwen2.5vl:latest` or configure it in Settings. The app enables image Q&A only when this separate model is configured and verifies model availability before each call. This project does not claim that the installed text-only Qwen2.5 model can interpret image pixels.

### Background asset and license

The Dashboard uses a locally stored Pexels technology photograph with a dark overlay and a CSS gradient fallback. No image is downloaded on page refresh, and the same asset is staged under `static/` so Streamlit can serve it locally. Attribution and the license link are recorded in [assets/backgrounds/ATTRIBUTION.txt](./assets/backgrounds/ATTRIBUTION.txt). The Pexels license permits use in websites and apps; attribution is included as a courtesy. If the asset is unavailable, the hero panel remains readable on its fallback gradient.

## What the application does

- **Dashboard:** database-backed document, indexed-page, table, and embedded-image counts; recent upload status; quick-start questions.
- **Upload Documents:** validates PDF uploads and per-file size, hashes files to avoid duplicate indexing, tracks Uploaded / Processing / Indexed / Failed state, and reports extraction/OCR errors.
- **Chat with Documents:** filters retrieval by document, displays assistant and user turns, persists search history, and shows source cards with document names, page numbers, excerpts, and a page preview. Clicking a citation selects that preview page. Answers are generated only when relevant retrieved evidence passes the configured relevance threshold; otherwise an explicit no-evidence response is shown.
- **Table Explorer:** browses extracted tables as pandas dataframes and charts detected numeric columns with Plotly.
- **Image Explorer:** browses extracted embedded images and scanned-page previews, filters OCR/description text, and supports optional selected-image Q&A using the configured vision model.
- **Search History:** locally persisted queries, answers, and evidence citations.
- **Settings:** Ollama URL and models, embedding model, chunk size/overlap, retrieval top-k and relevance threshold, upload size, and OCR configuration.

The app stores original PDFs in `data/uploads/`, extracted images in `data/extracted_images/`, rendered pages in `data/previews/`, SQLite metadata/history in `data/visionrag.sqlite3`, and vectors in `chroma_db/`. Uploaded files and extracted text are treated as untrusted data, are never executed, and are not sent to an external service by the app. Ollama and Hugging Face model downloads are separate local setup/runtime connections.

## Tests

Run the unit tests from the project root:

```powershell
python -m unittest discover -s tests -v
```

The PDF-processing test generates a small PDF in a temporary directory, so no sample document needs to be committed. The retrieval, citation, chunking, and Ollama model-error paths are tested without requiring a downloaded embedding model or a running LLM.

## Troubleshooting

- **Ollama unavailable:** start the Ollama application/service and check `http://localhost:11434/api/tags`. Confirm `OLLAMA_BASE_URL` if using another host.
- **Model missing:** run `ollama pull qwen2.5:latest`; for optional image questions, install and configure a separate vision model. `ollama list` is authoritative.
- **No searchable content / scanned PDF:** install Tesseract, add its language pack, configure `TESSERACT_CMD` if needed, and upload again.
- **Embedding model download fails:** connect once to download `sentence-transformers/all-MiniLM-L6-v2`, or pre-cache that model for offline environments. Once cached, indexing and embedding calls use the local copy.
- **Upload rejected:** verify the file opens as a PDF and stays below the configured per-file limit.
- **Missing preview/image:** do not remove or move files inside `data/`; previews and extracted images are referenced by local metadata.
- **Port conflict:** run `streamlit run app.py --server.port 8502`.
- **Reset local knowledge data:** stop the app, back up anything needed, then remove only the specific contents of `data/` and `chroma_db/`. Re-index the PDFs afterward.
