# Troubleshooting Guide — Obscura

> Common issues, their root causes, and step-by-step fixes for the Obscura pipeline. Organized by subsystem.

---

## Table of Contents

- [General Setup](#general-setup)
- [Virtual Environment & Dependencies](#virtual-environment--dependencies)
- [Local LLM (EXO / Ollama)](#local-llm-exo--ollama)
- [Cognee Memory Graph](#cognee-memory-graph)
- [Fastembed Embeddings](#fastembed-embeddings)
- [Document Ingest](#document-ingest)
- [PII Detection](#pii-detection)
- [Redaction Engine](#redaction-engine)
- [Image Vision (OpenCV / Tesseract)](#image-vision-opencv--tesseract)
- [Web UI (Flask)](#web-ui-flask)
- [Synthetic Dataset Generation](#synthetic-dataset-generation)
- [Airplane Mode / Offline Issues](#airplane-mode--offline-issues)
- [Performance & Memory](#performance--memory)

---

## General Setup

### Issue: `python` or `pip` not found

**Cause:** The virtual environment is not activated, or Python is not installed.

**Fix:**

```bash
# Check Python version
python3 --version  # Must be 3.10–3.14

# Create and activate venv
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate     # Windows

# Verify pip
which pip
pip --version
```

---

### Issue: `ModuleNotFoundError` on any import

**Cause:** Package not installed in the active environment, or environment not activated.

**Fix:**

```bash
source .venv/bin/activate
pip install -r requirements.txt

# Verify the specific import
python -c "import <module_name>; print('ok')"
```

---

### Issue: `.env` file not being read

**Cause:** Cognee reads `.env` from the **current working directory** (where you run the script). If you run a module from a different directory, `.env` won't be found.

**Fix:**

```bash
# Always run from the project root
cd /path/to/localhost-document-redaction
python -m cognee_mem.smoke_test

# Or explicitly set the working directory in code
import os
os.chdir("/path/to/localhost-document-redaction")
```

---

## Virtual Environment & Dependencies

### Issue: `uv` not available

**Cause:** `uv` is not installed on the system.

**Fix:**

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or use pip/venv as fallback
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### Issue: `pip install` fails with dependency conflicts

**Cause:** Stale environment or mixed package managers (e.g., `conda` + `pip`).

**Fix:**

```bash
# Nuclear option: rebuild the environment
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Issue: `cognee` import fails with `SyntaxError`

**Cause:** Python version is < 3.10. Cognee 1.1.3 requires Python 3.10+.

**Fix:**

```bash
python --version  # Must be 3.10–3.14
# If it's 3.9, install a newer Python and recreate the venv
```

---

### Issue: `PyMuPDF` installed but `import fitz` fails

**Cause:** PyMuPDF is installed under the package name `PyMuPDF` but imported as `fitz`. This is correct — the issue is usually a stale environment or wrong Python interpreter.

**Fix:**

```bash
source .venv/bin/activate
pip install "PyMuPDF>=1.24.0"
python -c "import fitz; print(fitz.__doc__[:50])"
```

---

## Local LLM (EXO / Ollama)

### Issue: `No local LLM endpoint found`

**Cause:** Neither EXO nor Ollama is running, or the port is wrong.

**Fix:**

```bash
# Check if EXO is running
curl http://localhost:52415/v1/models

# Check if Ollama is running
curl http://localhost:11434/v1/models

# Start EXO
cd /path/to/exo && exo

# Start Ollama
ollama serve

# Force Ollama fallback
export OBSCURA_USE_OLLAMA=1
```

---

### Issue: EXO model download fails or hangs

**Cause:** EXO tries to download the model on first run. On venue Wi-Fi, this may be slow or blocked. Or the model cache is corrupted.

**Fix:**

```bash
# Pre-download while you have good internet
cd /path/to/exo
exo
# Wait for the model to download fully (~2.4 GB, 5–10 min)

# Check the cache
ls ~/.cache/huggingface/

# If corrupted, clear and re-download
rm -rf ~/.cache/huggingface/
# Then re-run exo
```

---

### Issue: EXO returns 500 or empty response

**Cause:** The model is too large for available RAM, or the prompt is too long for the context window.

**Fix:**

- Reduce chunk size: edit `ingest/chunk.py` and lower `DEFAULT_WINDOW` (e.g., 80 instead of 120).
- Use a smaller model: `export OBSCURA_LLM_MODEL=mlx-community/Llama-3.2-3B-Instruct-4bit`.
- Switch to Ollama with `llama3.2:3b` which is more stable on limited RAM.

---

### Issue: LLM detection is very slow (10+ seconds per chunk)

**Cause:** Small model on CPU. Normal for 3B models on laptops. M1/M2 Macs with MLX are faster.

**Fix:**

- Use regex-only detection for speed: omit `--use-llm` flag.
- Reduce chunk size to process fewer tokens per call.
- Use a smaller model (e.g., 1B instead of 3B).
- Run on Apple Silicon with MLX (EXO uses MLX on M1/M2 automatically).

---

### Issue: `Ollama` returns `model not found`

**Cause:** The model was not pulled before running.

**Fix:**

```bash
ollama pull llama3.2:3b
ollama list  # verify it's there
ollama serve
```

---

## Cognee Memory Graph

### Issue: `cognee` import fails

**Cause:** `cognee[fastembed]` was not installed, or the environment is not activated.

**Fix:**

```bash
source .venv/bin/activate
pip install "cognee[fastembed]==1.1.3"
python -c "import cognee; print('cognee import ok')"
```

---

### Issue: `cognee.remember()` hangs or times out

**Cause:** The LLM preflight test fails because the local model is too slow to respond, or the endpoint is wrong.

**Fix:**

```bash
# Add to .env
COGNEE_SKIP_CONNECTION_TEST="true"

# Verify the LLM endpoint is reachable
curl $LLM_ENDPOINT/models
# Should return a list with the loaded model
```

---

### Issue: `Embedding dimension mismatch` or `stale vector collections`

**Cause:** The embedding model changed (e.g., different dimensions) or the vector store has stale data from a previous run.

**Fix:**

```python
import asyncio
import cognee

async def main():
    await cognee.prune.prune_system(metadata=True)

asyncio.run(main())
```

Or set a new `SYSTEM_ROOT_DIRECTORY` in `.env` to start completely fresh:

```dotenv
SYSTEM_ROOT_DIRECTORY="./cognee_data_v2"
```

---

### Issue: Cognee graph is empty after ingest

**Cause:** The `node_set` parameter wasn't passed to `remember()`, or the LLM failed to extract entities from the text.

**Fix:**

```bash
# Run the dry-run to see what Cognee would receive
python -m cognee_mem.ingest --dry-run

# Verify the payloads look correct
# Then run the full ingest with exo up
python -m cognee_mem.ingest
```

---

### Issue: `ladybug` graph adapter not found

**Cause:** Cognee 1.1.3 dropped the NetworkX adapter. `ladybug` is the new default. If an older config still references NetworkX, it fails.

**Fix:**

```bash
# Ensure .env does NOT set GRAPH_DATABASE_PROVIDER="networkx"
# Leave it unset or use:
GRAPH_DATABASE_PROVIDER="ladybug"
```

---

## Fastembed Embeddings

### Issue: `Fastembed` download fails in airplane mode

**Cause:** The embedding model was not pre-downloaded before going offline.

**Fix:**

```bash
# Do this WHILE ONLINE
python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')"

# Verify the cache
ls ~/.cache/fastembed/
```

---

### Issue: `ONNXRuntimeError` or `libonnxruntime` not found

**Cause:** Fastembed's ONNX runtime dependency is missing or incompatible with the architecture.

**Fix:**

```bash
# Reinstall with explicit architecture
pip install --force-reinstall fastembed

# On Apple Silicon, ensure you're running native (not Rosetta)
arch  # should say "arm64" on M1/M2
```

---

## Document Ingest

### Issue: `parse_document` raises `ValueError: unsupported file type`

**Cause:** The file extension is not `.pdf`, `.docx`, or `.txt`.

**Fix:**

```bash
# Check the extension
file my_document  # or check the filename
# Rename if needed
mv my_document my_document.pdf
```

---

### Issue: PDF text extraction returns empty or garbled text

**Cause:** The PDF is a scanned image (not text-based), or the font encoding is unusual.

**Fix:**

- For scanned PDFs, the pipeline cannot extract text directly. Use OCR (Tesseract) on the page images first, or use a pre-OCR'd version.
- For unusual fonts, PyMuPDF usually handles them. If not, try `page.get_text("text")` instead of the word-level extraction.

---

### Issue: `char_bboxes` lookup is empty for a PDF page

**Cause:** The PDF page has no extractable words (e.g., scanned image, or text in a vector path).

**Fix:**

- The redaction engine falls back to `page.search_for(span.text)` which is less precise but works without bboxes.
- For critical cases, consider pre-processing the PDF with OCR to extract text + positions.

---

### Issue: DOCX parse returns text with no geometry

**Cause:** This is expected. DOCX is flow-based with no fixed page geometry. The `NO_GEOMETRY_CONTRACT` means `width=height=0.0` and `word_boxes=[]`.

**Fix:**

- No fix needed — the redaction engine handles this by falling back to text-only redaction (no rectangles to draw).
- If you need spatial redaction, convert DOCX to PDF first.

---

## PII Detection

### Issue: Regex detects nothing in a document

**Cause:** The document uses non-UK formats, or the PII is in an image, or the text is garbled.

**Fix:**

- The regex patterns are UK-focused (e.g., `+44` phones, UK postcodes, NI numbers). For US documents, add US patterns:
  ```python
  _RE_SSN_US = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
  ```
- Use `--use-llm` to catch contextual PII that regex misses.
- Check if the PDF is scanned (no extractable text).

---

### Issue: LLM detects entities but offsets are wrong

**Cause:** The LLM returns chunk-relative offsets, or the model hallucinates wrong positions.

**Fix:**

- The `detect_llm` function in `pipeline/detect.py` fixes this by searching for the LLM text within the chunk text using `_find_text_in_chunk()`.
- If the text is not found, the span is dropped with a warning.
- This is expected behavior — LLM offsets are unreliable, and the search-based fix is the correct approach.

---

### Issue: Duplicate spans from regex + LLM overlap

**Cause:** Both detectors found the same PII. The merge step should deduplicate.

**Fix:**

- `merge_spans()` in `pipeline/detect.py` handles this. Regex wins on overlaps by construction (sorted first, wider spans kept).
- If you're seeing duplicates in the output, check that `merge_spans` is being called before redaction.

---

### Issue: `detect_llm` fails on one chunk but continues

**Cause:** This is by design. The function catches `RuntimeError` per chunk and logs a warning, so one bad chunk doesn't crash the whole pipeline.

**Fix:**

- Check the warning message for the specific chunk and error.
- If it's a timeout, reduce chunk size or increase timeout in `ExoClient`.
- If it's a JSON parse error, the LLM may have returned malformed output — check the raw response.

---

## Redaction Engine

### Issue: Redacted PDF still has text underneath (copy-paste test fails)

**Cause:** `page.apply_redactions()` was not called, or the redaction annotation was not added correctly.

**Fix:**

```python
import fitz
doc = fitz.open("redacted.pdf")
for page in doc:
    print(page.get_text())  # Should show pseudonyms, not original PII
```

If original text is still there:
- Check that `apply_redactions()` is called after all `add_redact_annot()` calls.
- Check that `page.apply_redactions()` is called on EVERY page, not just the first.
- Check that the PDF is saved AFTER `apply_redactions()`.

---

### Issue: Redacted PDF has black boxes but text is still selectable

**Cause:** This is the "black rectangle" anti-pattern. The annotation is a cover, not a true redaction.

**Fix:**

- Ensure `page.apply_redactions()` is called. This is the critical step that deletes the text stream.
- Verify in the code: `pipeline/redact.py` line ~115 should have `page.apply_redactions()` inside the page loop.

---

### Issue: Redacted text is misaligned or covers wrong area

**Cause:** The `char_bboxes` lookup is missing entries, or the search fallback is matching the wrong occurrence of the text.

**Fix:**

- For PDFs, ensure the ingest module built the `char_bboxes` lookup correctly. Check that `page.word_boxes` is populated.
- The search fallback (`page.search_for()`) can match the wrong instance if the text appears multiple times. The `char_bboxes` lookup is more precise.
- For DOCX/TXT, there is no geometry — this is expected. Spatial redaction only works for PDFs.

---

### Issue: `fitz` throws `RuntimeError: cannot apply redactions`  

**Cause:** The PDF is encrypted, corrupted, or the redaction annotation overlaps a complex layout (tables, multi-column text).

**Fix:**

- Check if the PDF is encrypted: `doc.is_encrypted`. If so, decrypt it first.
- For complex layouts, try flattening the page before redaction:
  ```python
  page.clean_contents()
  page.apply_redactions()
  ```
- If the PDF has form fields, flatten the form first.

---

## Image Vision (OpenCV / Tesseract)

### Issue: `TesseractNotFound` error

**Cause:** Tesseract OCR is not installed or not in PATH.

**Fix:**

- **macOS:** `brew install tesseract`
- **Linux:** `sudo apt install tesseract-ocr`
- **Windows:** Download from [UB Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
- Set the path explicitly:
  ```python
  import pytesseract
  pytesseract.pytesseract.tesseract_cmd = "/usr/local/bin/tesseract"
  ```

**Note:** The pipeline falls back to mock `TEXT_PII` regions if Tesseract is not found. The demo still works.

---

### Issue: Face detection returns no results

**Cause:** The OpenCV Haar cascade may not detect faces at certain angles, sizes, or lighting conditions. Or the image is too small.

**Fix:**

- Ensure the image is at least 100x100 pixels for the cascade to work reliably.
- Try rotating the image if the face is at an angle.
- The cascade is conservative by design. For production, consider MediaPipe or a DNN-based detector (but these need larger models).

---

### Issue: Image redaction produces blurry output for non-face regions

**Cause:** The `ImageRegion` label may be misclassified, or the bounding box is too large.

**Fix:**

- Check the `ImageRegion` objects in the detection output to verify labels and bboxes.
- Adjust the blur kernel size in `image_vision/redaction.py` if needed.

---

## Web UI (Flask)

### Issue: `Address already in use` on port 5001

**Cause:** Another process is using the port.

**Fix:**

```bash
# Find the process
lsof -i :5001
# Kill it
kill -9 <PID>

# Or use a different port
python -m web.app  # edit app.py to change port=5002
```

---

### Issue: Upload fails with `413 Request Entity Too Large`

**Cause:** The file exceeds the `MAX_CONTENT_LENGTH` limit (50 MB by default).

**Fix:**

```python
# In web/app.py, increase the limit
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB
```

---

### Issue: Job status stays at "ingesting" forever

**Cause:** The background thread crashed, but the exception wasn't caught or the job status wasn't updated.

**Fix:**

- Check the Flask console output for the exception traceback.
- Common causes: `fitz` import error, EXO not running, or file permission issues.
- Fix the root cause, restart the Flask app, and re-upload.

---

### Issue: Download button doesn't appear after processing

**Cause:** The job finished with an error, or the `output_path` was not set.

**Fix:**

- Check the job status via the API:
  ```bash
  curl http://localhost:5001/api/job/<job_id>
  ```
- If `status` is `"error"`, read the `error` field for the traceback.

---

## Synthetic Dataset Generation

### Issue: `generate.py` fails with `ModuleNotFoundError: No module named 'dataset'`

**Cause:** Running from the wrong directory. Python can't find the `dataset` package.

**Fix:**

```bash
cd /path/to/localhost-document-redaction
python -m dataset.generate  # -m ensures the project root is in PYTHONPATH
```

---

### Issue: Generated PDF is empty or has no text

**Cause:** The `DocBuilder` didn't append any text, or `reportlab` failed to render.

**Fix:**

- Check the `DocBuilder` output: `print(doc.text)` should show the full text.
- Check the `reportlab` canvas operations: `c.save()` must be called.
- If the `pad_to_words` function didn't run, the document may be too short — check `WORDS_PER_PAGE` and `target_words`.

---

### Issue: Generated images (ID card, signature) are low quality

**Cause:** Pillow's default settings produce small images. The `make_id_card` and `make_signature` functions in `dataset/images.py` may need DPI adjustment.

**Fix:**

- Increase `dpi` in the image generation functions.
- For the demo, the current quality is sufficient. For production, use higher-resolution templates.

---

## Airplane Mode / Offline Issues

### Issue: Everything works online but fails offline

**Cause:** A dependency is trying to download something on first use (model, embedding, font, etc.).

**Fix:**

1. **Pre-download all models while online:**
   ```bash
   # Fastembed
   python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')"
   # EXO model
   exo  # let it download fully
   # Ollama model
   ollama pull llama3.2:3b
   ```

2. **Check cache directories:**
   ```bash
   ls ~/.cache/huggingface/     # EXO models
   ls ~/.cache/fastembed/       # Embedding models
   ls ~/.ollama/models/         # Ollama models
   ```

3. **Test offline before the demo:**
   ```bash
   # Turn off Wi-Fi
   ping -c 1 google.com  # should fail
   # Run the full pipeline
   python -m pipeline.cli --input ... --output ... --use-llm
   ```

---

### Issue: Cognee fails offline even with pre-downloaded models

**Cause:** Cognee may try to verify the LLM connection on startup, which times out on a small local model.

**Fix:**

```bash
# Add to .env
COGNEE_SKIP_CONNECTION_TEST="true"
```

---

### Issue: `fontTools` or `reportlab` tries to download fonts

**Cause:** `reportlab` may auto-download standard fonts on first PDF generation.

**Fix:**

```bash
# Pre-generate one PDF while online to cache fonts
python -m dataset.generate
```

---

## Performance & Memory

### Issue: Pipeline runs out of memory on large documents

**Cause:** Large PDFs with many pages consume RAM during text extraction + LLM detection.

**Fix:**

- Process documents page by page instead of loading the whole document.
- Reduce chunk size (e.g., `window=60` instead of `120`) to reduce per-LLM-call memory.
- Use a smaller model (1B instead of 3B or 7B).
- Close PDFs explicitly: `doc.close()` after processing.

---

### Issue: EXO uses 100% CPU and the laptop fans spin up

**Cause:** Normal for LLM inference on CPU. Small models on laptops are CPU-intensive.

**Fix:**

- Reduce batch size: process one chunk at a time.
- Use Apple Silicon (M1/M2) with MLX — EXO auto-detects and uses the GPU.
- Use a smaller model (1B) for the demo; quality is lower but speed is acceptable.

---

### Issue: Web UI becomes unresponsive during large uploads

**Cause:** The background thread is CPU-bound (LLM detection), starving the Flask thread.

**Fix:**

- Use a process pool instead of a thread for the background worker:
  ```python
  from multiprocessing import Process
  # Replace threading.Thread with Process
  ```
- Or use a task queue (Celery, RQ) for production. For the hackathon demo, the current threading is acceptable for small documents.

---

## Still Stuck?

1. Check the **full error traceback** — don't just read the last line. The root cause is often 5 lines up.
2. Check if the issue is **environment-specific** — does it work on a colleague's machine?
3. Check the **spec.md** for the original design decisions — they may explain why something is implemented a certain way.
4. Check **PERSON_B_COGNEE_TASKS.md** for Cognee-specific issues.
5. Check **cognee-quickstart.md** for Cognee installation and configuration troubleshooting.
6. Open an issue on the repo with:
   - The exact command you ran.
   - The full error traceback.
   - Your Python version (`python --version`).
   - Your OS and architecture.
   - Whether you're online or offline.

---

> **Hackathon:** Localhost: On-Device Agent Hackathon — Dawn Capital, London — 20 June 2026
> **Last updated:** 20 June 2026
