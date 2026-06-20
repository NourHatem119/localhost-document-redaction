# Setup Guide — Obscura

> Complete installation, configuration, and first-run instructions for getting the Obscura pipeline running locally.

This guide assumes you are on **macOS or Linux** (Windows with WSL works similarly). All steps are verified on Python 3.12.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Clone the Repository](#clone-the-repository)
3. [Create a Virtual Environment](#create-a-virtual-environment)
4. [Install Dependencies](#install-dependencies)
5. [Configure Environment Variables](#configure-environment-variables)
6. [Pre-Download Models (Online Phase)](#pre-download-models-online-phase)
7. [Start the Local LLM](#start-the-local-llm)
8. [Verify the Setup](#verify-the-setup)
9. [Generate Synthetic Documents](#generate-synthetic-documents)
10. [Run the Pipeline](#run-the-pipeline)
11. [Run the Web UI](#run-the-web-ui)
12. [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10–3.14 | 3.12 recommended |
| pip / uv | Latest | `uv` is faster; pip works fine |
| RAM | 8 GB+ | 16 GB recommended for smooth LLM inference |
| Disk | 10 GB free | Models + dataset + outputs |
| OS | macOS / Linux / WSL | Native Windows not fully tested |
| Tesseract (optional) | Latest | For image OCR. Falls back to mock if absent. |

---

## Clone the Repository

```bash
git clone git@github.com:NourHatem119/localhost-document-redaction.git
cd localhost-document-redaction
```

---

## Create a Virtual Environment

### Option A: Using `uv` (fast, recommended)

```bash
uv venv .venv
source .venv/bin/activate
```

### Option B: Using `venv` + `pip`

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

---

## Install Dependencies

### Option A: Using `uv`

```bash
uv pip install -r requirements.txt
```

### Option B: Using `pip`

```bash
pip install -r requirements.txt
```

### What gets installed

| Package | Purpose |
|---|---|
| `PyMuPDF` | PDF parsing + true redaction engine |
| `python-docx` | DOCX parsing |
| `requests` | HTTP client for EXO / Ollama |
| `Flask` | Web UI |
| `cognee[fastembed]` | Memory graph + local embeddings (no API key) |
| `Faker` | Synthetic dataset generation |
| `reportlab` | PDF rendering for synthetic documents |
| `Pillow` | Image generation for synthetic documents |
| `opencv-python` | Face / ID / signature detection in images |
| `pytesseract` | OCR for text-PII inside images (optional) |
| `presidio-analyzer` (optional) | Structured PII detection (regex is built-in) |

> **Note:** `presidio-analyzer` and `presidio-anonymizer` are commented out in `requirements.txt` — the built-in regex patterns cover the same PII types with zero extra dependency. Uncomment them if you want Presidio's additional entity types.

---

## Configure Environment Variables

Cognee reads configuration from `.env` in the project root. Copy the template and edit:

```bash
cp .env.example .env
```

Edit `.env`:

```dotenv
# ── LLM Provider: exo (OpenAI-compatible, local) ────────────────────
# exo serves an OpenAI-compatible API at :52415.
LLM_PROVIDER="openai"
LLM_MODEL="openai/llama-3.2-3b"          # Must match the model exo loaded
LLM_ENDPOINT="http://localhost:52415/v1"
LLM_API_KEY="sk-local-exo"               # exo ignores this; must be non-empty

# ── Embeddings: Fastembed (local ONNX, CPU, no API key) ───────────
# This is the offline win: no dependency on exo serving /embeddings.
EMBEDDING_PROVIDER="fastembed"
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS="384"

# ── Cognee: skip LLM preflight for small local models ─────────────
# Small local models (3B–7B) may fail the connection test due to timeout.
COGNEE_SKIP_CONNECTION_TEST="true"

# ── Graph / Vector DB (default: in-process, no setup needed) ────────
# GRAPH_DATABASE_PROVIDER="ladybug"
# VECTOR_DB_PROVIDER="lancedb"
```

### Key decisions made

| Setting | Why |
|---|---|
| **LLM → exo** | Prize eligibility + local inference. OpenAI-compatible shape. |
| **Embeddings → Fastembed** | Removes the risk that exo doesn't serve `/embeddings`. Runs fully offline. |
| **Skip connection test** | Small models are slow on the first call; the preflight times out. |
| **ladybug + lancedb** | In-process, zero DB setup, no server, no network. |

---

## Pre-Download Models (Online Phase)

**Do this while you still have internet.** Once you switch to airplane mode, model downloads will fail.

### 1. Fastembed embedding model

```bash
python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')"
```

This downloads the ONNX model into `~/.cache/fastembed/` (~100 MB). After this, embedding works offline.

### 2. EXO model (if using EXO)

```bash
git clone https://github.com/exo-explore/exo.git
cd exo
pip install -e .
exo
```

EXO auto-downloads the model on first run. The default is `mlx-community/Llama-3.1-Nemotron-Nano-4B-v1.1-4bit` (~2.4 GB). On first launch, it downloads and caches. This can take 5–10 minutes on the first run — do it once, then it's instant.

### 3. Ollama model (if using Ollama fallback)

```bash
ollama pull llama3.2:3b
ollama serve
```

The model is cached in `~/.ollama/models/` (~2 GB).

---

## Start the Local LLM

### Option A: EXO (primary)

```bash
cd exo  # if you cloned exo separately
exo
```

Verify it's running:

```bash
curl http://localhost:52415/v1/models
```

You should see a JSON response listing the loaded model.

### Option B: Ollama (fallback)

```bash
ollama serve
```

Verify:

```bash
curl http://localhost:11434/v1/models
```

### Override the endpoint

If EXO is on a different machine (e.g., a second laptop):

```bash
export OBSCURA_LLM_URL=http://192.168.1.42:52415/v1/chat/completions
```

Or force Ollama:

```bash
export OBSCURA_USE_OLLAMA=1
```

---

## Verify the Setup

### 1. Cognee smoke test

```bash
python -m cognee_mem.smoke_test
```

This runs three levels of verification:

- `--config-only` — checks `.env` loads correctly.
- `--embed-only` — verifies Fastembed imports and runs a sample embedding.
- Full — runs `remember()` + `recall()` against the local Cognee backend.

**Expected output:**

```
[ok] config loaded
[ok] Fastembed imported and embedding dimension = 384
[ok] remember() / recall() succeeded
```

### 2. Pipeline smoke test (regex only, no LLM needed)

```bash
python -m pipeline.cli \
    --input dataset/docs/doc_0001/doc_0001.pdf \
    --output /tmp/test_redacted.pdf
```

**Expected output:**

```
[INFO] Ingested: 1 page(s), 3 chunk(s), 0 image(s)
[INFO] Detected 6 spans (6 regex, 0 LLM)
[INFO] Redacted PDF written to: /tmp/test_redacted.pdf
```

### 3. Cognee memory pipeline (offline, no exo)

```bash
python -m cognee_mem.pipeline
```

**Expected output:**

```
=== MASTER MAPPING (deterministic, offline) ===
  Agent A          <- ['John Smith', 'J. Smith', 'Smith, John']
  Organization 1   <- ['Acme Corporation', 'Acme Corp', 'Acme']
  ...
[RESULT] REDACTION PASS CORRECT ✅
```

---

## Generate Synthetic Documents

If the dataset hasn't been generated yet (or you want to regenerate):

```bash
python -m dataset.generate
```

This creates 4 documents in `dataset/docs/`:

| ID | Type | Persona | Key PII Types |
|---|---|---|---|
| `doc_0001` | Employment letter | Recurring | PERSON, ORG, EMAIL, PHONE, NI, ADDRESS, DOB |
| `doc_0002` | HR onboarding | John Smith | PERSON, ORG, EMAIL, PHONE, NI, ADDRESS, DOB + images (ID, signature) |
| `doc_0003` | Bank statement | Daniel Walsh | PERSON, ORG, IBAN, CREDIT_CARD, SSN, EMAIL, PHONE, ADDRESS, DOB |
| `doc_0004` | Medical record | Priya Patel | PERSON, ORG, MEDICAL, EMAIL, PHONE, NI, ADDRESS, DOB |

Each document folder contains:
- `.pdf` — realistic input PDF
- `.docx` — same content as DOCX
- `.txt` — canonical plain text (offsets index into this)
- `.labels.json` — ground-truth PII spans (for evaluation only)
- `images/` (doc_0002 only) — embedded ID card + signature images

---

## Run the Pipeline

### CLI (single document)

```bash
# Regex-only (fast, no LLM)
python -m pipeline.cli \
    --input dataset/docs/doc_0001/doc_0001.pdf \
    --output /tmp/doc_0001_redacted.pdf

# With LLM (contextual names, addresses, orgs) — requires EXO/Ollama
python -m pipeline.cli \
    --input dataset/docs/doc_0001/doc_0001.pdf \
    --output /tmp/doc_0001_redacted.pdf \
    --use-llm
```

### Python API

```python
from ingest.parse import parse_document
from ingest.chunk import chunk_document
from pipeline.detect import detect_regex, detect_llm, merge_spans
from pipeline.redact import apply_redactions, create_redaction_job
from pipeline.exo_client import ExoClient

# 1. Ingest
doc = parse_document("dataset/docs/doc_0001/doc_0001.pdf", doc_id="doc_0001")
chunks = chunk_document(doc)

# 2. Detect
full_text = "".join(p.text for p in doc.pages)
regex_spans = detect_regex(full_text, "doc_0001")
client = ExoClient()
llm_spans = detect_llm(chunks, client)
merged = merge_spans(regex_spans, llm_spans)

# 3. Redact
job = create_redaction_job(doc_id="doc_0001", spans=merged, output_path="/tmp/redacted.pdf")
apply_redactions(job, input_path="dataset/docs/doc_0001/doc_0001.pdf", output_path="/tmp/redacted.pdf")
```

---

## Run the Web UI

```bash
python -m web.app
```

Open `http://localhost:5001` in your browser.

Features:
- **Drag-and-drop** or click to upload PDF, DOCX, or TXT.
- **Live progress** bar with status updates (ingesting, detecting, redacting).
- **PII list** showing detected entities with type and source (regex vs LLM).
- **Download** the redacted PDF when processing completes.

The backend runs the same pipeline as the CLI but in a background thread.

---

## Troubleshooting

### `No module named cognee`

The virtual environment is not activated, or `cognee` was installed in the wrong environment.

```bash
source .venv/bin/activate
which python
pip install "cognee[fastembed]==1.1.3"
```

### `LLM request failed: Connection refused`

EXO or Ollama is not running. Start one:

```bash
exo         # or
ollama serve
```

Verify the endpoint:

```bash
curl http://localhost:52415/v1/models  # EXO
curl http://localhost:11434/v1/models  # Ollama
```

### `Embedding dimension mismatch` or `stale vector collections`

Reset Cognee's local metadata:

```python
import asyncio
import cognee
async def main():
    await cognee.prune.prune_system(metadata=True)
asyncio.run(main())
```

Or set a new `SYSTEM_ROOT_DIRECTORY` in `.env` to start fresh.

### `TesseractNotFound` (image vision)

Tesseract OCR is optional. The pipeline falls back to mock `TEXT_PII` regions. To install Tesseract:

- **macOS:** `brew install tesseract`
- **Linux:** `sudo apt install tesseract-ocr`
- **Windows:** Download from [UB Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)

### `ModuleNotFoundError: No module named 'fitz'`

PyMuPDF is installed as `PyMuPDF` but imported as `fitz`. This is correct. Ensure the environment is activated:

```bash
pip install "PyMuPDF>=1.24.0"
python -c "import fitz; print(fitz.__doc__)"
```

### Slow first LLM call

Small models on CPU take 4–6 seconds per chunk. Pre-warm by running a small test:

```bash
python -c "from pipeline.exo_client import ExoClient; c = ExoClient(); print('ready:', c.base_url)"
```

### EXO model download fails on venue Wi-Fi

Pre-download **all** weights while you have internet. The model cache is at:

- `~/.cache/huggingface/` (for EXO)
- `~/.ollama/models/` (for Ollama)
- `~/.cache/fastembed/` (for embeddings)

---

## Next Steps

- See `spec.md` for the full hackathon build specification and architecture decisions.
- See `PERSON_B_COGNEE_TASKS.md` for the Cognee memory graph task breakdown.
- See `cognee-quickstart.md` for Cognee-specific troubleshooting.
- See `dataset/README.md` for synthetic dataset generation details.
- See `pipeline/README.md` for the text pipeline API reference.
- See `image_vision/README.md` for image PII detection details.
