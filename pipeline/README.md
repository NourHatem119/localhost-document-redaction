# Obscura Pipeline — Person A (Text Core)

End-to-end text pipeline: **PDF/DOCX in → redacted PDF out**.

Runs entirely on-device using [EXO Labs](https://github.com/exo-explore/exo) local LLM inference (primary) with Ollama fallback. No cloud, no network required.

---

## Architecture

```
PDF/DOCX ──► [INGEST] ──► Document (pages, images)
                │
                ▼
           [CHUNK] ──► Chunk[] (text + char offsets)
                │
                ├──► [Regex] ──► PIISpan[] (emails, phones, NI, cards)
                │
                └──► [EXO LLM] ──► PIISpan[] (names, addresses, orgs)
                            │
                            ▼
                      [MERGE] ──► PIISpan[] (deduplicated)
                            │
                            ▼
                      [REDACT] ──► redacted PDF/DOCX
```

**Key principle:** `char_start`/`char_end` offsets are sacred. Every downstream step depends on them being exact. The ingest module must preserve them from the source document.

---

## Input

The pipeline expects **two things** from the ingest module (Person A's teammate):

### 1. `Chunk` objects

One per page or logical chunk. Each must have exact character offsets into the full document text.

```python
from schema import Chunk

chunk = Chunk(
    chunk_id="chunk-0",        # unique per chunk
    doc_id="doc_0001",         # document identifier
    page_no=0,                 # source page number
    text="Dear John Smith...", # the chunk text
    char_start=0,              # absolute offset in full document
    char_end=145,              # exclusive
)
```

### 2. `char_bboxes` lookup (optional, recommended)

A mapping from `(doc_id, char_offset)` to `[(page_no, fitz.Rect), ...]` so the redaction engine can place exact bounding boxes instead of using `search_for()` fallback.

```python
char_bboxes = {
    ("doc_0001", 0): [(0, fitz.Rect(100, 500, 120, 520))],
    ("doc_0001", 1): [(0, fitz.Rect(120, 500, 140, 520))],
    # ... one entry per character offset
}
```

If you don't provide this, the redaction engine falls back to `page.search_for(span.text)` which is less precise (may miss wrapped text).

---

## Output

The pipeline returns a **redacted PDF** (or DOCX) where the underlying text is **permanently deleted**, not merely covered with a black rectangle.

```python
from pipeline.redact import apply_redactions, create_redaction_job

job = create_redaction_job(
    doc_id="doc_0001",
    spans=merged_spans,           # from detect step
    regions=image_regions,        # from Person B (optional)
    output_path="doc_0001_redacted.pdf",
)

result_path = apply_redactions(
    job,
    input_path="doc_0001.pdf",
    output_path="doc_0001_redacted.pdf",
    char_bboxes=char_bboxes,    # optional
)
# result_path → absolute path to the redacted PDF
```

**Verification:** copy-paste text from the output PDF — the redacted content is gone. This is a legal-grade redaction, not cosmetic.

---

## Quick Start — CLI

```bash
# Full end-to-end on a single PDF (uses minimal built-in ingest)
python -m pipeline.cli \
    --input dataset/docs/doc_0001/doc_0001.pdf \
    --output doc_0001_redacted.pdf
```

The CLI includes a minimal PDF parser. In production, replace it with the full ingest module from your teammate.

---

## Quick Start — Library

```python
from pipeline import ExoClient, detect_regex, detect_llm, merge_spans
from pipeline.redact import apply_redactions, create_redaction_job
from schema import Chunk

# 1. Produce chunks from your ingest module
chunks = [
    Chunk(chunk_id="c0", doc_id="d1", page_no=0, text="Dear John Smith...", char_start=0, char_end=145),
]

# 2. Detect
full_text = "".join(c.text for c in chunks)
regex_spans = detect_regex(full_text, doc_id="d1")
llm_spans = detect_llm(chunks, client=ExoClient())
merged = merge_spans(regex_spans, llm_spans)

# 3. Redact
job = create_redaction_job(doc_id="d1", spans=merged, output_path="out.pdf")
result = apply_redactions(job, input_path="in.pdf", output_path="out.pdf")
print(f"Redacted: {result}")
```

---

## EXO Integration Details

| | Value |
|---|---|
| **Primary endpoint** | `http://localhost:52415/v1/chat/completions` |
| **Fallback** | `http://localhost:11434/v1/chat/completions` (Ollama) |
| **Loaded model** | `mlx-community/Llama-3.1-Nemotron-Nano-4B-v1.1-4bit` |
| **Model size** | ~2.4 GB |
| **API** | OpenAI-compatible (`/v1/chat/completions`) |
| **Auto-load** | `ExoClient` calls `/place_instance` if node is idle |

**Environment variables:**

```bash
# Override endpoint (e.g., remote EXO node)
export OBSCURA_LLM_URL=http://192.168.1.42:52415/v1/chat/completions

# Override model (must be available on the EXO node)
export OBSCURA_LLM_MODEL=mlx-community/Llama-3.1-Nemotron-Nano-4B-v1.1-4bit

# Force Ollama fallback
export OBSCURA_USE_EXO=0
```

---

## Detection Coverage

| Type | Source | Notes |
|---|---|---|
| `EMAIL` | Regex | Fast, exact |
| `PHONE` | Regex | UK mobile format |
| `NI_NUMBER` | Regex | UK format: `AB123456C` |
| `CREDIT_CARD` | Regex | Visa, MC, Amex, Discover |
| `IBAN` | Regex | ISO 13616 format |
| `DOB` | Regex | `YYYY-MM-DD`, `DD/MM/YYYY` |
| `PERSON` | EXO LLM | Contextual names |
| `ADDRESS` | EXO LLM | Street, city, postcode |
| `ORG` | EXO LLM | Companies, institutions |
| `MEDICAL` | EXO LLM | Conditions, medications |
| `OTHER` | EXO LLM | Catch-all contextual |

Regex wins on exact matches. LLM fills in contextual gaps. Merge deduplicates overlaps.

---

## Performance (M1 Mac, 16 GB RAM)

| Step | Time |
|---|---|
| Regex detection | < 10 ms |
| LLM detection (per chunk) | ~4–6 s |
| Redaction (single page) | ~100 ms |
| **End-to-end (single page)** | **~6–7 s** |

For multi-page documents, chunk and process pages in parallel or batch LLM calls.

---

## Team Integration Notes

- **Person B (Cognee + Vision):** receives `PIISpan[]` for graph population; receives `ImageRef[]` for face/ID/signature detection. Pass `ImageRegion[]` back to `RedactionJob.regions` for compositing.
- **Person C (UI + Overmind):** calls `create_redaction_job()` after human review; writes `Trace` objects with `decisions` (auto) and `corrections` (human overrides) for the fine-tuning loop.
- **Schema:** all shapes live in `../schema.py` — never drift them without telling the team.

---

## Files

| File | Role |
|---|---|
| `exo_client.py` | OpenAI client for EXO/Ollama with auto-discovery |
| `detect.py` | Regex + LLM hybrid detection, deduplication |
| `redact.py` | PyMuPDF true redaction engine |
| `cli.py` | End-to-end CLI for quick testing |
| `__init__.py` | Public API exports |
