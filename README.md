# Obscura — On-Device Enterprise Document Redaction

> **Tagline:** Privacy-grade document redaction that never leaves your machine.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Obscura** is a fully offline, air-gapped document redaction pipeline for enterprises. It ingests PDF, DOCX, and TXT files, detects PII in both text and embedded images, deduplicates and normalizes entities across documents, and produces legally redacted outputs where the underlying content is **permanently deleted** — not merely covered with a black rectangle.

Built for the **Localhost: On-Device Agent Hackathon** (Dawn Capital, London), Obscura runs entirely on-device with zero cloud dependency. During the demo, the machine is in airplane mode.

---

## Table of Contents

- [Why Obscura](#why-obscura)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Supported PII Types](#supported-pii-types)
- [Detection Sources](#detection-sources)
- [True Redaction](#true-redaction)
- [Cross-Document Consistency](#cross-document-consistency)
- [Tech Stack](#tech-stack)
- [Team & Roles](#team--roles)
- [License](#license)

---

## Why Obscura

Enterprise redaction is a real, regulated, and painful problem:

- Legal discovery firms must redact client information before public filing.
- Healthcare providers must anonymize records before research sharing.
- Financial institutions must scrub statements before audit publication.
- Government bodies must process FOIA requests with full PII removal.

The catch: the documents are **exactly** the kind of sensitive material that **cannot legally leave the building** to hit a cloud API. Redaction-as-a-service is not an option. Obscura's architecture — ingest, detect, remember, redact, improve — all on-device — is the only correct approach for the actual use case.

---

## Key Features

| Feature | Detail |
|---|---|
| **Fully Offline** | No cloud calls, no API keys, no network dependency. Runs in airplane mode. |
| **Multi-Format Ingest** | PDF, DOCX, TXT — with geometry extraction for PDFs. |
| **Hybrid Detection** | Fast regex (structured PII) + local LLM (contextual names, addresses, orgs). |
| **Image PII Detection** | Face detection, OCR text-PII, ID card layouts, signature regions via OpenCV. |
| **True Redaction** | PyMuPDF `apply_redactions()` — permanently deletes content, not cosmetic black boxes. |
| **Cross-Document Consistency** | Cognee memory graph ensures "J. Smith", "John Smith", and "Smith, John" are redacted the same way everywhere. |
| **Deterministic Redaction** | Exact offset-based replacement — no LLM in the redaction loop, reproducible and verifiable. |
| **Web UI** | Flask-based drag-and-drop interface with live progress and download. |
| **Self-Improvement** | Overmind integration captures misses and fine-tunes the model on its own failures. |
| **Synthetic Dataset** | Faker-based UK document generation with ground-truth labels for evaluation. |

---

## Architecture

```
                 ┌─────────────────────────────────────────────┐
PDF / DOCX / TXT ──▶│  INGEST → PARSE → CHUNK (offsets preserved) │  (A)
                    └───────────────┬──────────────┬──────────────┘
                                    │ text chunks  │ embedded images
                                    ▼              ▼
              ┌──────────────────────────┐   ┌──────────────────────────┐
 regex        │  TEXT PII DETECTION      │   │ IMAGE PII DETECTION      │
 Presidio     │  + EXO LLM (contextual)  │   │ OpenCV + Tesseract OCR   │  (B)
 (A)          └────────────┬─────────────┘   │ (faces / IDs / signatures)│
                           │ entities        └────────────┬─────────────┘
                           ▼                              │ image regions
                  ┌────────────────────┐                  │
                  │ COGNEE memory graph│ (B)             │
                  │ entity dedup +     │                  │
                  │ cross-doc consistency                │
                  └─────────┬──────────┘                  │
                            ▼                             ▼
                    ┌─────────────────────────────────────────────┐
                    │ REDACTION ENGINE — true removal, not cover  │ (A)
                    │ exact offset-based replacement                │
                    └───────────────────┬─────────────────────────┘
                                        ▼
                    ┌─────────────────────────────────────────────┐
                    │ WEB UI (Flask) — review, accept, reject,  │ (C)
                    │ download redacted PDF                         │
                    └───────────────────┬─────────────────────────┘
                                        │ human corrections
                                        ▼
                    ┌─────────────────────────────────────────────┐
                    │ OVERMIND — traces → failure patterns →      │ (C)
                    │ fine-tune model on misses → redeploy        │
                    └─────────────────────────────────────────────┘
```

**The three-person team split:**

- **Person A** — Pipeline & Text Core: ingest, parse, chunk, regex detection, redaction engine.
- **Person B** — Memory Graph & Image Vision: Cognee entity graph, dedup, image PII detection.
- **Person C** — App, Monitoring & Self-Improvement: Flask UI, Overmind trace capture, integration.

---

## Quick Start

### Prerequisites

- Python 3.10 or higher
- macOS, Linux, or Windows with WSL
- ~4 GB free RAM for local LLM (Llama 3.1 4B ~2.4 GB)

### Install

```bash
# Clone and enter the project
git clone git@github.com:NourHatem119/localhost-document-redaction.git
cd localhost-document-redaction

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Pre-download the Fastembed embedding model (while online)
python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')"
```

### Configure

Copy `.env.example` to `.env` and adjust:

```bash
cp .env.example .env
```

```dotenv
# LLM: exo (OpenAI-compatible, local)
LLM_PROVIDER="openai"
LLM_MODEL="openai/llama-3.2-3b"
LLM_ENDPOINT="http://localhost:52415/v1"
LLM_API_KEY="sk-local-exo"

# Embeddings: Fastembed (local ONNX, CPU, no API key)
EMBEDDING_PROVIDER="fastembed"
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS="384"

# Skip LLM connection preflight (small local models)
COGNEE_SKIP_CONNECTION_TEST="true"
```

### Run the LLM (EXO or Ollama)

```bash
# Option 1: EXO (primary)
# git clone https://github.com/exo-explore/exo && cd exo && pip install -e .
# exo

# Option 2: Ollama (fallback)
ollama pull llama3.2:3b
ollama serve
```

### Generate Synthetic Documents

```bash
# Generate 4 demo documents with realistic UK PII
python -m dataset.generate
```

This produces:
- `dataset/docs/doc_0001/` — Employment letter
- `dataset/docs/doc_0002/` — HR onboarding (with ID card + signature images)
- `dataset/docs/doc_0003/` — Bank statement
- `dataset/docs/doc_0004/` — Medical record

### Run the Web UI

```bash
# Start the Flask app
python -m web.app
# Open http://localhost:5001
```

Drop a PDF, DOCX, or TXT file. The pipeline shows progress, detected PII, and a redacted output download.

### Run the CLI

```bash
# Regex-only detection (fast, no LLM needed)
python -m pipeline.cli \
    --input dataset/docs/doc_0001/doc_0001.pdf \
    --output doc_0001_redacted.pdf

# With LLM detection (requires EXO/Ollama running)
python -m pipeline.cli \
    --input dataset/docs/doc_0001/doc_0001.pdf \
    --output doc_0001_redacted.pdf \
    --use-llm
```

### Verify True Redaction

Copy-paste text from the output PDF — the redacted content is gone. The underlying text is permanently removed, not just covered.

---

## Project Structure

```
localhost-document-redaction/
├── schema.py                    # Sacred data contract — shared types, enums, dataclasses
├── requirements.txt             # Python dependencies
├── .env.example                 # Cognee + LLM configuration template
├── spec.md                      # Hackathon build specification (the source of truth)
├── PERSON_B_COGNEE_TASKS.md     # Cognee sub-track task breakdown
├── cognee-quickstart.md         # Cognee setup quick-reference
│
├── pipeline/                    # Person A — Text pipeline
│   ├── README.md                # Pipeline documentation
│   ├── cli.py                   # End-to-end CLI entry point
│   ├── detect.py                # Regex + LLM hybrid detection, dedup
│   ├── redact.py                # PyMuPDF true redaction engine
│   └── exo_client.py            # OpenAI-compatible EXO/Ollama client
│
├── ingest/                      # Person A — Document parsing
│   ├── README.md                # Ingest documentation
│   ├── parse.py                 # Dispatcher (PDF/DOCX/TXT)
│   ├── parse_pdf.py             # PyMuPDF extraction (words + boxes + images)
│   ├── parse_docx.py            # python-docx extraction (text + images)
│   ├── parse_txt.py             # Plain text extraction
│   └── chunk.py                 # Overlapping word-window chunking
│
├── cognee_mem/                  # Person B — Memory graph & dedup
│   ├── config.py                # .env loader + Cognee config
│   ├── fixtures.py              # Mock PIISpan fixtures for TDD
│   ├── ingest.py                # PIISpan[] → Cognee graph
│   ├── dedup.py                 # Entity dedup / canonical resolution (T4)
│   ├── pseudonyms.py            # Phase 2: pseudonym assignment
│   ├── redact.py                # Phase 3: deterministic offset-based redaction
│   └── pipeline.py              # Full pipeline: run() / run_with_memory()
│
├── image_vision/                # Person B — Image PII detection
│   ├── README.md                # Vision documentation
│   ├── detector.py              # Face/OCR/ID/signature detection
│   ├── redaction.py             # Blur/pixelate redaction on images
│   └── demo.py                  # Standalone demo script
│
├── web/                         # Person C — Flask UI
│   ├── app.py                   # Flask app (upload, process, download)
│   └── templates/index.html     # Dark-mode UI with drag-and-drop
│
├── dataset/                     # Synthetic data generation
│   ├── README.md                # Dataset documentation
│   ├── schema.py                # Generation-side shapes (Persona, LabeledSpan, etc.)
│   ├── personas.py              # Recurring cast + Faker generator
│   ├── generate.py              # Document generator (PDF, DOCX, TXT + labels)
│   ├── generate_personas.py     # CLI persona generator
│   ├── filler.py                # Text padding to target word counts
│   ├── images.py                # Synthetic image generation (ID cards, signatures)
│   └── utils/                   # Helper functions for read/write
│
└── exp/                         # Experimental / analysis scripts
    ├── inspect_ingest.py        # Inspect parsed document output
    └── pdf_txt_divergence.py    # Quantify PDF vs .txt extraction gap
```

---

## Supported PII Types

| Type | Description | Detection Source |
|---|---|---|
| `PERSON` | Names (e.g., "John Smith") | LLM |
| `EMAIL` | Email addresses | Regex |
| `PHONE` | UK mobile numbers (e.g., `+44 7700 900123`) | Regex |
| `ADDRESS` | Street addresses, postcodes | LLM |
| `SSN` | Social Security Numbers | LLM |
| `NI_NUMBER` | UK National Insurance (e.g., `QQ123456C`) | Regex |
| `CREDIT_CARD` | Visa, MC, Amex, Discover | Regex |
| `IBAN` | International bank account numbers | Regex |
| `DOB` | Dates of birth (`YYYY-MM-DD`, `DD/MM/YYYY`) | Regex |
| `ORG` | Organizations, companies, institutions | LLM |
| `MEDICAL` | Conditions, medications | LLM |
| `OTHER` | Catch-all contextual PII | LLM |

---

## Detection Sources

- **Regex / Presidio** — Fast, exact, deterministic. Covers structured PII (emails, phones, NI numbers, credit cards, IBANs, DOBs). Zero model dependency.
- **LLM (EXO / Ollama)** — Contextual, covers names, addresses, organizations, medical terms, and catch-all PII. Served locally via OpenAI-compatible API.
- **Merge** — Regex wins on exact overlaps. LLM fills in gaps.

---

## True Redaction

A common redaction failure is covering text with a black rectangle. The underlying text is still extractable — copy-paste it, and it's there. Judges and auditors know this trick.

Obscura uses **PyMuPDF's `page.add_redact_annot()` + `page.apply_redactions()`**, which permanently deletes the text stream beneath the annotation. The result is a legally defensible redaction where the content is truly gone.

For images, the pipeline applies Gaussian blur + pixelation to detected regions (faces, IDs, signatures, text-PII inside images).

---

## Cross-Document Consistency

The same person may appear as "John Smith" in an employment letter, "J. Smith" in an invoice, and "Smith, John" in an HR memo. Without consistency, each variant gets redacted differently (or not at all), leaking information via reconstruction.

Obscura's Cognee memory graph solves this:

1. **Dedup** — Collapse variants into canonical entities (e.g., all "Smith" variants → one entity).
2. **Assign** — Give each canonical entity a stable pseudonym (e.g., "Agent A").
3. **Redact** — Replace every surface form with the same pseudonym across every document.

The result: "John Smith" and "J. Smith" both become "Agent A" everywhere. This is deterministic, offline, and reproducible — no LLM in the redaction loop.

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Ingest | PyMuPDF (fitz), python-docx | PDF/DOCX parsing + geometry extraction |
| Chunking | Custom (word-window overlap) | Preserve spans across chunk boundaries |
| Detection | Regex, Presidio, EXO LLM | Structured + contextual PII detection |
| Memory | Cognee (ladybug + LanceDB) | Entity graph + cross-document consistency |
| Embeddings | Fastembed (ONNX, CPU) | Local embeddings, no cloud, no API key |
| Vision | OpenCV, Tesseract OCR | Face, ID, signature, text-PII in images |
| Redaction | PyMuPDF `apply_redactions()` | True deletion, not cosmetic cover |
| UI | Flask + vanilla HTML/JS | Drag-and-drop upload, progress, download |
| LLM | EXO / Ollama | Local inference, OpenAI-compatible API |
| Self-Improvement | Overmind | Trace capture, failure analysis, fine-tuning |
| Dataset | Faker (en_GB), reportlab, pillow | Synthetic UK document generation |

---

## Team & Roles

Built by a 3-person team during the Localhost: On-Device Agent Hackathon (7.5-hour build window).

| Role | Responsibilities | Key Deliverables |
|---|---|---|
| **Person A** — Pipeline & Text Core | Ingest, parse, chunk, detect, redact | `pipeline/`, `ingest/` |
| **Person B** — Memory Graph & Vision | Cognee graph, entity dedup, image PII | `cognee_mem/`, `image_vision/` |
| **Person C** — App, Monitoring & Self-Improvement | Flask UI, Overmind integration, demo | `web/`, integration scripts |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

> **Hackathon:** Localhost: On-Device Agent Hackathon — Dawn Capital, London — 20 June 2026
> **Status:** Built and verified offline. All core functionality runs in airplane mode.
