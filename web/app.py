"""Obscura Web UI — Flask app for local document redaction.

Full pipeline: PDF/DOCX/TXT → ingest → detect (regex + LLM) →
Cognee text redaction (cross-doc consistent pseudonyms) → PDF true redaction →
Cognee memory graph (optional, needs exo).

Output layout:
    dataset/output/redacted/      — redacted PDFs/TXTs
    dataset/output/pseudonyms/    — entity→pseudonym JSON mappings
    dataset/output/extracted/     — images extracted during ingest (redirected)

Run:
    python -m web.app

Then open http://localhost:5000 in your browser.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, render_template, request, jsonify, send_file, abort

from ingest.parse import parse_document
from ingest.chunk import chunk_document
from pipeline.detect import detect_regex, detect_llm, merge_spans
from pipeline.redact import apply_redactions, create_redaction_job
from pipeline.exo_client import ExoClient
from schema import PIISpan, Document
from cognee_mem.pipeline import (
    PipelineOutput, run as cognee_run, run_with_memory_first
)
from cognee_mem.pseudonyms import MasterMapping
from cognee_mem.memory_store import (
    add_to_memory, get_memory_stats, MEMORY_PATH
)


app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB uploads

# In-memory job store (for demo — no DB needed).
_jobs: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Output directories — clean, organised layout
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_DIR = _PROJECT_ROOT / "dataset" / "output"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(_OUTPUT_DIR / "redacted").mkdir(exist_ok=True)
(_OUTPUT_DIR / "pseudonyms").mkdir(exist_ok=True)
(_OUTPUT_DIR / "extracted").mkdir(exist_ok=True)

# Uploads still land in temp (they are originals, not outputs).
_UPLOAD_DIR = Path(tempfile.gettempdir()) / "obscura_uploads"
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _use_llm() -> bool:
    return os.environ.get("OBSCURA_USE_LLM", "true").lower() in ("1", "true", "yes")


def _use_cognee_memory() -> bool:
    return os.environ.get("OBSCURA_USE_COGNEE_MEMORY", "true").lower() in ("1", "true", "yes")


def _mapping_to_dict(mapping: MasterMapping) -> List[dict]:
    """Convert MasterMapping to a flat list for the UI."""
    out = []
    for ent, pseudonym in mapping.entities:
        out.append({
            "canonical_id": ent.canonical_id,
            "type": ent.type.value,
            "canonical_text": ent.canonical_text,
            "aliases": list(ent.aliases),
            "pseudonym": pseudonym,
            "doc_ids": list(ent.doc_ids),
        })
    return out


def _span_to_dict(span: PIISpan) -> dict:
    return {
        "type": span.type.value,
        "text": span.text,
        "source": span.source,
        "confidence": round(span.confidence, 2),
    }


def _set_job_status(job: dict, status: str, progress: int, **extra: Any) -> None:
    job["status"] = status
    job["progress"] = progress
    job.update(extra)


def _build_output_paths(original_filename: str, job_id: str) -> tuple[str, str, str]:
    """Return (redacted_path, pseudonym_path, extracted_dir) based on the original filename."""
    stem = Path(original_filename).stem
    suffix = Path(original_filename).suffix.lower()
    
    # Redacted output: same suffix as input for PDF, .txt for others.
    redacted_ext = suffix if suffix == ".pdf" else ".txt"
    redacted_path = str(_OUTPUT_DIR / "redacted" / f"{stem}_redacted{redacted_ext}")
    
    # Pseudonym mapping: always JSON.
    pseudonym_path = str(_OUTPUT_DIR / "pseudonyms" / f"{stem}_pseudonyms.json")
    
    # Extracted images land in a per-document subfolder.
    extracted_dir = str(_OUTPUT_DIR / "extracted" / stem)
    
    return redacted_path, pseudonym_path, extracted_dir

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    """Return EXO / Ollama connection status + pipeline config."""
    try:
        client = ExoClient()
        return jsonify({
            "status": "ok",
            "endpoint": client.base_url,
            "model": client.model,
            "llm_enabled": _use_llm(),
            "cognee_memory_enabled": _use_cognee_memory(),
        })
    except RuntimeError as exc:
        return jsonify({
            "status": "error",
            "message": str(exc),
            "llm_enabled": _use_llm(),
            "cognee_memory_enabled": _use_cognee_memory(),
        }), 503


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Accept a file upload and return a job ID."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".pdf", ".docx", ".txt"):
        return jsonify({"error": f"Unsupported file type: {suffix}"}), 400

    job_id = f"job_{int(time.time() * 1000)}"
    upload_path = _UPLOAD_DIR / f"{job_id}{suffix}"
    file.save(str(upload_path))

    _jobs[job_id] = {
        "id": job_id,
        "filename": file.filename,
        "upload_path": str(upload_path),
        "status": "uploaded",
        "progress": 0,
        "spans": [],
        "output_path": None,
        "redacted_text": None,
        "mapping": [],
        "error": None,
        "stages": {},
    }

    # Start processing in background thread.
    thread = threading.Thread(target=_process_job, args=(job_id,))
    thread.daemon = True
    thread.start()

    return jsonify({"job_id": job_id, "status": "uploaded"})


@app.route("/api/job/<job_id>")
def api_job(job_id: str):
    """Poll for job status."""
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({
        "id": job["id"],
        "status": job["status"],
        "progress": job["progress"],
        "filename": job["filename"],
        "spans": [_span_to_dict(s) for s in job["spans"]],
        "mapping": job.get("mapping", []),
        "output_path": job["output_path"],
        "redacted_text": job.get("redacted_text"),
        "stages": job.get("stages", {}),
        "error": job["error"],
    })


@app.route("/api/download/<job_id>")
def api_download(job_id: str):
    """Download the redacted output (PDF or text)."""
    job = _jobs.get(job_id)
    if not job or not job["output_path"]:
        abort(404)
    return send_file(
        job["output_path"],
        as_attachment=True,
        download_name=f"{job['filename']}_redacted{Path(job['output_path']).suffix}",
    )


@app.route("/api/download_text/<job_id>")
def api_download_text(job_id: str):
    """Download the redacted text as a .txt file."""
    job = _jobs.get(job_id)
    if not job or not job.get("redacted_text"):
        abort(404)
    txt_path = _UPLOAD_DIR / f"{job_id}_redacted.txt"
    txt_path.write_text(job["redacted_text"], encoding="utf-8")
    return send_file(
        str(txt_path),
        as_attachment=True,
        download_name=f"{job['filename']}_redacted.txt",
    )


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

def _process_job(job_id: str) -> None:
    """Background worker: memory-first pipeline.
    
    1. Ingest document
    2. Regex detect (fast)
    3. Check memory - skip known entities
    4. LLM detect only on unknown chunks (limited to avoid timeout)
    5. Merge spans + dedup + pseudonym assignment
    6. Store new entities in memory
    7. Redact text deterministically
    8. True PDF redaction with pseudonyms
    """
    job = _jobs[job_id]
    doc_id = job_id
    stages: dict = {}

    try:
        # ------------------------------------------------------------------
        # 1. INGESTION
        # ------------------------------------------------------------------
        _set_job_status(job, "ingesting", 5, stages=stages)
        document = parse_document(job["upload_path"], doc_id=doc_id)
        chunks = chunk_document(document)

        # Build char→bbox lookup from word boxes (PDF only; DOCX/TXT have no geometry).
        char_bboxes: Dict[tuple, list] = {}
        doc_offset = 0
        for page in document.pages:
            for char_start, char_end, bbox in page.word_boxes:
                for offset in range(char_start + doc_offset, char_end + doc_offset):
                    key = (doc_id, offset)
                    char_bboxes.setdefault(key, []).append((page.page_no, bbox))
            doc_offset += len(page.text)

        stages["ingestion"] = {
            "pages": len(document.pages),
            "chunks": len(chunks),
            "images": len(document.images),
            "has_geometry": len(document.pages) > 0 and document.pages[0].width > 0,
        }
        _set_job_status(job, "ingesting", 20, stages=stages)

        # ------------------------------------------------------------------
        # 2. REGEX DETECTION (fast, exact)
        # ------------------------------------------------------------------
        full_text = "".join(p.text for p in document.pages)
        stages["text_length"] = len(full_text)

        _set_job_status(job, "detecting_regex", 30, stages=stages)
        regex_spans = detect_regex(full_text, doc_id)
        stages["regex_spans"] = len(regex_spans)

        # ------------------------------------------------------------------
        # 3. MEMORY-FIRST: Find known entities in text + check memory
        # ------------------------------------------------------------------
        _set_job_status(job, "checking_memory", 35, stages=stages)
        
        # Find known entities from memory that appear in this document
        from cognee_mem.memory_store import find_known_spans_in_text
        memory_spans = find_known_spans_in_text(full_text, doc_id)
        stages["memory_spans"] = len(memory_spans)
        
        # Merge all spans before running the pipeline
        all_spans = list({s.span_id: s for s in (regex_spans + memory_spans)}.values())
        
        # Select chunks to run LLM on (unknown content, limited to 3)
        unknown_chunks = []
        if _use_llm():
            unknown_chunks = [c for c in chunks if len(c.text) > 30][:3]
            stages["llm_chunks"] = len(unknown_chunks)
        
        # Run the full memory-first pipeline
        pipeline_out = run_with_memory_first(
            doc_texts={doc_id: full_text},
            spans=all_spans,
            memory_path=None,
            llm_chunks=unknown_chunks if _use_llm() else None,
        )
        
        stages["memory_entities"] = pipeline_out.memory_stats["total_entities"]
        stages["memory_after"] = get_memory_stats()["total_entities"]
        stages["new_entities"] = stages["memory_after"] - stages["memory_entities"]
        stages["llm_spans"] = len(pipeline_out.llm_spans)
        
        # Persist new entities to memory for future runs
        add_to_memory(pipeline_out.mapping)
        stages["memory_after"] = get_memory_stats()["total_entities"]
        stages["new_entities"] = stages["memory_after"] - stages["memory_entities"]
        
        all_spans = list({s.span_id: s for s in (regex_spans + memory_spans + pipeline_out.llm_spans)}.values())
        job["spans"] = all_spans
        stages["total_spans"] = len(all_spans)
        stages["regex_spans"] = len(regex_spans)
        
        redacted_text = pipeline_out.redacted_texts.get(doc_id, full_text)
        job["redacted_text"] = redacted_text
        job["mapping"] = _mapping_to_dict(pipeline_out.mapping)
        stages["replacements"] = sum(r.replacements for r in pipeline_out.results)
        stages["entities"] = len(pipeline_out.mapping.entities)

        _set_job_status(job, "redacting_text", 70, stages=stages)

        # ------------------------------------------------------------------
        # 6. REDACTED PDF OUTPUT (with pseudonyms)
        # ------------------------------------------------------------------
        _set_job_status(job, "redacting_pdf", 75, stages=stages)
        redacted_path, pseudonym_path, extracted_dir = _build_output_paths(
            job["filename"], job_id
        )
        Path(redacted_path).parent.mkdir(parents=True, exist_ok=True)
        Path(pseudonym_path).parent.mkdir(parents=True, exist_ok=True)
        Path(extracted_dir).mkdir(parents=True, exist_ok=True)

        is_pdf = Path(job["upload_path"]).suffix.lower() == ".pdf"
        if is_pdf:
            redact_job = create_redaction_job(
                doc_id=doc_id,
                spans=all_spans,
                output_path=redacted_path,
            )
            apply_redactions(
                redact_job,
                input_path=job["upload_path"],
                output_path=redacted_path,
                char_bboxes=char_bboxes if char_bboxes else None,
                mapping=pipeline_out.mapping,
            )
            job["output_path"] = redacted_path
            stages["pdf_output"] = redacted_path
        else:
            Path(redacted_path).write_text(redacted_text, encoding="utf-8")
            job["output_path"] = redacted_path
            stages["text_output"] = redacted_path

        _set_job_status(job, "writing_outputs", 85, stages=stages)

        # Write pseudonym mapping
        with open(pseudonym_path, "w", encoding="utf-8") as f:
            json.dump({
                "doc_id": doc_id,
                "original_filename": job["filename"],
                "entities": job["mapping"],
                "redacted_path": redacted_path,
            }, f, indent=2)
        stages["pseudonym_path"] = pseudonym_path

        # ------------------------------------------------------------------
        # 7. COGNEE MEMORY GRAPH (optional, real cognee package)
        # ------------------------------------------------------------------
        if _use_cognee_memory():
            _set_job_status(job, "building_memory", 90, stages=stages)
            try:
                import asyncio
                asyncio.run(_run_cognee_with_memory({doc_id: full_text}, all_spans))
                stages["cognee_memory"] = "ok"
            except Exception as exc:
                stages["cognee_memory"] = f"skipped: {str(exc)}"
        else:
            stages["cognee_memory"] = "disabled (using JSON memory)"

        _set_job_status(job, "done", 100, stages=stages)

    except Exception as exc:
        traceback_str = traceback.format_exc()
        job["status"] = "error"
        job["error"] = f"{exc}\n\n{traceback_str}"
        job["progress"] = 0
        stages["error"] = str(exc)
        job["stages"] = stages


async def _run_cognee_with_memory(doc_texts: dict, spans: List[PIISpan]) -> None:
    """Async wrapper to run Cognee memory pipeline."""
    from cognee_mem.pipeline import run_with_memory
    await run_with_memory(doc_texts, spans)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
