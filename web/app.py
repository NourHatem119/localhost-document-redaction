"""Obscura Web UI — Flask app for local document redaction.

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
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from flask import Flask, render_template, request, jsonify, send_file, abort

from ingest.parse import parse_document
from ingest.chunk import chunk_document
from pipeline.detect import detect_regex, detect_llm, merge_spans
from pipeline.redact import apply_redactions, create_redaction_job
from pipeline.exo_client import ExoClient
from schema import PIISpan

app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB uploads

# In-memory job store (for demo — no DB needed).
_jobs: dict[str, dict] = {}
_UPLOAD_DIR = Path(tempfile.gettempdir()) / "obscura_uploads"
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    """Return EXO / Ollama connection status."""
    try:
        client = ExoClient()
        return jsonify({
            "status": "ok",
            "endpoint": client.base_url,
            "model": client.model,
        })
    except RuntimeError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 503


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
        "error": None,
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
        "spans": [span_to_dict(s) for s in job["spans"]],
        "output_path": job["output_path"],
        "error": job["error"],
    })


@app.route("/api/download/<job_id>")
def api_download(job_id: str):
    """Download the redacted PDF."""
    job = _jobs.get(job_id)
    if not job or not job["output_path"]:
        abort(404)
    return send_file(job["output_path"], as_attachment=True, download_name=f"{job['filename']}_redacted.pdf")


def _process_job(job_id: str) -> None:
    """Background worker: ingest → detect → redact."""
    job = _jobs[job_id]
    try:
        job["status"] = "ingesting"
        job["progress"] = 10

        doc = parse_document(job["upload_path"], doc_id=job_id)
        chunks = chunk_document(doc)
        job["progress"] = 25

        # Build char→bbox lookup (PDF only).
        char_bboxes = {}
        doc_offset = 0
        for page in doc.pages:
            for c_start, c_end, bbox in page.word_boxes:
                for offset in range(c_start + doc_offset, c_end + doc_offset):
                    key = (job_id, offset)
                    char_bboxes.setdefault(key, []).append((page.page_no, bbox))
            doc_offset += len(page.text)

        job["status"] = "detecting"
        job["progress"] = 40

        full_text = "".join(p.text for p in doc.pages)
        regex_spans = detect_regex(full_text, job_id)
        job["progress"] = 60

        client = ExoClient()
        llm_spans = detect_llm(chunks, client)
        job["progress"] = 80

        merged = merge_spans(regex_spans, llm_spans)
        job["spans"] = merged
        job["status"] = "redacting"

        output_path = str(_UPLOAD_DIR / f"{job_id}_redacted.pdf")
        redact_job = create_redaction_job(
            doc_id=job_id,
            spans=merged,
            output_path=output_path,
        )
        apply_redactions(
            redact_job,
            input_path=job["upload_path"],
            output_path=output_path,
            char_bboxes=char_bboxes if char_bboxes else None,
        )

        job["output_path"] = output_path
        job["status"] = "done"
        job["progress"] = 100

    except Exception as exc:
        job["status"] = "error"
        job["error"] = str(exc)
        job["progress"] = 0


def span_to_dict(span: PIISpan) -> dict:
    return {
        "type": span.type.value,
        "text": span.text,
        "source": span.source,
        "confidence": round(span.confidence, 2),
    }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
