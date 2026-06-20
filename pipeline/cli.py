"""End-to-end pipeline CLI: PDF/DOCX/TXT in → redacted PDF out.

Usage:
    python -m pipeline.cli --input dataset/docs/doc_0001/doc_0001.pdf --output out.pdf

This is the MVP integration path for Person A. It uses the full ingest module
(parse_document + chunk_document) which produces proper chunks with exact
char offsets and, for PDF, word-level bounding boxes.
"""

from __future__ import annotations

import argparse
import os
import sys

from ingest.parse import parse_document
from ingest.chunk import chunk_document
from pipeline.detect import detect_regex, detect_llm, merge_spans
from pipeline.redact import apply_redactions, create_redaction_job
from pipeline.exo_client import ExoClient


def run(input_path: str, output_path: str) -> str:
    """Run the full pipeline on a single document."""
    doc_id = os.path.splitext(os.path.basename(input_path))[0]

    # 1. Ingest (full module: PDF, DOCX, TXT).
    document = parse_document(input_path, doc_id=doc_id)
    chunks = chunk_document(document)
    print(f"[INFO] Ingested: {len(document.pages)} page(s), {len(chunks)} chunk(s), {len(document.images)} image(s)")

    # Build char→bbox lookup from word boxes (PDF only; DOCX/TXT have no geometry).
    char_bboxes = {}
    doc_offset = 0
    for page in document.pages:
        for char_start, char_end, bbox in page.word_boxes:
            for offset in range(char_start + doc_offset, char_end + doc_offset):
                key = (doc_id, offset)
                char_bboxes.setdefault(key, []).append((page.page_no, bbox))
        doc_offset += len(page.text)
    # 2. Detect.
    full_text = "".join(p.text for p in document.pages)
    regex_spans = detect_regex(full_text, doc_id)

    client = ExoClient()
    llm_spans = detect_llm(chunks, client)

    merged = merge_spans(regex_spans, llm_spans)
    print(f"[INFO] Detected {len(merged)} spans ({len(regex_spans)} regex, {len(llm_spans)} LLM)")

    # 3. Redact.
    job = create_redaction_job(doc_id=doc_id, spans=merged, output_path=output_path)
    result_path = apply_redactions(job, input_path, output_path, char_bboxes=char_bboxes if char_bboxes else None)
    print(f"[INFO] Redacted PDF written to: {result_path}")
    return result_path


def main():
    parser = argparse.ArgumentParser(description="Obscura redaction pipeline")
    parser.add_argument("--input", required=True, help="Path to input PDF/DOCX/TXT")
    parser.add_argument("--output", required=True, help="Path to output PDF")
    args = parser.parse_args()

    run(args.input, args.output)


if __name__ == "__main__":
    main()
