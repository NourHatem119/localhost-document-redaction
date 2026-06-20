"""End-to-end pipeline CLI: PDF in → redacted PDF out.

Usage:
    python -m pipeline.cli --input dataset/docs/doc_0001/doc_0001.pdf --output out.pdf

This is the MVP integration path for Person A. It expects the input PDF to
already have extractable text (PyMuPDF handles this). In production, the
ingest module (Person A track) parses the PDF, builds chunks + char→bbox
lookup, and feeds it here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import fitz  # PyMuPDF

from schema import Chunk, Document
from pipeline.detect import detect_regex, detect_llm, merge_spans
from pipeline.redact import apply_redactions, create_redaction_job
from pipeline.exo_client import ExoClient


def _pdf_to_document(pdf_path: str, doc_id: str) -> Document:
    """Minimal PDF ingest: extract text per page, no images."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_no in range(len(doc)):
        page = doc.load_page(page_no)
        pages.append(
            {
                "page_no": page_no,
                "width": page.rect.width,
                "height": page.rect.height,
                "text": page.get_text(),
            }
        )
    doc.close()
    return Document(
        doc_id=doc_id,
        source_path=pdf_path,
        pages=pages,
    )


def _chunk_document(document: Document) -> list[Chunk]:
    """Simple chunking: one chunk per page."""
    chunks = []
    char_pos = 0
    for page in document.pages:
        text = page["text"]
        chunks.append(
            Chunk(
                chunk_id=f"chunk-{page['page_no']}",
                doc_id=document.doc_id,
                page_no=page["page_no"],
                text=text,
                char_start=char_pos,
                char_end=char_pos + len(text),
            )
        )
        char_pos += len(text)
    return chunks


def run(input_path: str, output_path: str) -> str:
    """Run the full pipeline on a single PDF."""
    doc_id = os.path.splitext(os.path.basename(input_path))[0]

    # 1. Ingest (minimal, to be replaced by full ingest module).
    document = _pdf_to_document(input_path, doc_id)
    chunks = _chunk_document(document)

    # 2. Detect.
    full_text = "".join(p["text"] for p in document.pages)
    regex_spans = detect_regex(full_text, doc_id)

    client = ExoClient()
    llm_spans = detect_llm(chunks, client)

    merged = merge_spans(regex_spans, llm_spans)
    print(f"[INFO] Detected {len(merged)} spans ({len(regex_spans)} regex, {len(llm_spans)} LLM)")

    # 3. Redact.
    job = create_redaction_job(doc_id=doc_id, spans=merged, output_path=output_path)
    result_path = apply_redactions(job, input_path, output_path)
    print(f"[INFO] Redacted PDF written to: {result_path}")
    return result_path


def main():
    parser = argparse.ArgumentParser(description="Obscura redaction pipeline")
    parser.add_argument("--input", required=True, help="Path to input PDF")
    parser.add_argument("--output", required=True, help="Path to output PDF")
    args = parser.parse_args()

    run(args.input, args.output)


if __name__ == "__main__":
    main()
