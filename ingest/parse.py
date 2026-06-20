"""Ingest dispatcher: route a file to the right parser by extension.

Returns a contract `Document`. Use `chunk_document` (ingest.chunk) to produce
offset-preserving chunks for detection.
"""

from __future__ import annotations

from pathlib import Path

from schema import Document

from ingest.parse_docx import parse_docx
from ingest.parse_pdf import parse_pdf
from ingest.parse_txt import parse_txt

SUPPORTED_SUFFIXES = (".pdf", ".docx", ".txt")


def parse_document(
    path: str | Path, doc_id: str | None = None, image_dir: str | Path | None = None
) -> Document:
    """Parse a PDF, DOCX or TXT into a Document, extracting images to `image_dir`.

    `doc_id` defaults to the file stem; `image_dir` defaults to a sibling
    `extracted/<doc_id>/` folder next to the source file. Plain text has no
    images, so `image_dir` is unused for `.txt`.
    """
    path = Path(path)
    doc_id = doc_id or path.stem
    if image_dir is None:
        image_dir = path.parent / "extracted" / doc_id

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path, doc_id, image_dir)
    if suffix == ".docx":
        return parse_docx(path, doc_id, image_dir)
    if suffix == ".txt":
        return parse_txt(path, doc_id)
    raise ValueError(
        f"unsupported file type: {suffix!r} (expected one of {SUPPORTED_SUFFIXES})"
    )
