"""Plain-text ingest.

A .txt file has no page geometry and no embedded images: the whole file is one
logical page with canonical text only (no word boxes). This mirrors the DOCX
treatment minus image extraction. Spatial redaction does not apply.
"""

from __future__ import annotations

from pathlib import Path

from schema import Document, Page

# No page dimensions for plain text; 0.0 signals "no geometry".
_NO_GEOMETRY = 0.0


def parse_txt(path: str | Path, doc_id: str) -> Document:
    """Parse a .txt file into a single-page, text-only Document."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    # NO-GEOMETRY CONTRACT: TXT has no page geometry, so width/height are 0.0
    # and word_boxes is empty. Downstream redaction must treat this as a
    # text-only source (no rectangles to draw); branch on `word_boxes == []`
    # / `bbox is None` rather than assuming every page has boxes.
    page = Page(
        page_no=1,
        width=_NO_GEOMETRY,
        height=_NO_GEOMETRY,
        text=text,
        word_boxes=[],
    )
    return Document(
        doc_id=doc_id, source_path=str(path), pages=[page], images=[]
    )
