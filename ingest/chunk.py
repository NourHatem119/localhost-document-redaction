"""Word-window chunking over canonical page text.

Chunks are built from fixed-size, overlapping word windows so that a span
straddling a window boundary is still wholly contained in a neighbouring
window. Each Chunk carries char_start/char_end indexing into the page's
canonical text — offsets are preserved end to end. When the page has word
boxes (PDF), a chunk also gets a union bbox covering its words.
"""

from __future__ import annotations

import re
from typing import List, Optional

from schema import BBox, Chunk, Document, Page, WordBox

# Defaults: ~120-word windows with 20-word overlap. Big enough to give the LLM
# real context for names/addresses, small enough to keep prompts cheap.
DEFAULT_WINDOW = 120
DEFAULT_OVERLAP = 20

_WORD_RE = re.compile(r"\S+")


def _word_spans(text: str) -> List[tuple[int, int]]:
    """Char offsets of each whitespace-delimited word in `text`."""
    return [(m.start(), m.end()) for m in _WORD_RE.finditer(text)]


def _union_bbox(boxes: List[WordBox], start: int, end: int) -> Optional[BBox]:
    """Union of word boxes whose char ranges fall within [start, end)."""
    # NO-GEOMETRY CONTRACT: DOCX/TXT pages have no word_boxes, so `boxes` is
    # empty and this returns None — the chunk's bbox is None. Consumers must
    # handle bbox=None as a text-only source (no rectangles to draw).
    inside = [b for (s, e, b) in boxes if s >= start and e <= end]
    if not inside:
        return None
    x0 = min(b[0] for b in inside)
    y0 = min(b[1] for b in inside)
    x1 = max(b[2] for b in inside)
    y1 = max(b[3] for b in inside)
    return (x0, y0, x1, y1)


def chunk_page(
    page: Page,
    doc_id: str,
    doc_offset: int = 0,
    window: int = DEFAULT_WINDOW,
    overlap: int = DEFAULT_OVERLAP,
) -> List[Chunk]:
    """Chunk one page's canonical text into overlapping word windows.

    `doc_offset` is the cumulative character offset of this page within the
    full document. All chunk `char_start`/`char_end` values are document-level.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    if not 0 <= overlap < window:
        raise ValueError("overlap must be >= 0 and < window")

    spans = _word_spans(page.text)
    if not spans:
        return []

    step = window - overlap
    chunks: List[Chunk] = []
    idx = 0
    for w_start in range(0, len(spans), step):
        w_slice = spans[w_start : w_start + window]
        if not w_slice:
            break
        char_start = w_slice[0][0] + doc_offset
        char_end = w_slice[-1][1] + doc_offset
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}_p{page.page_no}_c{idx}",
                doc_id=doc_id,
                page_no=page.page_no,
                text=page.text[w_slice[0][0] : w_slice[-1][1]],
                char_start=char_start,
                char_end=char_end,
                bbox=_union_bbox(page.word_boxes, w_slice[0][0], w_slice[-1][1]),
            )
        )
        idx += 1
        # Last window reached the end; stop to avoid a trailing dup chunk.
        if w_start + window >= len(spans):
            break
    return chunks


def chunk_document(
    doc: Document,
    window: int = DEFAULT_WINDOW,
    overlap: int = DEFAULT_OVERLAP,
) -> List[Chunk]:
    """Chunk every page of a Document, preserving document-level offsets."""
    chunks: List[Chunk] = []
    doc_offset = 0
    for page in doc.pages:
        page_chunks = chunk_page(page, doc.doc_id, doc_offset, window, overlap)
        chunks.extend(page_chunks)
        doc_offset += len(page.text)
    return chunks
