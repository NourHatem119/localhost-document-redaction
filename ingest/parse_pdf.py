"""PDF ingest via PyMuPDF (fitz).

PyMuPDF's extraction is treated as the canonical text for a PDF (we do not
align against any external .txt). Page text is rebuilt by joining positioned
words, so each word's [char_start, char_end) range and its bounding box come
from the same pass and cannot drift apart. Embedded images are written to disk
and handed off to Person B via ImageRef.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import fitz

from schema import BBox, Document, ImageRef, Page, WordBox

# Index into the tuple returned by page.get_text("words"):
#   (x0, y0, x1, y1, word, block_no, line_no, word_no)
_WORDS_SORT_KEY = (5, 6, 7)  # block, line, word — i.e. reading order


def _page_text_and_boxes(page: "fitz.Page") -> Tuple[str, List[WordBox]]:
    """Build canonical page text from positioned words.

    Words are joined with single spaces within a line and newlines between
    lines, so the flat text stays readable while every word keeps an exact
    char range aligned to its bbox.
    """
    words = page.get_text("words")
    if not words:
        return "", []

    words = sorted(words, key=lambda w: (w[5], w[6], w[7]))

    parts: List[str] = []
    boxes: List[WordBox] = []
    cursor = 0
    prev_line_key: Tuple[int, int] | None = None

    for x0, y0, x1, y1, word, block_no, line_no, _word_no in words:
        line_key = (block_no, line_no)
        if prev_line_key is None:
            sep = ""
        elif line_key != prev_line_key:
            sep = "\n"
        else:
            sep = " "
        if sep:
            parts.append(sep)
            cursor += len(sep)

        start = cursor
        parts.append(word)
        cursor += len(word)
        end = cursor

        bbox: BBox = (x0, y0, x1, y1)
        boxes.append((start, end, bbox))
        prev_line_key = line_key

    return "".join(parts), boxes


def _extract_images(
    doc: "fitz.Document", page: "fitz.Page", doc_id: str, out_dir: Path
) -> List[ImageRef]:
    """Extract embedded images on a page to disk, recording placement bbox."""
    refs: List[ImageRef] = []
    page_no = page.number + 1
    for img in page.get_images(full=True):
        xref = img[0]
        extracted = doc.extract_image(xref)
        ext = extracted.get("ext", "png")
        image_id = f"{doc_id}_p{page_no}_x{xref}"
        out_path = out_dir / f"{image_id}.{ext}"
        out_path.write_bytes(extracted["image"])

        rects = page.get_image_rects(xref)
        bbox: BBox | None = None
        if rects:
            r = rects[0]
            bbox = (r.x0, r.y0, r.x1, r.y1)

        refs.append(
            ImageRef(
                image_id=image_id,
                doc_id=doc_id,
                page_no=page_no,
                path=str(out_path),
                bbox=bbox,
            )
        )
    return refs


def parse_pdf(path: str | Path, doc_id: str, image_dir: str | Path) -> Document:
    """Parse a PDF into a Document with per-page canonical text + word boxes,
    and extract embedded images to `image_dir`.
    """
    path = Path(path)
    out_dir = Path(image_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(path)
    pages: List[Page] = []
    images: List[ImageRef] = []
    try:
        for page in doc:
            text, boxes = _page_text_and_boxes(page)
            pages.append(
                Page(
                    page_no=page.number + 1,
                    width=page.rect.width,
                    height=page.rect.height,
                    text=text,
                    word_boxes=boxes,
                )
            )
            images.extend(_extract_images(doc, page, doc_id, out_dir))
    finally:
        doc.close()

    return Document(
        doc_id=doc_id, source_path=str(path), pages=pages, images=images
    )
