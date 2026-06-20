"""True redaction engine using PyMuPDF.

Uses `page.add_redact_annot()` + `page.apply_redactions()` so the underlying
text is permanently removed, not merely covered with a black rectangle.

Accepts a RedactionJob (spans + image regions) and produces a redacted PDF.

Important: the pipeline must map text offsets to page bboxes. This module
provides two strategies:
  1. `apply_redactions` — if the ingest module already built a char→bbox
     lookup, use it directly.
  2. `apply_redactions_search` — fallback that uses PyMuPDF `search_for()` to
     find text on the page, slower but works without a lookup table.
"""

from __future__ import annotations

import os
import uuid
from typing import List, Optional, Tuple

import fitz  # PyMuPDF

from schema import PIISpan, ImageRegion, RedactionJob, BBox


# Colour for redaction rectangles (black).
_REDACT_FILL = (0, 0, 0)
_REDACT_FILL_IMAGE = (0, 0, 0)


def _to_fitz_rect(bbox: BBox) -> fitz.Rect:
    return fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])


def apply_redactions(
    job: RedactionJob,
    input_path: str,
    output_path: Optional[str] = None,
    char_bboxes: Optional[dict] = None,
) -> str:
    """Apply true redactions to a PDF and return the output path.

    Parameters
    ----------
    job : RedactionJob
        Contains spans (text) and regions (image areas) to redact.
        Page numbers are 1-indexed (PDF convention) and are converted to
        0-indexed internally for PyMuPDF.
    input_path : str
        Path to the original PDF.
    output_path : str, optional
        Where to write the redacted PDF. Defaults to `<input>_redacted.pdf`.
    char_bboxes : dict, optional
        Mapping from `(doc_id, char_offset)` → `[(page_no, fitz_rect), ...]`.
        `page_no` must be 1-indexed (matching the PDF convention).

    Returns
    -------
    str
        Absolute path to the redacted PDF.
    """
    if not output_path:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_redacted{ext}"

    doc = fitz.open(input_path)

    # Group spans by page (convert from 1-indexed to 0-indexed for PyMuPDF).
    spans_by_page: dict[int, List[PIISpan]] = {}
    for span in job.spans:
        if char_bboxes:
            bboxes = []
            for offset in range(span.char_start, span.char_end):
                key = (span.doc_id, offset)
                if key in char_bboxes:
                    bboxes.extend(char_bboxes[key])
            for page_no, bbox in bboxes:
                spans_by_page.setdefault(page_no - 1, []).append(span)
        else:
            # Fallback: search on every page (0-indexed).
            for page_no in range(len(doc)):
                page = doc.load_page(page_no)
                rects = page.search_for(span.text)
                if rects:
                    spans_by_page.setdefault(page_no, []).append(span)

    # Group image regions by page (1-indexed → 0-indexed).
    regions_by_page: dict[int, List[ImageRegion]] = {}
    for region in job.regions:
        regions_by_page.setdefault(region.page_no - 1, []).append(region)

    # Apply redaction annotations page by page.
    for page_idx in range(len(doc)):
        page = doc.load_page(page_idx)
        page_no = page_idx + 1  # 1-indexed for lookup matching

        # Text redactions.
        for span in spans_by_page.get(page_idx, []):
            # Use char_bboxes for regex (exact offsets) and search_for for LLM (unreliable offsets).
            use_bbox = char_bboxes and span.source == "regex"
            if use_bbox:
                rects = []
                for offset in range(span.char_start, span.char_end):
                    key = (span.doc_id, offset)
                    if key in char_bboxes:
                        for p_no, bbox in char_bboxes[key]:
                            if p_no == page_no:
                                rects.append(_to_fitz_rect(bbox))
                # Merge overlapping rects for the same span on this page.
                if rects:
                    union = rects[0]
                    for r in rects[1:]:
                        union |= r
                    page.add_redact_annot(union, fill=_REDACT_FILL)
                else:
                    # Fallback: bbox lookup failed, use search_for.
                    rects = page.search_for(span.text)
                    for r in rects:
                        page.add_redact_annot(r, fill=_REDACT_FILL)
            else:
                # LLM spans: search_for is more reliable than LLM offsets.
                rects = page.search_for(span.text)
                for r in rects:
                    page.add_redact_annot(r, fill=_REDACT_FILL)

        # Image region redactions (black box).
        for region in regions_by_page.get(page_idx, []):
            page.add_redact_annot(_to_fitz_rect(region.bbox), fill=_REDACT_FILL_IMAGE)

        page.apply_redactions()

    doc.save(output_path, deflate=True, garbage=4)
    doc.close()
    return os.path.abspath(output_path)


def create_redaction_job(
    doc_id: str,
    spans: List[PIISpan],
    regions: Optional[List[ImageRegion]] = None,
    output_path: Optional[str] = None,
) -> RedactionJob:
    """Convenience factory for a RedactionJob."""
    return RedactionJob(
        doc_id=doc_id,
        spans=spans,
        regions=regions or [],
        output_path=output_path,
    )
