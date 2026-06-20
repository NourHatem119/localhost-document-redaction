"""Paragraph chunking with neighbour overlap over canonical page text.

One base chunk per paragraph. Each chunk is then extended to include a slice of
its neighbouring paragraphs so spans near a boundary still sit wholly inside a
chunk. The overlap target is ~20% (by word count) of the adjacent paragraph,
then snapped outward to a sentence boundary so the borrowed context is always
whole sentences, never a mid-sentence fragment: the previous paragraph's tail is
prepended from the start of the sentence containing its 80% mark, and the next
paragraph's head is appended through the end of the sentence containing its 20%
mark. Snapping only ever grows the overlap, so it stays >= 20%. Because the
overlap regions are contiguous with the paragraph in canonical text,
`chunk.text` always equals `page.text[char_start:char_end]` — the offset
contract is preserved end to end. When the page has word boxes (PDF), a chunk
also gets a union bbox.
"""

from __future__ import annotations

import math
import re
from typing import List, Optional, Tuple

from schema import BBox, Chunk, Document, Page, WordBox

# Fraction of an adjacent paragraph (by word count) pulled into a chunk as
# overlap on each side. 0.2 == 20%.
DEFAULT_OVERLAP_RATIO = 0.2

_WORD_RE = re.compile(r"\S+")
# Paragraph separator: a blank line (one newline, optional whitespace, newline).
_PARA_SEP_RE = re.compile(r"\n[^\S\n]*\n")
# Sentence terminator: ., !, or ? followed by whitespace/newline or end of text.
# Good enough for clean synthetic docs; abbreviations like "Dr." may over-split,
# which only widens overlap slightly and never breaks the offset contract.
_SENTENCE_END_RE = re.compile(r"[.!?](?=\s|$)")


def _paragraph_word_spans(text: str) -> List[List[Tuple[int, int]]]:
    """Split text into paragraphs (on blank lines) and return, per paragraph,
    the char spans of its words. Empty paragraphs are dropped.
    """
    paragraphs: List[List[Tuple[int, int]]] = []
    # Build [start, end) ranges between separators, then the final tail.
    ranges: List[Tuple[int, int]] = []
    cursor = 0
    for sep in _PARA_SEP_RE.finditer(text):
        ranges.append((cursor, sep.start()))
        cursor = sep.end()
    ranges.append((cursor, len(text)))

    for r_start, r_end in ranges:
        words = [
            (r_start + m.start(), r_start + m.end())
            for m in _WORD_RE.finditer(text[r_start:r_end])
        ]
        if words:
            paragraphs.append(words)
    return paragraphs


def _sentence_ranges(text: str, start: int, end: int) -> List[Tuple[int, int]]:
    """Char ranges of sentences within text[start:end).

    Splits on sentence terminators; the final fragment (no trailing terminator)
    is its own sentence. Leading whitespace is trimmed so a range begins at the
    first non-space char.
    """
    ranges: List[Tuple[int, int]] = []
    cursor = start
    for m in _SENTENCE_END_RE.finditer(text, start, end):
        s_end = m.end()
        while cursor < s_end and text[cursor].isspace():
            cursor += 1
        if cursor < s_end:
            ranges.append((cursor, s_end))
        cursor = s_end
    if cursor < end:
        lead = cursor
        while lead < end and text[lead].isspace():
            lead += 1
        if lead < end:
            ranges.append((lead, end))
    return ranges


def _snap_start_to_sentence(text: str, p_start: int, p_end: int, target: int) -> int:
    """Start of the sentence (within the paragraph) containing `target`."""
    for s_start, s_end in _sentence_ranges(text, p_start, p_end):
        if s_start <= target < s_end:
            return s_start
    return target


def _snap_end_to_sentence(text: str, p_start: int, p_end: int, target: int) -> int:
    """End of the sentence (within the paragraph) containing `target`."""
    for s_start, s_end in _sentence_ranges(text, p_start, p_end):
        if s_start < target <= s_end:
            return s_end
    return target


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


def _overlap_word_count(num_words: int, ratio: float) -> int:
    """Number of words to borrow from a neighbour: round up, but at least 1
    when the neighbour is non-empty and ratio > 0.
    """
    if ratio <= 0 or num_words == 0:
        return 0
    return max(1, math.ceil(num_words * ratio))


def chunk_page(
    page: Page,
    doc_id: str,
    doc_offset: int = 0,
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
) -> List[Chunk]:
    """Chunk one page into per-paragraph chunks with neighbour overlap.

    `doc_offset` is the cumulative character offset of this page within the
    full document. All chunk `char_start`/`char_end` values are document-level;
    page-level offsets are used internally for the `text` slice and bbox lookup.
    """
    if not 0 <= overlap_ratio < 1:
        raise ValueError("overlap_ratio must be in [0, 1)")

    paragraphs = _paragraph_word_spans(page.text)
    if not paragraphs:
        return []

    chunks: List[Chunk] = []
    for idx, words in enumerate(paragraphs):
        # Page-level offsets into page.text (for slicing + bbox).
        p_start = words[0][0]
        p_end = words[-1][1]

        # Prepend the previous paragraph's tail, snapped to a sentence start so
        # the borrowed context begins at a whole sentence (may exceed 20%).
        if idx > 0:
            prev = paragraphs[idx - 1]
            take = _overlap_word_count(len(prev), overlap_ratio)
            if take:
                target = prev[-take][0]
                p_start = _snap_start_to_sentence(
                    page.text, prev[0][0], prev[-1][1], target
                )

        # Append the next paragraph's head, snapped to a sentence end.
        if idx < len(paragraphs) - 1:
            nxt = paragraphs[idx + 1]
            take = _overlap_word_count(len(nxt), overlap_ratio)
            if take:
                target = nxt[take - 1][1]
                p_end = _snap_end_to_sentence(
                    page.text, nxt[0][0], nxt[-1][1], target
                )

        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}_p{page.page_no}_c{idx}",
                doc_id=doc_id,
                page_no=page.page_no,
                text=page.text[p_start:p_end],
                char_start=p_start + doc_offset,
                char_end=p_end + doc_offset,
                bbox=_union_bbox(page.word_boxes, p_start, p_end),
            )
        )
    return chunks


def chunk_document(
    doc: Document,
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
) -> List[Chunk]:
    """Chunk every page of a Document, preserving document-level offsets."""
    chunks: List[Chunk] = []
    doc_offset = 0
    for page in doc.pages:
        chunks.extend(chunk_page(page, doc.doc_id, doc_offset, overlap_ratio))
        doc_offset += len(page.text)
    return chunks
