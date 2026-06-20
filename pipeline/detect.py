"""Hybrid PII detection: regex (fast, exact) + EXO LLM (contextual).

Both detectors emit PIISpan objects with document-level char offsets. The
merge step deduplicates overlaps with a simple priority: regex wins on exact
structured matches (emails, phones, cards, NI numbers), LLM fills in names,
addresses, and contextual PII.
"""

from __future__ import annotations

import re
import uuid
from typing import List, Optional

from schema import Chunk, PIISpan, PIIType, PIISource
from pipeline.exo_client import ExoClient


# ── Regex patterns (UK-focused, fast, exact) ──────────────────────────────

# ── Regex patterns (UK-focused, fast, exact) ──────────────────────────────

_RE_EMAIL = re.compile(
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", re.IGNORECASE
)
_RE_PHONE = re.compile(
    r"(?:\+44\s?7\d{3}|\(?07\d{3}\)?)\s?\d{3}\s?\d{3}"
)
_RE_NI = re.compile(
    r"\b[A-CEGHJ-PR-TW-Z]{2}\d{6}[A-D]\b", re.IGNORECASE
)
_RE_CREDIT_CARD = re.compile(
    r"\b(?:4\d{3}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}|5[1-5]\d{2}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}|3[47]\d{2}[-\s]?\d{6}[-\s]?\d{5}|6(?:011|5\d{2})[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4})\b"
)
_RE_IBAN = re.compile(
    r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}(?:[A-Z0-9]?){0,16}\b", re.IGNORECASE
)
_RE_DOB = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4})\b"
)
# UK postcode: B15 3DH, SW1A 1AA, etc.
_RE_POSTCODE = re.compile(
    r"\b([A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2})\b", re.IGNORECASE
)
# UK street address pattern: "number + street + city + postcode"
_RE_ADDRESS = re.compile(
    r"(\d+\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?,\s*(?:[A-Z][a-zA-Z]+\s*)+,?\s*[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2})\b",
    re.IGNORECASE
)

_PATTERNS: List[tuple] = [
    (_RE_EMAIL, PIIType.EMAIL),
    (_RE_PHONE, PIIType.PHONE),
    (_RE_NI, PIIType.NI_NUMBER),
    (_RE_CREDIT_CARD, PIIType.CREDIT_CARD),
    (_RE_IBAN, PIIType.IBAN),
    (_RE_DOB, PIIType.DOB),
    (_RE_POSTCODE, PIIType.ADDRESS),
    (_RE_ADDRESS, PIIType.ADDRESS),
]


def _drop_contained(spans: List[PIISpan]) -> List[PIISpan]:
    """Drop spans fully contained within a larger span of the same type.

    Two patterns can match overlapping text — e.g. the postcode pattern matches
    "B15 3DH" while the street pattern matches the whole "5 Elmwood Court,
    Birmingham, B15 3DH". The embedded postcode is redundant: keep the wider
    address span so it resolves to a single entity (one pseudonym). A standalone
    postcode elsewhere is not contained by anything and survives.
    """
    kept: List[PIISpan] = []
    for s in spans:
        s_len = s.char_end - s.char_start
        contained = any(
            o is not s
            and o.type == s.type
            and o.char_start <= s.char_start
            and o.char_end >= s.char_end
            and (o.char_end - o.char_start) > s_len
            for o in spans
        )
        if not contained:
            kept.append(s)
    return kept


def detect_regex(text: str, doc_id: str) -> List[PIISpan]:
    """Run compiled regex patterns over the full document text."""
    spans: List[PIISpan] = []
    for compiled, pii_type in _PATTERNS:
        for m in compiled.finditer(text):
            spans.append(
                PIISpan(
                    span_id=f"re-{uuid.uuid4().hex[:8]}",
                    doc_id=doc_id,
                    chunk_id=None,
                    type=pii_type,
                    text=m.group(0),
                    char_start=m.start(),
                    char_end=m.end(),
                    confidence=1.0,
                    source="regex",
                    status="auto",
                )
            )
    return _drop_contained(spans)


def _find_text_in_chunk(text: str, search: str) -> int:
    """Find `search` inside `text`, handling whitespace differences (e.g. newline vs space)."""
    # Exact match.
    idx = text.find(search)
    if idx != -1:
        return idx
    # Regex match: allow any whitespace between words.
    parts = search.split()
    if not parts:
        return -1
    pattern = r'\s*'.join(re.escape(p) for p in parts)
    m = re.search(pattern, text)
    if m:
        return m.start()
    return -1


def detect_llm(chunks: List[Chunk], client: Optional[ExoClient] = None) -> List[PIISpan]:
    """Run the local LLM over each chunk. Returns spans with document-level offsets.

    LLM offsets are often unreliable (return 0:0 or wrong values). We fix them
    by searching for the exact text the LLM returned within the chunk text,
    handling whitespace differences (newlines vs spaces in phone numbers, etc.).
    """
    client = client or ExoClient()
    spans: List[PIISpan] = []
    for chunk in chunks:
        try:
            raw_spans = client.detect(chunk)
        except RuntimeError as exc:
            # Log but do not crash the pipeline on one bad chunk.
            print(f"[WARN] LLM detection failed for chunk {chunk.chunk_id}: {exc}")
            continue

        for span in raw_spans:
            # Skip empty/whitespace-only spans from the LLM.
            if not span.text or not span.text.strip():
                continue
            # Fix offsets: search for the LLM text in chunk text.
            idx = _find_text_in_chunk(chunk.text, span.text)
            if idx == -1:
                print(f"[WARN] LLM span text not found in chunk: {span.text!r}")
                continue
            # Chunk offset → document offset.
            doc_start = chunk.char_start + idx
            doc_end = doc_start + len(span.text)
            spans.append(
                PIISpan(
                    span_id=span.span_id,
                    doc_id=span.doc_id,
                    chunk_id=span.chunk_id,
                    type=span.type,
                    text=span.text,
                    char_start=doc_start,
                    char_end=doc_end,
                    confidence=span.confidence,
                    source=span.source,
                    status=span.status,
                )
            )
    return spans


def _overlap(a: PIISpan, b: PIISpan) -> bool:
    """True if two spans overlap by at least 1 character."""
    return a.char_start < b.char_end and b.char_start < a.char_end


def merge_spans(regex_spans: List[PIISpan], llm_spans: List[PIISpan]) -> List[PIISpan]:
    """Dedupe overlapping spans. Regex wins on exact matches."""
    all_spans = sorted(regex_spans + llm_spans, key=lambda s: (s.char_start, -s.char_end))
    merged: List[PIISpan] = []
    for span in all_spans:
        if merged and _overlap(merged[-1], span):
            # If the new span is wider, keep regex and drop the narrower one.
            # If same range, regex wins by construction (sorted first).
            continue
        merged.append(span)
    return merged
