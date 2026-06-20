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

_PATTERNS: List[tuple] = [
    (_RE_EMAIL, PIIType.EMAIL),
    (_RE_PHONE, PIIType.PHONE),
    (_RE_NI, PIIType.NI_NUMBER),
    (_RE_CREDIT_CARD, PIIType.CREDIT_CARD),
    (_RE_IBAN, PIIType.IBAN),
    (_RE_DOB, PIIType.DOB),
]


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
    return spans


def detect_llm(chunks: List[Chunk], client: Optional[ExoClient] = None) -> List[PIISpan]:
    """Run the local LLM over each chunk. Returns spans with document-level offsets."""
    client = client or ExoClient()
    spans: List[PIISpan] = []
    for chunk in chunks:
        try:
            spans.extend(client.detect(chunk))
        except RuntimeError as exc:
            # Log but do not crash the pipeline on one bad chunk.
            print(f"[WARN] LLM detection failed for chunk {chunk.chunk_id}: {exc}")
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
