"""T2 — Mock PIISpan fixtures for the Cognee sub-track.

Lets T3 (ingest), T4 (dedup), and T5 (cross-doc lookup) be built and tested
*before* Person A's real detector output lands (~14:00). Swap these for A's
real `PIISpan[]` at T6.

The fixtures are deliberately constructed with two kinds of duplication so the
dedup/consistency logic has real targets:

  1. ENTITY VARIANTS (same entity, different surface forms):
       PERSON  "John Smith"  / "J. Smith" / "Smith, John"
       ORG     "Acme Corporation" / "ACME Corp" / "Acme"
  2. CROSS-DOC REPEATS (identical value in >1 doc):
       EMAIL   j.smith@acme.com   -> D1, D2
       PERSON  Jane Doe           -> D2, D3

`EXPECTED_CANONICAL` records the intended resolution so T4/T5 can assert
against it.

Char offsets are REAL: each span's char_start/char_end is computed by locating
the surface text inside its document's page text (offsets matter downstream per
spec section 3).
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

# Repo-root schema.py is the sacred data contract (same import trick as image_vision).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schema import PIISpan, PIIType  # noqa: E402


# ---------------------------------------------------------------------------
# Page texts — short, realistic, so offsets are genuine.
# ---------------------------------------------------------------------------
DOC_TEXTS: Dict[str, str] = {
    "D1_offer_letter": (
        "Dear John Smith,\n"
        "We are pleased to offer you a position at Acme Corporation. "
        "Please confirm via j.smith@acme.com or call +44 7700 900123. "
        "For payroll we hold your National Insurance number QQ123456C."
    ),
    "D2_invoice_4471": (
        "Invoice 4471 issued to J. Smith (Acme Corp).\n"
        "Remittance confirmation sent to j.smith@acme.com. "
        "Approved by Jane Doe. Card on file ending 4111 1111 1111 1111."
    ),
    "D3_hr_memo": (
        "HR memo re: Smith, John.\n"
        "Cc: Jane Doe (jane.doe@acme.com). "
        "Home address on record: 221B Baker Street, London. "
        "Employer: Acme."
    ),
}


# (PIIType, surface_text, source) per document. source mirrors how A would tag
# it: regex for structured PII, llm for contextual (names/orgs/addresses).
_DOC_ENTITIES = {
    "D1_offer_letter": [
        (PIIType.PERSON, "John Smith", "llm"),
        (PIIType.ORG, "Acme Corporation", "llm"),
        (PIIType.EMAIL, "j.smith@acme.com", "regex"),
        (PIIType.PHONE, "+44 7700 900123", "regex"),
        (PIIType.NI_NUMBER, "QQ123456C", "regex"),
    ],
    "D2_invoice_4471": [
        (PIIType.PERSON, "J. Smith", "llm"),
        (PIIType.ORG, "Acme Corp", "llm"),
        (PIIType.EMAIL, "j.smith@acme.com", "regex"),
        (PIIType.PERSON, "Jane Doe", "llm"),
        (PIIType.CREDIT_CARD, "4111 1111 1111 1111", "regex"),
    ],
    "D3_hr_memo": [
        (PIIType.PERSON, "Smith, John", "llm"),
        (PIIType.PERSON, "Jane Doe", "llm"),
        (PIIType.EMAIL, "jane.doe@acme.com", "regex"),
        (PIIType.ADDRESS, "221B Baker Street, London", "llm"),
        (PIIType.ORG, "Acme", "llm"),
    ],
}


def _build() -> List[PIISpan]:
    spans: List[PIISpan] = []
    for doc_id, entities in _DOC_ENTITIES.items():
        text = DOC_TEXTS[doc_id]
        for i, (ptype, surface, source) in enumerate(entities):
            start = text.find(surface)
            if start == -1:
                raise ValueError(f"{surface!r} not found in {doc_id} text — fix the fixture")
            spans.append(
                PIISpan(
                    span_id=f"{doc_id}-s{i}",
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}-c0",
                    type=ptype,
                    text=surface,
                    char_start=start,
                    char_end=start + len(surface),
                    bbox=None,
                    confidence=0.95 if source == "regex" else 0.80,
                    source=source,
                    status="auto",
                )
            )
    return spans


# All fixture spans, flat.
ALL_SPANS: List[PIISpan] = _build()


def all_spans() -> List[PIISpan]:
    """Every fixture span across all docs."""
    return list(ALL_SPANS)


def spans_by_doc() -> Dict[str, List[PIISpan]]:
    """Fixture spans grouped by doc_id."""
    out: Dict[str, List[PIISpan]] = {}
    for s in ALL_SPANS:
        out.setdefault(s.doc_id, []).append(s)
    return out


# ---------------------------------------------------------------------------
# Ground truth for T4/T5: how the entities SHOULD collapse + where they appear.
# Keyed by a human-readable canonical name -> list of (doc_id, surface).
# ---------------------------------------------------------------------------
EXPECTED_CANONICAL: Dict[str, List[tuple]] = {
    "John Smith (PERSON)": [
        ("D1_offer_letter", "John Smith"),
        ("D2_invoice_4471", "J. Smith"),
        ("D3_hr_memo", "Smith, John"),
    ],
    "Acme (ORG)": [
        ("D1_offer_letter", "Acme Corporation"),
        ("D2_invoice_4471", "Acme Corp"),
        ("D3_hr_memo", "Acme"),
    ],
    "j.smith@acme.com (EMAIL)": [
        ("D1_offer_letter", "j.smith@acme.com"),
        ("D2_invoice_4471", "j.smith@acme.com"),
    ],
    "Jane Doe (PERSON)": [
        ("D2_invoice_4471", "Jane Doe"),
        ("D3_hr_memo", "Jane Doe"),
    ],
}


if __name__ == "__main__":
    spans = all_spans()
    by_doc = spans_by_doc()
    print(f"[ok] {len(spans)} spans across {len(by_doc)} docs")
    for doc_id, doc_spans in by_doc.items():
        print(f"  {doc_id}: {len(doc_spans)} spans")
        for s in doc_spans:
            print(f"     {s.type.value:12} {s.text!r:30} [{s.char_start}:{s.char_end}] {s.source}")
    print("\n[expected canonical groups — T4/T5 targets]")
    for name, members in EXPECTED_CANONICAL.items():
        docs = ", ".join(f"{d}:{surf!r}" for d, surf in members)
        print(f"  {name}  <-  {docs}")
