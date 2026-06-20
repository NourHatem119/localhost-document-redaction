"""Phase 3 — Redaction pass (apply the master mapping).

Replaces every detected PII span with its Phase-2 pseudonym, so all variants of
an entity become the same token across every chunk and document:

    "...offer you a position. Confirm via j.smith@acme.com..."   (D1)
    "...issued to J. Smith (Acme Corp)..."                       (D2)
        ->  John Smith / J. Smith / Smith, John  ->  Agent A   everywhere

Why deterministic (no exo): we already have the exact `by_surface` map AND the
char offsets of every span, so replacement is exact substitution — it touches
*only* the detected spans and nothing else. An LLM rewrite can't make that
guarantee. exo's job is upstream (detecting the entities) and Cognee's graph;
the redaction itself is pure and reproducible.

Offset-based: replacements are applied right-to-left so earlier offsets don't
shift. Works on a whole page/doc text or on a single chunk (chunk-relative
offsets) — same operation either way; reassembly (Phase 4) is the user's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from schema import PIISpan
from cognee_mem.pseudonyms import MasterMapping, build_master_mapping


@dataclass
class RedactionResult:
    doc_id: str
    original: str
    redacted: str
    replacements: int


def redact_text(text: str, spans: List[PIISpan], mapping: MasterMapping) -> RedactionResult:
    """Replace each span's [char_start:char_end] slice with its pseudonym.

    Spans are applied in descending start order so offsets stay valid. Any span
    whose surface isn't in the map is left untouched (and reported by count).
    """
    doc_id = spans[0].doc_id if spans else ""
    ordered = sorted(spans, key=lambda s: s.char_start, reverse=True)
    out = text
    n = 0
    for s in ordered:
        pseudonym = mapping.pseudonym_for(s.text)
        if pseudonym is None:
            continue
        out = out[: s.char_start] + pseudonym + out[s.char_end :]
        n += 1
    return RedactionResult(doc_id=doc_id, original=text, redacted=out, replacements=n)


def redact_corpus(doc_texts: dict, spans: List[PIISpan], mapping: MasterMapping
                  ) -> List[RedactionResult]:
    """Redact every document, sharing one master mapping for cross-doc consistency."""
    by_doc: dict = {}
    for s in spans:
        by_doc.setdefault(s.doc_id, []).append(s)
    results = []
    for doc_id, text in doc_texts.items():
        results.append(redact_text(text, by_doc.get(doc_id, []), mapping))
    return results


if __name__ == "__main__":
    from cognee_mem.fixtures import all_spans, spans_by_doc, DOC_TEXTS

    spans = all_spans()
    mapping = build_master_mapping(spans)          # T4 dedup + Phase 2 assign
    results = redact_corpus(DOC_TEXTS, spans, mapping)

    for r in results:
        print(f"=== {r.doc_id}  ({r.replacements} replacements) ===")
        print(r.redacted)
        print()

    # ---- verification (all offline) ----
    print("[verify] checking redaction...")
    ok = True

    # 1. no original surface form survives in any redacted text
    all_text = "\n".join(r.redacted for r in results)
    leaked = sorted({s.text for s in spans if s.text in all_text})
    if leaked:
        ok = False
        print(f"  [FAIL] original PII still present: {leaked}")
    else:
        print("  [PASS] no original PII surface remains in any document")

    # 2. every span was replaced
    total_spans = len(spans)
    total_repl = sum(r.replacements for r in results)
    print(f"  [{'PASS' if total_repl == total_spans else 'FAIL'}] "
          f"replaced {total_repl}/{total_spans} spans")
    ok = ok and total_repl == total_spans

    # 3. cross-doc consistency: the same entity -> same pseudonym in every doc
    #    "Agent A" (John Smith) must appear in D1, D2 and D3.
    agent_a_docs = [r.doc_id for r in results if "Agent A" in r.redacted]
    print(f"  [{'PASS' if len(agent_a_docs) == 3 else 'FAIL'}] "
          f"'Agent A' appears across {len(agent_a_docs)} docs: {agent_a_docs}")
    ok = ok and len(agent_a_docs) == 3

    # 4. only spans changed: non-PII characters are preserved (length check per doc)
    for r, (doc_id, doc_spans) in zip(results, spans_by_doc().items()):
        removed = sum(s.char_end - s.char_start for s in doc_spans)
        added = sum(len(mapping.pseudonym_for(s.text)) for s in doc_spans)
        expected_len = len(r.original) - removed + added
        if len(r.redacted) != expected_len:
            ok = False
            print(f"  [FAIL] {doc_id}: unexpected length change")
    print("  [PASS] non-PII text preserved (only spans changed)") if ok else None

    print("\n[RESULT]", "REDACTION PASS CORRECT ✅" if ok else "PROBLEM ❌")
