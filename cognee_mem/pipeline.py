"""End-to-end Cognee redaction pipeline — DETERMINISTIC by design.

This is the canonical entry point for Person B's memory/redaction track. Given
detected PII spans + the source texts, it:

    1. dedup        (T4)       collapse entity variants -> canonical entities
    2. assign       (Phase 2)  one stable pseudonym per canonical entity
    3. redact       (Phase 3)  exact offset-based replacement in every doc

All three steps are deterministic and run fully offline — no exo, no network.
That's the point: the redaction itself must be exact and reproducible, never at
the mercy of a model. exo is used elsewhere (upstream detection produces the
spans; Cognee's graph holds the persistent memory + demo visual), not here.

`run()` is the offline core. `run_with_memory()` additionally writes the entity
graph + pseudonym map into Cognee (needs exo) for cross-session persistence and
the graph explorer — but the redacted output never depends on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from schema import PIISpan
from cognee_mem.pseudonyms import MasterMapping, build_master_mapping
from cognee_mem.redact import RedactionResult, redact_corpus


@dataclass
class PipelineOutput:
    mapping: MasterMapping
    results: List[RedactionResult]

    @property
    def redacted_texts(self) -> Dict[str, str]:
        return {r.doc_id: r.redacted for r in self.results}


def run(doc_texts: Dict[str, str], spans: List[PIISpan]) -> PipelineOutput:
    """Deterministic, offline: spans + texts -> redacted texts + master mapping."""
    mapping = build_master_mapping(spans)          # dedup (T4) + assign (Phase 2)
    results = redact_corpus(doc_texts, spans, mapping)   # redact (Phase 3)
    return PipelineOutput(mapping=mapping, results=results)


async def run_with_memory(doc_texts: Dict[str, str], spans: List[PIISpan]) -> PipelineOutput:
    """Same redaction, plus persist the graph + pseudonym map into Cognee.

    Requires exo. The redacted output is identical to run(); the Cognee write is
    purely for cross-session memory and the demo graph explorer.
    """
    from cognee_mem.ingest import ingest_spans
    from cognee_mem.pseudonyms import persist_to_cognee

    out = run(doc_texts, spans)
    await ingest_spans(spans)              # build the entity<->doc graph
    await persist_to_cognee(out.mapping)   # attach pseudonyms to the graph
    return out


if __name__ == "__main__":
    from cognee_mem.fixtures import all_spans, DOC_TEXTS

    out = run(DOC_TEXTS, all_spans())

    print("=== MASTER MAPPING (deterministic, offline) ===")
    seen = set()
    for ent, pseudonym in sorted(out.mapping.entities, key=lambda x: x[0].canonical_id):
        if pseudonym in seen:
            continue
        seen.add(pseudonym)
        print(f"  {pseudonym:16} <- {ent.aliases}")

    print("\n=== REDACTED DOCUMENTS ===")
    for r in out.results:
        print(f"\n--- {r.doc_id} ({r.replacements} replacements) ---")
        print(r.redacted)
