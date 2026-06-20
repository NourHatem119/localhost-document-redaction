"""End-to-end Cognee redaction pipeline with MEMORY-FIRST design.

Pipeline flow:
    1. Regex detect (fast, exact)
    2. Check memory for known entities (skip LLM for known ones)
    3. LLM detect only on unknown chunks (limited to avoid timeouts)
    4. Dedup + pseudonym assignment (Phase 2)
    5. Store new entities in memory (Cognee JSON store)
    6. Redact text deterministically (Phase 3)
    7. True PDF redaction with pseudonyms

Memory store: JSON-based entity memory that persists across runs.
When a document is processed, entities are stored and reused next time.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

# Repo-root schema.py is the sacred data contract.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schema import PIISpan
from cognee_mem.pseudonyms import MasterMapping, build_master_mapping
from cognee_mem.redact import RedactionResult, redact_corpus
from cognee_mem.memory_store import (
    load_memory, save_memory, merge_with_memory, build_mapping_with_memory,
    add_to_memory, reset_memory, get_memory_stats, MEMORY_PATH
)


@dataclass
class PipelineOutput:
    mapping: MasterMapping
    results: List[RedactionResult]
    memory_stats: dict
    llm_spans: List[PIISpan]

    @property
    def redacted_texts(self) -> Dict[str, str]:
        return {r.doc_id: r.redacted for r in self.results}


def run_with_memory_first(
    doc_texts: Dict[str, str],
    spans: List[PIISpan],
    memory_path: Optional[Path] = None,
    llm_chunks: Optional[List] = None,
    llm_client = None,
) -> PipelineOutput:
    """Memory-first pipeline: check memory, detect unknown, store new, redact.

    Parameters:
        doc_texts: {doc_id: full_text}
        spans: Regex-detected spans (already done upstream)
        memory_path: Path to entity memory JSON file
        llm_chunks: List of chunks to run LLM on (None = skip LLM)
        llm_client: ExoClient instance for LLM detection
    """
    from pipeline.detect import detect_llm
    from pipeline.exo_client import ExoClient

    # 1. Check memory for known entities
    print("[Pipeline] Checking memory...")
    existing_mapping, unknown_spans = merge_with_memory(spans, memory_path)
    memory_stats = get_memory_stats(memory_path)

    # 2. If we have LLM chunks and unknown spans, run LLM detection
    llm_spans: List[PIISpan] = []
    if llm_chunks and unknown_spans:
        print(f"[Pipeline] {len(unknown_spans)} unknown spans, running LLM on {len(llm_chunks)} chunks...")
        try:
            client = llm_client or ExoClient()
            llm_spans = detect_llm(llm_chunks, client)
            print(f"[Pipeline] LLM found {len(llm_spans)} new spans")
        except Exception as e:
            print(f"[Pipeline] LLM detection failed: {e}")
    
    # 3. Merge all spans (regex + LLM), deduplicate
    all_spans = list({s.span_id: s for s in (spans + llm_spans)}.values())
    
    # 4. Build mapping with memory (known + new entities)
    mapping = build_mapping_with_memory(all_spans, memory_path)
    
    # 5. Redact text
    results = redact_corpus(doc_texts, all_spans, mapping)
    
    return PipelineOutput(
        mapping=mapping,
        results=results,
        memory_stats=memory_stats,
        llm_spans=llm_spans,
    )


def run(doc_texts: Dict[str, str], spans: List[PIISpan]) -> PipelineOutput:
    """Deterministic, offline: spans + texts -> redacted texts + master mapping.
    
    Legacy entry point - no memory, no LLM, just redact.
    """
    mapping = build_master_mapping(spans)          # dedup (T4) + assign (Phase 2)
    results = redact_corpus(doc_texts, spans, mapping)   # redact (Phase 3)
    return PipelineOutput(
        mapping=mapping,
        results=results,
        memory_stats={"total_entities": 0, "types": {}},
        llm_spans=[],
    )


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
