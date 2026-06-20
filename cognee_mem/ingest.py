"""T3 — Ingest PIISpan[] into Cognee to build the entity graph.

API decision (verified on Cognee 1.1.3): use the NEW API.
  - `remember(text, dataset_name=..., node_set=[doc_id], self_improvement=False)`
      builds the persistent graph in one call. `node_set` tags every entity
      extracted from that text with its `doc_id`, so entity nodes link to doc
      nodes inside ONE shared graph (cross-doc consistency needs them together).
      `self_improvement=False` skips extra LLM enrichment passes — faster and
      fewer round-trips to the small local exo model.
  - `forget(everything=True)` resets the store between runs.
  - `add()`/`cognify()` remain a documented fallback if we ever need finer
    staging control; not used by default.

Granularity: one text block per document listing its entities by type. Cognee's
cognify step extracts the entities and (via node_set) associates them with the
doc. T4 handles dedup/normalization; T5 the cross-doc lookup.

Run:
  python -m cognee_mem.ingest --dry-run   # no exo: print the per-doc payloads
  python -m cognee_mem.ingest             # full ingest (needs exo running)
"""

from __future__ import annotations

import asyncio
import sys
from collections import defaultdict
from typing import Dict, List

import cognee_mem.config as _cfg  # noqa: F401  (loads .env on import)
from schema import PIISpan  # repo-root schema; path set up by config import chain

# Single dataset holds the whole corpus so entities resolve across documents.
DATASET = "obscura_memory"


def group_by_doc(spans: List[PIISpan]) -> Dict[str, List[PIISpan]]:
    out: Dict[str, List[PIISpan]] = defaultdict(list)
    for s in spans:
        out[s.doc_id].append(s)
    return dict(out)


def build_entity_text(doc_id: str, spans: List[PIISpan]) -> str:
    """A compact, extraction-friendly description of a doc's PII entities.

    Phrased so Cognee's entity extraction reliably picks up each value and its
    type, and ties them to the named document.
    """
    lines = [f'Document "{doc_id}" contains the following sensitive entities:']
    for s in spans:
        lines.append(f"- {s.type.value}: {s.text}")
    return "\n".join(lines)


def build_payloads(spans: List[PIISpan]) -> List[dict]:
    """The exact (text, dataset, node_set) we will hand to remember(), per doc."""
    payloads = []
    for doc_id, doc_spans in group_by_doc(spans).items():
        payloads.append(
            {
                "doc_id": doc_id,
                "dataset_name": DATASET,
                "node_set": [doc_id],
                "text": build_entity_text(doc_id, doc_spans),
                "n_spans": len(doc_spans),
            }
        )
    return payloads


async def ingest_spans(
    spans: List[PIISpan],
    *,
    reset: bool = True,
    self_improvement: bool = False,
) -> dict:
    """Ingest spans into Cognee, building one cross-doc entity graph.

    Requires exo to be reachable at the configured LLM endpoint.
    """
    import cognee

    if reset:
        await cognee.forget(everything=True)

    payloads = build_payloads(spans)
    for p in payloads:
        await cognee.remember(
            p["text"],
            dataset_name=p["dataset_name"],
            node_set=p["node_set"],
            self_improvement=self_improvement,
        )

    return {
        "dataset": DATASET,
        "docs": len(payloads),
        "spans": len(spans),
        "self_improvement": self_improvement,
    }


def dry_run(spans: List[PIISpan]) -> None:
    """Show exactly what would be sent to Cognee — no exo, no network."""
    payloads = build_payloads(spans)
    print(f"[dry-run] {len(spans)} spans -> {len(payloads)} remember() calls "
          f"into dataset {DATASET!r}\n")
    for p in payloads:
        print(f"--- doc {p['doc_id']}  (node_set={p['node_set']}, {p['n_spans']} spans) ---")
        print(p["text"])
        print()
    print("[next] run without --dry-run on the exo machine to build the real graph.")


if __name__ == "__main__":
    from cognee_mem.fixtures import all_spans

    spans = all_spans()
    if "--dry-run" in sys.argv:
        dry_run(spans)
    else:
        result = asyncio.run(ingest_spans(spans))
        print("[ok] ingest complete:", result)
