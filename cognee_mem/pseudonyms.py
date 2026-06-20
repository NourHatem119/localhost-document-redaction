"""Phase 2 — Pseudonym assign pass (the "memory pass").

Turns T4's canonical entities into a **master dictionary**: every surface form
of an entity maps to one stable pseudonym, e.g.

    John Smith  ->  Agent A
    J. Smith    ->  Agent A
    Smith, John ->  Agent A
    Jane Doe    ->  Agent B
    Acme Corp   ->  Organization 1

Two outputs:
  1. A deterministic, offline `MasterMapping` (our authoritative source of truth
     for Phase 3 redaction — reliable, no model in the loop).
  2. Cognee memory statements written via `remember()` so the graph carries the
     pseudonyms (drives the demo visual + lets Phase 3 "ask Cognee"). This part
     needs exo; everything else is offline.

Assignment is stable: entities are pseudonymized in `canonical_id` order, so the
same input always yields the same pseudonyms.
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from schema import PIIType
from cognee_mem.dedup import CanonicalEntity, resolve


# ---------------------------------------------------------------------------
# Pseudonym policy — how each type is labelled.
# ---------------------------------------------------------------------------
def _agent_label(n: int) -> str:
    # 0->A, 25->Z, 26->AA, ...
    letters = string.ascii_uppercase
    s = ""
    n += 1
    while n:
        n, r = divmod(n - 1, 26)
        s = letters[r] + s
    return f"Agent {s}"


def _numbered(prefix: str):
    return lambda n: f"{prefix} {n + 1}"


def _bracket(tag: str):
    return lambda n: f"[{tag}-{n + 1}]"


# type -> function(index) -> pseudonym string
_POLICY = {
    PIIType.PERSON: _agent_label,
    PIIType.ORG: _numbered("Organization"),
    PIIType.EMAIL: _bracket("EMAIL"),
    PIIType.PHONE: _bracket("PHONE"),
    PIIType.NI_NUMBER: _bracket("NI"),
    PIIType.CREDIT_CARD: _bracket("CARD"),
    PIIType.IBAN: _bracket("IBAN"),
    PIIType.SSN: _bracket("SSN"),
    PIIType.ADDRESS: _bracket("ADDRESS"),
    PIIType.DOB: _bracket("DOB"),
    PIIType.MEDICAL: _bracket("MEDICAL"),
    PIIType.OTHER: _bracket("REDACTED"),
}


@dataclass
class MasterMapping:
    by_canonical_id: Dict[str, str] = field(default_factory=dict)   # cid -> pseudonym
    by_surface: Dict[str, str] = field(default_factory=dict)        # surface form -> pseudonym
    entities: List[Tuple[CanonicalEntity, str]] = field(default_factory=list)

    def pseudonym_for(self, surface: str) -> str | None:
        return self.by_surface.get(surface)


def assign_pseudonyms(entities: List[CanonicalEntity]) -> MasterMapping:
    """Assign one stable pseudonym per canonical entity. Pure / offline."""
    mapping = MasterMapping()
    counters: Dict[PIIType, int] = {}
    # Stable order so pseudonyms are reproducible run-to-run.
    for ent in sorted(entities, key=lambda e: e.canonical_id):
        idx = counters.get(ent.type, 0)
        counters[ent.type] = idx + 1
        label_fn = _POLICY.get(ent.type, _bracket("REDACTED"))
        pseudonym = label_fn(idx)

        mapping.by_canonical_id[ent.canonical_id] = pseudonym
        for surface in ent.aliases:
            mapping.by_surface[surface] = pseudonym
        mapping.entities.append((ent, pseudonym))
    return mapping


def build_master_mapping(spans) -> MasterMapping:
    """Convenience: dedup (T4) + assign (Phase 2) in one step."""
    return assign_pseudonyms(resolve(spans))


# ---------------------------------------------------------------------------
# Cognee persistence — write the pseudonym map into memory (needs exo).
# ---------------------------------------------------------------------------
def cognee_statements(mapping: MasterMapping) -> List[dict]:
    """Natural-language memory statements + node_set tags for each entity."""
    out = []
    for ent, pseudonym in mapping.entities:
        aliases = ", ".join(ent.aliases)
        docs = ", ".join(ent.doc_ids)
        text = (
            f'The {ent.type.value} known as "{ent.canonical_text}" '
            f"(aliases: {aliases}) must always be redacted as \"{pseudonym}\". "
            f"It appears in documents: {docs}."
        )
        out.append({
            "canonical_id": ent.canonical_id,
            "pseudonym": pseudonym,
            "node_set": [ent.canonical_id, pseudonym],
            "text": text,
        })
    return out


async def persist_to_cognee(mapping: MasterMapping, *, dataset: str = "obscura_memory"):
    """Write the master mapping into Cognee memory. Requires exo running."""
    import cognee

    for stmt in cognee_statements(mapping):
        await cognee.remember(
            stmt["text"],
            dataset_name=dataset,
            node_set=stmt["node_set"],
            self_improvement=False,
        )
    return {"statements": len(mapping.entities), "dataset": dataset}


if __name__ == "__main__":
    import sys
    from cognee_mem.fixtures import all_spans, EXPECTED_CANONICAL

    mapping = build_master_mapping(all_spans())

    print(f"[ok] {len(mapping.entities)} entities pseudonymized "
          f"-> {len(mapping.by_surface)} surface forms mapped\n")
    print("Master dictionary (surface -> pseudonym):")
    for ent, pseudonym in sorted(mapping.entities, key=lambda x: x[0].canonical_id):
        for surface in ent.aliases:
            print(f"  {surface!r:30} -> {pseudonym}")

    # ---- assertions ----
    print("\n[verify] testing the assignment...")
    ok = True

    # 1. all aliases of an entity share one pseudonym
    for ent, pseudonym in mapping.entities:
        ps = {mapping.by_surface[a] for a in ent.aliases}
        if ps != {pseudonym}:
            ok = False
            print(f"  [FAIL] {ent.canonical_id} aliases map to {ps}")

    # 2. distinct entities -> distinct pseudonyms
    all_ps = [p for _, p in mapping.entities]
    if len(all_ps) != len(set(all_ps)):
        ok = False
        print(f"  [FAIL] pseudonym collision: {all_ps}")
    else:
        print(f"  [PASS] {len(set(all_ps))} entities -> {len(set(all_ps))} distinct pseudonyms")

    # 3. every fixture surface is covered
    surfaces = {s.text for s in all_spans()}
    missing = surfaces - set(mapping.by_surface)
    print(f"  [{'PASS' if not missing else 'FAIL'}] all {len(surfaces)} surfaces mapped"
          + (f" (missing {missing})" if missing else ""))
    ok = ok and not missing

    # 4. known variant groups collapse to a single pseudonym
    for name, members in EXPECTED_CANONICAL.items():
        ps = {mapping.by_surface[surf] for _, surf in members}
        status = "PASS" if len(ps) == 1 else "FAIL"
        ok = ok and status == "PASS"
        print(f"  [{status}] {name}: variants -> {ps}")

    print("\n[RESULT]", "PSEUDONYM ASSIGNMENT CORRECT ✅" if ok else "PROBLEM ❌")

    if "--show-cognee" in sys.argv:
        print("\n[cognee statements that would be written (needs exo to run)]")
        for stmt in cognee_statements(mapping):
            print(f"  node_set={stmt['node_set']}")
            print(f"    {stmt['text']}")
