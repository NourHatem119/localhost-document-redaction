"""T4 — Entity dedup / canonical resolution.

Collapses PIISpan variants into canonical entities, so "John Smith", "J. Smith"
and "Smith, John" become one entity. This is the foundation the Phase-2 assign
pass pseudonymizes (one pseudonym per canonical entity) and the Phase-3 redact
pass applies.

Design choice: this runs **deterministically and fully offline — no exo**. For
a redaction tool, consistency must be reliable, not at the mercy of a small
local model. Cognee's graph + embeddings can *augment* fuzzy matching later
(and drives the demo visual), but the authoritative collapse is rule-based here.

Matching per type:
  - Structured (EMAIL, PHONE, NI_NUMBER, CREDIT_CARD, IBAN, SSN): normalize
    (case/whitespace/punctuation) then exact-match.
  - PERSON: name signature — last name + first-token, with "Last, First"
    reordering and first-initial compatibility (J. ~ John).
  - ORG: core token after dropping corporate suffixes (Corp/Ltd/Inc/...).
  - Everything else (ADDRESS, DOB, MEDICAL, OTHER): normalized exact-match.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List

from schema import PIISpan, PIIType

# Types matched by normalized exact value.
_STRUCTURED = {
    PIIType.EMAIL,
    PIIType.PHONE,
    PIIType.NI_NUMBER,
    PIIType.CREDIT_CARD,
    PIIType.IBAN,
    PIIType.SSN,
}

_ORG_SUFFIXES = {
    "corp", "corporation", "inc", "incorporated", "ltd", "limited", "llc",
    "co", "company", "plc", "group", "holdings", "gmbh", "sa", "ag",
}


@dataclass
class CanonicalEntity:
    canonical_id: str
    type: PIIType
    canonical_text: str         # chosen display form (most complete surface)
    aliases: List[str]          # distinct surface forms seen
    spans: List[PIISpan] = field(default_factory=list)
    doc_ids: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Normalizers / signatures
# ---------------------------------------------------------------------------
def _norm_structured(ptype: PIIType, text: str) -> str:
    t = text.strip().lower()
    if ptype in (PIIType.PHONE, PIIType.NI_NUMBER, PIIType.CREDIT_CARD,
                 PIIType.IBAN, PIIType.SSN):
        return re.sub(r"[\s\-\.\(\)]", "", t)  # keep only significant chars
    return t  # email: lowercase is enough


def _norm_basic(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower()).rstrip(".,")


def _person_signature(text: str):
    """(last_name, first_token) lowercased, '.' stripped. Handles 'Last, First'."""
    t = text.strip()
    if "," in t:
        last, _, first = t.partition(",")
    else:
        parts = t.split()
        last = parts[-1] if parts else t
        first = " ".join(parts[:-1])
    last = re.sub(r"[^\w]", "", last).lower()
    first_tokens = [re.sub(r"[^\w]", "", tok).lower() for tok in first.split()]
    first0 = first_tokens[0] if first_tokens else ""
    return last, first0


def _persons_compatible(a, b) -> bool:
    (la, fa), (lb, fb) = a, b
    if la != lb:
        return False
    if not fa or not fb:
        return True                      # one has only a last name
    return fa == fb or fa.startswith(fb) or fb.startswith(fa)  # John ~ J


def _org_core(text: str) -> str:
    toks = [re.sub(r"[^\w]", "", tok).lower() for tok in text.split()]
    toks = [tok for tok in toks if tok and tok not in _ORG_SUFFIXES]
    return " ".join(toks)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def _cluster_by_key(spans: List[PIISpan], key: Callable[[PIISpan], str]):
    buckets: Dict[str, List[PIISpan]] = {}
    for s in spans:
        buckets.setdefault(key(s), []).append(s)
    return list(buckets.values())


def _cluster_compatible(spans: List[PIISpan], sig, compatible):
    """Greedy union for non-transitive matchers (PERSON)."""
    clusters: List[dict] = []  # {"sig": signature, "spans": [...]}
    for s in spans:
        ssig = sig(s.text)
        for c in clusters:
            if compatible(ssig, c["sig"]):
                c["spans"].append(s)
                # keep the more complete signature (longer first token)
                if len(ssig[1]) > len(c["sig"][1]):
                    c["sig"] = ssig
                break
        else:
            clusters.append({"sig": ssig, "spans": [s]})
    return [c["spans"] for c in clusters]


def _pick_canonical_text(ptype: PIIType, surfaces: List[str]) -> str:
    if ptype == PIIType.PERSON:
        # prefer natural order (no comma) and most complete; tiebreak length.
        return sorted(surfaces, key=lambda x: ("," in x, -len(x)))[0]
    if ptype == PIIType.ORG:
        return max(surfaces, key=len)          # "Acme Corporation"
    return max(surfaces, key=len)


def _make_entity(cid: str, ptype: PIIType, spans: List[PIISpan]) -> CanonicalEntity:
    surfaces = list(dict.fromkeys(s.text for s in spans))   # distinct, ordered
    doc_ids = list(dict.fromkeys(s.doc_id for s in spans))
    return CanonicalEntity(
        canonical_id=cid,
        type=ptype,
        canonical_text=_pick_canonical_text(ptype, surfaces),
        aliases=surfaces,
        spans=spans,
        doc_ids=doc_ids,
    )


def resolve(spans: List[PIISpan]) -> List[CanonicalEntity]:
    """Collapse spans into canonical entities. Pure, deterministic, offline."""
    by_type: Dict[PIIType, List[PIISpan]] = {}
    for s in spans:
        by_type.setdefault(s.type, []).append(s)

    entities: List[CanonicalEntity] = []
    counters: Dict[str, int] = {}
    for ptype, group in by_type.items():
        if ptype == PIIType.PERSON:
            clusters = _cluster_compatible(group, _person_signature, _persons_compatible)
        elif ptype == PIIType.ORG:
            clusters = _cluster_by_key(group, lambda s: _org_core(s.text))
        elif ptype in _STRUCTURED:
            clusters = _cluster_by_key(group, lambda s, pt=ptype: _norm_structured(pt, s.text))
        else:
            clusters = _cluster_by_key(group, lambda s: _norm_basic(s.text))

        for cluster in clusters:
            n = counters.get(ptype.value, 0)
            counters[ptype.value] = n + 1
            entities.append(_make_entity(f"{ptype.value}-{n:04d}", ptype, cluster))
    return entities


if __name__ == "__main__":
    from cognee_mem.fixtures import all_spans, EXPECTED_CANONICAL

    ents = resolve(all_spans())
    print(f"[ok] {len(all_spans())} spans -> {len(ents)} canonical entities\n")
    for e in sorted(ents, key=lambda x: x.canonical_id):
        multi = " *" if len(e.spans) > 1 else ""
        print(f"  {e.canonical_id:14} {e.canonical_text!r:28} "
              f"aliases={e.aliases} docs={e.doc_ids}{multi}")

    # --- assert against fixture ground truth ---
    print("\n[verify] checking EXPECTED_CANONICAL groups...")
    span_to_cid = {id(s): e.canonical_id for e in ents for s in e.spans}
    span_lookup = {(s.doc_id, s.text): s for s in all_spans()}
    ok = True
    for name, members in EXPECTED_CANONICAL.items():
        cids = {span_to_cid[id(span_lookup[(d, surf)])] for d, surf in members}
        status = "PASS" if len(cids) == 1 else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"  [{status}] {name}: {len(members)} surfaces -> {len(cids)} entity")
    print("\n[RESULT]", "ALL GROUPS COLLAPSED CORRECTLY ✅" if ok else "MISMATCH ❌")
