"""JSON-based memory store for entity mappings across pipeline runs.

Acts as a local Cognee replacement when the full Cognee package is not available.
Stores entity->pseudonym mappings in a JSON file and can be queried for
known entities before running LLM detection.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from schema import PIISpan, PIIType
from cognee_mem.dedup import CanonicalEntity, resolve
from cognee_mem.pseudonyms import MasterMapping, assign_pseudonyms


# Default memory file path
MEMORY_PATH = Path(os.path.dirname(os.path.abspath(__file__))) / "entity_memory.json"


def load_memory(path: Optional[Path] = None) -> Dict[str, dict]:
    """Load the entity memory from JSON.
    
    Returns a dict mapping canonical_id -> entity info dict.
    """
    path = path or MEMORY_PATH
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("entities", {})
    except (json.JSONDecodeError, IOError):
        return {}


def save_memory(entities: Dict[str, dict], path: Optional[Path] = None) -> None:
    """Save entity memory to JSON."""
    path = path or MEMORY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "entities": entities,
            "version": "1.0",
        }, f, indent=2, ensure_ascii=False)


def entity_to_dict(entity: CanonicalEntity, pseudonym: str) -> dict:
    """Convert a CanonicalEntity to a serializable dict."""
    return {
        "canonical_id": entity.canonical_id,
        "type": entity.type.value,
        "canonical_text": entity.canonical_text,
        "aliases": list(entity.aliases),
        "doc_ids": list(entity.doc_ids),
        "pseudonym": pseudonym,
    }


def dict_to_entity(data: dict) -> Tuple[CanonicalEntity, str]:
    """Convert a dict back to (CanonicalEntity, pseudonym)."""
    entity = CanonicalEntity(
        canonical_id=data["canonical_id"],
        type=PIIType(data["type"]),
        canonical_text=data["canonical_text"],
        aliases=list(data.get("aliases", [])),
        doc_ids=list(data.get("doc_ids", [])),
    )
    return entity, data["pseudonym"]


def find_known_spans_in_text(
    text: str,
    doc_id: str,
    memory_path: Optional[Path] = None,
) -> List[PIISpan]:
    """Scan text for all known entities in memory and return PIISpans for them.
    
    This ensures that entities learned from a previous document (e.g. doc_0002)
    are redacted when they appear in a new document (e.g. doc_0001 re-run).
    """
    memory = load_memory(memory_path)
    spans = []
    
    for cid, data in memory.items():
        pii_type = PIIType(data.get("type", "OTHER"))
        for alias in data.get("aliases", []):
            # Find all occurrences in text (case-sensitive for exact match)
            start = 0
            while True:
                idx = text.find(alias, start)
                if idx == -1:
                    break
                spans.append(PIISpan(
                    span_id=f"{cid}_{idx}",
                    doc_id=doc_id,
                    chunk_id=None,
                    type=pii_type,
                    text=alias,
                    char_start=idx,
                    char_end=idx + len(alias),
                    confidence=1.0,
                    source="memory",
                ))
                start = idx + len(alias)
    
    # Deduplicate overlapping spans: keep the longest one
    spans.sort(key=lambda s: s.char_start)
    deduped = []
    for s in spans:
        if not deduped or s.char_start >= deduped[-1].char_end:
            deduped.append(s)
        elif s.char_end > deduped[-1].char_end:
            # Overlapping: keep the longer one
            deduped[-1] = s
    
    return deduped


def merge_with_memory(
    spans: List[PIISpan],
    memory_path: Optional[Path] = None,
) -> Tuple[MasterMapping, List[PIISpan]]:
    """Check which spans are already in memory, return those + unknown ones.
    
    Returns:
        - MasterMapping with known entities pre-loaded
        - List of spans that need LLM detection (unknown entities)
    """
    memory = load_memory(memory_path)
    
    # Build a quick lookup by surface form
    known_surfaces: Dict[str, Tuple[str, str]] = {}  # surface -> (canonical_id, pseudonym)
    for cid, data in memory.items():
        for alias in data.get("aliases", []):
            known_surfaces[alias] = (cid, data["pseudonym"])
    
    # Separate known vs unknown spans
    known_spans = []
    unknown_spans = []
    
    for span in spans:
        surface = span.text.strip()
        if surface in known_surfaces:
            known_spans.append(span)
        else:
            unknown_spans.append(span)
    
    # Build partial mapping from memory
    mapping = MasterMapping()
    for cid, data in memory.items():
        entity, pseudonym = dict_to_entity(data)
        mapping.by_canonical_id[cid] = pseudonym
        for alias in data.get("aliases", []):
            mapping.by_surface[alias] = pseudonym
        mapping.entities.append((entity, pseudonym))
    
    print(f"[Memory] {len(known_spans)} spans from memory, {len(unknown_spans)} new spans need LLM")
    return mapping, unknown_spans


def add_to_memory(
    mapping: MasterMapping,
    memory_path: Optional[Path] = None,
) -> Dict[str, dict]:
    """Add new entities from a mapping to the persistent memory.
    
    Merges by canonical_text (not canonical_id which is counter-based and
    not globally unique across runs)."""
    memory = load_memory(memory_path)
    
    for entity, pseudonym in mapping.entities:
        # Find existing entity by canonical_text + type (not by canonical_id)
        existing_key = None
        for key, data in memory.items():
            if data.get("type") == entity.type.value and data.get("canonical_text") == entity.canonical_text:
                existing_key = key
                break
        
        if existing_key:
            # Merge: update aliases and doc_ids
            existing = memory[existing_key]
            existing["aliases"] = list(set(existing.get("aliases", [])) | set(entity.aliases))
            existing["doc_ids"] = list(set(existing.get("doc_ids", [])) | set(entity.doc_ids))
        else:
            # Create new entry with a unique key
            base = entity.type.value
            n = 0
            new_key = f"{base}-{n:04d}"
            while new_key in memory:
                n += 1
                new_key = f"{base}-{n:04d}"
            memory[new_key] = entity_to_dict(entity, pseudonym)
    
    save_memory(memory, memory_path)
    return memory


def build_mapping_with_memory(
    spans: List[PIISpan],
    memory_path: Optional[Path] = None,
) -> MasterMapping:
    """Build a master mapping, merging with existing memory.
    
    This is the main entry point: it checks memory first, then
    dedups remaining spans and assigns new pseudonyms.
    """
    memory = load_memory(memory_path)
    
    # Build mapping from memory entities
    mapping = MasterMapping()
    for cid, data in memory.items():
        entity, pseudonym = dict_to_entity(data)
        mapping.by_canonical_id[cid] = pseudonym
        for alias in data.get("aliases", []):
            mapping.by_surface[alias] = pseudonym
        mapping.entities.append((entity, pseudonym))
    
    # Find spans that are NOT in memory
    known_surfaces = set(mapping.by_surface.keys())
    unknown_spans = [s for s in spans if s.text.strip() not in known_surfaces]
    
    if unknown_spans:
        # Dedup and assign pseudonyms for new spans only, continuing from existing indices
        from cognee_mem.pseudonyms import _get_next_indices
        new_entities = resolve(unknown_spans)
        existing_pseudonyms = [p for _, p in mapping.entities]
        starting_counters = _get_next_indices(existing_pseudonyms)
        new_mapping = assign_pseudonyms(new_entities, starting_counters)
        
        # Merge into existing mapping
        for ent, pseudonym in new_mapping.entities:
            mapping.by_canonical_id[ent.canonical_id] = pseudonym
            for alias in ent.aliases:
                mapping.by_surface[alias] = pseudonym
            mapping.entities.append((ent, pseudonym))
    
    print(f"[Memory] Loaded {len(memory)} entities from memory, {len(unknown_spans)} new spans")
    return mapping


def reset_memory(memory_path: Optional[Path] = None) -> None:
    """Clear all memory."""
    path = memory_path or MEMORY_PATH
    if path.exists():
        path.unlink()
    print("[Memory] Cleared all entity memory")


def get_memory_stats(memory_path: Optional[Path] = None) -> dict:
    """Get statistics about the memory store."""
    memory = load_memory(memory_path)
    types = {}
    for data in memory.values():
        t = data.get("type", "UNKNOWN")
        types[t] = types.get(t, 0) + 1
    
    return {
        "total_entities": len(memory),
        "types": types,
        "file_path": str(memory_path or MEMORY_PATH),
    }


if __name__ == "__main__":
    stats = get_memory_stats()
    print(f"Memory stats: {stats}")
