# Person B — Cognee Memory Graph: Task Breakdown

> Scope: **only** the Cognee sub-track (not Image Vision). Goal per spec §5/§6:
> graph populated from real entities + cross-doc redaction consistency by **16:00**,
> `cognee ui` graph explorer as the demo "wow" by feature freeze.
>
> Status: **draft for review** — nothing built yet. Review, then I start on T1.

---

## Interfaces (lock these before coding)

- **Input from A:** `PIISpan[]` (see `schema.py`). Treat `text`, `type`, `doc_id`, `span_id` as the fields that matter for memory. Build against **mock fixtures** until A ships real spans (~14:00).
- **Output to C (UI):** a lookup the UI can read — given a `doc_id`, return the canonical/deduped entity list + which spans map to which canonical entity. Shape TBD in T5; propose `{canonical_id, type, canonical_text, aliases:[], spans:[span_id]}`.
- **LLM provider:** exo's OpenAI-compatible endpoint `http://localhost:52415/v1` (from A). Cognee must point here, not OpenAI cloud — demo is air-gapped.

---

## Tasks

### T1 — Install & local config ✅ *(done — built via the cognee-quickstart skill)*
What landed:
- `uv pip install "cognee[fastembed]"` (== 1.1.3), pinned in `requirements.txt`. Verified imports.
- Config lives in **`.env`** (Cognee reads it automatically) — see `.env.example`; working `.env` created (gitignored).
- `cognee_mem/config.py` loads the `.env`; `cognee_mem/smoke_test.py` has 3 levels: `--config-only`, `--embed-only`, full.
- **LLM → exo** (OpenAI-compatible, `http://localhost:52415/v1`). **Embeddings → Fastembed** (local ONNX, CPU, no API key).
- Verified: config loads correctly; Fastembed engine imports and runs.

**Two deviations from the spec discovered (tell the team):**
1. **No NetworkX in Cognee 1.1.3** → using in-process `ladybug` (default) instead. Same "zero DB, no server" intent; LanceDB unchanged.
2. **Embeddings** → switched from "point at exo" to **Fastembed**. Removes the risk of exo not serving an embeddings route; runs fully offline, no key.

**Still to do on the demo machine (only place exo + internet exist):**
- Pre-download the embedding model WHILE ONLINE: `python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')"`
- Set `LLM_MODEL` in `.env` to whatever model exo actually loaded.
- **Offline gate:** with exo up + Wi-Fi OFF, run `python -m cognee_mem.smoke_test` — `remember()`/`recall()` must succeed with zero cloud calls.

### T2 — Mock fixtures *(target 11:45, unblocks everything)*
- Create `cognee_mem/fixtures.py`: 2–3 fake docs' worth of `PIISpan[]` with deliberate duplicates across docs ("J. Smith" / "John Smith" / "Smith, John"; same email in 2 docs).
- These let me build T3–T5 before A's real output lands.
- **Done when:** importable fixture spans matching `schema.py`.

### T3 — Ingestion adapter: PIISpan[] → Cognee *(target 13:00)*
- Write `cognee_mem/ingest.py`: take `PIISpan[]`, format each entity into text Cognee can graph (type + value), one block per doc.
- **API decision (verified on 1.1.3):** use the **new API** — `remember(text, dataset_name, node_set=[doc_id], self_improvement=False)`. `node_set` tags each entity with its `doc_id` so entity nodes link to doc nodes in one shared graph. `self_improvement=False` avoids extra LLM passes on the small local model. `forget(everything=True)` resets between runs. `add()/cognify()` kept as a documented fallback only.
- One Cognee dataset for all docs (cross-doc consistency needs them in one graph); doc identity carried via `node_set`.
- `--dry-run` builds the per-doc texts with no Cognee/exo calls (offline-verifiable); full run needs exo.
- **Done when:** fixtures ingest → graph has entity nodes linked to doc nodes.

### T4 — Entity dedup / normalization *(target 14:30)*
- Collapse variants of the same entity: "J. Smith" == "John Smith", case/whitespace/punctuation normalization, same email/phone across docs → one canonical node.
- Strategy: normalize-then-match for structured types (email/phone/NI); use Cognee's graph + embedding similarity (via exo) for fuzzy PERSON/ORG names.
- Keep aliases on the canonical node so the UI can show "also seen as…".
- **Done when:** the 3 "Smith" variants in fixtures resolve to one canonical entity with 3 aliases.

### T5 — Cross-doc consistency lookup *(target 15:30 — the core deliverable)*
- API: given a `doc_id` (or a span), return the canonical entity + **every** other span/doc it appears in.
- This is what enforces "same person redacted everywhere" — A/C use it so a confirmed redaction in doc 1 auto-applies to doc 3.
- Expose as a small Python function C can import + (optionally) a tiny FastAPI/in-process call.
- **Done when:** querying "John Smith" returns all spans across all docs; demonstrable on fixtures, then on A's real spans.

### T6 — Integrate real entities from A *(target 16:00 — CHECKPOINT 2)*
- Swap fixtures for A's live `PIISpan[]` once available (~14:00+).
- Confirm schema hasn't drifted; handle missing/optional fields gracefully.
- **Done when:** graph populates from a real ingested PDF's entities end-to-end.

### T7 — Graph explorer UI for the demo *(target 16:30, polish to freeze)*
- Stand up `cognee ui` (or equivalent) showing the entity graph on the big screen.
- Verify the demo beat reads clearly: one person → three documents → consistent redaction (spec §7 step 4).
- Pre-load the demo doc set so the graph looks populated on stage.
- **Done when:** graph explorer opens offline and visibly shows cross-doc entity links.

---

## Two-Pass Redaction Pipeline (user's design — drives the assign + redact passes)

Ingestion (Phase 1, smart chunking with 15–20% overlap so names aren't severed)
and Reassembly (Phase 4, merge overlaps by string match) are **already done** by
the user. The Cognee sub-track owns Phases 2 and 3:

- **Phase 2 — Memory pass (build the map).** No redaction yet. Feed chunks
  sequentially; extract entities (names, orgs); for each, check Cognee for an
  existing alias → if new, create a node + assign a **pseudonym** (Agent X); if
  alias, link to the existing node. Result: a master dictionary for the whole
  doc, e.g. `John Smith / J. Smith / Mr. Smith -> Agent X`.
  → Built on **T4** (dedup gives the canonical groups; this pass pseudonymizes
  them). Tracked as task "Phase 2 — Pseudonym assign pass".
- **Phase 3 — Redaction pass (apply the map). DETERMINISTIC (locked).** Replace
  every detected span with its pseudonym by **exact offset-based substitution**
  from the master map — no LLM. Reason: redaction must be exact and reproducible;
  an LLM rewrite can't guarantee "replace these and touch nothing else." Leaves
  all non-span text byte-for-byte intact. Result: consistent redaction across
  chunks/docs. Tracked as task "Phase 3 — Redaction pass".

Order: **T4 → Phase 2 assign → Phase 3 redact**. T5 (cross-doc lookup) and the
Phase 3 retrieval are the same need from two angles.

**Status — built & verified offline (no exo):** T1✅ T2✅ T3✅ T4✅ Phase 2✅
Phase 3✅. One entry point: `cognee_mem/pipeline.py` — `run(doc_texts, spans)`
does dedup→assign→redact deterministically; `run_with_memory(...)` adds the
Cognee graph + pseudonym write (needs exo) for persistence + the demo visual.
exo is required only for (a) upstream detection that produces the spans and
(b) the Cognee memory/graph — never for applying redactions.

## Risks / fallbacks (Cognee-specific)
- **cognify() flaky or slow offline** → fallback: skip Cognee's LLM enrichment, do dedup with plain normalization + embedding similarity directly against exo; still feed a NetworkX graph for the visual.
- **Cognee can't hit exo cleanly** → wrap exo behind the exact OpenAI client config Cognee expects; test in T1, not hour 6.
- **UI graph empty on stage** → T7 pre-loads a fixed doc set; keep a screenshot/recording as backup.
- **Time tight** → spec says ship Cognee before Image Vision. T1–T5 are the must-haves; T7 is the flourish.

---

## Suggested file layout
```
cognee_mem/
  __init__.py
  config.py      # exo endpoint + local backend setup (T1)
  fixtures.py    # mock PIISpan[] (T2)
  ingest.py      # PIISpan[] -> add() -> cognify() (T3)
  dedup.py       # normalization + canonical resolution (T4)
  lookup.py      # cross-doc consistency API for A/C (T5)
  README.md
```

## Critical path within the sub-track
`T1 → T3 → T4 → T5` is the spine. T2 unblocks T3–T5 before A is ready. T6 is the swap-in. T7 is demo polish.
