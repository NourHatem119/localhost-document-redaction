# Obscura — On-Device Enterprise Redaction

> **One-liner:** An air-gapped pipeline that ingests enterprise documents, finds every piece of PII in text *and* images, redacts it permanently, remembers entities across documents, and fine-tunes itself on its own misses — all running locally with no cloud, no internet.

**Hackathon:** Localhost: On-Device Agent Hackathon · Dawn Capital, London · Sat 20 June 2026
**Constraint that defines everything:** must run **fully offline** during the demo. Airplane mode on. No cloud calls.
**Team:** 3 people · **Build window:** 11:00 → 18:30 (~7.5 hrs, lunch rolling)

---

## 1. Why this wins the room

Enterprise redaction is a real, painful, regulated problem (legal discovery, healthcare records, finance, FOIA). The catch: the documents are *exactly* the kind of sensitive material that **cannot legally leave the building** to hit a cloud API. So "redaction that runs 100% on-device" isn't a gimmick for this hackathon — it's the only correct architecture for the actual use case. That's your pitch.

**Prize targeting** — this build uses 4 of the 5 track partners centrally (not bolted on):

| Partner | Prize | How we use it (must be non-trivial to win) |
|---|---|---|
| **Exo Labs** | TBC | Local LLM inference engine for contextual PII detection + the model Overmind fine-tunes |
| **Cognee** | £500 | Memory graph of PII entities → cross-document redaction consistency + the demo's "wow" visual |
| **Captur** | £500 | On-device image validation to flag photos containing PII (faces, IDs, signatures) |
| **Overmind** | £500 | Capture redaction traces, identify misses, fine-tune the model on its own mistakes |
| **Cosine** | £500 + £1.5k credits | *Dev-time:* use their coding agent to build Obscura → eligible for "built with Cosine" |

Go say hi to the **Overmind and Captur reps in the first 30 minutes** — their SDKs are the two unknowns and the reps will save you an hour each.

---

## 2. Architecture

```
                      ┌─────────────────────────────────────────────┐
   PDF / DOCX  ─────▶│  INGEST → PARSE → CHUNK (offsets preserved)  │  (A)
                      └───────────────┬──────────────┬──────────────┘
                                      │ text chunks  │ embedded images
                                      ▼              ▼
                ┌──────────────────────────┐   ┌──────────────────────────┐
 regex/Presidio │  TEXT PII DETECTION      │   │ IMAGE PII DETECTION      │
 (fast, exact)  │  + exo LLM (contextual)  │   │ Captur + CV (faces/OCR/  │ (B)
          (A)   └────────────┬─────────────┘   │ signatures/ID layouts)   │
                             │ entities        └────────────┬─────────────┘
                             ▼                              │ image regions
                    ┌────────────────────┐                  │
                    │ COGNEE memory graph│ (B)              │
                    │ entity dedup +     │                  │
                    │ cross-doc consistency                 │
                    └─────────┬──────────┘                  │
                              ▼                             ▼
                      ┌─────────────────────────────────────────────┐
                      │ REDACTION ENGINE — true removal, not cover  │ (A)
                      │ → redacted PDF/DOCX output                  │
                      └───────────────────┬─────────────────────────┘
                                          ▼
                      ┌─────────────────────────────────────────────┐
                      │ REVIEW UI + MONITORING  (accept/reject/add) │ (C)
                      └───────────────────┬─────────────────────────┘
                                          │ human corrections = ground truth
                                          ▼
                      ┌─────────────────────────────────────────────┐
                      │ OVERMIND — traces → failure patterns →      │ (C)
                      │ fine-tune exo model on misses → redeploy    │
                      └─────────────────────────────────────────────┘
```

**The clean integration trick:** exo serves an OpenAI-compatible API at `http://localhost:52415/v1/chat/completions`. Point *both* your detection code *and* Cognee's **LLM** provider at that URL. One model, one endpoint, everything local.

> **Update (build day):** this applies to the *LLM* only. Cognee's **embeddings** now run on **Fastembed** (local ONNX, CPU, no API key) instead of exo — so we don't depend on exo serving an `/embeddings` route, and the embedding side is guaranteed offline. Set this in `.env`, not in code (see §9).

---

## 3. The data contract (lock this at 11:15 — it's what lets 3 people work in parallel)

Everyone codes against these shapes from minute one, using mock fixtures, so nobody is blocked waiting on anyone else. Put it in `schema.py` / `types.ts` and treat it as sacred.

```jsonc
Document      { doc_id, source_path, pages: [Page], images: [ImageRef] }
Page          { page_no, width, height, text }
Chunk         { chunk_id, doc_id, page_no, text, char_start, char_end, bbox? }   // bbox=[x0,y0,x1,y1]
PIISpan       { span_id, doc_id, chunk_id, type, text, char_start, char_end,
                bbox, confidence, source: "regex"|"llm"|"cognee", status: "auto"|"confirmed"|"rejected" }
ImageRegion   { region_id, doc_id, image_id, page_no, bbox, label: "FACE"|"ID"|"SIGNATURE"|"TEXT_PII",
                confidence, source: "captur"|"cv" }
RedactionJob  { doc_id, spans: [PIISpan], regions: [ImageRegion], output_path }
Trace         { trace_id, doc_id, decisions: [PIISpan], corrections: [PIISpan] }  // for Overmind
```

`type` enum (agree the list now): `PERSON, EMAIL, PHONE, ADDRESS, SSN, NI_NUMBER, CREDIT_CARD, IBAN, DOB, ORG, MEDICAL, OTHER`.

**Critical rule for real redaction:** covering text with a black rectangle is *not* redaction — the text is still extractable underneath, and judges who've seen real redaction tools will catch it. Use PyMuPDF's `page.add_redact_annot()` + `page.apply_redactions()`, which **actually deletes** the underlying content. Mention this on stage; it's an enterprise-credibility point.

---

## 4. Scope — MVP vs stretch (be ruthless)

**MVP (must work for the demo — this is the floor):**
- Ingest one PDF, extract text with positions + extract embedded images
- Detect PII via regex (emails/phones/cards/NI numbers) **+** exo LLM (names/addresses)
- Produce a genuinely redacted output PDF
- Show detected PII in a UI, side-by-side original vs redacted

**Core (what makes it a *real* project, target by ~16:30):**
- Cognee graph populated with entities + cross-document consistency
- Image PII detection (at least CV face-blur + OCR) compositing into the output
- Review UI where a human can accept/reject/add a redaction

**Stretch (the closing flourish — only if core is solid):**
- Overmind: capture traces, feed corrections, run **one** fine-tune pass, show fewer misses on a held-out doc
- Captur integration proper (vs pure-CV fallback)
- Cognee graph-explorer visual on the big screen during the pitch

> If you're behind at 16:00, **drop stretch, not core.** A polished MVP+core that runs flawlessly offline beats a half-working full vision.

---

## 5. The 3-person split

The seams are chosen so each person owns a vertical slice with a clean interface, and only **A is on the true critical path**. B and C build against mock fixtures until A's real output lands (~14:00), then integrate.

### Person A — Pipeline & Text Core *(owns the spine)*
The demo cannot exist without this, so A protects the critical path and ships the MVP first.
- Stand up exo on the demo machine; load a small model (Llama 3.2 3B or Qwen 2.5 7B — must fit in RAM); verify `localhost:52415`. **Have the fallback ready (see Risks).**
- Ingest + parse: PDF via **PyMuPDF (fitz)**, DOCX via **python-docx** → text *with* bounding boxes + extract embedded images (hand to B).
- Chunk text preserving `char_start/char_end` (offsets are everything downstream).
- Hybrid detection: **Presidio/regex** for structured PII + **exo LLM** prompt for contextual PII. Return `PIISpan[]`.
- Redaction engine: apply true redactions, write output PDF. Own `RedactionJob → output_path`.
- **Deliverable by 14:00:** one PDF in → redacted PDF out, end to end.

### Person B — Memory Graph & Image Vision *(Cognee + Captur)*
Two related sub-tracks; if time gets tight, ship Cognee first (it's the bigger visual + more partners care).
- **Cognee:** `uv pip install "cognee[fastembed]"` (==1.1.3), run local (LanceDB + **`ladybug`** in-process, zero DB setup), config via **`.env`** (LLM → exo, embeddings → Fastembed). Feed A's `PIISpan[]` → **`cognee.remember(..., node_set=[doc_id], self_improvement=False)`** to build the entity graph → dedup/normalize so "J. Smith" and "John Smith" collapse → cross-doc consistency (same entity redacted everywhere). Stand up `cognee ui` graph explorer for the demo.
  > **Updates (build day):** (1) Cognee 1.1.3 **dropped the NetworkX graph adapter** — in-process default is now `ladybug` (same "no server, zero setup" property). (2) Using the **new API** (`remember`/`recall`/`forget`); old `add`/`cognify`/`search` still work as a fallback. (3) **Entity dedup runs deterministically offline (no exo)** for reliable consistency; Cognee's graph/embeddings augment fuzzy cases + power the demo visual. (4) Implements the **two-pass pipeline** (full design in `PERSON_B_COGNEE_TASKS.md`): Phase 2 builds an entity→pseudonym map ("John Smith"/"J. Smith" → *Agent A*) in memory; Phase 3 applies it for consistent replacement. **Decision (locked): Phase 3 redaction is DETERMINISTIC** — exact offset-based substitution from the master map, no LLM in the loop, so it's exact, reproducible, and verifiable offline. (An LLM rewrite can't guarantee "replace these and touch nothing else.") exo is used upstream (detection) and for Cognee's graph, not for applying redactions. Note vs §3: pseudonymization is the *cross-doc consistency* layer; Person A's PyMuPDF `apply_redactions()` is still what truly removes content in the final PDF — the two compose, they don't conflict. One offline entry point: `cognee_mem/pipeline.py` (`run()` = dedup→assign→redact; `run_with_memory()` adds the Cognee write, needs exo).
- **Image PII:** receive images from A → run **Captur** on-device validation + CV: face detection (OpenCV/MediaPipe), **Tesseract OCR** to catch text-PII inside images, simple signature/ID-layout heuristics → emit `ImageRegion[]` with blur/box regions back to A for compositing.
- **Deliverable by 16:00:** graph populated from real entities + image regions flowing for at least one image.

### Person C — App, Monitoring & Self-Improvement *(Overmind + UI + integrator)*
C is the integrator and demo owner — builds against mocks early so the UI exists before the pipeline is ready, then wires in the real thing.
- **UI** (use **Gradio or Streamlit** for speed — do not hand-roll React today): upload → side-by-side original/redacted → PII list (read from Cognee) → review panel (accept/reject/add → writes `corrections`) → monitoring dashboard (counts, confidence, misses).
- **Overmind:** instrument the agent to emit `Trace` per doc; pipe human corrections as the failure signal; run Overmind's capture → identify → fine-tune loop; **the killer beat is "model v1 missed X, we corrected it, model v2 catches it on a fresh doc."** Talk to their rep at 11:00 to scope what's realistic in 7 hrs (likely: trace capture + failure ID + one small LoRA pass).
- **Integration + demo:** own the orchestration script, the shared `schema.py`, the 3-min demo script, and a **backup screen-recording** in case live fails.
- **Deliverable by 16:30:** end-to-end click-through working in the UI on one doc.

---

## 6. Timeline with integration checkpoints

| Time | A — Pipeline/Text | B — Cognee/Vision | C — UI/Overmind/Integration |
|---|---|---|---|
| 11:00–11:30 | exo up + model loaded | cognee install + local up | repo scaffold, **lock `schema.py`**, mock fixtures |
| 11:30–13:00 | parse + chunk + regex | cognee ingest entities (mock) | UI skeleton on mocks + Overmind hello-world |
| 13:00–14:00 | *lunch — keep building* | | |
| **14:00 ✅ CHECKPOINT 1** | **text path: PDF → redacted PDF** | | confirm schema hasn't drifted |
| 14:00–16:00 | exo LLM detection + render | graph from real entities + image CV | wire UI to real pipeline + trace capture |
| **16:00 ✅ CHECKPOINT 2** | | **end-to-end on one doc** | |
| 16:00–17:30 | image regions composited | Captur proper / polish graph | Overmind fine-tune loop + dashboard |
| **17:30 🧊 FEATURE FREEZE** | | | |
| 17:30–18:30 | rehearse in **airplane mode**, record backup video, write submission | | |
| **18:30** | **SUBMIT** | | |

---

## 7. Demo script (3 minutes — rehearse it twice)

1. **Hook (15s):** "This is a redaction tool for documents too sensitive to ever touch the cloud. Wi-Fi is off right now." *(show airplane mode — this is the whole hackathon's theme, lean in)*
2. **Ingest (30s):** drop in a messy PDF with names, an email, an NI number, and a photo with a face + an ID card.
3. **Detect + redact (45s):** show the side-by-side; the output PDF with text *truly removed* (try to copy-paste from under a redaction — nothing there).
4. **Memory (30s):** show the Cognee graph — "the same person appears in three documents and gets redacted consistently everywhere."
5. **Self-improvement (45s):** correct one miss in the review panel → run the Overmind loop → re-run on a held-out doc → "v2 now catches what v1 missed." *(if stretch landed)*
6. **Close (15s):** "Ingest, redact, remember, improve — entirely on-device."

---

## 8. Risks & mitigations (read before you start)

| Risk | Mitigation — set up *now*, not at hour 6 |
|---|---|
| exo setup eats hours / model too big | **Fallback ready hour 1:** Ollama or llama.cpp single-machine, same OpenAI endpoint shape. Pick a *small* model. |
| Models won't download on venue Wi-Fi | **Pre-download all weights to local cache immediately** while you have internet. HF cache at `~/.cache/huggingface`. |
| Overmind fine-tune too ambitious for 7h | Scope with their rep at 11:00. Acceptable demo = trace capture + failure ID + one tiny fine-tune on a toy example. |
| Captur SDK unknown | Talk to rep at 11:00. **Pure-CV fallback** (OpenCV face-blur + Tesseract) means image redaction works regardless. |
| "Redaction" that's just a black box | Use PyMuPDF `apply_redactions()` — truly deletes content. Test by copy-pasting under a redaction. |
| Integration hell at hour 6 | The data contract + mock fixtures + the two checkpoints exist precisely to prevent this. |
| Live demo fails | C records a backup video at 17:30 and the demo runs offline-rehearsed. |

---

## 9. Setup quick-reference

```bash
# Exo (Python 3.12+, from source) — serves OpenAI-compatible API at :52415
git clone https://github.com/exo-explore/exo && cd exo && pip install -e .
exo   # then hit http://localhost:52415/v1/chat/completions

# Cognee — local, in-process graph+vector store (verified: cognee 1.1.3)
uv pip install "cognee[fastembed]"     # [fastembed] = local CPU embeddings, no key, offline
#   config lives in .env at the project root (Cognee reads it automatically):
#     LLM_PROVIDER="openai"   LLM_ENDPOINT="http://localhost:52415/v1"   LLM_MODEL="openai/<exo-model>"   LLM_API_KEY="sk-local-exo"
#     EMBEDDING_PROVIDER="fastembed"   EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"   EMBEDDING_DIMENSIONS="384"
#     COGNEE_SKIP_CONNECTION_TEST="true"      # local small model: skip LLM preflight
#   PRE-DOWNLOAD the embedding model WHILE ONLINE (else airplane mode fails on first embed):
#     python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')"
#   core calls (new API): cognee.remember(...) -> cognee.recall(...) ; cognee.forget(everything=True) to reset ; cognee ui  for the graph explorer
#   graph provider is in-process `ladybug` (NetworkX adapter removed in 1.1.3)

# Pipeline + vision
pip install pymupdf python-docx presidio-analyzer presidio-anonymizer \
            opencv-python pytesseract mediapipe gradio
# Captur SDK + Overmind SDK: get install/auth from their reps at the booths
```

---

*Working name "Obscura" — rename freely. Now go lock the schema and talk to the Overmind + Captur reps before you write a line of detection code.*