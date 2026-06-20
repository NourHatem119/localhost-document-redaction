# Demo Guide — Obscura

> How to pitch, demonstrate, and present Obscura at the hackathon (or any venue). This is a 3-minute script with setup, rehearsal, and backup plans.

---

## The Pitch (3 minutes)

### 1. Hook — 15 seconds

> "This is a redaction tool for documents too sensitive to ever touch the cloud. Wi-Fi is off right now."

**Action:** Show airplane mode on the demo machine. Open a terminal, run `ifconfig` or `networksetup -getairportpower en0` to prove no Wi-Fi. This is the whole hackathon's theme — lean into it.

**Why this wins:** Enterprise redaction is a real, regulated problem. The documents are exactly the kind that cannot legally leave the building. Obscura is the only correct architecture for the actual use case.

---

### 2. Ingest — 30 seconds

> "Drop in a messy document — names, emails, NI numbers, and a photo with a face plus an ID card."

**Action:** Use the web UI (`http://localhost:5001`). Drag and drop `dataset/docs/doc_0002/doc_0002.pdf` (HR onboarding with embedded ID card + signature images). Or use the CLI:

```bash
python -m pipeline.cli \
    --input dataset/docs/doc_0002/doc_0002.pdf \
    --output /tmp/demo_redacted.pdf \
    --use-llm
```

**What to show:**
- The UI accepts the file instantly.
- Progress bar: "Ingesting → Detecting → Redacting → Done."
- The detected PII list populates: 5+ spans from regex + 3+ from LLM.

---

### 3. Detect + Redact — 45 seconds

> "Side-by-side: the original document and the redacted output. The text is truly gone — not just covered."

**Action:** Open the original PDF in one window, the redacted PDF in another. Scroll to the first page with PII.

**The credibility test:**

```bash
# Extract text from the redacted PDF to prove it's gone
python -c "
import fitz
doc = fitz.open('/tmp/demo_redacted.pdf')
for page in doc:
    print(page.get_text())
"
```

**What to show:**
- "John Smith" is replaced with "[PERSON-1]" or "Agent A".
- "QQ123456C" is replaced with "[NI-1]".
- The email and phone are masked.
- The ID card image has a black box over the face and photo area.
- The signature is blacked out.

**The true-redaction proof:** Copy-paste from the PDF — the text is literally not there. Mention this on stage: "A black rectangle is not redaction. The text is still underneath. We permanently delete it."

---

### 4. Memory (Cross-Document Consistency) — 30 seconds

> "The same person appears in three documents and gets redacted consistently everywhere."

**Action:** Show the Cognee graph explorer (if T7 landed) or run the deterministic pipeline offline:

```bash
python -m cognee_mem.pipeline
```

**What to show:**

```
=== MASTER MAPPING (deterministic, offline) ===
  Agent A          <- ['John Smith', 'J. Smith', 'Smith, John']
  Organization 1   <- ['Acme Corporation', 'Acme Corp', 'Acme']
  [EMAIL-1]        <- ['j.smith@acme.com']
```

**The pitch:** "In an employment letter, he's 'John Smith'. In an invoice, he's 'J. Smith'. In an HR memo, he's 'Smith, John'. Without consistency, each variant gets a different mask — you can reconstruct the original. Obscura's memory graph collapses all three to one entity: 'Agent A'. Everywhere."

**If the Cognee UI is running:**

```bash
cognee ui
```

Open the browser to the graph explorer. Zoom to the node for "Agent A" and show the three document edges connected to it.

---

### 5. Self-Improvement (Stretch — if landed) — 45 seconds

> "Model v1 missed something. We corrected it. Model v2 catches it on a fresh document."

**Action:** This requires the Overmind integration to be complete. If it's not, skip this beat and close stronger on the offline + consistency story.

If it is complete:

```bash
# 1. Run on a document, note a miss
# 2. Correct it in the UI (accept/reject/add)
# 3. Feed the correction to Overmind
# 4. Fine-tune one pass
# 5. Run on a held-out document
# 6. Show the model now catches the miss
```

**What to say:** "This isn't just redaction. It's a system that learns from its own mistakes — entirely on-device. No data ever leaves this machine."

---

### 6. Close — 15 seconds

> "Ingest, redact, remember, improve — entirely on-device. Obscura."

**Action:** Kill the Wi-Fi if you turned it back on. Show the airplane mode icon one last time. Thank the judges.

---

## Pre-Demo Checklist

### 30 minutes before

- [ ] Machine is charged (or plugged in).
- [ ] EXO or Ollama is running and warmed up.
- [ ] `python -m cognee_mem.smoke_test` passes.
- [ ] `python -m pipeline.cli --input dataset/docs/doc_0001/doc_0001.pdf --output /tmp/test.pdf` works.
- [ ] The web UI loads at `http://localhost:5001`.
- [ ] Synthetic documents are generated (`dataset/docs/doc_0001/` through `doc_0004/` exist).
- [ ] The demo PDF (`doc_0002`) is the right one — it has images for the visual wow.

### 10 minutes before

- [ ] Turn on airplane mode. Verify:
  ```bash
  ping -c 1 google.com  # should fail
  curl http://localhost:52415/v1/models  # should succeed (local)
  ```
- [ ] Close all unrelated apps and browser tabs.
- [ ] Set terminal font to 16pt+ for readability.
- [ ] Open the web UI in the browser (full screen).
- [ ] Have a second terminal ready with the CLI command typed but not entered.
- [ ] Have the `fitz` text-extraction test ready (for the true-redaction proof).

### 5 minutes before

- [ ] Rehearse the 3-minute script once, out loud, to yourself.
- [ ] Check the redacted output file exists and looks correct.
- [ ] Have the backup video ready (see below).
- [ ] Deep breath. You've got this.

---

## Backup Plans

### If the live demo fails

**Plan A — Recorded video**

Record a 2-minute screen capture at 17:30 (before feature freeze):

```bash
# macOS
osascript -e 'tell application "QuickTime Player" to start new screen recording'

# Or use a simple Python script with pyautogui / mss if you prefer
```

Show: upload → progress → PII list → download → text-extraction proof → side-by-side.

**Plan B — Screenshot deck**

If even the video fails, have 5 screenshots ready:
1. Web UI with the uploaded document.
2. PII detection list.
3. Side-by-side original vs redacted.
4. Text-extraction proof (terminal showing empty text where PII was).
5. Cognee graph explorer showing cross-document entity links.

**Plan C — CLI only**

If the web UI breaks, the CLI is the fallback:

```bash
python -m pipeline.cli --input dataset/docs/doc_0002/doc_0002.pdf --output /tmp/demo.pdf --use-llm
python -c "import fitz; print(fitz.open('/tmp/demo.pdf')[0].get_text())"
```

This is boring but bulletproof. Emphasize the pipeline, not the UI.

### If EXO crashes

Switch to Ollama instantly:

```bash
export OBSCURA_USE_OLLAMA=1
ollama serve &
python -m pipeline.cli --input ... --output ... --use-llm
```

Have the Ollama model pre-pulled. The `ExoClient` auto-discovers Ollama if EXO is down.

### If Cognee graph is empty

The deterministic pipeline (`cognee_mem.pipeline.py`) works offline with zero Cognee dependency. Show the master mapping printout — it's equally impressive and more reliable.

---

## Judges' FAQ (Be Ready For These)

### "How is this different from just using a black highlighter?"

> "A black rectangle is cosmetic. The text is still extractable underneath. Obscura uses PyMuPDF's `apply_redactions()`, which permanently deletes the underlying content stream. We can prove it — copy-paste from the output and the text is gone."

### "Why not just use a cloud API?"

> "These documents are legal discovery files, medical records, and financial statements. They cannot legally leave the building. A cloud API is a compliance violation. Obscura is the only architecture that satisfies the actual use case."

### "What if the LLM misses something?"

> "Two layers: regex catches structured PII (emails, phones, NI numbers) with 100% precision. The LLM catches contextual PII (names, addresses). The human reviewer can accept, reject, or add redactions in the UI. Overmind captures these corrections and fine-tunes the model."

### "How does cross-document consistency work?"

> "The same person may appear as 'John Smith', 'J. Smith', and 'Smith, John' across documents. Our Cognee memory graph collapses all variants to one canonical entity. Every occurrence gets the same pseudonym. Without this, you could reconstruct the original by comparing different masks."

### "What about images?"

> "Embedded images are extracted, scanned for faces, IDs, signatures, and text-PII via OCR, then redacted with Gaussian blur and pixelation. The result is composited back into the output PDF."

### "Can this run on a server?"

> "Yes, but the design point is air-gapped. It runs on a laptop with no network. For a server deployment, you'd run the same stack on a secure internal machine. The architecture is identical — just bigger hardware."

---

## Post-Demo (After the Pitch)

1. **Stay at the booth** for questions. Judges often come back.
2. **Have the repo open** in case they want to see code.
3. **Have the `schema.py`** ready to show the data contract — it's a clean design signal.
4. **Mention the synthetic dataset** — it's a thoughtful touch that shows you thought about evaluation.
5. **Be honest about stretch goals** — if Overmind didn't land, say so and explain what the next step would be.

---

> **Hackathon:** Localhost: On-Device Agent Hackathon — Dawn Capital, London — 20 June 2026
> **Slot:** 3 minutes. Rehearse twice. Have a backup. Close on "airplane mode."
