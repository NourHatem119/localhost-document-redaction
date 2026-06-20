# Ingest & Parse

Turns a PDF, DOCX, or TXT file into a contract `Document` (canonical text +
word boxes + extracted images), then into offset-preserving `Chunk`s for
detection. This is Person A's ingest layer (spec §5).

## Core principle: independent canonical text per source

PDF extraction (PyMuPDF) and any external `.txt` are produced independently and
do **not** match character-for-character (measured ~0.99 similarity, non-zero
char delta on the sample doc). So each source defines its **own** canonical
text, and all downstream char offsets index into that:

- **PDF** — PyMuPDF's extraction is canonical. Page text is rebuilt by joining
  positioned words from `get_text("words")`, so every word's `[char_start,
  char_end)` and its bounding box come from the same pass and cannot drift.
- **DOCX** — flow-based, no page geometry. The whole document is one logical
  page with text only (no word boxes). Spatial redaction does not apply.
- **TXT** — plain text, no geometry and no images. One logical page, text only
  (no word boxes). Same as DOCX minus image extraction.

### The "no geometry" contract

DOCX and TXT pages carry `width = height = 0.0` and `word_boxes = []`, and their
chunks have `bbox = None`. Downstream code (especially the redaction engine)
**must treat "no geometry" as a first-class case**: there are no rectangles to
draw, so redaction for these sources is text-only. Branch on
`page.word_boxes`/`bbox is None` rather than assuming every page has boxes.

## Layout

| Path | Purpose |
|---|---|
| `parse.py` | Dispatcher: routes by extension → `Document` |
| `parse_pdf.py` | PyMuPDF parse: canonical text, word boxes, image extraction |
| `parse_docx.py` | python-docx parse: single-page text, image extraction |
| `parse_txt.py` | Plain-text parse: single-page text, no geometry, no images |
| `chunk.py` | Word-window chunking over canonical page text |

## Usage

```python
from ingest.parse import parse_document
from ingest.chunk import chunk_document

doc = parse_document("dataset/docs/doc_0001/doc_0001.pdf")
chunks = chunk_document(doc, overlap_ratio=0.2)

# Offsets are sacred: this holds for every chunk.
page = doc.pages[0]
assert page.text[chunks[0].char_start:chunks[0].char_end] == chunks[0].text
```

`parse_document(path, doc_id=None, image_dir=None)`:
- `doc_id` defaults to the file stem.
- `image_dir` defaults to `extracted/<doc_id>/` beside the source file
  (unused for `.txt`, which has no images).
- raises `ValueError` for anything other than `.pdf` / `.docx` / `.txt`.

## Output shapes (from `schema.py`)

- `Document { doc_id, source_path, pages: [Page], images: [ImageRef] }`
- `Page { page_no, width, height, text, word_boxes: [(char_start, char_end, bbox)] }`
- `ImageRef { image_id, doc_id, page_no, path, bbox? }` — handed to Person B
- `Chunk { chunk_id, doc_id, page_no, text, char_start, char_end, bbox? }`

`word_boxes` is empty for DOCX and TXT. A `Chunk.bbox` is the union of its
words' boxes (PDF only); `None` when no geometry is available.

## Chunking

**One chunk per paragraph**, with **sentence-snapped overlap** into the
neighbouring paragraphs so a span near a boundary still sits wholly inside a
chunk and the LLM always sees whole sentences as context.

- Paragraphs are split on blank lines (`\n\n`). PDF gets these boundaries from
  PyMuPDF block separators; DOCX/TXT already have them.
- For each side, the overlap target is ~20% (by word count) of the adjacent
  paragraph, then **extended outward to the enclosing sentence**:
  - the previous paragraph's tail is prepended from the **start of the sentence**
    containing its 80% mark;
  - the next paragraph's head is appended through the **end of the sentence**
    containing its 20% mark.
- Snapping only ever grows the overlap, so effective overlap is `>= 20%` and is
  never a mid-sentence fragment.

| Param | Default | Notes |
|---|---|---|
| `overlap_ratio` | `0.2` | fraction of an adjacent paragraph (by words) targeted before sentence-snapping; range `[0, 1)` |

Because overlap regions are contiguous with the paragraph in canonical text,
`chunk.text == page.text[char_start:char_end]` always holds.

Sentence segmentation is a simple regex on `.!?` followed by whitespace/end.
Abbreviations like "Dr." may over-split a sentence, which only widens overlap
slightly and never breaks the offset contract.

## Image extraction

- **PDF** — `page.get_images()` + `extract_image()` to disk; placement `bbox`
  from `get_image_rects()`.
- **DOCX** — image parts pulled from the package relationships; `bbox=None`.

Images land in `image_dir` and are referenced by `ImageRef.path` for Person B.

## Experimental

`exp/pdf_txt_divergence.py` quantifies the PDF-vs-`.txt` gap per doc:

```bash
python -m exp.pdf_txt_divergence --docs dataset/docs
```
