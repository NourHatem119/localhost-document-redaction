"""Document generator: turns personas into realistic synthetic documents
with gold-standard PII labels.

Design: text is assembled through a `DocBuilder` that records each PII span's
character offset *at the moment it is appended*, so `text[start:end]` always
equals the span text by construction — offsets can never drift.

Each generated document writes three artefacts into `dataset/docs/`:
  - <doc_id>.pdf          the realistic input an enterprise would drop in
  - <doc_id>.txt          the canonical plain text the offsets index into
  - <doc_id>.labels.json  ground-truth PII spans (eval-only, never an input)

The labels are an internal evaluation/fine-tuning artefact. The redaction
pipeline never reads them; in production the human reviewer's corrections are
the ground-truth signal. See README for the framing.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import List

from docx import Document as DocxDocument
from docx.shared import Inches as DocxInches
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

from schema import PIIType
from dataset.schema import DocType, ImageAsset, LabeledSpan, SyntheticDocument
from dataset.personas import _recurring_personas, get_persona, seed
from dataset.images import make_id_card, make_signature

DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")


class DocBuilder:
    """Assembles document text while recording PII spans with exact offsets.

    `add()` appends literal (non-PII) text. `pii()` appends a PII value and
    records a `LabeledSpan` whose char_start/char_end bracket exactly that
    value in the final text.
    """

    def __init__(self) -> None:
        self._parts: List[str] = []
        self._pos = 0
        self.spans: List[LabeledSpan] = []

    def add(self, text: str) -> "DocBuilder":
        self._parts.append(text)
        self._pos += len(text)
        return self

    def line(self, text: str = "") -> "DocBuilder":
        return self.add(text + "\n")

    def pii(self, value: str, pii_type: PIIType, persona_id: str | None = None) -> "DocBuilder":
        start = self._pos
        self._parts.append(value)
        self._pos += len(value)
        self.spans.append(
            LabeledSpan(
                type=pii_type,
                text=value,
                char_start=start,
                char_end=self._pos,
                persona_id=persona_id,
            )
        )
        return self

    def text(self) -> str:
        return "".join(self._parts)


def build_employment_letter(doc_id: str, persona) -> SyntheticDocument:
    """An offer/employment-confirmation letter. Exercises PERSON, ADDRESS,
    DOB, NI_NUMBER, EMAIL, PHONE, ORG.
    """
    pid = persona.persona_id
    b = DocBuilder()

    b.line(persona.org).line()
    b.add("Date: 20 June 2026").line().line()

    b.add("Dear ").pii(persona.full_name, PIIType.PERSON, pid).line(",")
    b.line()
    b.add("We are pleased to confirm your employment with ")
    b.pii(persona.org, PIIType.ORG, pid).add(". This letter sets out the ")
    b.line("key details we hold on file for you.")
    b.line()

    b.add("Home address: ").pii(persona.address, PIIType.ADDRESS, pid).line(".")
    b.add("Date of birth: ").pii(persona.dob, PIIType.DOB, pid).line(".")
    b.add("National Insurance number: ").pii(persona.ni_number, PIIType.NI_NUMBER, pid).line(".")
    b.line()

    b.add("Our HR team will contact you at ")
    b.pii(persona.email, PIIType.EMAIL, pid).add(" or by phone on ")
    b.pii(persona.phone, PIIType.PHONE, pid).line(" if any of the above")
    b.line("details are incorrect.")
    b.line()

    b.line("Yours sincerely,").line()
    b.add("HR Department, ").pii(persona.org, PIIType.ORG, pid).line()

    return SyntheticDocument(
        doc_id=doc_id,
        doc_type=DocType.EMPLOYMENT_LETTER,
        text=b.text(),
        spans=b.spans,
        image_paths=[],
        persona_ids=[pid],
    )


def build_hr_onboarding(doc_id: str, persona, images_dir: str) -> SyntheticDocument:
    """An HR onboarding / right-to-work verification form. Exercises text PII
    (PERSON, ADDRESS, DOB, NI_NUMBER, EMAIL, PHONE, ORG, OTHER) and embeds two
    images: a photo ID card carrying a real face (FACE), the card layout (ID)
    and printed details (TEXT_PII); and a separate handwritten signature
    (SIGNATURE) below it.
    """
    pid = persona.persona_id
    b = DocBuilder()

    b.line(persona.org).line()
    b.line("NEW STARTER ONBOARDING & RIGHT-TO-WORK VERIFICATION").line()
    b.add("Date: 20 June 2026").line().line()

    b.add("Employee: ").pii(persona.full_name, PIIType.PERSON, pid).line()
    b.add("Home address: ").pii(persona.address, PIIType.ADDRESS, pid).line()
    b.add("Date of birth: ").pii(persona.dob, PIIType.DOB, pid).line()
    b.add("National Insurance number: ").pii(persona.ni_number, PIIType.NI_NUMBER, pid).line()
    if persona.other_id:
        b.add("Passport number: ").pii(persona.other_id, PIIType.OTHER, pid).line()
    b.add("Contact email: ").pii(persona.email, PIIType.EMAIL, pid).line()
    b.add("Contact phone: ").pii(persona.phone, PIIType.PHONE, pid).line()
    b.line()

    b.line("Identity verified by photo ID card (scan attached below).")
    b.line()
    b.line("I confirm the above details are correct and that I have the right")
    b.add("to work for ").pii(persona.org, PIIType.ORG, pid).line(".")
    b.line()
    b.line("Signed (employee signature below):")

    os.makedirs(images_dir, exist_ok=True)
    id_card = make_id_card(
        os.path.join(images_dir, "id_card.png"),
        full_name=persona.full_name,
        dob_iso=persona.dob,
        address=persona.address,
        persona_id=pid,
    )
    signature = make_signature(
        os.path.join(images_dir, "signature.png"),
        full_name=persona.full_name,
        persona_id=pid,
    )
    images = [id_card, signature]

    return SyntheticDocument(
        doc_id=doc_id,
        doc_type=DocType.HR_ONBOARDING,
        text=b.text(),
        spans=b.spans,
        images=images,
        image_paths=[a.path for a in images],
        persona_ids=[pid],
    )


def render_pdf(doc: SyntheticDocument, pdf_path: str) -> None:
    """Render the canonical text to a simple, single-column A4 PDF.

    Layout is intentionally plain: the offsets index the canonical .txt, and
    the PDF is the realistic artefact the detection pipeline ingests.
    """
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4
    font_name = "Helvetica"
    font_size = 11
    left = 25 * mm
    right = 25 * mm
    top = height - 25 * mm
    bottom = 25 * mm
    leading = 6 * mm
    max_width = width - left - right
    c.setFont(font_name, font_size)

    def wrap(line: str) -> List[str]:
        """Greedy word-wrap so text stays within the right margin."""
        if not line:
            return [""]
        wrapped: List[str] = []
        current = ""
        for word in line.split(" "):
            candidate = word if not current else current + " " + word
            if stringWidth(candidate, font_name, font_size) <= max_width:
                current = candidate
            else:
                if current:
                    wrapped.append(current)
                current = word
        wrapped.append(current)
        return wrapped

    y = top
    for raw_line in doc.text.split("\n"):
        for visual_line in wrap(raw_line):
            if y < bottom:
                c.showPage()
                c.setFont(font_name, font_size)
                y = top
            c.drawString(left, y, visual_line)
            y -= leading

    # Embed images below the text, scaled to fit the text column.
    for asset in doc.images:
        draw_w = min(asset.width, max_width)
        draw_h = asset.height * (draw_w / asset.width)
        if y - draw_h < bottom:
            c.showPage()
            c.setFont(font_name, font_size)
            y = top
        y -= leading
        c.drawImage(asset.path, left, y - draw_h, width=draw_w, height=draw_h)
        y -= draw_h + leading

    c.save()


def render_docx(doc: SyntheticDocument, docx_path: str) -> None:
    """Render the canonical text to a DOCX, one paragraph per text line.

    Word handles wrapping natively, so each canonical line maps to a single
    paragraph. Blank lines become empty paragraphs to preserve spacing.
    """
    document = DocxDocument()
    for raw_line in doc.text.split("\n"):
        document.add_paragraph(raw_line)
    for asset in doc.images:
        # Scale to a sensible on-page width while preserving aspect ratio.
        document.add_picture(asset.path, width=DocxInches(min(asset.width / 96.0, 5.0)))
    document.save(docx_path)


def write_document(doc: SyntheticDocument) -> dict:
    """Write every artefact for a document into its own folder and return a
    manifest entry. Each doc gets `dataset/docs/<doc_id>/` holding the .txt,
    .pdf, .docx and .labels.json.
    """
    doc_dir = os.path.join(DOCS_DIR, doc.doc_id)
    os.makedirs(doc_dir, exist_ok=True)
    txt_path = os.path.join(doc_dir, f"{doc.doc_id}.txt")
    pdf_path = os.path.join(doc_dir, f"{doc.doc_id}.pdf")
    docx_path = os.path.join(doc_dir, f"{doc.doc_id}.docx")
    labels_path = os.path.join(doc_dir, f"{doc.doc_id}.labels.json")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(doc.text)

    render_pdf(doc, pdf_path)
    render_docx(doc, docx_path)

    labels = {
        "doc_id": doc.doc_id,
        "doc_type": doc.doc_type.value,
        "persona_ids": doc.persona_ids,
        "spans": [
            {**asdict(s), "type": s.type.value} for s in doc.spans
        ],
        "images": [
            {
                "image_id": a.image_id,
                "path": os.path.relpath(a.path, doc_dir),
                "width": a.width,
                "height": a.height,
                "regions": [asdict(r) for r in a.regions],
            }
            for a in doc.images
        ],
    }
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    return {
        "doc_id": doc.doc_id,
        "doc_type": doc.doc_type.value,
        "dir": os.path.relpath(doc_dir, DOCS_DIR),
        "pdf": os.path.basename(pdf_path),
        "docx": os.path.basename(docx_path),
        "txt": os.path.basename(txt_path),
        "labels": os.path.basename(labels_path),
        "n_spans": len(doc.spans),
        "n_images": len(doc.images),
        "n_regions": sum(len(a.regions) for a in doc.images),
    }


def main() -> None:
    seed(20260620)

    persona1 = get_persona(recurring_prob=1.0)
    doc1 = build_employment_letter("doc_0001", persona1)
    entry1 = write_document(doc1)
    print(json.dumps(entry1, indent=2))

    # doc_0002: HR onboarding for John Smith, with embedded ID/face/signature.
    persona2 = next(p for p in _recurring_personas() if p.persona_id == "p_smith")
    images_dir = os.path.join(DOCS_DIR, "doc_0002", "images")
    doc2 = build_hr_onboarding("doc_0002", persona2, images_dir)
    entry2 = write_document(doc2)
    print(json.dumps(entry2, indent=2))


if __name__ == "__main__":
    main()
