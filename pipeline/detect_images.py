"""Image PII detection for the pipeline.

Runs the offline image detectors (YuNet faces + Tesseract OCR) over each image
extracted by ingest, and returns regions grouped per image for redaction.

Two safety gates:
  * TEXT_PII regions are kept only when Tesseract is actually available.
    Without it, `detect_image_pii` emits *mock* boxes; we drop those so the
    pipeline never blurs fabricated rectangles.
  * The noisy ID/SIGNATURE contour heuristics are excluded — this pass covers
    faces and OCR text only.
"""

from __future__ import annotations

from typing import Dict, List

from schema import Document, ImageRef, ImageRegion
from image_vision.detector import detect_image_pii, is_tesseract_available

# Labels this pass will redact. FACE (YuNet) is always real; TEXT_PII is only
# trustworthy when Tesseract is installed (otherwise it is mock output).
_FACE_LABEL = "FACE"
_TEXT_LABEL = "TEXT_PII"


def detect_document_images(
    document: Document, progress: bool = True
) -> Dict[str, List[ImageRegion]]:
    """Detect PII regions in every extracted image of a document.

    Returns a mapping of `image_id -> [ImageRegion, ...]` containing only the
    regions we intend to redact (faces, plus OCR text when Tesseract is real).
    Images with no redactable regions are omitted.
    """
    ocr_ready = is_tesseract_available()
    if progress:
        n = len(document.images)
        ocr_state = "real OCR" if ocr_ready else "OCR disabled (Tesseract missing)"
        print(f"[INFO] Image detection over {n} image(s) — faces + {ocr_state}")

    results: Dict[str, List[ImageRegion]] = {}
    for ref in document.images:
        regions = _detect_one(ref, ocr_ready)
        if regions:
            results[ref.image_id] = regions
            if progress:
                faces = sum(1 for r in regions if r.label == _FACE_LABEL)
                texts = sum(1 for r in regions if r.label == _TEXT_LABEL)
                print(
                    f"[INFO]   {ref.image_id}: {faces} face(s), {texts} text region(s)"
                )
    return results


def _detect_one(ref: ImageRef, ocr_ready: bool) -> List[ImageRegion]:
    """Detect and filter regions for a single image reference."""
    raw = detect_image_pii(
        ref.path, doc_id=ref.doc_id, image_id=ref.image_id, page_no=ref.page_no
    )
    kept: List[ImageRegion] = []
    for region in raw:
        if region.label == _FACE_LABEL:
            kept.append(region)
        elif region.label == _TEXT_LABEL and ocr_ready:
            kept.append(region)
        # ID / SIGNATURE heuristics and mock text boxes are intentionally dropped.
    return kept
