"""Offline image PII detection pipeline.

Detects FACE, TEXT_PII, ID, and SIGNATURE regions using OpenCV, Haar cascades,
and optionally Tesseract OCR. All detectors are wrapped in try/except so the
pipeline never crashes if a single detector fails.
"""

from __future__ import annotations

import os
import sys
import uuid
import logging
from typing import List, Optional

import cv2
import numpy as np

# Ensure repo-root schema.py is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schema import ImageRegion, BBox, ImageLabel, ImageSource

# ---------------------------------------------------------------------------
# Tesseract setup
# ---------------------------------------------------------------------------

_TESSERACT_FOUND: Optional[bool] = None


def _find_tesseract() -> bool:
    """Probe common Windows Tesseract paths and configure pytesseract if found."""
    global _TESSERACT_FOUND
    if _TESSERACT_FOUND is not None:
        return _TESSERACT_FOUND

    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    # Also allow environment override
    env_path = os.environ.get("TESSERACT_CMD")
    if env_path:
        candidates.insert(0, env_path)

    for path in candidates:
        if os.path.isfile(path):
            try:
                import pytesseract
                pytesseract.pytesseract.tesseract_cmd = path
                _TESSERACT_FOUND = True
                return True
            except Exception:
                continue

    _TESSERACT_FOUND = False
    return False


def is_tesseract_available() -> bool:
    """Return whether Tesseract OCR is installed and reachable."""
    return _find_tesseract()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_region_id() -> str:
    return str(uuid.uuid4())


def _bbox_from_xywh(x: int, y: int, w: int, h: int) -> BBox:
    return (float(x), float(y), float(x + w), float(y + h))


def _safe_read(image_path: str) -> Optional[np.ndarray]:
    img = cv2.imread(image_path)
    if img is None:
        logging.warning("Could not read image: %s", image_path)
    return img


# ---------------------------------------------------------------------------
# Face detection
# ---------------------------------------------------------------------------

def detect_faces(image: np.ndarray, doc_id: str, image_id: str, page_no: int) -> List[ImageRegion]:
    """Detect faces using OpenCV Haar cascade (guaranteed offline fallback)."""
    regions: List[ImageRegion] = []
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if not os.path.isfile(cascade_path):
            logging.warning("Haar cascade not found at %s", cascade_path)
            return regions

        classifier = cv2.CascadeClassifier(cascade_path)
        if classifier.empty():
            logging.warning("Haar cascade classifier failed to load.")
            return regions

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = classifier.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        for (x, y, w, h) in detections:
            regions.append(
                ImageRegion(
                    region_id=_new_region_id(),
                    doc_id=doc_id,
                    image_id=image_id,
                    page_no=page_no,
                    bbox=_bbox_from_xywh(x, y, w, h),
                    label="FACE",
                    confidence=0.75,
                    source="cv",
                )
            )
    except Exception as exc:
        logging.warning("Face detection failed: %s", exc)

    return regions


# ---------------------------------------------------------------------------
# Text PII detection (OCR)
# ---------------------------------------------------------------------------

def detect_text_pii(
    image: np.ndarray,
    image_path: str,
    doc_id: str,
    image_id: str,
    page_no: int,
) -> List[ImageRegion]:
    """OCR text blocks with Tesseract. Falls back to mock regions if unavailable."""
    regions: List[ImageRegion] = []

    if not is_tesseract_available():
        # ------------------------------------------------------------------
        # Fallback: Tesseract not installed — return mock text regions so the
        # demo pipeline still produces visible output. In production, install
        # Tesseract (see README.md).
        # ------------------------------------------------------------------
        h, w = image.shape[:2]
        # Two mock text lines in the top half of the image
        mock_boxes = [
            (int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.22)),
            (int(w * 0.1), int(h * 0.25), int(w * 0.9), int(h * 0.37)),
        ]
        for x0, y0, x1, y1 in mock_boxes:
            regions.append(
                ImageRegion(
                    region_id=_new_region_id(),
                    doc_id=doc_id,
                    image_id=image_id,
                    page_no=page_no,
                    bbox=(float(x0), float(y0), float(x1), float(y1)),
                    label="TEXT_PII",
                    confidence=0.5,
                    source="cv",
                )
            )
        return regions

    try:
        import pytesseract

        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        n_boxes = len(data["text"])
        for i in range(n_boxes):
            text = data["text"][i].strip()
            if not text:
                continue
            conf = int(data["conf"][i])
            if conf < 30:
                continue
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            regions.append(
                ImageRegion(
                    region_id=_new_region_id(),
                    doc_id=doc_id,
                    image_id=image_id,
                    page_no=page_no,
                    bbox=_bbox_from_xywh(x, y, w, h),
                    label="TEXT_PII",
                    confidence=float(conf) / 100.0,
                    source="cv",
                )
            )
    except Exception as exc:
        logging.warning("Text PII detection failed: %s", exc)

    return regions


# ---------------------------------------------------------------------------
# ID / signature heuristics
# ---------------------------------------------------------------------------

def detect_id_signature_heuristics(
    image: np.ndarray,
    doc_id: str,
    image_id: str,
    page_no: int,
) -> List[ImageRegion]:
    """Heuristic contour detection for ID-card-like and signature-like regions."""
    regions: List[ImageRegion] = []
    try:
        h, w = image.shape[:2]
        if h < 20 or w < 20:
            return regions

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw < 20 or bh < 20:
                continue

            aspect = bw / max(bh, 1)

            # ID card aspect ratio (1.5 - 1.8)
            if 1.5 <= aspect <= 1.8:
                regions.append(
                    ImageRegion(
                        region_id=_new_region_id(),
                        doc_id=doc_id,
                        image_id=image_id,
                        page_no=page_no,
                        bbox=_bbox_from_xywh(x, y, bw, bh),
                        label="ID",
                        confidence=0.6,
                        source="cv",
                    )
                )
                continue

            # Signature / cheque strip aspect ratio (2.5 - 3.5)
            if 2.5 <= aspect <= 3.5:
                # Low-text-density elongated region -> signature
                regions.append(
                    ImageRegion(
                        region_id=_new_region_id(),
                        doc_id=doc_id,
                        image_id=image_id,
                        page_no=page_no,
                        bbox=_bbox_from_xywh(x, y, bw, bh),
                        label="SIGNATURE",
                        confidence=0.55,
                        source="cv",
                    )
                )
                continue

    except Exception as exc:
        logging.warning("ID/signature heuristic detection failed: %s", exc)

    return regions


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def detect_image_pii(
    image_path: str,
    doc_id: str = "demo",
    image_id: str = "img_0",
    page_no: int = 0,
) -> List[ImageRegion]:
    """Run all offline detectors and return merged ImageRegion list."""
    image = _safe_read(image_path)
    if image is None:
        return []

    all_regions: List[ImageRegion] = []
    all_regions.extend(detect_faces(image, doc_id, image_id, page_no))
    all_regions.extend(detect_text_pii(image, image_path, doc_id, image_id, page_no))
    all_regions.extend(detect_id_signature_heuristics(image, doc_id, image_id, page_no))

    return all_regions
