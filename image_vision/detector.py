"""Offline image PII detection pipeline.

Detects FACE, TEXT_PII, ID, and SIGNATURE regions using OpenCV, Haar cascades,
and optionally Tesseract OCR. All detectors are wrapped in try/except so the
pipeline never crashes if a single detector fails.
"""

from __future__ import annotations

import os
import shutil
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
# YuNet face detector setup
# ---------------------------------------------------------------------------

# YuNet ONNX weights are bundled in-repo so face detection runs fully offline.
# Requires OpenCV >= 4.7 for the 2023mar model format.
_YUNET_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "face_detection_yunet_2023mar.onnx",
)
# Score/NMS thresholds: 0.9 score keeps confident faces only; 0.3 NMS dedups
# overlapping boxes. top_k caps candidate boxes before NMS.
_YUNET_SCORE_THRESHOLD = 0.9
_YUNET_NMS_THRESHOLD = 0.3
_YUNET_TOP_K = 5000

_yunet_detector: Optional["cv2.FaceDetectorYN"] = None


def _get_yunet() -> Optional["cv2.FaceDetectorYN"]:
    """Lazily build the YuNet detector, or None if weights/OpenCV unavailable.

    The input size is a placeholder; it is reset per-image in detect_faces.
    """
    global _yunet_detector
    if _yunet_detector is not None:
        return _yunet_detector
    if not hasattr(cv2, "FaceDetectorYN") or not os.path.isfile(_YUNET_MODEL_PATH):
        return None
    try:
        _yunet_detector = cv2.FaceDetectorYN.create(
            _YUNET_MODEL_PATH,
            "",
            (320, 320),
            _YUNET_SCORE_THRESHOLD,
            _YUNET_NMS_THRESHOLD,
            _YUNET_TOP_K,
        )
        return _yunet_detector
    except Exception as exc:
        logging.warning("YuNet detector failed to initialize: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Tesseract setup
# ---------------------------------------------------------------------------

_TESSERACT_FOUND: Optional[bool] = None


def _find_tesseract() -> bool:
    """Locate the Tesseract binary cross-platform and configure pytesseract.

    Resolution order: TESSERACT_CMD env override, then the binary on PATH
    (Linux/macOS/Windows via shutil.which), then common fixed install paths.
    Requires the `pytesseract` module; if it is missing, OCR is unavailable.
    """
    global _TESSERACT_FOUND
    if _TESSERACT_FOUND is not None:
        return _TESSERACT_FOUND

    try:
        import pytesseract
    except Exception:
        # pip package not installed — OCR can't run regardless of the binary.
        _TESSERACT_FOUND = False
        return False

    candidates = []
    env_path = os.environ.get("TESSERACT_CMD")
    if env_path:
        candidates.append(env_path)
    # PATH lookup covers Linux/macOS (`tesseract`) and Windows (`tesseract.exe`).
    on_path = shutil.which("tesseract")
    if on_path:
        candidates.append(on_path)
    # Common fixed locations as a last resort.
    candidates.extend(
        [
            "/usr/bin/tesseract",
            "/usr/local/bin/tesseract",
            "/opt/homebrew/bin/tesseract",
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
    )

    for path in candidates:
        if not (os.path.isfile(path) or shutil.which(path)):
            continue
        try:
            pytesseract.pytesseract.tesseract_cmd = path
            # Confirm the binary actually runs, not just that it exists.
            pytesseract.get_tesseract_version()
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
    """Detect faces, preferring YuNet DNN and falling back to a Haar cascade.

    YuNet (cv2.FaceDetectorYN) is far more robust to angle/occlusion than Haar
    and runs offline from bundled ONNX weights. If the weights or OpenCV's YuNet
    support are missing, fall back to the always-available Haar cascade.
    """
    detector = _get_yunet()
    if detector is not None:
        try:
            return _detect_faces_yunet(detector, image, doc_id, image_id, page_no)
        except Exception as exc:
            logging.warning("YuNet face detection failed, falling back to Haar: %s", exc)
    return _detect_faces_haar(image, doc_id, image_id, page_no)


def _detect_faces_yunet(
    detector: "cv2.FaceDetectorYN",
    image: np.ndarray,
    doc_id: str,
    image_id: str,
    page_no: int,
) -> List[ImageRegion]:
    """Run YuNet on a single image and return FACE regions."""
    regions: List[ImageRegion] = []
    h, w = image.shape[:2]
    if h < 1 or w < 1:
        return regions

    # YuNet requires the input size to match the image it scores.
    detector.setInputSize((w, h))
    _retval, faces = detector.detect(image)
    if faces is None:
        return regions

    for face in faces:
        # face = [x, y, w, h, 5x landmark coords..., score]
        x, y, fw, fh = face[:4]
        score = float(face[-1])
        # Clamp to image bounds; YuNet can return slightly out-of-frame boxes.
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(w, int(x + fw))
        y1 = min(h, int(y + fh))
        if x1 <= x0 or y1 <= y0:
            continue
        regions.append(
            ImageRegion(
                region_id=_new_region_id(),
                doc_id=doc_id,
                image_id=image_id,
                page_no=page_no,
                bbox=(float(x0), float(y0), float(x1), float(y1)),
                label="FACE",
                confidence=round(score, 3),
                source="cv",
            )
        )
    return regions


def _detect_faces_haar(
    image: np.ndarray, doc_id: str, image_id: str, page_no: int
) -> List[ImageRegion]:
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
            # Redaction must favour catching text over precision: low-confidence
            # tokens are often the most distorted PII (e.g. an ID number OCR'd at
            # conf 14). Only drop Tesseract's "no text" sentinel (conf < 0) and
            # require at least one alphanumeric char to avoid boxing pure noise.
            if conf < 0 or not any(c.isalnum() for c in text):
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
