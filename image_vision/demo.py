"""Demo script for image_vision package.

Creates synthetic test images, runs PII detection, and produces redacted output.
Can be run via:

    python -m image_vision.demo
    python image_vision/demo.py
"""

from __future__ import annotations

import os
import sys
import json
from typing import List

import cv2
import numpy as np

# Ensure repo-root schema.py is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from image_vision.detector import detect_image_pii, is_tesseract_available
from image_vision.redaction import redact_image
from schema import ImageRegion


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(SCRIPT_DIR, "test_images")
REDACTED_DIR = os.path.join(SCRIPT_DIR, "redacted_output")


def _make_face_text_image(path: str) -> None:
    """Create a synthetic image with a face-like oval + text lines."""
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255

    # Face-like oval (skin tone)
    cv2.ellipse(img, (300, 180), (80, 100), 0, 0, 360, (180, 150, 120), -1)
    # Eyes
    cv2.circle(img, (270, 160), 10, (50, 50, 50), -1)
    cv2.circle(img, (330, 160), 10, (50, 50, 50), -1)
    # Mouth
    cv2.ellipse(img, (300, 220), (30, 15), 0, 0, 180, (50, 50, 50), -1)

    # Text lines
    cv2.putText(img, "John Doe", (50, 340), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.putText(img, "johndoe@example.com", (50, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    cv2.imwrite(path, img)


def _make_id_signature_image(path: str) -> None:
    """Create a synthetic image with an ID-card rectangle + signature area."""
    img = np.ones((400, 600, 3), dtype=np.uint8) * 240

    # ID card rectangle (aspect ratio ~1.586, close to credit-card size)
    x0, y0 = 50, 50
    w, h = 240, 151
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (200, 200, 200), -1)
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (100, 100, 100), 2)
    cv2.putText(img, "ID CARD", (x0 + 60, y0 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)

    # Signature strip (elongated rectangle, aspect ratio ~3.0)
    sx0, sy0 = 50, 250
    sw, sh = 300, 100
    cv2.rectangle(img, (sx0, sy0), (sx0 + sw, sy0 + sh), (220, 220, 220), -1)
    cv2.rectangle(img, (sx0, sy0), (sx0 + sw, sy0 + sh), (100, 100, 100), 2)
    # Fake signature scribble
    for i in range(20):
        x1 = sx0 + 20 + i * 13
        y1 = sy0 + 30 + (i % 5) * 10
        x2 = x1 + 10
        y2 = y1 + 8
        cv2.line(img, (x1, y1), (x2, y2), (80, 80, 80), 2)

    cv2.imwrite(path, img)


def _ensure_test_images() -> List[str]:
    """Create synthetic test images if they don't already exist."""
    os.makedirs(TEST_DIR, exist_ok=True)
    paths = [
        os.path.join(TEST_DIR, "face_text.png"),
        os.path.join(TEST_DIR, "id_signature.png"),
    ]
    if not os.path.isfile(paths[0]):
        _make_face_text_image(paths[0])
        print(f"Created test image: {paths[0]}")
    if not os.path.isfile(paths[1]):
        _make_id_signature_image(paths[1])
        print(f"Created test image: {paths[1]}")
    return paths


def _pretty_print_regions(regions: List[ImageRegion]) -> None:
    if not regions:
        print("  No regions detected.")
        return
    for r in regions:
        bbox = f"({r.bbox[0]:.1f}, {r.bbox[1]:.1f}, {r.bbox[2]:.1f}, {r.bbox[3]:.1f})"
        print(f"  {r.label:12s} conf={r.confidence:.2f} bbox={bbox} source={r.source}")


def main() -> None:
    print("=" * 60)
    print("Obscura image_vision demo")
    print("=" * 60)
    print(f"Tesseract available: {is_tesseract_available()}")
    print()

    image_paths = _ensure_test_images()
    os.makedirs(REDACTED_DIR, exist_ok=True)

    for idx, img_path in enumerate(image_paths):
        image_id = f"img_{idx}"
        print(f"--- Processing {os.path.basename(img_path)} (image_id={image_id}) ---")

        regions = detect_image_pii(img_path, doc_id="demo", image_id=image_id, page_no=idx)
        print(f"Detected {len(regions)} region(s):")
        _pretty_print_regions(regions)

        out_path = os.path.join(REDACTED_DIR, f"redacted_{os.path.basename(img_path)}")
        redact_image(img_path, regions, out_path)
        print(f"Redacted output: {out_path}")
        print()

    print("=" * 60)
    print("Demo complete.")


if __name__ == "__main__":
    main()
