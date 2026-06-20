"""Image redaction utilities.

Applies irreversible Gaussian blur + pixelation to each bounding box in an image.
"""

from __future__ import annotations

import os
import logging
from typing import List

import cv2
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schema import ImageRegion


# Padding (px) added around text boxes so glyph overhang is fully covered.
_TEXT_PAD = 3
# Fill colour for destroyed text regions (black).
_TEXT_FILL = (0, 0, 0)


def redact_image(image_path: str, regions: List[ImageRegion], output_path: str) -> str:
    """Load image and irreversibly redact each region, then save.

    Text regions (TEXT_PII) are filled solid with padding — blur/pixelation
    leaves short text recoverable by OCR, so text must be destroyed, not
    softened. Other regions (faces, ID, signature) are blurred + pixelated,
    which is unrecoverable enough and visually reads as redaction.

    Parameters
    ----------
    image_path : str
        Path to the original image.
    regions : List[ImageRegion]
        Detected regions to redact.
    output_path : str
        Where to write the redacted image.

    Returns
    -------
    str
        The output path.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    img_h, img_w = image.shape[:2]
    for region in regions:
        x0, y0, x1, y1 = region.bbox
        is_text = region.label == "TEXT_PII"
        pad = _TEXT_PAD if is_text else 0

        x0_i = max(0, int(x0) - pad)
        y0_i = max(0, int(y0) - pad)
        x1_i = min(img_w, int(x1) + pad)
        y1_i = min(img_h, int(y1) + pad)

        if x1_i <= x0_i or y1_i <= y0_i:
            continue

        if is_text:
            # Solid fill: OCR cannot recover anything from a flat rectangle.
            cv2.rectangle(image, (x0_i, y0_i), (x1_i, y1_i), _TEXT_FILL, thickness=-1)
            continue

        roi = image[y0_i:y1_i, x0_i:x1_i]
        if roi.size == 0:
            continue

        # Strong Gaussian blur (51x51 kernel) for irreversible smoothing
        blurred = cv2.GaussianBlur(roi, (51, 51), 0)

        # Pixelation: resize down then up
        h, w = blurred.shape[:2]
        if h > 10 and w > 10:
            small = cv2.resize(blurred, (w // 10, h // 10), interpolation=cv2.INTER_LINEAR)
            pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            pixelated = blurred

        image[y0_i:y1_i, x0_i:x1_i] = pixelated

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, image)
    logging.info("Redacted image saved to %s", output_path)
    return output_path
