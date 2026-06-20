"""Image PII detection and redaction package for Obscura.

Pure offline computer-vision pipeline for detecting faces, text PII, ID cards,
and signatures in images. No cloud services required.
"""

import os
import sys

# Ensure repo-root schema.py is importable when running as a module.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from image_vision.detector import detect_image_pii, is_tesseract_available
from image_vision.redaction import redact_image

__all__ = [
    "detect_image_pii",
    "redact_image",
    "is_tesseract_available",
]
