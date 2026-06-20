"""Image redaction + reinsertion into the PDF.

Approach: blur/pixelate detected regions in the image's own pixel space (where
the detector's coordinates are valid), then swap the redacted bytes back into
the PDF by xref via PyMuPDF `page.replace_image`. This keeps redaction in pixel
space — avoiding any image-pixel vs PDF-point coordinate mismatch — and the
result is true pixel destruction, not a black box over the original.
"""

from __future__ import annotations

import os
import tempfile
from typing import Dict, List

import fitz  # PyMuPDF

from schema import Document, ImageRef, ImageRegion
from image_vision.redaction import redact_image


def _xref_from_image_id(image_id: str) -> int | None:
    """Recover the PDF image xref encoded by the parser as `..._x{xref}`."""
    marker = image_id.rfind("_x")
    if marker == -1:
        return None
    try:
        return int(image_id[marker + 2 :])
    except ValueError:
        return None


def reinsert_redacted_images(
    document: Document,
    regions_by_image: Dict[str, List[ImageRegion]],
    input_path: str,
    output_path: str,
    progress: bool = True,
) -> int:
    """Blur detected regions and write the redacted images back into the PDF.

    Reads `input_path`, replaces each affected embedded image with a redacted
    copy, and saves to `output_path`. Returns the number of images redacted.
    Images with no regions are left untouched.
    """
    refs_by_id: Dict[str, ImageRef] = {r.image_id: r for r in document.images}
    doc = fitz.open(input_path)
    redacted = 0
    tmp_paths: List[str] = []
    try:
        for image_id, regions in regions_by_image.items():
            ref = refs_by_id.get(image_id)
            if ref is None or not regions:
                continue
            xref = _xref_from_image_id(image_id)
            if xref is None:
                if progress:
                    print(f"[WARN] no xref for image {image_id}; skipping")
                continue

            # Redact in pixel space to a temp file.
            fd, blurred_path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            tmp_paths.append(blurred_path)
            redact_image(ref.path, regions, blurred_path)

            page = doc.load_page(ref.page_no - 1)  # ImageRef.page_no is 1-indexed
            page.replace_image(xref, filename=blurred_path)
            redacted += 1
            if progress:
                print(f"[INFO]   redacted image {image_id} ({len(regions)} region(s))")

        # PyMuPDF refuses a full (non-incremental) save back onto the file it
        # opened. Save to a temp file in the SAME directory as the output (so
        # os.replace stays on one filesystem), then move it into place. This
        # lets input_path == output_path work.
        out_dir = os.path.dirname(os.path.abspath(output_path))
        fd, save_tmp = tempfile.mkstemp(suffix=".pdf", dir=out_dir)
        os.close(fd)
        tmp_paths.append(save_tmp)
        doc.save(save_tmp, deflate=True, garbage=4)
        doc.close()
        os.replace(save_tmp, output_path)
        tmp_paths.remove(save_tmp)
    finally:
        if not doc.is_closed:
            doc.close()
        for path in tmp_paths:
            try:
                os.remove(path)
            except OSError:
                pass
    return redacted
