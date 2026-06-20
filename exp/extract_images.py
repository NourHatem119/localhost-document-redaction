"""Experimental: extract and save embedded images from an input PDF, and
optionally detect + redact faces in each extracted image.

Thin wrapper over the ingest layer — `parse_document` already writes embedded
images to `image_dir` and records each as an `ImageRef`. This script points
extraction at an output folder and reports what was saved (id, page, bbox, size,
on-disk path). With --redact-faces it runs the offline YuNet/Haar face detector
on each image and writes a redacted copy (blurred + pixelated faces) beside it.

Usage:
    python -m exp.extract_images --doc dataset/docs/doc_0002/doc_0002.pdf
    python -m exp.extract_images --doc some.pdf --out exp/_extracted_images
    python -m exp.extract_images --doc some.pdf --redact-faces
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from image_vision.detector import detect_faces
from image_vision.redaction import redact_image
from ingest.parse import parse_document


def _fmt_bbox(bbox) -> str:
    if bbox is None:
        return "None"
    return "(" + ", ".join(f"{v:.1f}" for v in bbox) + ")"


def _redact_faces(image_path: str, doc_id: str, image_id: str, page_no: int) -> str | None:
    """Detect faces in one image and write a redacted copy beside it.

    Returns the redacted path, or None if the image is unreadable or has no
    detected faces (nothing to redact).
    """
    image = cv2.imread(image_path)
    if image is None:
        return None
    faces = detect_faces(image, doc_id, image_id, page_no)
    if not faces:
        return None
    src = Path(image_path)
    out_path = src.with_suffix(f".redacted{src.suffix}")
    redact_image(image_path, faces, str(out_path))
    return str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract images from a PDF.")
    parser.add_argument(
        "--doc", default="dataset/docs/doc_0001/doc_0001.pdf", help="path to a PDF"
    )
    parser.add_argument(
        "--out", default="dataset/data", help="output directory for images"
    )
    parser.add_argument(
        "--redact-faces",
        action="store_true",
        help="detect and blur faces in each extracted image",
    )
    args = parser.parse_args()

    doc_path = Path(args.doc)
    if not doc_path.exists():
        raise SystemExit(f"file not found: {doc_path}")
    if doc_path.suffix.lower() != ".pdf":
        raise SystemExit(f"this script extracts from .pdf; got {doc_path.suffix!r}")

    out_dir = Path(args.out) / doc_path.stem
    doc = parse_document(doc_path, image_dir=out_dir)

    print(f"doc_id    : {doc.doc_id}")
    print(f"source    : {doc.source_path}")
    print(f"out dir   : {out_dir}")
    print(f"images    : {len(doc.images)}")

    if not doc.images:
        print("\nNo embedded images found in this PDF.")
        return

    print(f"\n{'image_id':28} {'page':>4} {'bytes':>8}  bbox")
    for ref in doc.images:
        size = Path(ref.path).stat().st_size
        print(f"{ref.image_id:28} {ref.page_no:>4} {size:>8}  {_fmt_bbox(ref.bbox)}")
        print(f"    -> {ref.path}")
        if args.redact_faces:
            redacted = _redact_faces(ref.path, ref.doc_id, ref.image_id, ref.page_no)
            if redacted:
                print(f"    faces redacted -> {redacted}")
            else:
                print("    no faces detected")


if __name__ == "__main__":
    main()
