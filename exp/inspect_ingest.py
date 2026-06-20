"""Experimental: inspect the output of the ingest module for one document.

Runs parse + chunk and prints a readable breakdown — page geometry, canonical
text, a sample of word boxes, extracted images, and each chunk with its char
range and union bbox. Also asserts every chunk offset round-trips against the
canonical page text so you can eyeball correctness.

Usage:
    python -m exp.inspect_ingest --doc dataset/docs/doc_0001/doc_0001.pdf
    python -m exp.inspect_ingest --doc dataset/docs/doc_0001/doc_0001.docx \\
        --overlap-ratio 0.2 --text-preview 400
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ingest.chunk import chunk_document
from ingest.parse import SUPPORTED_SUFFIXES, parse_document


def _rule(label: str) -> None:
    print(f"\n=== {label} ===")


def _fmt_bbox(bbox) -> str:
    if bbox is None:
        return "None"
    return "(" + ", ".join(f"{v:.1f}" for v in bbox) + ")"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect ingest output for a doc.")
    parser.add_argument(
        "--doc", default="dataset/docs/doc_0001/doc_0001.pdf", help="path to a PDF/DOCX"
    )
    parser.add_argument(
        "--overlap-ratio",
        type=float,
        default=0.2,
        help="fraction of an adjacent paragraph to overlap (0-1)",
    )
    parser.add_argument(
        "--text-preview", type=int, default=240, help="chars of page text to show"
    )
    parser.add_argument(
        "--boxes", type=int, default=8, help="number of word boxes to sample per page"
    )
    parser.add_argument(
        "--image-dir", default="exp/_tmp_images", help="where to extract images"
    )
    args = parser.parse_args()

    doc_path = Path(args.doc)
    if not doc_path.exists():
        raise SystemExit(f"file not found: {doc_path}")
    if doc_path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise SystemExit(
            f"ingest handles {SUPPORTED_SUFFIXES}, got {doc_path.suffix!r}. "
            f"Point --doc at a supported file (e.g. {doc_path.with_suffix('.pdf')})."
        )
    image_dir = Path(args.image_dir) / doc_path.stem
    doc = parse_document(doc_path, image_dir=image_dir)

    _rule("DOCUMENT")
    print(f"doc_id      : {doc.doc_id}")
    print(f"source_path : {doc.source_path}")
    print(f"pages       : {len(doc.pages)}")
    print(f"images      : {len(doc.images)}")

    for page in doc.pages:
        _rule(f"PAGE {page.page_no}")
        print(f"size        : {page.width:.1f} x {page.height:.1f} pt")
        print(f"text chars  : {len(page.text)}")
        print(f"word boxes  : {len(page.word_boxes)}")
        preview = page.text[: args.text_preview].replace("\n", "\\n")
        suffix = "..." if len(page.text) > args.text_preview else ""
        print(f"text        : {preview}{suffix}")

        if page.word_boxes:
            print(f"\nfirst {min(args.boxes, len(page.word_boxes))} word boxes "
                  f"[char_start:char_end] -> word | bbox")
            for start, end, bbox in page.word_boxes[: args.boxes]:
                word = page.text[start:end]
                print(f"  [{start:>4}:{end:>4}] {word!r:24} {_fmt_bbox(bbox)}")

    if doc.images:
        _rule("IMAGES")
        for ref in doc.images:
            print(f"  {ref.image_id}  page={ref.page_no}  bbox={_fmt_bbox(ref.bbox)}")
            print(f"      -> {ref.path}")

    chunks = chunk_document(doc, overlap_ratio=args.overlap_ratio)
    _rule(f"CHUNKS (overlap_ratio={args.overlap_ratio})")
    print(f"count       : {len(chunks)}")
    page_by_no = {p.page_no: p for p in doc.pages}
    mismatches = 0
    for chunk in chunks:
        page_text = page_by_no[chunk.page_no].text
        ok = page_text[chunk.char_start : chunk.char_end] == chunk.text
        mismatches += 0 if ok else 1
        words = len(chunk.text.split())
        print(
            f"  {chunk.chunk_id}  [{chunk.char_start}:{chunk.char_end}] "
            f"words={words} bbox={_fmt_bbox(chunk.bbox)} offset_ok={ok}"
        )

    _rule("OFFSET CHECK")
    if mismatches == 0:
        print("all chunk offsets round-trip against canonical page text OK")
    else:
        print(f"WARNING: {mismatches} chunk(s) failed offset round-trip")


if __name__ == "__main__":
    main()
