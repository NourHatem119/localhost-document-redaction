"""Experimental: quantify how PDF-extracted text diverges from the .txt.

PDF extraction (PyMuPDF) and the generator's .txt are produced independently,
so their character streams need not match. This script reports that gap per doc
so we know how unsafe it would be to reuse .txt offsets against the PDF.

Usage:
    python -m exp.pdf_txt_divergence --docs dataset/docs
"""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path

from ingest.parse_pdf import parse_pdf


def _pdf_text(pdf_path: Path, doc_id: str, tmp_dir: Path) -> str:
    doc = parse_pdf(pdf_path, doc_id, tmp_dir)
    return "\n".join(p.text for p in doc.pages)


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def compare_doc(doc_dir: Path, tmp_dir: Path) -> dict:
    doc_id = doc_dir.name
    pdf_path = doc_dir / f"{doc_id}.pdf"
    txt_path = doc_dir / f"{doc_id}.txt"
    txt = txt_path.read_text(encoding="utf-8")
    pdf = _pdf_text(pdf_path, doc_id, tmp_dir / doc_id)
    return {
        "doc_id": doc_id,
        "txt_chars": len(txt),
        "pdf_chars": len(pdf),
        "char_delta": len(pdf) - len(txt),
        "similarity": round(_similarity(txt, pdf), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PDF text vs .txt.")
    parser.add_argument("--docs", default="dataset/docs", help="docs root")
    parser.add_argument(
        "--tmp", default="exp/_tmp_images", help="scratch dir for extracted images"
    )
    args = parser.parse_args()

    docs_root = Path(args.docs)
    tmp_dir = Path(args.tmp)
    doc_dirs = sorted(d for d in docs_root.iterdir() if d.is_dir())

    print(f"{'doc_id':12} {'txt':>6} {'pdf':>6} {'delta':>6} {'similarity':>10}")
    for doc_dir in doc_dirs:
        if not (doc_dir / f"{doc_dir.name}.pdf").exists():
            continue
        r = compare_doc(doc_dir, tmp_dir)
        print(
            f"{r['doc_id']:12} {r['txt_chars']:>6} {r['pdf_chars']:>6} "
            f"{r['char_delta']:>6} {r['similarity']:>10.4f}"
        )


if __name__ == "__main__":
    main()
