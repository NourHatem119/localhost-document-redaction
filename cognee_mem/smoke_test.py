"""T1 smoke test — Cognee quickstart style (remember / recall / forget).

Levels:
  1. CONFIG ONLY (works anywhere, no exo, no model):
        python -m cognee_mem.smoke_test --config-only
  2. EMBEDDINGS ONLY (no exo; verifies Fastembed runs offline once cached):
        python -m cognee_mem.smoke_test --embed-only
  3. FULL end-to-end (needs exo running + Fastembed model cached):
        python -m cognee_mem.smoke_test

The full run is the real T1 "done" gate: run it on the demo machine with exo up
and Wi-Fi OFF to prove zero cloud calls.
"""

from __future__ import annotations

import asyncio
import sys

import cognee_mem.config as cfg  # loads .env on import


def config_only() -> None:
    import json

    print(json.dumps(cfg.config_summary(), indent=2))
    print("\n[ok] config loaded. Next: --embed-only (no exo), then full run on the exo box.")


def embed_only() -> None:
    """Prove Fastembed produces vectors locally — no exo, no API key."""
    import os

    from fastembed import TextEmbedding

    name = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    model = TextEmbedding(name)
    vec = list(model.embed(["John Smith emailed jane.doe@acme.com about invoice 4471."]))
    print(f"[ok] Fastembed produced a {len(vec[0])}-dim vector with no network.")


async def full() -> None:
    import cognee

    print("[..] forget(everything=True)")
    await cognee.forget(everything=True)
    print("[..] remember(...)  (hits exo for LLM + Fastembed for embeddings)")
    await cognee.remember("John Smith emailed jane.doe@acme.com about invoice 4471.")
    print("[..] recall('Who emailed about the invoice?')")
    results = await cognee.recall(query_text="Who emailed about the invoice?")
    for r in results:
        print("     ->", getattr(r, "text", r))
    print("\n[done] end-to-end ran. If Wi-Fi was off, T1 offline gate is PASSED.")


if __name__ == "__main__":
    if "--config-only" in sys.argv:
        config_only()
    elif "--embed-only" in sys.argv:
        embed_only()
    else:
        asyncio.run(full())
