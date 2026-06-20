"""T1 — Load Cognee's local/offline config from the project-root `.env`.

Per the Cognee quickstart, configuration lives in `.env` (read automatically by
Cognee via pydantic-settings), not in code. This module just loads that `.env`
early and exposes a tiny summary/verify helper so other modules can import one
thing and know the environment is set.

See `.env.example` for the values. Key points:
  - LLM  -> exo (OpenAI-compatible, http://localhost:52415/v1)
  - Embeddings -> Fastembed (local ONNX, CPU, no API key, offline)
  - Graph + vector stores -> in-process (ladybug + lancedb), no servers

Deviation from spec §5: Cognee 1.1.3 has no NetworkX graph adapter; the
in-process default is `ladybug` (same "zero DB setup" intent).
"""

from __future__ import annotations

import os
from pathlib import Path

# Load the project-root .env before cognee reads its settings.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _PROJECT_ROOT / ".env"

try:
    from dotenv import load_dotenv

    load_dotenv(_ENV_PATH)
except ModuleNotFoundError:  # python-dotenv ships with cognee, but be defensive
    if _ENV_PATH.exists():
        for line in _ENV_PATH.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def config_summary() -> dict:
    """Return the effective Cognee env config (no secrets) for logging/asserts."""
    return {
        "env_file": str(_ENV_PATH),
        "env_file_found": _ENV_PATH.exists(),
        "llm_provider": os.environ.get("LLM_PROVIDER"),
        "llm_model": os.environ.get("LLM_MODEL"),
        "llm_endpoint": os.environ.get("LLM_ENDPOINT"),
        "embedding_provider": os.environ.get("EMBEDDING_PROVIDER"),
        "embedding_model": os.environ.get("EMBEDDING_MODEL"),
        "embedding_dimensions": os.environ.get("EMBEDDING_DIMENSIONS"),
        "skip_connection_test": os.environ.get("COGNEE_SKIP_CONNECTION_TEST"),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(config_summary(), indent=2))
