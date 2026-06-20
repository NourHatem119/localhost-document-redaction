"""Obscura pipeline: EXO detection + redaction engine.

Person A vertical slice — owns the full text pipeline from PDF ingestion to
redacted PDF output. Uses EXO as the primary LLM inference engine (running
locally on localhost:52415) with Ollama as an automatic fallback.
"""

from pipeline.detect import detect_regex, detect_llm, merge_spans
from pipeline.redact import apply_redactions, create_redaction_job
from pipeline.exo_client import ExoClient

__all__ = [
    "detect_regex",
    "detect_llm",
    "merge_spans",
    "apply_redactions",
    "create_redaction_job",
    "ExoClient",
]
