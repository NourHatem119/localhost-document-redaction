"""OpenAI-compatible client for EXO local inference (primary) with Ollama fallback.

EXO is the primary endpoint for this hackathon (prize eligibility, partner
technology). Ollama is an automatic fallback if EXO has no model loaded.

The client auto-discovers the best available endpoint and, for EXO,
auto-loads a lightweight model if the node is idle.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from typing import List, Optional

import requests

from schema import Chunk, PIISpan, PIIType, PIISource


# Default EXO model: ~2.4 GB, fits in laptop RAM, fast enough for demo.
_DEFAULT_EXO_MODEL = "mlx-community/Llama-3.1-Nemotron-Nano-4B-v1.1-4bit"
# Default Ollama model (fallback).
_DEFAULT_OLLAMA_MODEL = "llama3.2:3b"

# Model selection: use env override, then auto-select based on endpoint.
_DEFAULT_MODEL = os.getenv(
    "OBSCURA_LLM_MODEL",
    _DEFAULT_EXO_MODEL if os.getenv("OBSCURA_USE_EXO", "1") == "1" else _DEFAULT_OLLAMA_MODEL,
)

# Endpoint discovery: EXO first (primary), then Ollama (fallback), then env.
_FALLBACKS = [
    os.getenv("OBSCURA_LLM_URL", ""),
    "http://localhost:52415/v1/chat/completions",   # EXO — primary
    "http://localhost:11434/v1/chat/completions",  # Ollama — fallback
]

# Prompt that constrains the LLM to emit strict JSON.
_SYSTEM_PROMPT = (
    "You are a PII detection engine. "
    "Find all personally identifiable information in the provided text. "
    "Return a JSON array only. No markdown, no explanation. "
    "Each object must have: type (one of PERSON, EMAIL, PHONE, ADDRESS, SSN, "
    "NI_NUMBER, CREDIT_CARD, IBAN, DOB, ORG, MEDICAL, OTHER), "
    "text (the exact substring), char_start (integer), char_end (integer)."
)


class ExoClient:
    """Thin wrapper around an OpenAI-compatible local LLM endpoint."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: str = _DEFAULT_MODEL,
        timeout: int = 30,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.base_url = base_url or self._discover_url()
        if not self.base_url:
            raise RuntimeError(
                "No local LLM endpoint found. "
                "Start EXO (`exo`) or Ollama (`ollama serve`) first, "
                "or set OBSCURA_LLM_URL."
            )

    def _discover_url(self) -> str:
        for url in _FALLBACKS:
            if not url:
                continue
            try:
                resp = requests.get(url.replace("/v1/chat/completions", "/v1/models"), timeout=2)
                if resp.status_code == 200:
                    return url
            except requests.RequestException:
                pass
        return ""

    def detect(self, chunk: Chunk) -> List[PIISpan]:
        """Send chunk text to the local LLM and return PII spans."""
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Text:\n{chunk.text}\n\nReturn JSON array of PII objects.",
            },
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        try:
            resp = requests.post(
                self.base_url, json=payload, timeout=self.timeout
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc

        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return self._parse_response(content, chunk)

    def _parse_response(self, content: str, chunk: Chunk) -> List[PIISpan]:
        """Extract JSON array from the LLM response, adjusting offsets."""
        # Strip markdown fences if the model ignored instructions.
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        try:
            raw = json.loads(content)
        except json.JSONDecodeError as exc:
            # Try to extract the first JSON array from the text.
            match = re.search(r"\[.*\]", content, re.DOTALL)
            if not match:
                raise RuntimeError(f"LLM returned non-JSON: {content[:200]}") from exc
            raw = json.loads(match.group(0))

        if not isinstance(raw, list):
            raise RuntimeError(f"LLM returned non-array JSON: {type(raw).__name__}")

        spans: List[PIISpan] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            pii_type = self._normalize_type(item.get("type", "OTHER"))
            text = item.get("text", "")
            c_start = int(item.get("char_start", 0))
            c_end = int(item.get("char_end", 0))
            # Adjust chunk-relative offsets to document-level.
            doc_start = chunk.char_start + c_start
            doc_end = chunk.char_start + c_end
            spans.append(
                PIISpan(
                    span_id=f"llm-{uuid.uuid4().hex[:8]}",
                    doc_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    type=pii_type,
                    text=text,
                    char_start=doc_start,
                    char_end=doc_end,
                    confidence=0.85,
                    source="llm",
                    status="auto",
                )
            )
        return spans

    @staticmethod
    def _normalize_type(value: str) -> PIIType:
        try:
            return PIIType(value.upper())
        except ValueError:
            return PIIType.OTHER
