"""Shared data contract for Obscura.

This is the sacred schema referenced in spec section 3. Every component
(text detection, image vision, Cognee memory, redaction, UI, Overmind traces)
codes against these shapes. Do not drift them without telling the whole team.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class PIIType(str, Enum):
    """Enumerated PII categories. Locked at 11:15 per spec."""

    PERSON = "PERSON"
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    ADDRESS = "ADDRESS"
    SSN = "SSN"
    NI_NUMBER = "NI_NUMBER"
    CREDIT_CARD = "CREDIT_CARD"
    IBAN = "IBAN"
    DOB = "DOB"
    ORG = "ORG"
    MEDICAL = "MEDICAL"
    OTHER = "OTHER"


# Source of a detection or memory record.
PIISource = str  # "regex" | "llm" | "cognee"
# Review status of a span.
PIIStatus = str  # "auto" | "confirmed" | "rejected"
# Image region label.
ImageLabel = str  # "FACE" | "ID" | "SIGNATURE" | "TEXT_PII"
# Image region source.
ImageSource = str  # "captur" | "cv"

# Axis-aligned bounding box [x0, y0, x1, y1].
BBox = Tuple[float, float, float, float]


@dataclass
class Page:
    page_no: int
    width: float
    height: float
    text: str


@dataclass
class ImageRef:
    image_id: str
    doc_id: str
    page_no: int
    path: str
    bbox: Optional[BBox] = None


@dataclass
class Document:
    doc_id: str
    source_path: str
    pages: List[Page] = field(default_factory=list)
    images: List[ImageRef] = field(default_factory=list)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    page_no: int
    text: str
    char_start: int
    char_end: int
    bbox: Optional[BBox] = None


@dataclass
class PIISpan:
    span_id: str
    doc_id: str
    chunk_id: Optional[str]
    type: PIIType
    text: str
    char_start: int
    char_end: int
    bbox: Optional[BBox] = None
    confidence: float = 1.0
    source: PIISource = "regex"
    status: PIIStatus = "auto"


@dataclass
class ImageRegion:
    region_id: str
    doc_id: str
    image_id: str
    page_no: int
    bbox: BBox
    label: ImageLabel
    confidence: float = 1.0
    source: ImageSource = "cv"


@dataclass
class RedactionJob:
    doc_id: str
    spans: List[PIISpan] = field(default_factory=list)
    regions: List[ImageRegion] = field(default_factory=list)
    output_path: Optional[str] = None


@dataclass
class Trace:
    trace_id: str
    doc_id: str
    decisions: List[PIISpan] = field(default_factory=list)
    corrections: List[PIISpan] = field(default_factory=list)
