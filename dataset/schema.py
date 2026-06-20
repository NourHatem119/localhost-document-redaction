"""Schema for synthetic document generation (Person A dataset track).

These shapes are generation-side only. The generator produces synthetic
documents plus ground-truth labels, where each label maps cleanly onto a
`PIISpan` from the shared contract (schema.py). Personas model the recurring
cast that drives Cognee's cross-document consistency demo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from schema import BBox, ImageLabel, PIIType


class DocType(str, Enum):
    """Synthetic document templates. Each exercises a different PII mix."""

    EMPLOYMENT_LETTER = "EMPLOYMENT_LETTER"
    MEDICAL_RECORD = "MEDICAL_RECORD"
    INVOICE = "INVOICE"
    BANK_STATEMENT = "BANK_STATEMENT"
    HR_ONBOARDING = "HR_ONBOARDING"


@dataclass
class Persona:
    """A recurring fake person. Holds canonical PII plus name variants so the
    same entity can appear as 'John Smith' / 'J. Smith' / 'Mr Smith' across
    documents for the cross-doc consistency demo.
    """

    persona_id: str
    full_name: str
    name_variants: List[str]
    email: str
    phone: str
    address: str
    ni_number: str
    dob: str
    org: str
    iban: Optional[str] = None
    credit_card: Optional[str] = None
    ssn: Optional[str] = None
    medical_note: Optional[str] = None
    other_id: Optional[str] = None  # passport / driving licence number
    recurring: bool = False


@dataclass
class LabeledSpan:
    """Ground-truth PII occurrence with exact character offsets into the
    document text. Maps 1:1 onto a contract `PIISpan` (source='regex',
    status='auto') when emitted as gold labels.
    """

    type: PIIType
    text: str
    char_start: int
    char_end: int
    persona_id: Optional[str] = None  # links occurrences to the same entity


@dataclass
class LabeledImageRegion:
    """Ground-truth PII region inside an embedded image. `bbox` is in pixel
    coordinates of the image itself ([x0, y0, x1, y1]) — the space the CV /
    OCR vision track operates in. Maps onto a contract `ImageRegion`.
    """

    label: ImageLabel  # "FACE" | "ID" | "SIGNATURE" | "TEXT_PII"
    bbox: BBox
    persona_id: Optional[str] = None  # links the region to an entity


@dataclass
class ImageAsset:
    """An image embedded in a document plus the gold regions inside it."""

    image_id: str
    path: str
    width: int
    height: int
    regions: List[LabeledImageRegion] = field(default_factory=list)


@dataclass
class SyntheticDocument:
    """A generated document with text and gold-standard PII labels.

    `images` carries gold image regions (faces / IDs / signatures / text-PII)
    for the vision track. `image_paths` is kept as a convenience list of the
    same paths for any consumer that only needs the files.
    """

    doc_id: str
    doc_type: DocType
    text: str
    spans: List[LabeledSpan] = field(default_factory=list)
    images: List[ImageAsset] = field(default_factory=list)
    image_paths: List[str] = field(default_factory=list)
    persona_ids: List[str] = field(default_factory=list)
