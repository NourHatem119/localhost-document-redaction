"""Persona factory for synthetic document generation.

Kept deliberately simple: a hand-written list of dicts for the recurring cast
(the entities that drive Cognee's cross-document consistency demo) and a
Faker-backed generator for one-off personas. `get_persona()` either picks a
recurring persona or mints a fresh one.
"""

from __future__ import annotations

import random
from typing import List, Optional

from faker import Faker

from dataset.schema import Persona

# UK-flavoured fakes; seed at call sites for determinism.
_faker = Faker("en_GB")


def seed(value: int) -> None:
    """Seed both Faker and the stdlib RNG for reproducible datasets."""
    Faker.seed(value)
    random.seed(value)


def _ni_number() -> str:
    """UK National Insurance number: 2 letters, 6 digits, 1 suffix letter.
    Faker has no NI generator, so synthesize a format-valid one.
    """
    prefix = "".join(random.choice("ABCEGHJKLMNOPRSTWXYZ") for _ in range(2))
    digits = "".join(random.choice("0123456789") for _ in range(6))
    suffix = random.choice("ABCD")
    return f"{prefix}{digits}{suffix}"


def _passport_number() -> str:
    """Nine-digit UK passport number."""
    return "".join(random.choice("0123456789") for _ in range(9))


def _name_variants(full_name: str) -> List[str]:
    """Derive 'J. Smith' / 'Mr Smith' style variants from a full name."""
    parts = full_name.split()
    first, last = parts[0], parts[-1]
    return [full_name, f"{first[0]}. {last}", f"Mr {last}"]


# Recurring cast: appears across multiple documents in varied forms so Cognee
# can collapse them into single entities. Canonical PII is fixed here.
RECURRING_CAST: List[dict] = [
    {
        "persona_id": "p_smith",
        "full_name": "John Smith",
        "name_variants": ["John Smith", "J. Smith", "Mr Smith"],
        "email": "john.smith@meridian-health.co.uk",
        "phone": "07700 900142",
        "address": "14 Carlisle Road, Manchester, M20 4PL",
        "ni_number": "QQ123456C",
        "dob": "1984-03-12",
        "org": "Meridian Health NHS Trust",
        "iban": "GB29MIDL40051512345678",
        "credit_card": "4716 2210 4456 1809",
        "ssn": None,
        "medical_note": "Type 2 diabetes; metformin 500mg twice daily",
        "other_id": "503271845",
        "recurring": True,
    },
    {
        "persona_id": "p_okafor",
        "full_name": "Amara Okafor",
        "name_variants": ["Amara Okafor", "A. Okafor", "Ms Okafor"],
        "email": "a.okafor@brightside-legal.co.uk",
        "phone": "07700 900318",
        "address": "8 Foundry Lane, Bristol, BS1 6RT",
        "ni_number": "AB654321A",
        "dob": "1990-11-27",
        "org": "Brightside Legal LLP",
        "iban": "GB94BARC20038412345679",
        "credit_card": "5500 0055 1234 5678",
        "ssn": None,
        "medical_note": None,
        "other_id": "712904553",
        "recurring": True,
    },
    {
        "persona_id": "p_walsh",
        "full_name": "Daniel Walsh",
        "name_variants": ["Daniel Walsh", "D. Walsh", "Mr Walsh"],
        "email": "daniel.walsh@northgate-finance.com",
        "phone": "07700 900775",
        "address": "27 Kingsway, Leeds, LS1 2HQ",
        "ni_number": "JT998877B",
        "dob": "1978-06-04",
        "org": "Northgate Finance Ltd",
        "iban": "GB33BUKB20201512345680",
        "credit_card": "4024 0071 5523 4412",
        "ssn": "078-05-1120",
        "medical_note": None,
        "other_id": "664218907",
        "recurring": True,
    },
    {
        "persona_id": "p_patel",
        "full_name": "Priya Patel",
        "name_variants": ["Priya Patel", "P. Patel", "Ms Patel"],
        "email": "priya.patel@meridian-health.co.uk",
        "phone": "07700 900261",
        "address": "5 Elmwood Court, Birmingham, B15 3DH",
        "ni_number": "WK112233C",
        "dob": "1987-09-15",
        "org": "Meridian Health NHS Trust",
        "iban": "GB68CITI18500812345681",
        "credit_card": "6011 0009 9013 9424",
        "ssn": None,
        "medical_note": "Penicillin allergy; asthma, salbutamol inhaler PRN",
        "other_id": "590183274",
        "recurring": True,
    },
    {
        "persona_id": "p_novak",
        "full_name": "Stefan Novak",
        "name_variants": ["Stefan Novak", "S. Novak", "Mr Novak"],
        "email": "stefan.novak@northgate-finance.com",
        "phone": "07700 900503",
        "address": "92 Harbour View, Liverpool, L3 4BG",
        "ni_number": "PR445566A",
        "dob": "1982-01-22",
        "org": "Northgate Finance Ltd",
        "iban": "GB17NWBK60161312345682",
        "credit_card": "4539 1488 0343 6467",
        "ssn": None,
        "medical_note": None,
        "other_id": "638295107",
        "recurring": True,
    },
]


def _recurring_personas() -> List[Persona]:
    return [Persona(**d) for d in RECURRING_CAST]


def generate_persona(persona_id: str) -> Persona:
    """Mint a fresh one-off persona using Faker."""
    full_name = _faker.name()
    # Strip Faker honorific prefixes so name variants stay clean.
    for prefix in ("Dr ", "Mr ", "Mrs ", "Ms ", "Miss "):
        if full_name.startswith(prefix):
            full_name = full_name[len(prefix):]
            break
    address = _faker.address().replace("\n", ", ")
    return Persona(
        persona_id=persona_id,
        full_name=full_name,
        name_variants=_name_variants(full_name),
        email=_faker.email(),
        phone=_faker.phone_number(),
        address=address,
        ni_number=_ni_number(),
        dob=_faker.date_of_birth(minimum_age=22, maximum_age=65).isoformat(),
        org=_faker.company(),
        iban=_faker.iban(),
        credit_card=_faker.credit_card_number(),
        ssn=None,
        medical_note=None,
        other_id=_passport_number(),
        recurring=False,
    )


def get_persona(recurring_prob: float = 0.5, used_ids: Optional[set] = None) -> Persona:
    """Pick a recurring persona or generate a one-off.

    With probability `recurring_prob`, return a member of the recurring cast
    (preferring ones not already in `used_ids` so the cast spreads across docs).
    Otherwise mint a fresh persona with a unique id.
    """
    cast = _recurring_personas()
    if random.random() < recurring_prob:
        if used_ids:
            unused = [p for p in cast if p.persona_id not in used_ids]
            if unused:
                return random.choice(unused)
        return random.choice(cast)
    new_id = f"p_oneoff_{random.randint(100000, 999999)}"
    return generate_persona(new_id)
