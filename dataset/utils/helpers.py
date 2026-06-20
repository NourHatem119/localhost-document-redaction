"""Helper functions for generating and persisting synthetic personas.

Reuses the Faker (en_GB) generator and the NI/passport helpers defined in
`dataset.personas`, so persona construction stays in one place.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Union

from dataset.personas import generate_persona, seed as seed_rng
from dataset.schema import Persona


def generate_personas(n: int, seed: Optional[int] = None) -> List[Persona]:
    """Generate N one-off personas via Faker (en_GB locale).

    Each persona gets a unique sequential id. Pass `seed` for a reproducible
    batch; the NI and passport numbers are produced by the helpers in
    `dataset.personas`.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if seed is not None:
        seed_rng(seed)
    return [generate_persona(f"p_gen_{i:04d}") for i in range(n)]


def write_personas(personas: List[Persona], path: Union[str, Path]) -> Path:
    """Write personas to a JSON file, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [asdict(p) for p in personas]
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return path


def read_personas(path: Union[str, Path]) -> List[Persona]:
    """Read personas back from a JSON file into Persona objects."""
    path = Path(path)
    records = json.loads(path.read_text(encoding="utf-8"))
    return [Persona(**record) for record in records]
