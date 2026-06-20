# Dataset Generation

Synthetic UK document data for Obscura's text PII pipeline (Person A track).
All fakes use Faker's `en_GB` locale. National Insurance and passport numbers
are synthesized by local helpers since Faker has no UK generators for them.

## Layout

| Path | Purpose |
|---|---|
| `schema.py` | Generation-side shapes: `Persona`, `LabeledSpan`, `SyntheticDocument`, `DocType` |
| `personas.py` | Recurring cast (hand-written) + Faker one-off generator + `get_persona()` |
| `utils/helpers.py` | Generate / write / read personas |
| `generate_personas.py` | CLI runner that produces a JSON persona file |
| `data/` | Generated output (JSON) |

## Generate personas

```bash
python -m dataset.generate_personas --count 20 --out dataset/data/personas.json --seed 42
```

| Flag | Default | Meaning |
|---|---|---|
| `-n`, `--count` | `20` | number of one-off personas to generate |
| `-o`, `--out` | `dataset/data/personas.json` | output JSON path |
| `-s`, `--seed` | none | RNG seed for reproducible batches |

## Helper functions (`dataset/utils/helpers.py`)

```python
from dataset.utils.helpers import generate_personas, write_personas, read_personas

personas = generate_personas(20, seed=42)   # List[Persona] via Faker en_GB
write_personas(personas, "dataset/data/personas.json")
personas = read_personas("dataset/data/personas.json")
```

- `generate_personas(n, seed=None)` — N one-off personas with unique ids; reuses
  the NI/passport helpers from `personas.py`.
- `write_personas(personas, path)` — serialize to JSON, creating parent dirs.
- `read_personas(path)` — load JSON back into `Persona` objects.

## Recurring cast

`personas.py` also holds `RECURRING_CAST`: five hand-written personas with
stable, checksum-valid IBAN/credit-card values and name variants
(`John Smith` / `J. Smith` / `Mr Smith`). These drive Cognee's cross-document
consistency demo. Use `get_persona(recurring_prob, used_ids)` to mix recurring
and one-off personas when building documents.
