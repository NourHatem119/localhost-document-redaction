"""Generate N synthetic personas and store them to a JSON file.

Usage:
    python -m dataset.generate_personas --count 20 --out dataset/data/personas.json --seed 42
"""

from __future__ import annotations

import argparse

from dataset.utils.helpers import generate_personas, read_personas, write_personas


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic personas as JSON.")
    parser.add_argument("-n", "--count", type=int, default=20, help="number of personas")
    parser.add_argument(
        "-o", "--out", default="dataset/data/personas.json", help="output JSON path"
    )
    parser.add_argument("-s", "--seed", type=int, default=None, help="RNG seed")
    args = parser.parse_args()

    personas = generate_personas(args.count, seed=args.seed)
    path = write_personas(personas, args.out)
    # Round-trip read confirms the file is well-formed.
    loaded = read_personas(path)
    print(f"Wrote {len(loaded)} personas to {path}")


if __name__ == "__main__":
    main()
