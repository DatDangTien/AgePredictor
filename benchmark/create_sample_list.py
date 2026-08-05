#!/usr/bin/env python3
"""Create a reusable sample-id list from a canonical benchmark manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

BENCHMARK_DIR = Path(__file__).resolve().parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from manifest import load_manifest  # noqa: E402
from sampling import select_records  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--strategy",
        choices=(
            "all",
            "evenly_spaced",
            "random",
            "stratified_age",
            "stratified_age_gender",
        ),
        default="stratified_age_gender",
    )
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    records = load_manifest(args.manifest)
    selected, _ = select_records(
        records,
        limit=None if args.limit <= 0 else args.limit,
        strategy=args.strategy,
        seed=args.seed,
    )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(record.sample_id for record in selected) + "\n",
        encoding="utf-8",
    )
    print(f"[sample-list] wrote {len(selected):,} sample ids to {output}")


if __name__ == "__main__":
    main()
