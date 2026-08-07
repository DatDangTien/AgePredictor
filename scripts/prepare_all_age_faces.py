#!/usr/bin/env python3
"""Create a canonical benchmark manifest for All-Age-Faces."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = REPO_ROOT / "benchmark"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from manifest import MANIFEST_COLUMNS  # noqa: E402
from metrics import AGE_INTERVALS  # noqa: E402

AAF_FILENAME_PATTERN = re.compile(
    r"^(?P<image_id>\d{5})A(?P<age>\d{2})\.jpg$",
    re.IGNORECASE,
)
AAF_LAST_FEMALE_ID = 7_380


def parse_aaf_filename(path: Path) -> tuple[int, int, str]:
    match = AAF_FILENAME_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"filename does not match NNNNNAxx.jpg: {path.name}")
    image_id = int(match.group("image_id"))
    age = int(match.group("age"))
    gender = "Female" if image_id <= AAF_LAST_FEMALE_ID else "Male"
    return image_id, age, gender


def prepare_manifest(
    *,
    data_dir: Path,
    dataset_id: str,
    dataset_version: str,
    output_path: Path,
    validation_report_path: Path,
) -> dict[str, Any]:
    data_root = data_dir.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"All-Age-Faces data directory not found: {data_root}")

    files = sorted(data_root.glob("*.jpg"), key=lambda path: path.name.lower())
    rows: list[dict[str, str]] = []
    invalid_filenames: list[str] = []
    decode_failures: list[str] = []
    duplicate_sample_ids: list[str] = []
    seen_sample_ids: set[str] = set()
    ages: list[int] = []
    gender_counts: Counter[str] = Counter()

    for path in files:
        try:
            image_id, age, gender = parse_aaf_filename(path)
        except ValueError:
            invalid_filenames.append(path.name)
            continue

        sample_id = f"{image_id:05d}"
        if sample_id in seen_sample_ids:
            duplicate_sample_ids.append(sample_id)
            continue
        seen_sample_ids.add(sample_id)

        if cv2.imread(str(path)) is None:
            decode_failures.append(path.name)
            continue

        ages.append(age)
        gender_counts[gender] += 1
        rows.append(
            {
                "dataset_id": dataset_id,
                "dataset_version": dataset_version,
                "sample_id": sample_id,
                "image_path": _repo_relative(path),
                "age": str(age),
                "gender": gender,
                "identity_id": "",
                "bbox_x1": "",
                "bbox_y1": "",
                "bbox_x2": "",
                "bbox_y2": "",
                "landmarks_path": "",
                "dataset_split": "all",
                "source": "All-Age-Faces",
                "is_valid": "true",
                "validation_error": "",
                "metadata_json": json.dumps({"image_id": image_id}, separators=(",", ":")),
            }
        )

    output = output_path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    validation = {
        "files_discovered": len(files),
        "valid_records": len(rows),
        "invalid_filenames": invalid_filenames,
        "decode_failures": decode_failures,
        "age_min": min(ages) if ages else None,
        "age_max": max(ages) if ages else None,
        "unique_ages": len(set(ages)),
        "age_bin_counts": _age_bin_counts(ages),
        "gender_counts": dict(gender_counts),
        "duplicate_sample_ids": duplicate_sample_ids,
    }
    report = validation_report_path.expanduser().resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    return validation


def _age_bin_counts(ages: Sequence[int]) -> dict[str, int]:
    return {
        name: sum(start <= age <= end for age in ages)
        for name, start, end in AGE_INTERVALS
    }


def _repo_relative(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--dataset-id", default="DS001")
    parser.add_argument("--dataset-version", default="official")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "manifests" / "DS001_all_age_faces.csv",
    )
    parser.add_argument(
        "--validation-report",
        type=Path,
        default=REPO_ROOT / "output" / "validation" / "DS001_validation.json",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validation = prepare_manifest(
        data_dir=args.data_dir,
        dataset_id=args.dataset_id,
        dataset_version=args.dataset_version,
        output_path=args.output,
        validation_report_path=args.validation_report,
    )
    print(
        f"[prepare-aaf] wrote {args.output} with "
        f"{validation['valid_records']:,} valid records"
    )


if __name__ == "__main__":
    main()
