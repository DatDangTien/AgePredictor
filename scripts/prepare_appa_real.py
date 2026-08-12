#!/usr/bin/env python3
"""Create one canonical benchmark manifest from all APPA-REAL splits."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Sequence

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = REPO_ROOT / "benchmark"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from manifest import MANIFEST_COLUMNS  # noqa: E402
from metrics import AGE_INTERVALS  # noqa: E402

DEFAULT_DATA_DIR = REPO_ROOT / "data" / "APPA-REAL" / "appa-real-release"
REQUIRED_LABEL_COLUMNS = {"file_name", "real_age"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SPLITS = (
    ("train", "train", "gt_train.csv"),
    ("valid", "validation", "gt_valid.csv"),
    ("test", "test", "gt_test.csv"),
)


def parse_real_age(value: str) -> int:
    """Parse an APPA-REAL age, accepting integral values such as ``25.0``."""
    normalized = value.strip()
    if not normalized:
        raise ValueError("real_age must not be blank")
    try:
        age_value = Decimal(normalized)
    except InvalidOperation as exc:
        raise ValueError(f"real_age must be numeric: {value!r}") from exc
    if not age_value.is_finite() or age_value != age_value.to_integral_value():
        raise ValueError(f"real_age must be a finite integer: {value!r}")
    age = int(age_value)
    if age < 0:
        raise ValueError(f"real_age must be non-negative: {age}")
    return age


def prepare_manifest(
    *,
    data_dir: Path,
    dataset_id: str,
    dataset_version: str,
    output_path: Path,
    validation_report_path: Path,
    progress_every: int = 1000,
) -> dict[str, Any]:
    """Build a single manifest containing APPA-REAL train, valid, and test."""
    data_root = data_dir.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"APPA-REAL data directory not found: {data_root}")
    if progress_every < 0:
        raise ValueError("progress_every must be non-negative")

    rows: list[dict[str, str]] = []
    invalid_labels: list[str] = []
    missing_images: list[str] = []
    decode_failures: list[str] = []
    duplicate_sample_ids: list[str] = []
    duplicate_label_files: list[str] = []
    seen_sample_ids: set[str] = set()
    seen_image_paths: set[str] = set()
    labelled_image_paths: set[str] = set()
    label_rows_by_split: Counter[str] = Counter()
    valid_records_by_split: Counter[str] = Counter()
    ages: list[int] = []

    print(
        f"[prepare-appa-real] data root: {data_root} | "
        f"version={dataset_version}",
        flush=True,
    )
    for folder_name, manifest_split, labels_filename in SPLITS:
        split_dir = data_root / folder_name
        labels_path = data_root / labels_filename
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"APPA-REAL split directory not found: {split_dir}"
            )
        if not labels_path.is_file():
            raise FileNotFoundError(f"APPA-REAL label CSV not found: {labels_path}")

        print(
            f"[prepare-appa-real] starting {manifest_split}: {labels_path}",
            flush=True,
        )
        with labels_path.open(newline="", encoding="utf-8-sig") as labels_file:
            reader = csv.DictReader(labels_file)
            fieldnames = {name.strip() for name in (reader.fieldnames or ())}
            missing_columns = REQUIRED_LABEL_COLUMNS - fieldnames
            if missing_columns:
                raise ValueError(
                    f"{labels_path} is missing label column(s): "
                    + ", ".join(sorted(missing_columns))
                )

            for row_number, source_row in enumerate(reader, start=2):
                label_rows_by_split[manifest_split] += 1
                split_rows = label_rows_by_split[manifest_split]
                if progress_every > 0 and split_rows % progress_every == 0:
                    print(
                        f"[prepare-appa-real] {manifest_split}: processed "
                        f"{split_rows:,} label rows; {len(rows):,} total valid",
                        flush=True,
                    )

                location = f"{labels_path.name}:{row_number}"
                try:
                    relative_path = _image_path(
                        source_row.get("file_name", ""),
                        expected_folder=folder_name,
                    )
                    age = parse_real_age(source_row.get("real_age", ""))
                except ValueError as exc:
                    invalid_labels.append(f"{location}: {exc}")
                    continue

                relative_text = relative_path.as_posix()
                if relative_text in seen_image_paths:
                    duplicate_label_files.append(relative_text)
                    continue
                seen_image_paths.add(relative_text)

                image_path = (data_root / relative_path).resolve()
                try:
                    image_path.relative_to(data_root)
                except ValueError:
                    invalid_labels.append(
                        f"{location}: file_name escapes the APPA-REAL data directory"
                    )
                    continue
                if not image_path.is_file():
                    missing_images.append(relative_text)
                    continue
                labelled_image_paths.add(relative_text)
                if cv2.imread(str(image_path)) is None:
                    decode_failures.append(relative_text)
                    continue

                sample_id = (
                    f"{folder_name}_{relative_path.stem}"
                    .replace("/", "_")
                    .replace("\\", "_")
                )
                if sample_id in seen_sample_ids:
                    duplicate_sample_ids.append(sample_id)
                    continue
                seen_sample_ids.add(sample_id)

                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "dataset_version": dataset_version,
                        "sample_id": sample_id,
                        "image_path": _repo_relative(image_path),
                        "age": str(age),
                        "gender": "",
                        "identity_id": "",
                        "bbox_x1": "",
                        "bbox_y1": "",
                        "bbox_x2": "",
                        "bbox_y2": "",
                        "landmarks_path": "",
                        "dataset_split": manifest_split,
                        "source": "APPA-REAL",
                        "is_valid": "true",
                        "validation_error": "",
                        "metadata_json": json.dumps(
                            {
                                "appa_real_file": relative_text,
                                "label_file": labels_path.name,
                                "label_row": row_number,
                            },
                            separators=(",", ":"),
                        ),
                    }
                )
                ages.append(age)
                valid_records_by_split[manifest_split] += 1

        print(
            f"[prepare-appa-real] finished {manifest_split}: "
            f"{label_rows_by_split[manifest_split]:,} label rows; "
            f"{valid_records_by_split[manifest_split]:,} valid records",
            flush=True,
        )

    if not rows:
        raise ValueError("No valid APPA-REAL records were found")

    output = output_path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[prepare-appa-real] writing {len(rows):,} records to {output}",
        flush=True,
    )
    with output.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print("[prepare-appa-real] scanning image folders for validation", flush=True)
    discovered_images = _discover_images(data_root)
    unlabeled_images = sorted(discovered_images - labelled_image_paths)
    validation = {
        "files_discovered": len(discovered_images),
        "files_discovered_by_split": {
            manifest_split: sum(
                path.startswith(f"{folder_name}/") for path in discovered_images
            )
            for folder_name, manifest_split, _ in SPLITS
        },
        "label_rows_by_split": dict(label_rows_by_split),
        "valid_records": len(rows),
        "valid_records_by_split": dict(valid_records_by_split),
        "gender_labels_available": False,
        "age_min": min(ages),
        "age_max": max(ages),
        "unique_ages": len(set(ages)),
        "age_bin_counts": _age_bin_counts(ages),
        "invalid_labels": invalid_labels,
        "missing_images": missing_images,
        "decode_failures": decode_failures,
        "duplicate_sample_ids": duplicate_sample_ids,
        "duplicate_label_files": duplicate_label_files,
        "unlabeled_image_count": len(unlabeled_images),
        "unlabeled_image_examples": unlabeled_images[:100],
    }
    report = validation_report_path.expanduser().resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(f"[prepare-appa-real] wrote validation report: {report}", flush=True)
    return validation


def _image_path(value: str, *, expected_folder: str) -> Path:
    normalized = value.strip().replace("\\", "/")
    if not normalized:
        raise ValueError("file_name must not be blank")
    relative = Path(normalized)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"file_name must be a relative image path: {value!r}")
    if relative.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"file_name has an unsupported image extension: {value!r}")

    split_folders = {folder_name for folder_name, _, _ in SPLITS}
    if len(relative.parts) > 1:
        if relative.parts[0] not in split_folders:
            raise ValueError(f"images must be stored directly in a split: {value!r}")
        if relative.parts[0] != expected_folder:
            raise ValueError(
                f"file_name belongs to {relative.parts[0]!r}, "
                f"expected {expected_folder!r}"
            )
        if len(relative.parts) != 2:
            raise ValueError(f"images must be stored directly in a split: {value!r}")
        return relative
    return Path(expected_folder) / relative


def _discover_images(data_root: Path) -> set[str]:
    discovered: set[str] = set()
    for folder_name, _, _ in SPLITS:
        split_dir = data_root / folder_name
        for path in split_dir.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                discovered.add(path.relative_to(data_root).as_posix())
    return discovered


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
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--dataset-id", default="DS005")
    parser.add_argument("--dataset-version", default="official")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "manifests" / "DS005_appa_real.csv",
    )
    parser.add_argument(
        "--validation-report",
        type=Path,
        default=REPO_ROOT / "output" / "validation" / "DS005_validation.json",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print preparation progress every N label rows; 0 disables periodic logs",
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
        progress_every=args.progress_every,
    )
    print(
        f"[prepare-appa-real] complete: {validation['valid_records']:,} records "
        f"across train, validation, and test in {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
