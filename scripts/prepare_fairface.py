#!/usr/bin/env python3
"""Create one canonical benchmark manifest from FairFace train and val data."""

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

DEFAULT_DATA_DIR = (
    REPO_ROOT / "data" / "FairFace" / "fairface-img-margin025-trainval"
)
DEFAULT_TRAIN_LABELS = "fairface_label_train.csv"
DEFAULT_VAL_LABELS = "fairface_label_val.csv"
REQUIRED_LABEL_COLUMNS = {"file", "age", "gender", "race"}
SPLITS = (("train", "train"), ("val", "validation"))

# FairFace supplies age intervals, not exact ages. Representative values let the
# existing exact-age benchmark run, while metadata preserves the source interval.
FAIRFACE_AGE_GROUPS: dict[str, tuple[int, int | None, int]] = {
    "0-2": (0, 2, 1),
    "3-9": (3, 9, 6),
    "10-19": (10, 19, 15),
    "20-29": (20, 29, 25),
    "30-39": (30, 39, 35),
    "40-49": (40, 49, 45),
    "50-59": (50, 59, 55),
    "60-69": (60, 69, 65),
    "70+": (70, None, 75),
}
AGE_GROUP_ALIASES = {
    "more than 70": "70+",
    "70 and above": "70+",
    "70 or older": "70+",
    "70+": "70+",
}
FAIRFACE_RACES = (
    "East Asian",
    "Southeast Asian",
    "Black",
    "White",
    "Indian",
    "Middle Eastern",
    "Latino_Hispanic",
)
RACE_BY_NORMALIZED = {
    " ".join(label.replace("_", " ").casefold().split()): label
    for label in FAIRFACE_RACES
}


def parse_age_group(value: str) -> tuple[str, int, int | None, int]:
    normalized = " ".join(value.strip().lower().split())
    canonical = AGE_GROUP_ALIASES.get(normalized, normalized)
    if canonical not in FAIRFACE_AGE_GROUPS:
        raise ValueError(f"unsupported FairFace age group: {value!r}")
    lower, upper, representative = FAIRFACE_AGE_GROUPS[canonical]
    return canonical, lower, upper, representative


def parse_gender(value: str) -> str:
    normalized = value.strip().casefold()
    if normalized == "female":
        return "Female"
    if normalized == "male":
        return "Male"
    raise ValueError(f"unsupported FairFace gender: {value!r}")


def parse_race(value: str) -> str:
    normalized = " ".join(value.replace("_", " ").casefold().split())
    try:
        return RACE_BY_NORMALIZED[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported FairFace race: {value!r}") from exc


def parse_service_test(value: str | None) -> bool | None:
    normalized = (value or "").strip().casefold()
    if not normalized:
        return None
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"service_test must be true, false, or blank: {value!r}")


def image_padding_from_version(dataset_version: str) -> float:
    normalized = dataset_version.strip().lower().replace("_", "")
    match = re.fullmatch(r"margin(\d+)", normalized)
    if match is None:
        raise ValueError(
            "cannot infer image padding from dataset version; use a value such "
            "as margin025 or margin125"
        )
    return int(match.group(1)) / 100.0


def prepare_manifest(
    *,
    data_dir: Path,
    train_labels_path: Path | None,
    val_labels_path: Path | None,
    dataset_id: str,
    dataset_version: str,
    output_path: Path,
    validation_report_path: Path,
    age_label_policy: str = "representative",
    image_padding: float | None = None,
    progress_every: int = 1000,
) -> dict[str, Any]:
    data_root = data_dir.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"FairFace data directory not found: {data_root}")
    if age_label_policy not in {"representative", "blank"}:
        raise ValueError("age_label_policy must be representative or blank")
    if progress_every < 0:
        raise ValueError("progress_every must be non-negative")
    resolved_image_padding = (
        image_padding
        if image_padding is not None
        else image_padding_from_version(dataset_version)
    )
    if resolved_image_padding < 0:
        raise ValueError("image_padding must be non-negative")

    label_paths = {
        "train": _resolve_label_path(
            data_root,
            train_labels_path,
            DEFAULT_TRAIN_LABELS,
        ),
        "val": _resolve_label_path(
            data_root,
            val_labels_path,
            DEFAULT_VAL_LABELS,
        ),
    }

    rows: list[dict[str, str]] = []
    invalid_labels: list[str] = []
    missing_images: list[str] = []
    decode_failures: list[str] = []
    duplicate_sample_ids: list[str] = []
    duplicate_label_files: list[str] = []
    seen_sample_ids: set[str] = set()
    seen_image_paths: set[str] = set()
    labelled_image_paths: set[str] = set()
    split_counts: Counter[str] = Counter()
    age_group_counts: Counter[str] = Counter()
    gender_counts: Counter[str] = Counter()
    race_counts: Counter[str] = Counter()
    label_rows_by_split: Counter[str] = Counter()

    print(
        f"[prepare-fairface] data root: {data_root} | "
        f"version={dataset_version} | padding={resolved_image_padding}",
        flush=True,
    )
    for folder_name, manifest_split in SPLITS:
        split_dir = data_root / folder_name
        if not split_dir.is_dir():
            raise FileNotFoundError(f"FairFace split directory not found: {split_dir}")

        labels_path = label_paths[folder_name]
        print(
            f"[prepare-fairface] starting {manifest_split} labels: {labels_path}",
            flush=True,
        )
        with labels_path.open(newline="", encoding="utf-8-sig") as labels_file:
            reader = csv.DictReader(labels_file)
            missing_columns = REQUIRED_LABEL_COLUMNS - set(reader.fieldnames or ())
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
                        f"[prepare-fairface] {manifest_split}: processed "
                        f"{split_rows:,} label rows; {len(rows):,} total valid",
                        flush=True,
                    )
                location = f"{labels_path.name}:{row_number}"
                try:
                    relative_path = _fairface_image_path(
                        source_row.get("file", ""),
                        expected_folder=folder_name,
                    )
                    age_group, age_min, age_max, representative_age = parse_age_group(
                        source_row.get("age", "")
                    )
                    gender = parse_gender(source_row.get("gender", ""))
                    race = parse_race(source_row.get("race", ""))
                    service_test = parse_service_test(source_row.get("service_test"))
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
                        f"{location}: image path escapes the FairFace data directory"
                    )
                    continue
                if not image_path.is_file():
                    missing_images.append(relative_text)
                    continue
                labelled_image_paths.add(relative_text)
                if cv2.imread(str(image_path)) is None:
                    decode_failures.append(relative_text)
                    continue

                sample_id = relative_path.with_suffix("").as_posix().replace("/", "_")
                if sample_id in seen_sample_ids:
                    duplicate_sample_ids.append(sample_id)
                    continue
                seen_sample_ids.add(sample_id)

                age_value = (
                    str(representative_age)
                    if age_label_policy == "representative"
                    else ""
                )
                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "dataset_version": dataset_version,
                        "sample_id": sample_id,
                        "image_path": _repo_relative(image_path),
                        "age": age_value,
                        "gender": gender,
                        "identity_id": "",
                        "bbox_x1": "",
                        "bbox_y1": "",
                        "bbox_x2": "",
                        "bbox_y2": "",
                        "landmarks_path": "",
                        "dataset_split": manifest_split,
                        "source": "FairFace",
                        "is_valid": "true",
                        "validation_error": "",
                        "metadata_json": json.dumps(
                            {
                                "fairface_file": relative_text,
                                "age_group": age_group,
                                "age_min": age_min,
                                "age_max": age_max,
                                "age_representative": representative_age,
                                "age_label_policy": age_label_policy,
                                "race_label": race,
                                "service_test": service_test,
                                "image_padding": resolved_image_padding,
                            },
                            separators=(",", ":"),
                        ),
                    }
                )
                split_counts[manifest_split] += 1
                age_group_counts[age_group] += 1
                gender_counts[gender] += 1
                race_counts[race] += 1
        print(
            f"[prepare-fairface] finished {manifest_split}: "
            f"{label_rows_by_split[manifest_split]:,} label rows; "
            f"{split_counts[manifest_split]:,} valid records",
            flush=True,
        )

    if not rows:
        raise ValueError("No valid FairFace records were found")

    output = output_path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print("[prepare-fairface] scanning image folders for validation", flush=True)
    discovered_images = _discover_images(data_root)
    unlabeled_images = sorted(discovered_images - labelled_image_paths)
    validation = {
        "files_discovered": len(discovered_images),
        "files_discovered_by_split": {
            folder_name: sum(
                path.startswith(f"{folder_name}/") for path in discovered_images
            )
            for folder_name, _ in SPLITS
        },
        "label_rows_by_split": dict(label_rows_by_split),
        "valid_records": len(rows),
        "valid_records_by_split": dict(split_counts),
        "age_label_policy": age_label_policy,
        "image_padding": resolved_image_padding,
        "age_group_counts": dict(age_group_counts),
        "gender_counts": dict(gender_counts),
        "race_counts": dict(race_counts),
        "invalid_labels": invalid_labels,
        "missing_images": missing_images,
        "decode_failures": decode_failures,
        "duplicate_sample_ids": duplicate_sample_ids,
        "duplicate_label_files": duplicate_label_files,
        "unlabeled_image_count": len(unlabeled_images),
        "unlabeled_image_examples": unlabeled_images[:100],
        "warnings": (
            [
                "FairFace ages are intervals. Manifest ages are representative "
                "proxy values and exact-age MAE/RMSE must be interpreted accordingly."
            ]
            if age_label_policy == "representative"
            else [
                "FairFace ages are intervals. Manifest ages were left blank, so "
                "the runtime will report age latency but not exact-age accuracy."
            ]
        ),
    }
    report = validation_report_path.expanduser().resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    return validation


def _resolve_label_path(
    data_root: Path,
    provided_path: Path | None,
    filename: str,
) -> Path:
    candidates = (
        [provided_path.expanduser().resolve()]
        if provided_path is not None
        else [data_root / filename, data_root.parent / filename]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"FairFace label CSV not found; checked: {checked}")


def _fairface_image_path(value: str, *, expected_folder: str) -> Path:
    normalized = value.strip().replace("\\", "/")
    if not normalized:
        raise ValueError("file must not be blank")
    relative = Path(normalized)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"file must be a relative FairFace path: {value!r}")
    if relative.suffix.lower() not in {".jpg", ".jpeg"}:
        raise ValueError(f"file must point to a JPEG image: {value!r}")

    if relative.parts[0] in {"train", "val"}:
        if relative.parts[0] != expected_folder:
            raise ValueError(
                f"file belongs to {relative.parts[0]!r}, expected {expected_folder!r}"
            )
        return relative
    return Path(expected_folder) / relative


def _discover_images(data_root: Path) -> set[str]:
    discovered: set[str] = set()
    for folder_name, _ in SPLITS:
        split_dir = data_root / folder_name
        for path in split_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}:
                discovered.add(path.relative_to(data_root).as_posix())
    return discovered


def _repo_relative(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--train-labels",
        type=Path,
        help=(
            "FairFace train label CSV; defaults to fairface_label_train.csv "
            "inside --data-dir or its parent"
        ),
    )
    parser.add_argument(
        "--val-labels",
        type=Path,
        help=(
            "FairFace validation label CSV; defaults to fairface_label_val.csv "
            "inside --data-dir or its parent"
        ),
    )
    parser.add_argument("--dataset-id", default="DS003")
    parser.add_argument("--dataset-version", default="margin025")
    parser.add_argument(
        "--image-padding",
        type=float,
        help="Image padding multiplier; inferred from --dataset-version by default",
    )
    parser.add_argument(
        "--age-label-policy",
        choices=("representative", "blank"),
        default="representative",
        help=(
            "Use representative ages for interval labels, or leave age blank "
            "to disable exact-age accuracy metrics"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "manifests" / "DS003_fairface.csv",
    )
    parser.add_argument(
        "--validation-report",
        type=Path,
        default=REPO_ROOT / "output" / "validation" / "DS003_validation.json",
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
        train_labels_path=args.train_labels,
        val_labels_path=args.val_labels,
        dataset_id=args.dataset_id,
        dataset_version=args.dataset_version,
        output_path=args.output,
        validation_report_path=args.validation_report,
        age_label_policy=args.age_label_policy,
        image_padding=args.image_padding,
        progress_every=args.progress_every,
    )
    print(
        f"[prepare-fairface] wrote {args.output} with "
        f"{validation['valid_records']:,} valid train/validation records",
        flush=True,
    )


if __name__ == "__main__":
    main()
