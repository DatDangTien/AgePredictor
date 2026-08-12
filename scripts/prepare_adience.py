#!/usr/bin/env python3
"""Create a canonical benchmark manifest for one Adience image variant."""

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

IMAGE_PREFIXES = {
    "aligned": "landmark_aligned_face",
    "faces": "coarse_tilt_aligned_face",
}
REQUIRED_LABEL_COLUMNS = {
    "user_id",
    "original_image",
    "face_id",
    "age",
    "gender",
    "x",
    "y",
    "dx",
    "dy",
    "tilt_ang",
    "fiducial_yaw_angle",
    "fiducial_score",
}
UNKNOWN_LABELS = {"", "none", "nan", "unknown", "u"}
AGE_INTERVAL_PATTERN = re.compile(
    r"^[\[(]?\s*(?P<lower>\d+)\s*[,\-]\s*(?P<upper>\d+)\s*[\])]?$"
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_age_label(value: str) -> tuple[str, int, int, int] | None:
    """Return canonical label, inclusive bounds, and a midpoint proxy age."""
    normalized = " ".join(value.strip().casefold().split())
    if normalized in UNKNOWN_LABELS:
        return None

    if normalized.isdigit():
        age = int(normalized)
        return str(age), age, age, age

    match = AGE_INTERVAL_PATTERN.fullmatch(normalized)
    if match is None:
        raise ValueError(f"unsupported Adience age label: {value!r}")
    lower = int(match.group("lower"))
    upper = int(match.group("upper"))
    if upper < lower:
        raise ValueError(f"Adience age interval is reversed: {value!r}")

    # Round .5 upward so (15, 20) becomes 18, matching the representative-age
    # convention used by the other interval-labelled dataset preparation code.
    representative = (lower + upper + 1) // 2
    return f"({lower}, {upper})", lower, upper, representative


def parse_gender(value: str) -> str | None:
    normalized = value.strip().casefold()
    if normalized in UNKNOWN_LABELS:
        return None
    if normalized in {"f", "female"}:
        return "Female"
    if normalized in {"m", "male"}:
        return "Male"
    raise ValueError(f"unsupported Adience gender: {value!r}")


def prepare_manifest(
    *,
    data_dir: Path,
    image_variant: str,
    labels_dir: Path | None,
    fold_set: str,
    dataset_id: str,
    dataset_version: str,
    output_path: Path,
    validation_report_path: Path,
    age_label_policy: str = "representative",
    progress_every: int = 1000,
) -> dict[str, Any]:
    data_root = data_dir.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Adience data directory not found: {data_root}")
    if image_variant not in IMAGE_PREFIXES:
        raise ValueError("image_variant must be aligned or faces")
    if fold_set not in {"all", "frontal"}:
        raise ValueError("fold_set must be all or frontal")
    if age_label_policy not in {"representative", "blank"}:
        raise ValueError("age_label_policy must be representative or blank")
    if progress_every < 0:
        raise ValueError("progress_every must be non-negative")

    image_root = data_root / image_variant
    if not image_root.is_dir():
        raise FileNotFoundError(
            f"Adience {image_variant} image directory not found: {image_root}"
        )
    label_root = (
        labels_dir.expanduser().resolve()
        if labels_dir is not None
        else data_root / "labels"
    )
    if not label_root.is_dir():
        raise FileNotFoundError(f"Adience labels directory not found: {label_root}")

    label_paths = _label_paths(label_root, fold_set)
    rows: list[dict[str, str]] = []
    invalid_labels: list[str] = []
    missing_images: list[str] = []
    decode_failures: list[str] = []
    duplicate_sample_ids: list[str] = []
    duplicate_label_images: list[str] = []
    seen_sample_ids: set[str] = set()
    seen_image_paths: set[str] = set()
    labelled_image_paths: set[str] = set()
    split_counts: Counter[str] = Counter()
    age_group_counts: Counter[str] = Counter()
    gender_counts: Counter[str] = Counter()
    label_rows_by_fold: Counter[str] = Counter()
    missing_age_labels = 0
    missing_gender_labels = 0

    print(
        f"[prepare-adience] data root: {data_root} | variant={image_variant} | "
        f"fold_set={fold_set} | age_policy={age_label_policy}",
        flush=True,
    )
    for fold_index, labels_path in enumerate(label_paths):
        dataset_split = f"fold_{fold_index}"
        valid_before = len(rows)
        missing_before = len(missing_images)
        decode_before = len(decode_failures)
        invalid_before = len(invalid_labels)
        print(
            f"[prepare-adience] starting {dataset_split}: {labels_path}",
            flush=True,
        )
        with labels_path.open(newline="", encoding="utf-8-sig") as labels_file:
            reader = csv.DictReader(labels_file, delimiter="\t")
            fieldnames = {name.strip() for name in (reader.fieldnames or ())}
            missing_columns = REQUIRED_LABEL_COLUMNS - fieldnames
            if missing_columns:
                raise ValueError(
                    f"{labels_path} is missing label column(s): "
                    + ", ".join(sorted(missing_columns))
                )

            for row_number, source_row in enumerate(reader, start=2):
                label_rows_by_fold[dataset_split] += 1
                fold_rows = label_rows_by_fold[dataset_split]
                if progress_every > 0 and fold_rows % progress_every == 0:
                    print(
                        f"[prepare-adience] {dataset_split}: processed "
                        f"{fold_rows:,} label rows; {len(rows):,} total valid",
                        flush=True,
                    )
                source_row = {
                    (key or "").strip(): (value or "").strip()
                    for key, value in source_row.items()
                }
                location = f"{labels_path.name}:{row_number}"
                try:
                    user_id = _path_component(
                        source_row["user_id"], field_name="user_id"
                    )
                    original_image = _path_component(
                        source_row["original_image"],
                        field_name="original_image",
                    )
                    face_id = _face_id(source_row["face_id"])
                    parsed_age = parse_age_label(source_row["age"])
                    gender = parse_gender(source_row["gender"])
                    source_bbox = _source_bbox(source_row)
                    pose = _pose_metadata(source_row)
                except ValueError as exc:
                    invalid_labels.append(f"{location}: {exc}")
                    continue

                image_filename = (
                    f"{IMAGE_PREFIXES[image_variant]}.{face_id}.{original_image}"
                )
                relative_image_path = Path(user_id) / image_filename
                relative_text = relative_image_path.as_posix()
                if relative_text in seen_image_paths:
                    duplicate_label_images.append(relative_text)
                    continue
                seen_image_paths.add(relative_text)

                image_path = image_root / relative_image_path
                if not image_path.is_file():
                    missing_images.append(relative_text)
                    continue
                labelled_image_paths.add(relative_text)
                if cv2.imread(str(image_path)) is None:
                    decode_failures.append(relative_text)
                    continue

                sample_id = _sample_id(user_id, face_id, original_image)
                if sample_id in seen_sample_ids:
                    duplicate_sample_ids.append(sample_id)
                    continue
                seen_sample_ids.add(sample_id)

                age_value = ""
                age_metadata: dict[str, Any] = {}
                if parsed_age is None:
                    missing_age_labels += 1
                else:
                    age_group, age_min, age_max, representative_age = parsed_age
                    if age_label_policy == "representative":
                        age_value = str(representative_age)
                    age_metadata = {
                        "age_group": age_group,
                        "age_min": age_min,
                        "age_max": age_max,
                        "age_representative": representative_age,
                    }
                    age_group_counts[age_group] += 1

                if gender is None:
                    missing_gender_labels += 1
                else:
                    gender_counts[gender] += 1

                metadata = {
                    "adience_file": relative_text,
                    "original_image": original_image,
                    "face_id": int(face_id),
                    "image_variant": image_variant,
                    "fold_set": fold_set,
                    "label_file": labels_path.name,
                    "label_row": row_number,
                    "age_label_policy": age_label_policy,
                    "source_age_label": source_row["age"],
                    "source_gender_label": source_row["gender"],
                    "source_bbox_xywh": source_bbox,
                    **pose,
                    **age_metadata,
                }
                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "dataset_version": dataset_version,
                        "sample_id": sample_id,
                        "image_path": _repo_relative(image_path),
                        "age": age_value,
                        "gender": gender or "",
                        "identity_id": user_id,
                        # These coordinates describe the face in the original
                        # Flickr photo, not in the supplied face crop.
                        "bbox_x1": "",
                        "bbox_y1": "",
                        "bbox_x2": "",
                        "bbox_y2": "",
                        "landmarks_path": "",
                        "dataset_split": dataset_split,
                        "source": "Adience",
                        "is_valid": "true",
                        "validation_error": "",
                        "metadata_json": json.dumps(
                            metadata,
                            separators=(",", ":"),
                        ),
                    }
                )
                split_counts[dataset_split] += 1
        print(
            f"[prepare-adience] finished {dataset_split}: "
            f"{label_rows_by_fold[dataset_split]:,} labels, "
            f"{len(rows) - valid_before:,} valid, "
            f"{len(missing_images) - missing_before:,} missing, "
            f"{len(decode_failures) - decode_before:,} decode failures, "
            f"{len(invalid_labels) - invalid_before:,} invalid labels",
            flush=True,
        )

    if not rows:
        raise ValueError(
            f"No valid Adience {image_variant} records were found for {fold_set} folds"
        )

    output = output_path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"[prepare-adience] scanning {image_root} for validation",
        flush=True,
    )
    discovered_images = _discover_images(image_root)
    unlabeled_images = sorted(discovered_images - labelled_image_paths)
    warnings = [
        "Adience ages are intervals. Use interval-aware age metrics for the "
        "primary accuracy result."
    ]
    if age_label_policy == "representative":
        warnings.append(
            "Manifest ages are midpoint proxy values; exact-age MAE/RMSE are "
            "not measurements against exact ground-truth ages."
        )
    else:
        warnings.append(
            "Manifest ages are blank; exact-age metrics are disabled while "
            "interval-aware metrics remain available."
        )

    validation = {
        "image_variant": image_variant,
        "fold_set": fold_set,
        "age_label_policy": age_label_policy,
        "files_discovered": len(discovered_images),
        "label_rows": sum(label_rows_by_fold.values()),
        "label_rows_by_fold": dict(label_rows_by_fold),
        "valid_records": len(rows),
        "valid_records_by_fold": dict(split_counts),
        "age_group_counts": dict(age_group_counts),
        "gender_counts": dict(gender_counts),
        "missing_age_labels": missing_age_labels,
        "missing_gender_labels": missing_gender_labels,
        "invalid_labels": invalid_labels,
        "missing_images": missing_images,
        "decode_failures": decode_failures,
        "duplicate_sample_ids": duplicate_sample_ids,
        "duplicate_label_images": duplicate_label_images,
        "unlabeled_image_count": len(unlabeled_images),
        "unlabeled_image_examples": unlabeled_images[:100],
        "warnings": warnings,
    }
    report = validation_report_path.expanduser().resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(
        f"[prepare-adience] validation: {len(rows):,} valid / "
        f"{sum(label_rows_by_fold.values()):,} labels; "
        f"{len(missing_images):,} missing; "
        f"{len(decode_failures):,} decode failures; "
        f"report={report}",
        flush=True,
    )
    return validation


def _label_paths(label_root: Path, fold_set: str) -> list[Path]:
    prefix = "fold_frontal" if fold_set == "frontal" else "fold"
    paths = [label_root / f"{prefix}_{index}_data.txt" for index in range(5)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing Adience fold label file(s): " + ", ".join(missing)
        )
    return paths


def _path_component(value: str, *, field_name: str) -> str:
    if not value:
        raise ValueError(f"{field_name} must not be blank")
    component = Path(value)
    if component.is_absolute() or len(component.parts) != 1 or value in {".", ".."}:
        raise ValueError(f"{field_name} must be one relative path component")
    return value


def _face_id(value: str) -> str:
    if not value.isdigit():
        raise ValueError(f"face_id must be a non-negative integer: {value!r}")
    return str(int(value))


def _source_bbox(row: dict[str, str]) -> list[int | float | str | None]:
    return [_number_or_source_text(row[field]) for field in ("x", "y", "dx", "dy")]


def _pose_metadata(row: dict[str, str]) -> dict[str, int | float | str | None]:
    metadata: dict[str, int | float | str | None] = {}
    for source_name, output_name in (
        ("tilt_ang", "tilt_angle"),
        ("fiducial_yaw_angle", "fiducial_yaw_angle"),
        ("fiducial_score", "fiducial_score"),
    ):
        metadata[output_name] = _number_or_source_text(row[source_name])
    return metadata


def _number_or_source_text(value: str) -> int | float | str | None:
    if not value:
        return None
    try:
        number = float(value)
    except ValueError:
        # Bounding-box and pose columns are informational for these already
        # cropped images. Preserve an unusual source value instead of dropping
        # an otherwise usable age/gender sample.
        return value
    return int(number) if number.is_integer() else number


def _sample_id(user_id: str, face_id: str, original_image: str) -> str:
    original_stem = Path(original_image).stem
    return f"{user_id}_{face_id}_{original_stem}"


def _discover_images(image_root: Path) -> set[str]:
    return {
        path.relative_to(image_root).as_posix()
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
    }


def _repo_relative(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data" / "adience",
        help="Directory containing aligned/, faces/, and labels/",
    )
    parser.add_argument(
        "--image-variant",
        choices=tuple(IMAGE_PREFIXES),
        required=True,
        help="Prepare exactly one image variant per manifest/benchmark run",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        help="Defaults to <data-dir>/labels",
    )
    parser.add_argument(
        "--fold-set",
        choices=("all", "frontal"),
        default="all",
        help="Use all-face folds or the official near-frontal subset",
    )
    parser.add_argument("--dataset-id", default="DS004")
    parser.add_argument("--dataset-version", default="official")
    parser.add_argument(
        "--age-label-policy",
        choices=("representative", "blank"),
        default="representative",
        help=(
            "Use midpoint proxy ages for exact-age metrics, or leave age blank; "
            "source intervals are always preserved for interval-aware metrics"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Defaults to manifests/DS004_adience_<variant>[_frontal].csv",
    )
    parser.add_argument(
        "--validation-report",
        type=Path,
        help="Defaults to output/validation/DS004_adience_<variant>[_frontal].json",
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
    suffix = f"adience_{args.image_variant}"
    if args.fold_set == "frontal":
        suffix += "_frontal"
    output_path = args.output or (
        REPO_ROOT / "manifests" / f"{args.dataset_id}_{suffix}.csv"
    )
    validation_report_path = args.validation_report or (
        REPO_ROOT
        / "output"
        / "validation"
        / f"{args.dataset_id}_{suffix}.json"
    )
    validation = prepare_manifest(
        data_dir=args.data_dir,
        image_variant=args.image_variant,
        labels_dir=args.labels_dir,
        fold_set=args.fold_set,
        dataset_id=args.dataset_id,
        dataset_version=args.dataset_version,
        output_path=output_path,
        validation_report_path=validation_report_path,
        age_label_policy=args.age_label_policy,
        progress_every=args.progress_every,
    )
    print(
        f"[prepare-adience] wrote {output_path} with "
        f"{validation['valid_records']:,} valid {args.image_variant} records",
        flush=True,
    )


if __name__ == "__main__":
    main()
