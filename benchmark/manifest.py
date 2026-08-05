from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from records import BenchmarkRecord


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_COLUMNS = (
    "dataset_id",
    "dataset_version",
    "sample_id",
    "image_path",
    "age",
    "gender",
    "identity_id",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "landmarks_path",
    "dataset_split",
    "source",
    "is_valid",
    "validation_error",
    "metadata_json",
)


def load_manifest(path: Path | str) -> list[BenchmarkRecord]:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    records: list[BenchmarkRecord] = []
    seen_sample_ids: set[str] = set()
    with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
        reader = csv.DictReader(manifest_file)
        if reader.fieldnames != list(MANIFEST_COLUMNS):
            raise ValueError(
                "Manifest header mismatch. Expected columns: "
                + ", ".join(MANIFEST_COLUMNS)
            )

        for row_number, row in enumerate(reader, start=2):
            if row.get("is_valid", "").strip().lower() == "false":
                continue

            sample_id = row["sample_id"].strip()
            if not sample_id:
                raise ValueError(f"Row {row_number}: sample_id must not be empty")
            if sample_id in seen_sample_ids:
                raise ValueError(f"Row {row_number}: duplicate sample_id {sample_id}")
            seen_sample_ids.add(sample_id)

            image_path = _resolve_existing_path(
                row["image_path"],
                manifest_path=manifest_path,
                row_number=row_number,
                field_name="image_path",
            )
            landmarks_path = _resolve_optional_path(
                row["landmarks_path"],
                manifest_path=manifest_path,
                row_number=row_number,
                field_name="landmarks_path",
            )
            metadata = _parse_metadata(row["metadata_json"], row_number=row_number)
            gender = optional_text(row["gender"])
            if gender not in ("Female", "Male", None):
                raise ValueError(f"Row {row_number}: unsupported gender {gender!r}")

            records.append(
                BenchmarkRecord(
                    dataset_id=row["dataset_id"].strip(),
                    dataset_version=row["dataset_version"].strip(),
                    sample_id=sample_id,
                    path=image_path,
                    age=optional_int(row["age"], row_number=row_number, field_name="age"),
                    gender_label=gender,
                    identity_id=optional_text(row["identity_id"]),
                    bbox_xyxy=parse_bbox(row, row_number=row_number),
                    landmarks_path=landmarks_path,
                    dataset_split=optional_text(row["dataset_split"]),
                    source=optional_text(row["source"]),
                    metadata=metadata,
                )
            )

    if not records:
        raise ValueError(f"Manifest has no valid records: {manifest_path}")
    return records


def optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def optional_int(
    value: str | None,
    *,
    row_number: int | None = None,
    field_name: str = "value",
) -> int | None:
    stripped = optional_text(value)
    if stripped is None:
        return None
    try:
        return int(stripped)
    except ValueError as exc:
        location = f"Row {row_number}: " if row_number is not None else ""
        raise ValueError(f"{location}{field_name} must be an integer") from exc


def optional_float(
    value: str | None,
    *,
    row_number: int | None = None,
    field_name: str = "value",
) -> float | None:
    stripped = optional_text(value)
    if stripped is None:
        return None
    try:
        return float(stripped)
    except ValueError as exc:
        location = f"Row {row_number}: " if row_number is not None else ""
        raise ValueError(f"{location}{field_name} must be numeric") from exc


def parse_bbox(
    row: dict[str, str],
    *,
    row_number: int | None = None,
) -> tuple[float, float, float, float] | None:
    fields = ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")
    values = [
        optional_float(row.get(field), row_number=row_number, field_name=field)
        for field in fields
    ]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        location = f"Row {row_number}: " if row_number is not None else ""
        raise ValueError(f"{location}bbox fields must all be present or all blank")
    return tuple(float(value) for value in values)  # type: ignore[arg-type]


def _parse_metadata(value: str | None, *, row_number: int) -> dict[str, Any]:
    stripped = optional_text(value)
    if stripped is None:
        return {}
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Row {row_number}: malformed metadata_json") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"Row {row_number}: metadata_json must be an object")
    return parsed


def _resolve_optional_path(
    value: str | None,
    *,
    manifest_path: Path,
    row_number: int,
    field_name: str,
) -> Path | None:
    if optional_text(value) is None:
        return None
    return _resolve_existing_path(
        value,
        manifest_path=manifest_path,
        row_number=row_number,
        field_name=field_name,
    )


def _resolve_existing_path(
    value: str | None,
    *,
    manifest_path: Path,
    row_number: int,
    field_name: str,
) -> Path:
    stripped = optional_text(value)
    if stripped is None:
        raise ValueError(f"Row {row_number}: {field_name} must not be empty")

    candidate = Path(stripped).expanduser()
    candidates = [candidate] if candidate.is_absolute() else [
        REPO_ROOT / candidate,
        manifest_path.parent / candidate,
    ]
    for path in candidates:
        resolved = path.resolve()
        if resolved.exists():
            return resolved
    raise FileNotFoundError(
        f"Row {row_number}: {field_name} does not exist: {stripped}"
    )
