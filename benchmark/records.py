from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkRecord:
    dataset_id: str
    dataset_version: str
    sample_id: str
    path: Path
    age: int | None = None
    gender_label: str | None = None
    identity_id: str | None = None
    bbox_xyxy: tuple[float, float, float, float] | None = None
    landmarks_path: Path | None = None
    dataset_split: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id must not be empty")
        if self.gender_label not in ("Female", "Male", None):
            raise ValueError("gender_label must be Female, Male, or None")
        if self.age is not None and self.age < 0:
            raise ValueError("age must be non-negative when present")
        if self.bbox_xyxy is not None and len(self.bbox_xyxy) != 4:
            raise ValueError("bbox_xyxy must contain four values when present")
