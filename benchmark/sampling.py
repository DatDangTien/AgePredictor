from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from metrics import AGE_INTERVALS
from records import BenchmarkRecord


def select_records(
    records: Sequence[BenchmarkRecord],
    *,
    limit: int | None,
    strategy: str,
    seed: int,
    sample_list_path: Path | None = None,
) -> tuple[list[BenchmarkRecord], dict[str, Any]]:
    if sample_list_path is not None:
        selected = _from_sample_list(records, sample_list_path)
        return selected, _metadata("from_sample_list", selected, sample_list_path)

    if strategy == "all":
        selected = list(records)
    elif strategy == "evenly_spaced":
        selected = _evenly_spaced(records, limit)
    elif strategy == "random":
        selected = _random(records, limit, seed)
    elif strategy == "stratified_age":
        selected = _stratified(records, limit, seed, group_by=_age_group)
    elif strategy == "stratified_age_gender":
        selected = _stratified(records, limit, seed, group_by=_age_gender_group)
    elif strategy == "from_sample_list":
        raise ValueError("--sample-list is required for from_sample_list sampling")
    else:
        raise ValueError(f"Unsupported sampling strategy: {strategy}")

    return selected, _metadata(strategy, selected, sample_list_path)


def _metadata(
    strategy: str,
    selected: Sequence[BenchmarkRecord],
    sample_list_path: Path | None,
) -> dict[str, Any]:
    group_counts = defaultdict(int)
    for record in selected:
        group_counts[_age_gender_group(record)] += 1
    return {
        "strategy": strategy,
        "sample_count": len(selected),
        "sample_ids": [record.sample_id for record in selected],
        "sample_list_path": str(sample_list_path.resolve()) if sample_list_path else None,
        "actual_group_counts": dict(sorted(group_counts.items())),
    }


def _evenly_spaced(
    records: Sequence[BenchmarkRecord],
    limit: int | None,
) -> list[BenchmarkRecord]:
    if limit is None or limit <= 0 or limit >= len(records):
        return list(records)
    if limit == 1:
        return [records[len(records) // 2]]
    last = len(records) - 1
    indices = [round(index * last / (limit - 1)) for index in range(limit)]
    return [records[index] for index in indices]


def _random(
    records: Sequence[BenchmarkRecord],
    limit: int | None,
    seed: int,
) -> list[BenchmarkRecord]:
    rng = random.Random(seed)
    selected = list(records)
    rng.shuffle(selected)
    return selected if limit is None or limit <= 0 else selected[:limit]


def _stratified(
    records: Sequence[BenchmarkRecord],
    limit: int | None,
    seed: int,
    *,
    group_by,
) -> list[BenchmarkRecord]:
    if limit is None or limit <= 0 or limit >= len(records):
        return list(records)

    rng = random.Random(seed)
    groups: dict[str, list[BenchmarkRecord]] = defaultdict(list)
    for record in records:
        groups[group_by(record)].append(record)

    selected: list[BenchmarkRecord] = []
    ordered_groups = sorted(groups.items())
    base = limit // len(ordered_groups)
    remainder = limit % len(ordered_groups)
    for index, (_, group_records) in enumerate(ordered_groups):
        shuffled = list(group_records)
        rng.shuffle(shuffled)
        take = min(len(shuffled), base + (1 if index < remainder else 0))
        selected.extend(shuffled[:take])

    if len(selected) < limit:
        selected_ids = {record.sample_id for record in selected}
        remaining = [
            record for record in records if record.sample_id not in selected_ids
        ]
        rng.shuffle(remaining)
        selected.extend(remaining[: limit - len(selected)])

    return sorted(selected, key=lambda record: record.sample_id)


def _from_sample_list(
    records: Sequence[BenchmarkRecord],
    sample_list_path: Path,
) -> list[BenchmarkRecord]:
    ids = [
        line.strip()
        for line in sample_list_path.expanduser().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {record.sample_id: record for record in records}
    missing = [sample_id for sample_id in ids if sample_id not in by_id]
    if missing:
        raise ValueError(f"Sample list includes unknown sample_id(s): {missing[:5]}")
    return [by_id[sample_id] for sample_id in ids]


def _age_group(record: BenchmarkRecord) -> str:
    if record.age is None:
        return "age_unknown"
    for name, start, end in AGE_INTERVALS:
        if start <= record.age <= end:
            return name
    return "age_out_of_range"


def _age_gender_group(record: BenchmarkRecord) -> str:
    gender = record.gender_label or "gender_unknown"
    return f"{_age_group(record)}:{gender}"
