from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = REPO_ROOT / "benchmark"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from records import BenchmarkRecord  # noqa: E402
from sampling import select_records  # noqa: E402


class SamplingTests(unittest.TestCase):
    def test_all_strategy_respects_limit(self) -> None:
        records = [_record(index) for index in range(10)]

        selected, metadata = select_records(
            records,
            limit=5,
            strategy="all",
            seed=42,
        )

        self.assertEqual([record.sample_id for record in selected], [str(i) for i in range(5)])
        self.assertEqual(metadata["strategy"], "all")
        self.assertEqual(metadata["sample_count"], 5)

    def test_all_strategy_with_no_limit_returns_everything(self) -> None:
        records = [_record(index) for index in range(3)]

        selected, metadata = select_records(
            records,
            limit=None,
            strategy="all",
            seed=42,
        )

        self.assertEqual(len(selected), 3)
        self.assertEqual(metadata["sample_count"], 3)


def _record(index: int) -> BenchmarkRecord:
    return BenchmarkRecord(
        dataset_id="DSTEST",
        dataset_version="test",
        sample_id=str(index),
        path=Path(f"{index}.jpg"),
        age=20 + index,
        gender_label="Female" if index % 2 == 0 else "Male",
    )


if __name__ == "__main__":
    unittest.main()
