from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_runtime import (
    AAFRecord,
    age_metrics,
    latency_summary,
    load_aaf_records,
    select_records,
    value_summary,
)


class AllAgeFacesParsingTests(unittest.TestCase):
    def test_loads_age_from_all_age_faces_filename(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "00000A02.jpg").touch()
            (root / "13321A80.jpg").touch()

            records = load_aaf_records(root, expected_count=2)

        self.assertEqual([record.image_id for record in records], [0, 13321])
        self.assertEqual([record.age for record in records], [2, 80])

    def test_rejects_legacy_underscore_filename(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "23_person_m.jpg").touch()

            with self.assertRaisesRegex(ValueError, "NNNNNAxx"):
                load_aaf_records(root, expected_count=None)

    def test_rejects_unexpected_corpus_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "00000A02.jpg").touch()

            with self.assertRaisesRegex(ValueError, "Expected 2"):
                load_aaf_records(root, expected_count=2)


class BenchmarkMathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            AAFRecord(Path(f"{index:05d}A{index:02d}.jpg"), index, index)
            for index in range(10)
        ]

    def test_even_selection_is_deterministic_and_spans_corpus(self) -> None:
        selected = select_records(self.records, 4)
        self.assertEqual([record.image_id for record in selected], [0, 3, 6, 9])
        self.assertEqual(selected, select_records(self.records, 4))

    def test_limit_none_selects_every_record(self) -> None:
        self.assertEqual(select_records(self.records, None), self.records)

    def test_age_metrics(self) -> None:
        metrics = age_metrics([10.0, 20.0, 40.0], [10, 25, 30])
        self.assertAlmostEqual(metrics["mae"], 5.0)
        self.assertAlmostEqual(metrics["rmse"], (125 / 3) ** 0.5)
        self.assertAlmostEqual(metrics["within_5_years"], 2 / 3)
        self.assertEqual(metrics["evaluated_images"], 3)

    def test_latency_summary(self) -> None:
        summary = latency_summary([10.0, 20.0, 30.0])
        self.assertEqual(summary["samples"], 3)
        self.assertEqual(summary["mean_ms"], 20.0)
        self.assertEqual(summary["p50_ms"], 20.0)
        self.assertEqual(summary["fps_from_mean"], 50.0)

    def test_value_summary(self) -> None:
        summary = value_summary([0.25, 0.5, 0.75])
        self.assertEqual(summary["samples"], 3)
        self.assertEqual(summary["mean"], 0.5)
        self.assertEqual(summary["p50"], 0.5)


if __name__ == "__main__":
    unittest.main()
