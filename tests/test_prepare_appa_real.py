from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from prepare_appa_real import parse_real_age, prepare_manifest  # noqa: E402
from manifest import load_manifest  # noqa: E402


class PrepareAppaRealTests(unittest.TestCase):
    def test_parses_integral_real_ages(self) -> None:
        self.assertEqual(parse_real_age("25"), 25)
        self.assertEqual(parse_real_age("25.0"), 25)
        with self.assertRaisesRegex(ValueError, "finite integer"):
            parse_real_age("25.5")

    def test_combines_all_splits_with_blank_gender_and_progress_logs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_rows = {
                "train": ("train_face.jpg", "20"),
                "valid": ("valid_face.jpg", "35.0"),
                "test": ("test_face.jpg", "70"),
            }
            for split, (file_name, age) in source_rows.items():
                split_dir = root / split
                split_dir.mkdir()
                _write_jpeg(split_dir / file_name)
                _write_labels(
                    root / f"gt_{split}.csv",
                    file_name=file_name,
                    real_age=age,
                )

            output = root / "manifest.csv"
            report = root / "validation.json"
            progress_output = io.StringIO()
            with redirect_stdout(progress_output):
                validation = prepare_manifest(
                    data_dir=root,
                    dataset_id="DS005",
                    dataset_version="official",
                    output_path=output,
                    validation_report_path=report,
                    progress_every=1,
                )

            with output.open(newline="", encoding="utf-8") as manifest_file:
                rows = list(csv.DictReader(manifest_file))
            records = load_manifest(output)
            report_data = json.loads(report.read_text(encoding="utf-8"))

        self.assertEqual(validation["valid_records"], 3)
        self.assertEqual(
            validation["valid_records_by_split"],
            {"train": 1, "validation": 1, "test": 1},
        )
        self.assertFalse(report_data["gender_labels_available"])
        self.assertEqual(
            [row["sample_id"] for row in rows],
            ["train_train_face", "valid_valid_face", "test_test_face"],
        )
        self.assertEqual(
            [row["dataset_split"] for row in rows],
            ["train", "validation", "test"],
        )
        self.assertEqual([row["age"] for row in rows], ["20", "35", "70"])
        self.assertEqual([row["gender"] for row in rows], ["", "", ""])
        self.assertTrue(all(record.gender_label is None for record in records))
        self.assertIn("train: processed 1 label rows", progress_output.getvalue())
        self.assertIn("finished validation", progress_output.getvalue())
        self.assertIn("writing 3 records", progress_output.getvalue())
        self.assertIn("wrote validation report", progress_output.getvalue())


def _write_labels(path: Path, *, file_name: str, real_age: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as labels_file:
        writer = csv.DictWriter(
            labels_file,
            fieldnames=("file_name", "real_age", "apparent_age_avg"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "file_name": file_name,
                "real_age": real_age,
                "apparent_age_avg": "999",
            }
        )


def _write_jpeg(path: Path) -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to write test image: {path}")


if __name__ == "__main__":
    unittest.main()
