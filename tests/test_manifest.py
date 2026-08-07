from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = REPO_ROOT / "benchmark"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from manifest import MANIFEST_COLUMNS, load_manifest  # noqa: E402


class ManifestTests(unittest.TestCase):
    def test_valid_row_and_blank_optional_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "sample.jpg"
            image.touch()
            manifest = root / "manifest.csv"
            _write_manifest(
                manifest,
                [
                    {
                        "dataset_id": "DSX",
                        "dataset_version": "v1",
                        "sample_id": "s1",
                        "image_path": str(image),
                        "age": "",
                        "gender": "",
                        "metadata_json": "{}",
                    }
                ],
            )

            records = load_manifest(manifest)

        self.assertEqual(len(records), 1)
        self.assertIsNone(records[0].age)
        self.assertIsNone(records[0].gender_label)

    def test_rejects_missing_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            _write_manifest(
                manifest,
                [{"sample_id": "s1", "image_path": str(root / "missing.jpg")}],
            )

            with self.assertRaisesRegex(FileNotFoundError, "image_path"):
                load_manifest(manifest)

    def test_reports_row_number_for_bad_json_and_gender(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "sample.jpg"
            image.touch()
            manifest = root / "manifest.csv"
            _write_manifest(
                manifest,
                [
                    {
                        "sample_id": "s1",
                        "image_path": str(image),
                        "gender": "Other",
                        "metadata_json": "{}",
                    }
                ],
            )
            with self.assertRaisesRegex(ValueError, "Row 2.*gender"):
                load_manifest(manifest)

            _write_manifest(
                manifest,
                [
                    {
                        "sample_id": "s1",
                        "image_path": str(image),
                        "metadata_json": "{bad",
                    }
                ],
            )
            with self.assertRaisesRegex(ValueError, "Row 2.*metadata_json"):
                load_manifest(manifest)

    def test_rejects_duplicate_sample_id_and_partial_bbox(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "sample.jpg"
            image.touch()
            manifest = root / "manifest.csv"
            _write_manifest(
                manifest,
                [
                    {"sample_id": "s1", "image_path": str(image)},
                    {"sample_id": "s1", "image_path": str(image)},
                ],
            )
            with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
                load_manifest(manifest)

            _write_manifest(
                manifest,
                [
                    {
                        "sample_id": "s1",
                        "image_path": str(image),
                        "bbox_x1": "1",
                    }
                ],
            )
            with self.assertRaisesRegex(ValueError, "bbox"):
                load_manifest(manifest)


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    defaults = {
        "dataset_id": "DSX",
        "dataset_version": "v1",
        "sample_id": "",
        "image_path": "",
        "age": "",
        "gender": "",
        "identity_id": "",
        "bbox_x1": "",
        "bbox_y1": "",
        "bbox_x2": "",
        "bbox_y2": "",
        "landmarks_path": "",
        "dataset_split": "all",
        "source": "test",
        "is_valid": "true",
        "validation_error": "",
        "metadata_json": json.dumps({}),
    }
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({**defaults, **row})


if __name__ == "__main__":
    unittest.main()
