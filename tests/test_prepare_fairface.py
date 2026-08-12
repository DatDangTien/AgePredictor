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

from prepare_fairface import (  # noqa: E402
    image_padding_from_version,
    parse_age_group,
    parse_gender,
    parse_race,
    prepare_manifest,
)
from manifest import load_manifest  # noqa: E402


class PrepareFairFaceTests(unittest.TestCase):
    def test_infers_padding_from_dataset_version(self) -> None:
        self.assertEqual(image_padding_from_version("margin025"), 0.25)
        self.assertEqual(image_padding_from_version("margin125"), 1.25)

    def test_normalizes_official_labels(self) -> None:
        self.assertEqual(parse_age_group("20-29"), ("20-29", 20, 29, 25))
        self.assertEqual(parse_age_group("more than 70"), ("70+", 70, None, 75))
        self.assertEqual(parse_gender("female"), "Female")
        self.assertEqual(parse_race("Latino Hispanic"), "Latino_Hispanic")

    def test_combines_train_and_val_into_one_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "fairface-img-margin025-trainval"
            train_dir = data_root / "train"
            val_dir = data_root / "val"
            train_dir.mkdir(parents=True)
            val_dir.mkdir(parents=True)
            _write_jpeg(train_dir / "1.jpg")
            _write_jpeg(val_dir / "2.jpg")

            train_labels = root / "fairface_label_train.csv"
            val_labels = root / "fairface_label_val.csv"
            _write_labels(
                train_labels,
                [
                    {
                        "file": "train/1.jpg",
                        "age": "20-29",
                        "gender": "Male",
                        "race": "East Asian",
                        "service_test": "True",
                    }
                ],
            )
            _write_labels(
                val_labels,
                [
                    {
                        "file": "val/2.jpg",
                        "age": "more than 70",
                        "gender": "Female",
                        "race": "White",
                        "service_test": "False",
                    }
                ],
            )

            output = root / "manifest.csv"
            report = root / "validation.json"
            progress_output = io.StringIO()
            with redirect_stdout(progress_output):
                validation = prepare_manifest(
                    data_dir=data_root,
                    train_labels_path=train_labels,
                    val_labels_path=val_labels,
                    dataset_id="DS003",
                    dataset_version="margin025",
                    output_path=output,
                    validation_report_path=report,
                    progress_every=1,
                )

            with output.open(newline="", encoding="utf-8") as manifest_file:
                rows = list(csv.DictReader(manifest_file))
            records = load_manifest(output)

        self.assertEqual(validation["valid_records"], 2)
        self.assertIn("starting train labels", progress_output.getvalue())
        self.assertIn("train: processed 1 label rows", progress_output.getvalue())
        self.assertIn("finished validation", progress_output.getvalue())
        self.assertIn("scanning image folders", progress_output.getvalue())
        self.assertEqual(
            validation["valid_records_by_split"],
            {"train": 1, "validation": 1},
        )
        self.assertEqual([row["sample_id"] for row in rows], ["train_1", "val_2"])
        self.assertEqual([row["dataset_split"] for row in rows], ["train", "validation"])
        self.assertEqual([row["age"] for row in rows], ["25", "75"])
        self.assertEqual([row["gender"] for row in rows], ["Male", "Female"])
        self.assertEqual(json.loads(rows[1]["metadata_json"])["age_group"], "70+")
        self.assertEqual(json.loads(rows[1]["metadata_json"])["image_padding"], 0.25)
        self.assertEqual([record.dataset_split for record in records], ["train", "validation"])

    def test_can_leave_interval_age_blank(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "images"
            (data_root / "train").mkdir(parents=True)
            (data_root / "val").mkdir(parents=True)
            _write_jpeg(data_root / "train" / "1.jpg")
            _write_jpeg(data_root / "val" / "2.jpg")
            train_labels = root / "fairface_label_train.csv"
            val_labels = root / "fairface_label_val.csv"
            _write_labels(
                train_labels,
                [{"file": "1.jpg", "age": "3-9", "gender": "Male", "race": "Indian"}],
            )
            _write_labels(
                val_labels,
                [{"file": "2.jpg", "age": "10-19", "gender": "Female", "race": "Black"}],
            )
            output = root / "manifest.csv"

            prepare_manifest(
                data_dir=data_root,
                train_labels_path=train_labels,
                val_labels_path=val_labels,
                dataset_id="DS003",
                dataset_version="margin025",
                output_path=output,
                validation_report_path=root / "validation.json",
                age_label_policy="blank",
            )
            with output.open(newline="", encoding="utf-8") as manifest_file:
                rows = list(csv.DictReader(manifest_file))

        self.assertEqual([row["age"] for row in rows], ["", ""])


def _write_labels(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ("file", "age", "gender", "race", "service_test")
    with path.open("w", newline="", encoding="utf-8") as labels_file:
        writer = csv.DictWriter(labels_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_jpeg(path: Path) -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to write test image: {path}")


if __name__ == "__main__":
    unittest.main()
