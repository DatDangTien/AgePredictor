from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
BENCHMARK_DIR = REPO_ROOT / "benchmark"
for import_root in (SCRIPTS_DIR, BENCHMARK_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from manifest import load_manifest  # noqa: E402
from prepare_adience import (  # noqa: E402
    parse_age_label,
    parse_gender,
    prepare_manifest,
)

LABEL_COLUMNS = (
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
)


class PrepareAdienceTests(unittest.TestCase):
    def test_parses_interval_exact_and_unknown_labels(self) -> None:
        self.assertEqual(parse_age_label("(15, 20)"), ("(15, 20)", 15, 20, 18))
        self.assertEqual(parse_age_label("35"), ("35", 35, 35, 35))
        self.assertIsNone(parse_age_label("None"))
        self.assertEqual(parse_gender("f"), "Female")
        self.assertEqual(parse_gender("male"), "Male")
        self.assertIsNone(parse_gender("u"))

    def test_prepares_aligned_and_faces_as_separate_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels_dir = root / "labels"
            labels_dir.mkdir()
            user_id = "100001@N00"
            original_image = "photo.jpg"
            row = {
                "user_id": user_id,
                "original_image": original_image,
                "face_id": "0",
                "age": "(25, 32)",
                "gender": "m",
                "x": "10",
                "y": "20",
                "dx": "30",
                "dy": "40",
                "tilt_ang": "5",
                "fiducial_yaw_angle": "-3",
                "fiducial_score": "80.5",
            }
            for index in range(5):
                rows = [row] if index == 0 else []
                _write_fold(labels_dir / f"fold_{index}_data.txt", rows)

            aligned_image = (
                root
                / "aligned"
                / user_id
                / f"landmark_aligned_face.0.{original_image}"
            )
            faces_image = (
                root
                / "faces"
                / user_id
                / f"coarse_tilt_aligned_face.0.{original_image}"
            )
            aligned_image.parent.mkdir(parents=True)
            faces_image.parent.mkdir(parents=True)
            _write_jpeg(aligned_image)
            _write_jpeg(faces_image)

            manifests: dict[str, list[dict[str, str]]] = {}
            for variant in ("aligned", "faces"):
                output = root / f"{variant}.csv"
                validation = prepare_manifest(
                    data_dir=root,
                    image_variant=variant,
                    labels_dir=None,
                    fold_set="all",
                    dataset_id="DS004",
                    dataset_version="official",
                    output_path=output,
                    validation_report_path=root / f"{variant}.json",
                )
                with output.open(newline="", encoding="utf-8") as manifest_file:
                    manifests[variant] = list(csv.DictReader(manifest_file))
                records = load_manifest(output)

                self.assertEqual(validation["valid_records"], 1)
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0].dataset_split, "fold_0")
                self.assertEqual(records[0].metadata["age_min"], 25)
                self.assertEqual(records[0].metadata["age_max"], 32)

        aligned_row = manifests["aligned"][0]
        faces_row = manifests["faces"][0]
        self.assertIn("landmark_aligned_face.0.photo.jpg", aligned_row["image_path"])
        self.assertIn(
            "coarse_tilt_aligned_face.0.photo.jpg",
            faces_row["image_path"],
        )
        self.assertEqual(aligned_row["sample_id"], faces_row["sample_id"])
        self.assertEqual(aligned_row["age"], "29")
        self.assertEqual(aligned_row["gender"], "Male")
        self.assertEqual(aligned_row["identity_id"], "100001@N00")
        metadata = json.loads(aligned_row["metadata_json"])
        self.assertEqual(metadata["source_bbox_xywh"], [10, 20, 30, 40])
        self.assertEqual(metadata["image_variant"], "aligned")

    def test_blank_policy_preserves_interval_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels_dir = root / "labels"
            labels_dir.mkdir()
            user_id = "subject"
            original_image = "image.jpg"
            row = {
                "user_id": user_id,
                "original_image": original_image,
                "face_id": "2",
                "age": "(60, 100)",
                "gender": "u",
                "x": "",
                "y": "",
                "dx": "",
                "dy": "",
                "tilt_ang": "",
                "fiducial_yaw_angle": "",
                "fiducial_score": "",
            }
            for index in range(5):
                _write_fold(
                    labels_dir / f"fold_{index}_data.txt",
                    [row] if index == 0 else [],
                )
            image = (
                root
                / "aligned"
                / user_id
                / f"landmark_aligned_face.2.{original_image}"
            )
            image.parent.mkdir(parents=True)
            _write_jpeg(image)
            output = root / "manifest.csv"

            prepare_manifest(
                data_dir=root,
                image_variant="aligned",
                labels_dir=None,
                fold_set="all",
                dataset_id="DS004",
                dataset_version="official",
                output_path=output,
                validation_report_path=root / "validation.json",
                age_label_policy="blank",
            )
            with output.open(newline="", encoding="utf-8") as manifest_file:
                manifest_row = next(csv.DictReader(manifest_file))

        self.assertEqual(manifest_row["age"], "")
        self.assertEqual(manifest_row["gender"], "")
        metadata = json.loads(manifest_row["metadata_json"])
        self.assertEqual(metadata["age_min"], 60)
        self.assertEqual(metadata["age_max"], 100)


def _write_fold(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as labels_file:
        writer = csv.DictWriter(
            labels_file,
            fieldnames=LABEL_COLUMNS,
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_jpeg(path: Path) -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to write test image: {path}")


if __name__ == "__main__":
    unittest.main()
