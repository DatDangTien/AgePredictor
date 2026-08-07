from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from prepare_all_age_faces import parse_aaf_filename, prepare_manifest  # noqa: E402


class PrepareAllAgeFacesTests(unittest.TestCase):
    def test_parses_age_and_gender_boundaries(self) -> None:
        self.assertEqual(
            parse_aaf_filename(Path("07380A80.jpg")),
            (7380, 80, "Female"),
        )
        self.assertEqual(
            parse_aaf_filename(Path("07381A02.jpg")),
            (7381, 2, "Male"),
        )

    def test_rejects_invalid_filename(self) -> None:
        with self.assertRaisesRegex(ValueError, "NNNNNAxx"):
            parse_aaf_filename(Path("23_person_m.jpg"))

    def test_writes_manifest_and_validation_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "07380A80.jpg"
            _write_jpeg(image)
            output = root / "manifest.csv"
            report = root / "validation.json"

            validation = prepare_manifest(
                data_dir=root,
                dataset_id="DS001",
                dataset_version="official",
                output_path=output,
                validation_report_path=report,
            )
            self.assertTrue(output.is_file())
            self.assertTrue(report.is_file())

        self.assertEqual(validation["valid_records"], 1)
        self.assertEqual(validation["gender_counts"], {"Female": 1})


def _write_jpeg(path: Path) -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    self_ok = cv2.imwrite(str(path), image)
    if not self_ok:
        raise RuntimeError(f"failed to write test image: {path}")


if __name__ == "__main__":
    unittest.main()
