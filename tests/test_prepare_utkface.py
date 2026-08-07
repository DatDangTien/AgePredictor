from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from prepare_utkface import parse_utkface_filename  # noqa: E402


class PrepareUTKFaceTests(unittest.TestCase):
    def test_parses_valid_filename_and_boundaries(self) -> None:
        age_zero = parse_utkface_filename(Path("0_0_2_20170101.jpg"))
        self.assertEqual(age_zero["age"], 0)
        self.assertEqual(age_zero["gender"], "Male")
        self.assertEqual(age_zero["race_label"], "Asian")
        self.assertEqual(age_zero["collection_timestamp"], "20170101")
        self.assertEqual(age_zero["image_variant"], "original")

        age_116 = parse_utkface_filename(Path("116_1_4_20170101.jpg"))
        self.assertEqual(age_116["age"], 116)
        self.assertEqual(age_116["gender"], "Female")
        self.assertEqual(age_116["race_label"], "Others")
        self.assertEqual(age_116["image_variant"], "original")

    def test_parses_aligned_chip_filename(self) -> None:
        parsed = parse_utkface_filename(Path("35_0_0_20170117145906651.jpg.chip.jpg"))
        self.assertEqual(parsed["age"], 35)
        self.assertEqual(parsed["gender"], "Male")
        self.assertEqual(parsed["race_label"], "White")
        self.assertEqual(parsed["collection_timestamp"], "20170117145906651")
        self.assertEqual(parsed["image_variant"], "chip")

    def test_rejects_invalid_shapes_and_values(self) -> None:
        invalid_names = (
            "117_0_0_20170101.jpg",
            "20_2_0_20170101.jpg",
            "20_0_5_20170101.jpg",
            "20_0_0_extra_20170101.jpg",
            "20_0_0_20170101.png",
        )
        for filename in invalid_names:
            with self.subTest(filename=filename):
                with self.assertRaises(ValueError):
                    parse_utkface_filename(Path(filename))


if __name__ == "__main__":
    unittest.main()
