from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = REPO_ROOT / "benchmark"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from benchmark_runtime import _log_wandb_result, wandb_metrics  # noqa: E402
from metrics import age_metrics_by_interval, gender_metrics  # noqa: E402


class GenderMetricsTests(unittest.TestCase):
    def test_computes_confusion_and_classification_metrics(self) -> None:
        metrics = gender_metrics(
            predictions=["Female", "Male", "Female", "Male"],
            labels=["Female", "Female", "Male", "Male"],
            female_probabilities=[0.9, 0.4, 0.8, 0.1],
        )

        self.assertEqual(
            metrics["confusion_matrix"],
            {
                "true_female_pred_female": 1,
                "true_female_pred_male": 1,
                "true_male_pred_female": 1,
                "true_male_pred_male": 1,
            },
        )
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["female_f1"], 0.5)
        self.assertAlmostEqual(metrics["male_f1"], 0.5)
        self.assertAlmostEqual(metrics["macro_f1"], 0.5)
        self.assertAlmostEqual(metrics["roc_auc_female"], 0.75)
        self.assertAlmostEqual(metrics["pr_auc_female"], 5 / 6)

    def test_auc_is_none_without_both_classes(self) -> None:
        metrics = gender_metrics(
            predictions=["Female", "Female"],
            labels=["Female", "Female"],
            female_probabilities=[0.9, 0.8],
        )

        self.assertIsNone(metrics["roc_auc_female"])
        self.assertAlmostEqual(metrics["pr_auc_female"], 1.0)


class AgeIntervalMetricsTests(unittest.TestCase):
    def test_computes_age_metrics_by_true_age_interval(self) -> None:
        metrics = age_metrics_by_interval(
            predictions=[10.0, 20.0, 30.0, 50.0],
            labels=[10, 14, 25, 45],
        )

        self.assertEqual(metrics["0-12"]["samples"], 1)
        self.assertAlmostEqual(metrics["0-12"]["mae"], 0.0)
        self.assertAlmostEqual(metrics["13-19"]["mae"], 6.0)
        self.assertAlmostEqual(metrics["13-19"]["mean_signed_error"], 6.0)
        self.assertAlmostEqual(metrics["20-29"]["within_5_years"], 1.0)
        self.assertAlmostEqual(metrics["40-49"]["rmse"], 5.0)
        self.assertEqual(metrics["110-119"]["samples"], 0)
        self.assertIsNone(metrics["110-119"]["mae"])


class WandbMetricFlatteningTests(unittest.TestCase):
    def test_flattens_new_metrics_and_confusion_matrix(self) -> None:
        metrics = wandb_metrics(
            {
                "gender": {
                    "accuracy": 0.75,
                    "macro_f1": 0.7,
                    "roc_auc_female": 0.8,
                    "confusion_matrix": {
                        "true_female_pred_female": 3,
                        "true_female_pred_male": 1,
                    },
                },
                "age": {
                    "by_true_age_interval": {
                        "0-12": {"samples": 2, "mae": 8.5}
                    }
                },
            }
        )

        self.assertEqual(metrics["gender/accuracy"], 0.75)
        self.assertEqual(metrics["gender/macro_f1"], 0.7)
        self.assertEqual(metrics["age/by_true_age_interval/0-12/mae"], 8.5)
        self.assertEqual(
            metrics["gender/confusion_matrix/true_female_pred_female"],
            3,
        )

    def test_artifact_metadata_stays_compact(self) -> None:
        calls: dict[str, object] = {}

        class FakeArtifact:
            def __init__(self, **kwargs: object) -> None:
                calls["artifact_kwargs"] = kwargs

            def add_file(self, **kwargs: object) -> None:
                calls["artifact_file"] = kwargs

        class FakeRun:
            id = "run-123"
            name = "benchmark-test"

            def log(self, metrics: dict[str, int | float]) -> None:
                calls["metrics"] = metrics

            def log_artifact(self, artifact: FakeArtifact) -> None:
                calls["artifact"] = artifact

        wandb = types.ModuleType("wandb")
        wandb.Artifact = FakeArtifact
        result = {
            "schema_version": "2.0",
            "created_at": "2026-08-05T00:00:00+00:00",
            "run": {"run_id": "RUN001", "dataset_id": "DS001"},
            "dataset": {"age_bins": {str(index): index for index in range(120)}},
            "age": {"mae": 4.2},
            "gender": {"accuracy": 0.75, "macro_f1": 0.7},
            "pipeline": {"processed_images": 10, "successful_images": 9},
        }

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "benchmark.json"
            output_path.write_text("{}", encoding="utf-8")
            with mock.patch.dict("sys.modules", {"wandb": wandb}):
                _log_wandb_result(FakeRun(), result, output_path)

        artifact_kwargs = calls["artifact_kwargs"]
        self.assertIsInstance(artifact_kwargs, dict)
        metadata = artifact_kwargs["metadata"]
        self.assertIsInstance(metadata, dict)
        self.assertLessEqual(len(metadata), 100)
        self.assertEqual(metadata["gender_accuracy"], 0.75)
        self.assertGreater(len(calls["metrics"]), 100)


if __name__ == "__main__":
    unittest.main()
