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

from benchmark_runtime import (  # noqa: E402
    _log_wandb_result,
    _record_age_interval,
    wandb_metrics,
    wandb_summary_values,
)
from metrics import (  # noqa: E402
    age_interval_metrics,
    age_metrics_by_interval,
    gender_metrics,
    interval_absolute_error,
)
from records import BenchmarkRecord  # noqa: E402


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
    def test_reads_fairface_interval_metadata_without_requiring_proxy_age(self) -> None:
        record = BenchmarkRecord(
            dataset_id="DS003",
            dataset_version="margin025",
            sample_id="val_1",
            path=Path("val/1.jpg"),
            age=None,
            metadata={"age_min": 70, "age_max": None, "age_group": "70+"},
        )

        self.assertEqual(_record_age_interval(record), (70.0, None))

    def test_interval_absolute_error_uses_nearest_boundary(self) -> None:
        self.assertEqual(interval_absolute_error(25.0, 20.0, 29.0), 0.0)
        self.assertEqual(interval_absolute_error(18.0, 20.0, 29.0), 2.0)
        self.assertEqual(interval_absolute_error(32.0, 20.0, 29.0), 3.0)

    def test_open_ended_interval_accepts_every_prediction_above_lower_bound(self) -> None:
        self.assertEqual(interval_absolute_error(60.0, 70.0, None), 10.0)
        self.assertEqual(interval_absolute_error(70.0, 70.0, None), 0.0)
        self.assertEqual(interval_absolute_error(100.0, 70.0, None), 0.0)

    def test_aggregates_interval_aware_age_metrics(self) -> None:
        metrics = age_interval_metrics(
            predictions=[18.0, 25.0, 32.0, 60.0, 90.0],
            lower_bounds=[20.0, 20.0, 20.0, 70.0, 70.0],
            upper_bounds=[29.0, 29.0, 29.0, None, None],
        )

        self.assertEqual(metrics["evaluated_images"], 5)
        self.assertAlmostEqual(metrics["mae"], 3.0)
        self.assertAlmostEqual(metrics["rmse"], (113 / 5) ** 0.5)
        self.assertAlmostEqual(metrics["within_interval"], 0.4)
        self.assertAlmostEqual(metrics["within_5_years"], 0.8)
        self.assertAlmostEqual(metrics["within_10_years"], 1.0)

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
                    "interval_aware": {
                        "evaluated_images": 2,
                        "mae": 1.5,
                        "within_interval": 0.5,
                    },
                    "by_true_age_interval": {
                        "0-12": {"samples": 2, "mae": 8.5}
                    }
                },
            }
        )

        self.assertEqual(metrics["gender/accuracy"], 0.75)
        self.assertEqual(metrics["gender/macro_f1"], 0.7)
        self.assertEqual(metrics["age/by_true_age_interval/0-12/mae"], 8.5)
        self.assertEqual(metrics["age/interval_aware/mae"], 1.5)
        self.assertEqual(
            metrics["gender/confusion_matrix/true_female_pred_female"],
            3,
        )

    def test_logs_all_populated_aggregate_fields_and_skips_large_rows(self) -> None:
        result = {
            "schema_version": "2.0",
            "run": {
                "run_id": "RUN001",
                "sample_count": 2,
                "selected_sample_ids": ["sample-1", "sample-2"],
                "git_commit": None,
            },
            "environment": {"python": "3.12.0"},
            "config": {
                "warmup_runs": 10,
                "providers_requested": ["CUDA"],
                "enabled": True,
                "empty_value": "",
            },
            "models": {"face": {"size_bytes": 123, "path": "/models/face.onnx"}},
            "dataset": {
                "record_count": 2,
                "sampling": {"sample_ids": ["sample-1", "sample-2"]},
            },
            "pipeline": {"success_rate": 1.0},
            "samples": [{"sample_id": "sample-1", "pipeline_ms": 1.0}],
            "failures": [],
            "exports": {"headline": "/output/headline.tsv"},
        }

        summary = wandb_summary_values(result)
        metrics = wandb_metrics(result)

        self.assertEqual(summary["run/run_id"], "RUN001")
        self.assertEqual(summary["environment/python"], "3.12.0")
        self.assertEqual(summary["config/providers_requested"], '["CUDA"]')
        self.assertIs(summary["config/enabled"], True)
        self.assertNotIn("config/empty_value", summary)
        self.assertEqual(summary["models/face/path"], "/models/face.onnx")
        self.assertNotIn("run/git_commit", summary)
        self.assertNotIn("run/selected_sample_ids", summary)
        self.assertFalse(any(key.startswith("samples/") for key in summary))
        self.assertFalse(any(key.startswith("failures/") for key in summary))
        self.assertFalse(any(key.startswith("exports/") for key in summary))
        self.assertEqual(metrics["run/sample_count"], 2)
        self.assertEqual(metrics["config/warmup_runs"], 10)
        self.assertEqual(metrics["models/face/size_bytes"], 123)
        self.assertEqual(metrics["pipeline/success_rate"], 1.0)

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
            summary: dict[str, object] = {}

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
        self.assertEqual(FakeRun.summary["run/run_id"], "RUN001")
        self.assertEqual(FakeRun.summary["schema_version"], "2.0")


if __name__ == "__main__":
    unittest.main()
