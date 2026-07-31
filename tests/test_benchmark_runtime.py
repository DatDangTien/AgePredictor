from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from scripts.benchmark_runtime import (
    AAFRecord,
    BenchmarkConfig,
    _init_wandb,
    _log_wandb_result,
    age_metrics,
    latency_summary,
    load_aaf_records,
    parse_args,
    select_records,
    value_summary,
    wandb_metrics,
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


class WandbLoggingTests(unittest.TestCase):
    def test_flattens_only_numeric_aggregate_metrics(self) -> None:
        metrics = wandb_metrics(
            {
                "dataset": {"image_count": 10, "selection": "all"},
                "face": {
                    "detection_coverage": 0.9,
                    "model_latency": {"mean_ms": 12.5, "p95_ms": None},
                },
                "age": {},
                "gender": {"prediction_counts": {"Female": 4, "Male": 6}},
                "pipeline": {"failed_images": 0},
                "samples": [{"pipeline_ms": 15.0}],
            }
        )

        self.assertEqual(metrics["dataset/image_count"], 10)
        self.assertEqual(metrics["face/detection_coverage"], 0.9)
        self.assertEqual(metrics["face/model_latency/mean_ms"], 12.5)
        self.assertEqual(metrics["gender/prediction_counts/Female"], 4)
        self.assertNotIn("face/model_latency/p95_ms", metrics)
        self.assertNotIn("samples", metrics)

    def test_wandb_cli_is_opt_in_and_supports_offline_runs(self) -> None:
        default_args = parse_args([])
        self.assertFalse(default_args.wandb)

        args = parse_args(
            [
                "--wandb",
                "--wandb-mode",
                "offline",
                "--wandb-tags",
                "cpu",
                "smoke",
            ]
        )
        self.assertTrue(args.wandb)
        self.assertEqual(args.wandb_mode, "offline")
        self.assertEqual(args.wandb_tags, ["cpu", "smoke"])

    def test_initializes_run_and_logs_json_artifact(self) -> None:
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

        run = FakeRun()
        wandb = types.ModuleType("wandb")
        wandb.Artifact = FakeArtifact

        def init(**kwargs: object) -> FakeRun:
            calls["init_kwargs"] = kwargs
            return run

        wandb.init = init

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "benchmark.json"
            output_path.write_text("{}", encoding="utf-8")
            config = BenchmarkConfig(output_path=output_path)
            with mock.patch.dict("sys.modules", {"wandb": wandb}):
                initialized_run = _init_wandb(
                    config,
                    project="benchmark-project",
                    entity="team",
                    run_name="benchmark-test",
                    mode="offline",
                    tags=("cpu",),
                )
                _log_wandb_result(
                    initialized_run,
                    {"pipeline": {"successful_images": 10}},
                    output_path,
                )

        init_kwargs = calls["init_kwargs"]
        self.assertIsInstance(init_kwargs, dict)
        self.assertEqual(init_kwargs["project"], "benchmark-project")
        self.assertEqual(init_kwargs["mode"], "offline")
        self.assertEqual(calls["metrics"], {"pipeline/successful_images": 10})
        self.assertIsInstance(calls["artifact"], FakeArtifact)
        self.assertEqual(
            calls["artifact_file"],
            {
                "local_path": str(output_path.resolve()),
                "name": "benchmark_runtime.json",
            },
        )


if __name__ == "__main__":
    unittest.main()
