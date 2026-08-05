#!/usr/bin/env python3
"""Benchmark the deployed face, age, and gender ONNX pipeline from a manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import cv2
import numpy as np
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = Path(__file__).resolve().parent
for import_root in (REPO_ROOT, BENCHMARK_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from app import inference as inference_module  # noqa: E402
from app.inference import AgePipeline  # noqa: E402
from exporters import export_all  # noqa: E402
from manifest import load_manifest  # noqa: E402
from metrics import (  # noqa: E402
    age_metrics,
    age_metrics_by_interval,
    gender_metrics,
    latency_summary,
    value_summary,
)
from records import BenchmarkRecord  # noqa: E402
from sampling import select_records  # noqa: E402

DEFAULT_MODELS_DIR = REPO_ROOT / "models"
DEFAULT_MANIFEST = REPO_ROOT / "manifests" / "DS001_all_age_faces.csv"
DEFAULT_WANDB_PROJECT = "AgeGenderPredictor"


@dataclass(frozen=True)
class BenchmarkConfig:
    manifest_path: Path
    output_path: Path
    dataset_id: str
    run_id: str
    config_id: str
    face_model_path: Path = (
        DEFAULT_MODELS_DIR / "face_det_lite-onnx-w8a8" / "face_det_lite.onnx"
    )
    age_model_path: Path = DEFAULT_MODELS_DIR / "age.onnx"
    gender_model_path: Path = (
        DEFAULT_MODELS_DIR / "adaface_ir50_ms1mv2_gender.onnx"
    )
    providers: tuple[str, ...] = ("CPUExecutionProvider",)
    limit: int | None = None
    sampling_strategy: str = "all"
    random_seed: int = 42
    sample_list_path: Path | None = None
    warmup_runs: int = 10
    progress_every: int = 100


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool = False
    project: str = DEFAULT_WANDB_PROJECT
    entity: str | None = None
    run_name: str | None = None
    mode: str = "online"
    tags: tuple[str, ...] = ()


@dataclass
class BenchmarkMeasurements:
    """Mutable measurements collected across benchmark records."""

    face_stage_ms: list[float] = field(default_factory=list)
    face_model_ms: list[float] = field(default_factory=list)
    face_counts: list[int] = field(default_factory=list)
    age_model_ms: list[float] = field(default_factory=list)
    gender_model_ms: list[float] = field(default_factory=list)
    pipeline_ms: list[float] = field(default_factory=list)
    age_predictions: list[float] = field(default_factory=list)
    age_labels: list[int] = field(default_factory=list)
    gender_predictions: list[str] = field(default_factory=list)
    gender_labels: list[str] = field(default_factory=list)
    gender_confidences: list[float] = field(default_factory=list)
    female_probabilities: list[float] = field(default_factory=list)
    age_inference_count: int = 0
    gender_inference_count: int = 0
    pipeline_success_count: int = 0
    samples: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)


class TimedSession:
    """Proxy an ONNX Runtime session and time each ``run`` call."""

    def __init__(self, session: ort.InferenceSession):
        self.session = session
        self.timings_ms: list[float] = []

    def run(self, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return self.session.run(*args, **kwargs)
        finally:
            self.timings_ms.append((time.perf_counter() - started) * 1000.0)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.session, name)


def wandb_metrics(result: dict[str, Any]) -> dict[str, int | float]:
    """Flatten numeric aggregate results into W&B metric names."""
    metrics: dict[str, int | float] = {}

    def collect(value: Any, prefix: str) -> None:
        if isinstance(value, dict):
            for key, nested_value in value.items():
                collect(nested_value, f"{prefix}/{key}")
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[prefix] = value

    for section in ("dataset", "face", "age", "gender", "pipeline"):
        collect(result.get(section, {}), section)
    return metrics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_metadata(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Model not found: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _female_probability(logits: np.ndarray) -> float:
    values = logits[0].astype(np.float64)
    probabilities = np.exp(values - values.max())
    probabilities /= probabilities.sum()
    return float(probabilities[0])


def _warmup(
    pipeline: AgePipeline,
    records: Sequence[BenchmarkRecord],
    runs: int,
) -> None:
    if runs <= 0:
        return

    image: np.ndarray | None = None
    face: tuple[list[float], float, list] | None = None
    for record in records:
        candidate = cv2.imread(str(record.path))
        if candidate is None:
            continue
        faces = pipeline.detect_faces(candidate)
        if faces:
            image = candidate
            face = max(faces, key=lambda item: item[1])
            break
    if image is None or face is None:
        raise RuntimeError("Warm-up failed: no detectable face in selected records")

    for _ in range(runs):
        pipeline.detect_faces(image)

    age_input = pipeline._preprocess_face_crop(image, face[0], face[2])
    if age_input is None:
        raise RuntimeError("Warm-up failed: production face crop returned no input")
    gender_input = pipeline._gender_input(age_input)
    for _ in range(runs):
        pipeline.age_sess.run([pipeline.age_out], {pipeline.age_in: age_input})
        pipeline.gender_sess.run(
            [pipeline.gender_out],
            {pipeline.gender_in: gender_input},
        )


def _provider_metadata(pipeline: AgePipeline) -> dict[str, list[str]]:
    return {
        "available": list(ort.get_available_providers()),
        "face": list(pipeline.face_sess.get_providers()),
        "age": list(pipeline.age_sess.get_providers()),
        "gender": list(pipeline.gender_sess.get_providers()),
    }


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def _benchmark_record(
    pipeline: AgePipeline,
    face_timer: TimedSession,
    record: BenchmarkRecord,
    measurements: BenchmarkMeasurements,
) -> None:
    pipeline_started = time.perf_counter()
    sample: dict[str, Any] = {
        "dataset_id": record.dataset_id,
        "sample_id": record.sample_id,
        "filename": record.path.name,
        "age_true": record.age,
        "gender_true": record.gender_label,
    }
    failure_stage = "unknown"

    try:
        failure_stage = "decode"
        image = cv2.imread(str(record.path))
        if image is None:
            raise ValueError("image_decode_failed")

        failure_stage = "face_detection"
        face_call_count = len(face_timer.timings_ms)
        stage_started = time.perf_counter()
        faces = pipeline.detect_faces(image)
        face_stage_ms = _elapsed_ms(stage_started)
        measurements.face_stage_ms.append(face_stage_ms)
        measurements.face_counts.append(len(faces))
        sample.update(face_count=len(faces), face_stage_ms=face_stage_ms)

        if len(face_timer.timings_ms) > face_call_count:
            face_model_ms = face_timer.timings_ms[-1]
            measurements.face_model_ms.append(face_model_ms)
            sample["face_model_ms"] = face_model_ms

        if not faces:
            raise ValueError("no_face_detected")

        failure_stage = "face_crop"
        xyxy, face_score, landmarks = max(faces, key=lambda item: item[1])
        sample["face_score"] = float(face_score)
        age_input = pipeline._preprocess_face_crop(image, xyxy, landmarks)
        if age_input is None:
            raise ValueError("face_crop_failed")

        failure_stage = "age_inference"
        model_started = time.perf_counter()
        age_output = pipeline.age_sess.run(
            [pipeline.age_out],
            {pipeline.age_in: age_input},
        )[0]
        age_model_ms = _elapsed_ms(model_started)
        predicted_age = pipeline._postprocess_age(age_output)
        measurements.age_model_ms.append(age_model_ms)
        measurements.age_inference_count += 1
        sample.update(age_predicted=predicted_age, age_model_ms=age_model_ms)
        if record.age is not None:
            measurements.age_predictions.append(predicted_age)
            measurements.age_labels.append(record.age)
            sample["age_absolute_error"] = abs(predicted_age - record.age)
            sample["age_signed_error"] = predicted_age - record.age

        failure_stage = "gender_inference"
        gender_input = pipeline._gender_input(age_input)
        model_started = time.perf_counter()
        gender_output = pipeline.gender_sess.run(
            [pipeline.gender_out],
            {pipeline.gender_in: gender_input},
        )[0]
        gender_model_ms = _elapsed_ms(model_started)
        predicted_gender, gender_confidence = pipeline._postprocess_gender(
            gender_output
        )
        female_probability = _female_probability(gender_output)
        measurements.gender_model_ms.append(gender_model_ms)
        measurements.gender_inference_count += 1
        sample.update(
            gender_predicted=predicted_gender,
            gender_confidence=gender_confidence,
            female_probability=female_probability,
            gender_model_ms=gender_model_ms,
        )
        if record.gender_label is not None:
            measurements.gender_labels.append(record.gender_label)
            measurements.gender_predictions.append(predicted_gender)
            measurements.gender_confidences.append(gender_confidence)
            measurements.female_probabilities.append(female_probability)
            sample["gender_correct"] = predicted_gender == record.gender_label

        measurements.pipeline_success_count += 1
    except Exception as exc:
        reason = str(exc) or type(exc).__name__
        sample["failure"] = reason
        measurements.failures.append(
            {
                "sample_id": record.sample_id,
                "filename": record.path.name,
                "stage": failure_stage,
                "reason": reason,
            }
        )
    finally:
        pipeline_ms = _elapsed_ms(pipeline_started)
        measurements.pipeline_ms.append(pipeline_ms)
        sample["pipeline_ms"] = pipeline_ms
        measurements.samples.append(sample)


def _dataset_summary(records: Sequence[BenchmarkRecord]) -> dict[str, Any]:
    gender_counts = Counter(record.gender_label for record in records if record.gender_label)
    age_values = [record.age for record in records if record.age is not None]
    dataset_ids = sorted({record.dataset_id for record in records})
    splits = Counter(record.dataset_split or "unspecified" for record in records)
    return {
        "record_count": len(records),
        "dataset_ids": dataset_ids,
        "age_ground_truth_available": bool(age_values),
        "gender_ground_truth_available": bool(gender_counts),
        "bounding_boxes_available": any(record.bbox_xyxy is not None for record in records),
        "age_min": min(age_values) if age_values else None,
        "age_max": max(age_values) if age_values else None,
        "unique_ages": len(set(age_values)),
        "gender_counts": {
            label: int(gender_counts.get(label, 0))
            for label in inference_module.GENDER_LABELS
        },
        "dataset_split_counts": dict(splits),
    }


def _build_result(
    config: BenchmarkConfig,
    all_records: Sequence[BenchmarkRecord],
    selected_records: Sequence[BenchmarkRecord],
    sampling_metadata: dict[str, Any],
    providers: dict[str, list[str]],
    measurements: BenchmarkMeasurements,
    wall_seconds: float,
) -> dict[str, Any]:
    gender_count = len(measurements.gender_predictions)
    prediction_counts = Counter(measurements.gender_predictions)
    manifest_path = Path(config.manifest_path).expanduser().resolve()

    return {
        "schema_version": "2.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run": {
            "run_id": config.run_id,
            "config_id": config.config_id,
            "dataset_id": config.dataset_id,
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "sample_count": len(selected_records),
            "sampling_strategy": sampling_metadata["strategy"],
            "random_seed": config.random_seed,
            "sample_list_path": sampling_metadata.get("sample_list_path"),
            "selected_sample_ids": sampling_metadata["sample_ids"],
            "git_commit": _git_commit(),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "onnxruntime": ort.__version__,
        },
        "config": {
            "providers_requested": list(config.providers),
            "warmup_runs": config.warmup_runs,
            "gender_female_threshold": (
                inference_module.GENDER_FEMALE_THRESHOLD
            ),
        },
        "providers": providers,
        "models": {
            "face": _model_metadata(Path(config.face_model_path)),
            "age": _model_metadata(Path(config.age_model_path)),
            "gender": _model_metadata(Path(config.gender_model_path)),
        },
        "dataset": {
            **_dataset_summary(all_records),
            "selected_images": len(selected_records),
            "sampling": sampling_metadata,
        },
        "face": {
            "measurement": "latency and detection coverage",
            "processed_images": len(selected_records),
            "images_with_detection": int(
                sum(count > 0 for count in measurements.face_counts)
            ),
            "detection_coverage": (
                float(
                    sum(count > 0 for count in measurements.face_counts)
                    / len(selected_records)
                )
                if selected_records
                else None
            ),
            "mean_faces_per_processed_image": (
                float(np.mean(measurements.face_counts))
                if measurements.face_counts
                else None
            ),
            "model_latency": latency_summary(measurements.face_model_ms),
            "stage_latency": latency_summary(measurements.face_stage_ms),
        },
        "age": {
            "measurement": "accuracy against available age labels",
            "inference_images": measurements.age_inference_count,
            **age_metrics(
                measurements.age_predictions,
                measurements.age_labels,
            ),
            "by_true_age_interval": age_metrics_by_interval(
                measurements.age_predictions,
                measurements.age_labels,
            ),
            "model_latency": latency_summary(measurements.age_model_ms),
        },
        "gender": {
            "measurement": "accuracy against available gender labels",
            "inference_images": measurements.gender_inference_count,
            "evaluated_images": gender_count,
            **gender_metrics(
                measurements.gender_predictions,
                measurements.gender_labels,
                measurements.female_probabilities,
            ),
            "prediction_counts": {
                label: int(prediction_counts.get(label, 0))
                for label in inference_module.GENDER_LABELS
            },
            "prediction_rates": {
                label: (
                    float(prediction_counts.get(label, 0) / gender_count)
                    if gender_count
                    else None
                )
                for label in inference_module.GENDER_LABELS
            },
            "confidence": value_summary(measurements.gender_confidences),
            "female_probability": value_summary(
                measurements.female_probabilities
            ),
            "model_latency": latency_summary(measurements.gender_model_ms),
        },
        "pipeline": {
            "processed_images": len(selected_records),
            "successful_images": measurements.pipeline_success_count,
            "failed_images": len(measurements.failures),
            "success_rate": (
                float(measurements.pipeline_success_count / len(selected_records))
                if selected_records
                else None
            ),
            "latency": latency_summary(measurements.pipeline_ms),
            "wall_seconds": wall_seconds,
            "throughput_images_per_second": (
                float(len(selected_records) / wall_seconds)
                if wall_seconds > 0
                else None
            ),
        },
        "failures": measurements.failures,
        "samples": measurements.samples,
    }


def _write_result(result: dict[str, Any], output_path: Path) -> Path:
    resolved_path = output_path.expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    result["exports"] = export_all(result, resolved_path)
    resolved_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return resolved_path


def run_benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    all_records = load_manifest(config.manifest_path)
    selected_records, sampling_metadata = select_records(
        all_records,
        limit=config.limit,
        strategy=config.sampling_strategy,
        seed=config.random_seed,
        sample_list_path=config.sample_list_path,
    )
    print(
        f"[benchmark] loaded {len(all_records):,} manifest records; "
        f"running {len(selected_records):,}"
    )

    pipeline = AgePipeline(
        face_model_path=str(config.face_model_path),
        age_model_path=str(config.age_model_path),
        gender_model_path=str(config.gender_model_path),
        providers=list(config.providers),
    )
    providers = _provider_metadata(pipeline)
    _warmup(pipeline, selected_records, config.warmup_runs)

    face_timer = TimedSession(pipeline.face_sess)
    pipeline.face_sess = face_timer  # type: ignore[assignment]
    measurements = BenchmarkMeasurements()

    wall_started = time.perf_counter()
    for position, record in enumerate(selected_records, start=1):
        _benchmark_record(pipeline, face_timer, record, measurements)
        if config.progress_every > 0 and (
            position % config.progress_every == 0
            or position == len(selected_records)
        ):
            print(
                f"[benchmark] processed {position:,}/"
                f"{len(selected_records):,}"
            )

    result = _build_result(
        config=config,
        all_records=all_records,
        selected_records=selected_records,
        sampling_metadata=sampling_metadata,
        providers=providers,
        measurements=measurements,
        wall_seconds=time.perf_counter() - wall_started,
    )
    output_path = _write_result(result, Path(config.output_path))
    print(f"[benchmark] wrote {output_path}")
    return result


def print_summary(result: dict[str, Any]) -> None:
    def number(value: float | None, digits: int = 2) -> str:
        return "n/a" if value is None else f"{value:.{digits}f}"

    def percent(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.2%}"

    face = result["face"]
    age = result["age"]
    gender = result["gender"]
    pipeline = result["pipeline"]

    print("\nFace detector")
    print(
        f"  coverage: {face['images_with_detection']}/"
        f"{face['processed_images']} ({percent(face['detection_coverage'])})"
    )
    print(
        f"  model latency mean/p50/p95: "
        f"{number(face['model_latency']['mean_ms'])} / "
        f"{number(face['model_latency']['p50_ms'])} / "
        f"{number(face['model_latency']['p95_ms'])} ms"
    )

    print("\nAge")
    print(
        f"  inference/evaluated: {age['inference_images']}/"
        f"{age['evaluated_images']} | MAE {number(age['mae'], 3)} | "
        f"RMSE {number(age['rmse'], 3)}"
    )
    print(
        f"  mean signed error: {number(age['mean_signed_error'], 3)} | "
        f"within 5/10 years: {percent(age['within_5_years'])} / "
        f"{percent(age['within_10_years'])}"
    )

    print("\nGender")
    print(
        f"  inference/evaluated: {gender['inference_images']}/"
        f"{gender['evaluated_images']} | accuracy {percent(gender['accuracy'])} | "
        f"balanced accuracy {percent(gender['balanced_accuracy'])} | "
        f"macro F1 {number(gender['macro_f1'], 3)}"
    )
    print(
        f"  female ROC-AUC/PR-AUC: "
        f"{number(gender['roc_auc_female'], 3)} / "
        f"{number(gender['pr_auc_female'], 3)}"
    )
    print(f"  prediction counts: {gender['prediction_counts']}")

    print("\nEnd-to-end pipeline")
    print(
        f"  success: {pipeline['successful_images']}/"
        f"{pipeline['processed_images']} ({percent(pipeline['success_rate'])})"
    )
    print(
        f"  latency mean/p50/p95: "
        f"{number(pipeline['latency']['mean_ms'])} / "
        f"{number(pipeline['latency']['p50_ms'])} / "
        f"{number(pipeline['latency']['p95_ms'])} ms"
    )
    print(
        f"  wall throughput: "
        f"{number(pipeline['throughput_images_per_second'])} images/s"
    )


def _providers_from_name(name: str) -> tuple[str, ...]:
    if name == "cpu":
        return ("CPUExecutionProvider",)
    if name == "coreml":
        return ("CoreMLExecutionProvider", "CPUExecutionProvider")
    if name == "cuda":
        return ("CUDAExecutionProvider", "CPUExecutionProvider")
    available = ort.get_available_providers()
    if "CoreMLExecutionProvider" in available:
        return ("CoreMLExecutionProvider", "CPUExecutionProvider")
    if "CUDAExecutionProvider" in available:
        return ("CUDAExecutionProvider", "CPUExecutionProvider")
    return ("CPUExecutionProvider",)


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return completed.stdout.strip() or None


def _init_wandb(
    config: BenchmarkConfig,
    wandb_config: WandbConfig,
) -> Any:
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B logging requires the 'wandb' package. "
            "Install the project requirements and try again."
        ) from exc

    return wandb.init(
        project=wandb_config.project,
        entity=wandb_config.entity,
        name=wandb_config.run_name,
        mode=wandb_config.mode,
        tags=list(wandb_config.tags),
        job_type="runtime-benchmark",
        config={
            "manifest_path": str(Path(config.manifest_path).expanduser().resolve()),
            "output_path": str(Path(config.output_path).expanduser().resolve()),
            "dataset_id": config.dataset_id,
            "run_id": config.run_id,
            "config_id": config.config_id,
            "providers": list(config.providers),
            "limit": config.limit,
            "sampling_strategy": config.sampling_strategy,
            "random_seed": config.random_seed,
            "sample_list_path": (
                str(config.sample_list_path.expanduser().resolve())
                if config.sample_list_path
                else None
            ),
            "warmup_runs": config.warmup_runs,
        },
        save_code=True,
    )


@contextmanager
def _wandb_session(
    config: BenchmarkConfig,
    wandb_config: WandbConfig,
) -> Iterator[Any | None]:
    if not wandb_config.enabled:
        yield None
        return

    run = _init_wandb(config, wandb_config)
    try:
        yield run
    except BaseException:
        run.finish(exit_code=1)
        raise
    else:
        run.finish(exit_code=0)


def _log_wandb_result(
    run: Any,
    result: dict[str, Any],
    output_path: Path,
) -> None:
    import wandb

    metrics = wandb_metrics(result)
    run.log(metrics)

    artifact_metadata = {
        "schema_version": result.get("schema_version"),
        "created_at": result.get("created_at"),
        "run_id": result.get("run", {}).get("run_id"),
        "dataset_id": result.get("run", {}).get("dataset_id"),
        "processed_images": result.get("pipeline", {}).get("processed_images"),
        "successful_images": result.get("pipeline", {}).get("successful_images"),
        "age_mae": result.get("age", {}).get("mae"),
        "gender_accuracy": result.get("gender", {}).get("accuracy"),
        "gender_macro_f1": result.get("gender", {}).get("macro_f1"),
    }
    artifact = wandb.Artifact(
        name=f"benchmark-runtime-{run.id}",
        type="benchmark-result",
        description="Runtime benchmark metrics, per-image results, and failures",
        metadata=artifact_metadata,
    )
    artifact.add_file(
        local_path=str(output_path.expanduser().resolve()),
        name=output_path.name,
    )
    run.log_artifact(artifact)
    print(f"[benchmark] logged {len(metrics)} metrics to W&B run {run.name}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset-id", default="DS001")
    parser.add_argument(
        "--run-id",
        default=datetime.now(timezone.utc).strftime("RUN_%Y%m%d_%H%M%S"),
    )
    parser.add_argument("--config-id", default="CFG001")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "output" / "benchmark_runtime.json",
        help="JSON output path",
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--sampling",
        choices=(
            "all",
            "evenly_spaced",
            "random",
            "stratified_age",
            "stratified_age_gender",
            "from_sample_list",
        ),
        default="all",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-list", type=Path)
    parser.add_argument(
        "--provider",
        choices=("auto", "cpu", "coreml", "cuda"),
        default="cpu",
    )
    parser.add_argument("--warmup-runs", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--face-model",
        type=Path,
        default=DEFAULT_MODELS_DIR
        / "face_det_lite-onnx-w8a8"
        / "face_det_lite.onnx",
    )
    parser.add_argument(
        "--age-model",
        type=Path,
        default=DEFAULT_MODELS_DIR / "age.onnx",
    )
    parser.add_argument(
        "--gender-model",
        type=Path,
        default=DEFAULT_MODELS_DIR / "adaface_ir50_ms1mv2_gender.onnx",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log aggregate metrics and the JSON result artifact to W&B",
    )
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline"),
        default="online",
    )
    parser.add_argument("--wandb-tags", nargs="*", default=())
    return parser.parse_args(argv)


def _benchmark_config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    return BenchmarkConfig(
        manifest_path=args.manifest,
        output_path=args.output,
        dataset_id=args.dataset_id,
        run_id=args.run_id,
        config_id=args.config_id,
        face_model_path=args.face_model,
        age_model_path=args.age_model,
        gender_model_path=args.gender_model,
        providers=_providers_from_name(args.provider),
        limit=None if args.limit <= 0 else args.limit,
        sampling_strategy=args.sampling,
        random_seed=args.seed,
        sample_list_path=args.sample_list,
        warmup_runs=args.warmup_runs,
        progress_every=args.progress_every,
    )


def _wandb_config_from_args(args: argparse.Namespace) -> WandbConfig:
    return WandbConfig(
        enabled=args.wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        mode=args.wandb_mode,
        tags=tuple(args.wandb_tags),
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = _benchmark_config_from_args(args)
    wandb_config = _wandb_config_from_args(args)

    with _wandb_session(config, wandb_config) as wandb_run:
        result = run_benchmark(config)
        print_summary(result)
        if wandb_run is not None:
            _log_wandb_result(
                wandb_run,
                result,
                Path(config.output_path),
            )


if __name__ == "__main__":
    main()
