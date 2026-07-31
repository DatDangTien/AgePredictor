#!/usr/bin/env python3
"""Benchmark the deployed face, age, and gender ONNX pipeline.

All-Age-Faces filenames encode age as ``NNNNNAxx.jpg``. They do not encode
gender and do not include face bounding boxes, so this benchmark reports:

* face-detector latency and detection coverage (not AP/IoU accuracy);
* age latency and accuracy against the filename age;
* gender latency, prediction distribution, and confidence (not accuracy);
* end-to-end pipeline latency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app import inference as inference_module  # noqa: E402
from app.inference import AgePipeline  # noqa: E402

AAF_FILENAME_PATTERN = re.compile(
    r"^(?P<image_id>\d{5})A(?P<age>\d{2})\.jpg$",
    re.IGNORECASE,
)
DEFAULT_EXPECTED_COUNT = 13_322
DEFAULT_MODELS_DIR = REPO_ROOT / "models"
DEFAULT_WANDB_PROJECT = "AgeGenderPredictor"


@dataclass(frozen=True)
class AAFRecord:
    path: Path
    image_id: int
    age: int


@dataclass(frozen=True)
class BenchmarkConfig:
    data_dir: Path = REPO_ROOT / "data"
    output_path: Path = REPO_ROOT / "output" / "benchmark_runtime.json"
    face_model_path: Path = (
        DEFAULT_MODELS_DIR / "face_det_lite-onnx-w8a8" / "face_det_lite.onnx"
    )
    age_model_path: Path = DEFAULT_MODELS_DIR / "age.onnx"
    gender_model_path: Path = (
        DEFAULT_MODELS_DIR / "adaface_ir50_ms1mv2_gender.onnx"
    )
    providers: tuple[str, ...] = ("CPUExecutionProvider",)
    limit: int | None = 100
    warmup_runs: int = 10
    expected_count: int | None = DEFAULT_EXPECTED_COUNT
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
    gender_confidences: list[float] = field(default_factory=list)
    female_probabilities: list[float] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, str]] = field(default_factory=list)


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


def load_aaf_records(
    data_dir: Path | str,
    *,
    expected_count: int | None = DEFAULT_EXPECTED_COUNT,
) -> list[AAFRecord]:
    """Load and validate flat All-Age-Faces JPEG filenames."""
    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"All-Age-Faces directory not found: {root}")

    paths = sorted(root.glob("*.jpg"), key=lambda path: path.name.lower())
    invalid_names: list[str] = []
    records: list[AAFRecord] = []
    for path in paths:
        match = AAF_FILENAME_PATTERN.fullmatch(path.name)
        if match is None:
            invalid_names.append(path.name)
            continue
        records.append(
            AAFRecord(
                path=path,
                image_id=int(match.group("image_id")),
                age=int(match.group("age")),
            )
        )

    if invalid_names:
        sample = ", ".join(invalid_names[:5])
        raise ValueError(
            f"{len(invalid_names)} JPEG filename(s) do not match NNNNNAxx.jpg: "
            f"{sample}"
        )
    if expected_count is not None and len(records) != expected_count:
        raise ValueError(
            f"Expected {expected_count:,} All-Age-Faces images, found "
            f"{len(records):,} in {root}"
        )
    if not records:
        raise ValueError(f"No All-Age-Faces JPEGs found in {root}")
    return records


def select_records(
    records: Sequence[AAFRecord],
    limit: int | None,
) -> list[AAFRecord]:
    """Select a deterministic, evenly spaced subset of the ordered corpus."""
    if limit is None or limit <= 0 or limit >= len(records):
        return list(records)
    if limit == 1:
        return [records[len(records) // 2]]
    last = len(records) - 1
    indices = [round(index * last / (limit - 1)) for index in range(limit)]
    return [records[index] for index in indices]


def summarize_dataset(records: Sequence[AAFRecord]) -> dict[str, Any]:
    ages = np.asarray([record.age for record in records], dtype=np.int64)
    return {
        "image_count": len(records),
        "age_min": int(ages.min()),
        "age_max": int(ages.max()),
        "unique_ages": int(np.unique(ages).size),
        "age_bins": {
            f"{start}-{end}": int(np.sum((ages >= start) & (ages <= end)))
            for start, end in (
                (0, 9),
                (10, 19),
                (20, 29),
                (30, 39),
                (40, 49),
                (50, 59),
                (60, 69),
                (70, 79),
                (80, 89),
            )
        },
    }


def latency_summary(values_ms: Sequence[float]) -> dict[str, Any]:
    if not values_ms:
        return {
            "samples": 0,
            "mean_ms": None,
            "min_ms": None,
            "max_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "fps_from_mean": None,
        }
    values = np.asarray(values_ms, dtype=np.float64)
    mean_ms = float(values.mean())
    return {
        "samples": int(values.size),
        "mean_ms": mean_ms,
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "fps_from_mean": float(1000.0 / mean_ms) if mean_ms > 0 else None,
    }


def value_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "samples": 0,
            "mean": None,
            "min": None,
            "max": None,
            "p50": None,
            "p95": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "samples": int(array.size),
        "mean": float(array.mean()),
        "min": float(array.min()),
        "max": float(array.max()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
    }


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


def age_metrics(
    predictions: Sequence[float],
    labels: Sequence[int],
) -> dict[str, Any]:
    if not predictions:
        return {
            "evaluated_images": 0,
            "mae": None,
            "rmse": None,
            "median_absolute_error": None,
            "p90_absolute_error": None,
            "within_5_years": None,
            "within_10_years": None,
        }
    predicted = np.asarray(predictions, dtype=np.float64)
    expected = np.asarray(labels, dtype=np.float64)
    error = predicted - expected
    absolute_error = np.abs(error)
    return {
        "evaluated_images": int(predicted.size),
        "mae": float(absolute_error.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "median_absolute_error": float(np.median(absolute_error)),
        "p90_absolute_error": float(np.percentile(absolute_error, 90)),
        "within_5_years": float(np.mean(absolute_error <= 5.0)),
        "within_10_years": float(np.mean(absolute_error <= 10.0)),
    }


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
    records: Sequence[AAFRecord],
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
    record: AAFRecord,
    measurements: BenchmarkMeasurements,
) -> None:
    """Benchmark one record while preserving failures in the final report."""
    pipeline_started = time.perf_counter()
    sample: dict[str, Any] = {
        "filename": record.path.name,
        "age_true": record.age,
    }

    try:
        image = cv2.imread(str(record.path))
        if image is None:
            raise ValueError("image_decode_failed")

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

        xyxy, face_score, landmarks = max(faces, key=lambda item: item[1])
        sample["face_score"] = float(face_score)
        age_input = pipeline._preprocess_face_crop(image, xyxy, landmarks)
        if age_input is None:
            raise ValueError("face_crop_failed")

        model_started = time.perf_counter()
        age_output = pipeline.age_sess.run(
            [pipeline.age_out],
            {pipeline.age_in: age_input},
        )[0]
        age_model_ms = _elapsed_ms(model_started)
        predicted_age = pipeline._postprocess_age(age_output)
        measurements.age_model_ms.append(age_model_ms)
        measurements.age_predictions.append(predicted_age)
        measurements.age_labels.append(record.age)
        sample.update(
            age_predicted=predicted_age,
            age_absolute_error=abs(predicted_age - record.age),
            age_model_ms=age_model_ms,
        )

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
        measurements.gender_predictions.append(predicted_gender)
        measurements.gender_confidences.append(gender_confidence)
        measurements.female_probabilities.append(female_probability)
        sample.update(
            gender_predicted=predicted_gender,
            gender_confidence=gender_confidence,
            female_probability=female_probability,
            gender_model_ms=gender_model_ms,
        )
    except Exception as exc:  # keep per-image failures visible
        reason = str(exc) or type(exc).__name__
        sample["failure"] = reason
        measurements.failures.append(
            {"filename": record.path.name, "reason": reason}
        )
    finally:
        pipeline_ms = _elapsed_ms(pipeline_started)
        measurements.pipeline_ms.append(pipeline_ms)
        sample["pipeline_ms"] = pipeline_ms
        measurements.samples.append(sample)


def _build_result(
    config: BenchmarkConfig,
    all_records: Sequence[AAFRecord],
    selected_records: Sequence[AAFRecord],
    providers: dict[str, list[str]],
    measurements: BenchmarkMeasurements,
    wall_seconds: float,
) -> dict[str, Any]:
    successful_images = len(measurements.age_predictions)
    gender_count = len(measurements.gender_predictions)
    prediction_counts = Counter(measurements.gender_predictions)

    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "onnxruntime": ort.__version__,
        },
        "config": {
            "data_dir": str(Path(config.data_dir).expanduser().resolve()),
            "limit": config.limit,
            "warmup_runs": config.warmup_runs,
            "expected_count": config.expected_count,
            "providers_requested": list(config.providers),
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
            **summarize_dataset(all_records),
            "selected_images": len(selected_records),
            "selection": (
                "all"
                if len(selected_records) == len(all_records)
                else "evenly_spaced_over_filename_order"
            ),
            "gender_ground_truth_available": False,
            "bounding_boxes_available": False,
        },
        "face": {
            "measurement": (
                "latency and detection coverage; no bounding-box accuracy"
            ),
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
            "measurement": "accuracy against age encoded in NNNNNAxx.jpg",
            **age_metrics(
                measurements.age_predictions,
                measurements.age_labels,
            ),
            "model_latency": latency_summary(measurements.age_model_ms),
        },
        "gender": {
            "measurement": (
                "latency and prediction distribution; no accuracy without labels"
            ),
            "evaluated_images": gender_count,
            "accuracy": None,
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
            "successful_images": successful_images,
            "failed_images": len(measurements.failures),
            "success_rate": (
                float(successful_images / len(selected_records))
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
    resolved_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return resolved_path


def run_benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    """Run the three-stage benchmark and write its JSON result."""
    all_records = load_aaf_records(
        config.data_dir,
        expected_count=config.expected_count,
    )
    selected_records = select_records(all_records, config.limit)
    print(
        f"[benchmark] validated {len(all_records):,} images; "
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
    print(
        f"  stage latency mean/p50/p95: "
        f"{number(face['stage_latency']['mean_ms'])} / "
        f"{number(face['stage_latency']['p50_ms'])} / "
        f"{number(face['stage_latency']['p95_ms'])} ms"
    )

    print("\nAge")
    print(
        f"  evaluated: {age['evaluated_images']} | "
        f"MAE {number(age['mae'], 3)} | RMSE {number(age['rmse'], 3)}"
    )
    print(
        f"  median/p90 absolute error: "
        f"{number(age['median_absolute_error'], 3)} / "
        f"{number(age['p90_absolute_error'], 3)} years"
    )
    print(
        f"  within 5/10 years: "
        f"{percent(age['within_5_years'])} / "
        f"{percent(age['within_10_years'])}"
    )
    print(
        f"  model latency mean/p50/p95: "
        f"{number(age['model_latency']['mean_ms'])} / "
        f"{number(age['model_latency']['p50_ms'])} / "
        f"{number(age['model_latency']['p95_ms'])} ms"
    )

    print("\nGender")
    print("  accuracy: unavailable (dataset has no gender labels)")
    print(f"  prediction counts: {gender['prediction_counts']}")
    print(
        f"  model latency mean/p50/p95: "
        f"{number(gender['model_latency']['mean_ms'])} / "
        f"{number(gender['model_latency']['p50_ms'])} / "
        f"{number(gender['model_latency']['p95_ms'])} ms"
    )

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
            "data_dir": str(Path(config.data_dir).expanduser().resolve()),
            "output_path": str(Path(config.output_path).expanduser().resolve()),
            "face_model_path": str(
                Path(config.face_model_path).expanduser().resolve()
            ),
            "age_model_path": str(
                Path(config.age_model_path).expanduser().resolve()
            ),
            "gender_model_path": str(
                Path(config.gender_model_path).expanduser().resolve()
            ),
            "providers": list(config.providers),
            "limit": config.limit,
            "warmup_runs": config.warmup_runs,
            "expected_count": config.expected_count,
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

    artifact = wandb.Artifact(
        name=f"benchmark-runtime-{run.id}",
        type="benchmark-result",
        description="Runtime benchmark metrics, per-image results, and failures",
        metadata=metrics,
    )
    artifact.add_file(
        local_path=str(output_path.expanduser().resolve()),
        name="benchmark_runtime.json",
    )
    run.log_artifact(artifact)
    print(f"[benchmark] logged {len(metrics)} metrics to W&B run {run.name}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data",
        help="Flat directory containing NNNNNAxx.jpg images",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "output" / "benchmark_runtime.json",
        help="JSON output path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Evenly spaced sample size; use 0 for all images",
    )
    parser.add_argument(
        "--provider",
        choices=("auto", "cpu", "coreml", "cuda"),
        default="cpu",
    )
    parser.add_argument("--warmup-runs", type=int, default=10)
    parser.add_argument("--expected-count", type=int, default=DEFAULT_EXPECTED_COUNT)
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
    parser.add_argument(
        "--wandb-project",
        default=DEFAULT_WANDB_PROJECT,
        help="W&B project used with --wandb",
    )
    parser.add_argument(
        "--wandb-entity",
        default=None,
        help="Optional W&B team or username",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Optional W&B run display name",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline"),
        default="online",
        help="Use offline mode to save the W&B run locally without uploading",
    )
    parser.add_argument(
        "--wandb-tags",
        nargs="*",
        default=(),
        help="Optional space-separated W&B run tags",
    )
    return parser.parse_args(argv)


def _benchmark_config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    return BenchmarkConfig(
        data_dir=args.data_dir,
        output_path=args.output,
        face_model_path=args.face_model,
        age_model_path=args.age_model,
        gender_model_path=args.gender_model,
        providers=_providers_from_name(args.provider),
        limit=None if args.limit <= 0 else args.limit,
        warmup_runs=args.warmup_runs,
        expected_count=(
            None if args.expected_count <= 0 else args.expected_count
        ),
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
