from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


HEADLINE_COLUMNS = (
    "Run_ID",
    "Face_Coverage",
    "Face_Model_Mean_ms",
    "Face_Model_P50_ms",
    "Face_Model_P95_ms",
    "Face_Stage_Mean_ms",
    "Face_Stage_P50_ms",
    "Face_Stage_P95_ms",
    "AGE_MAE",
    "AGE_RMSE",
    "AGE_Median_AE",
    "AGE_P90_AE",
    "AGE_Within5",
    "AGE_Within10",
    "GEN_Accuracy",
    "GEN_Balanced_Accuracy",
    "GEN_Macro_F1",
    "GEN_Female_ROC_AUC",
    "GEN_Female_PR_AUC",
    "PIPE_Success_Rate",
    "PIPE_Mean_ms",
    "PIPE_P50_ms",
    "PIPE_P95_ms",
    "PIPE_Throughput_img_s",
)
SLICE_COLUMNS = ("Run_ID", "Task", "Slice_Type", "Slice_Value", "Metric", "Value", "Unit", "Sample_N")


def export_headline_tsv(result: dict[str, Any], path: Path) -> Path:
    run_id = result["run"]["run_id"]
    face = result["face"]
    age = result["age"]
    gender = result["gender"]
    pipeline = result["pipeline"]
    row = {
        "Run_ID": run_id,
        "Face_Coverage": face.get("detection_coverage"),
        "Face_Model_Mean_ms": face["model_latency"].get("mean_ms"),
        "Face_Model_P50_ms": face["model_latency"].get("p50_ms"),
        "Face_Model_P95_ms": face["model_latency"].get("p95_ms"),
        "Face_Stage_Mean_ms": face["stage_latency"].get("mean_ms"),
        "Face_Stage_P50_ms": face["stage_latency"].get("p50_ms"),
        "Face_Stage_P95_ms": face["stage_latency"].get("p95_ms"),
        "AGE_MAE": age.get("mae"),
        "AGE_RMSE": age.get("rmse"),
        "AGE_Median_AE": age.get("median_absolute_error"),
        "AGE_P90_AE": age.get("p90_absolute_error"),
        "AGE_Within5": age.get("within_5_years"),
        "AGE_Within10": age.get("within_10_years"),
        "GEN_Accuracy": gender.get("accuracy"),
        "GEN_Balanced_Accuracy": gender.get("balanced_accuracy"),
        "GEN_Macro_F1": gender.get("macro_f1"),
        "GEN_Female_ROC_AUC": gender.get("roc_auc_female"),
        "GEN_Female_PR_AUC": gender.get("pr_auc_female"),
        "PIPE_Success_Rate": pipeline.get("success_rate"),
        "PIPE_Mean_ms": pipeline["latency"].get("mean_ms"),
        "PIPE_P50_ms": pipeline["latency"].get("p50_ms"),
        "PIPE_P95_ms": pipeline["latency"].get("p95_ms"),
        "PIPE_Throughput_img_s": pipeline.get("throughput_images_per_second"),
    }
    return _write_rows(path, HEADLINE_COLUMNS, [row], delimiter="\t")


def export_slice_tsv(result: dict[str, Any], path: Path) -> Path:
    rows: list[dict[str, Any]] = []
    run_id = result["run"]["run_id"]
    for interval, metrics in result["age"].get("by_true_age_interval", {}).items():
        for metric in ("mae", "rmse", "mean_signed_error", "within_5_years", "within_10_years"):
            rows.append(
                {
                    "Run_ID": run_id,
                    "Task": "age",
                    "Slice_Type": "true_age_interval",
                    "Slice_Value": interval,
                    "Metric": metric,
                    "Value": metrics.get(metric),
                    "Unit": "years" if metric in ("mae", "rmse", "mean_signed_error") else "proportion",
                    "Sample_N": metrics.get("samples"),
                }
            )
    return _write_rows(path, SLICE_COLUMNS, rows, delimiter="\t")


def export_samples_csv(result: dict[str, Any], path: Path) -> Path:
    samples = result.get("samples", [])
    columns = sorted({key for sample in samples for key in sample})
    return _write_rows(path, columns, samples)


def export_failures_csv(result: dict[str, Any], path: Path) -> Path:
    failures = result.get("failures", [])
    columns = ("sample_id", "filename", "stage", "reason")
    return _write_rows(path, columns, failures)


def export_all(result: dict[str, Any], output_path: Path) -> dict[str, str]:
    stem = output_path.with_suffix("")
    paths = {
        "headline_tsv": export_headline_tsv(result, stem.with_name(f"{stem.name}_headline.tsv")),
        "slices_tsv": export_slice_tsv(result, stem.with_name(f"{stem.name}_slices.tsv")),
        "samples_csv": export_samples_csv(result, stem.with_name(f"{stem.name}_samples.csv")),
        "failures_csv": export_failures_csv(result, stem.with_name(f"{stem.name}_failures.csv")),
    }
    return {key: str(path) for key, path in paths.items()}


def _write_rows(
    path: Path,
    fieldnames: tuple[str, ...] | list[str],
    rows: list[dict[str, Any]],
    *,
    delimiter: str = ",",
) -> Path:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})
    return path


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return str(value)
    return value
