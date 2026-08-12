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
    "AGE_Interval_Evaluated",
    "AGE_Interval_MAE",
    "AGE_Within_Label_Interval",
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
    interval_aware = age.get("interval_aware", {})
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
        "AGE_Interval_Evaluated": interval_aware.get("evaluated_images"),
        "AGE_Interval_MAE": interval_aware.get("mae"),
        "AGE_Within_Label_Interval": interval_aware.get("within_interval"),
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
        sample_n = metrics.get("samples")
        age_metrics = (
            ("Samples", "samples", "count"),
            ("MAE", "mae", "years"),
            ("RMSE", "rmse", "years"),
            ("Within 5 Years", "within_5_years", "proportion"),
            ("Within 10 Years", "within_10_years", "proportion"),
            ("Mean Signed Error", "mean_signed_error", "years"),
            ("Median Absolute Error", "median_absolute_error", "years"),
        )
        for label, metric, unit in age_metrics:
            rows.append(
                {
                    "Run_ID": run_id,
                    "Task": "Age Prediction",
                    "Slice_Type": "True Age Interval",
                    "Slice_Value": interval,
                    "Metric": label,
                    "Value": metrics.get(metric),
                    "Unit": unit,
                    "Sample_N": sample_n,
                }
            )
    rows.extend(_gender_slice_rows(result))
    return _write_rows(path, SLICE_COLUMNS, rows, delimiter="\t")


def _gender_slice_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    gender = result.get("gender", {})
    confusion = gender.get("confusion_matrix", {})
    ff = confusion.get("true_female_pred_female", 0)
    fm = confusion.get("true_female_pred_male", 0)
    mf = confusion.get("true_male_pred_female", 0)
    mm = confusion.get("true_male_pred_male", 0)
    run_id = result["run"]["run_id"]
    class_rows = [
        ("Female", ff + fm, (
            ("Precision", "female_precision"),
            ("Recall", "female_recall"),
            ("F1", "female_f1"),
            ("Accuracy", "female_accuracy"),
        )),
        ("Male", mf + mm, (
            ("Precision", "male_precision"),
            ("Recall", "male_recall"),
            ("F1", "male_f1"),
            ("Accuracy", "male_accuracy"),
        )),
    ]

    rows: list[dict[str, Any]] = []
    for label, sample_n, metrics in class_rows:
        rows.append(
            {
                "Run_ID": run_id,
                "Task": "Gender Prediction",
                "Slice_Type": "True Class",
                "Slice_Value": label,
                "Metric": "Samples",
                "Value": sample_n,
                "Unit": "count",
                "Sample_N": sample_n,
            }
        )
        for metric_label, metric_key in metrics:
            rows.append(
                {
                    "Run_ID": run_id,
                    "Task": "Gender Prediction",
                    "Slice_Type": "True Class",
                    "Slice_Value": label,
                    "Metric": metric_label,
                    "Value": gender.get(metric_key),
                    "Unit": "proportion",
                    "Sample_N": sample_n,
                }
            )

    confusion_rows = (
        ("True Female Pred Female", "true_female_pred_female"),
        ("True Female Pred Male", "true_female_pred_male"),
        ("True Male Pred Female", "true_male_pred_female"),
        ("True Male Pred Male", "true_male_pred_male"),
    )
    evaluated = gender.get("evaluated_images", ff + fm + mf + mm)
    for label, key in confusion_rows:
        rows.append(
            {
                "Run_ID": run_id,
                "Task": "Gender Prediction",
                "Slice_Type": "Confusion Matrix",
                "Slice_Value": label,
                "Metric": "Count",
                "Value": confusion.get(key),
                "Unit": "count",
                "Sample_N": evaluated,
            }
        )
    return rows


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
