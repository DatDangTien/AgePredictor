#!/usr/bin/env python3
"""Create workbook-import TSV files from a benchmark runtime JSON result."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]

DATASET_COLUMNS = [
    "Dataset_ID",
    "Dataset_Name",
    "Version",
    "Image_Count",
    "Age_Label",
    "Gender_Label",
    "Face_BBox",
    "Landmarks",
    "Identity_ID",
    "Age_Range",
    "Split_Method",
    "License",
    "Source_URL",
    "Local_Path",
    "Status",
    "Notes",
]
MODEL_COLUMNS = [
    "Model_ID",
    "Task",
    "Model_Name",
    "Version",
    "Format",
    "Input_Size",
    "Precision",
    "Parameters",
    "Training_Dataset",
    "File_Path",
    "File_Hash",
    "Status",
    "Baseline?",
    "Notes",
]
RUN_COLUMNS = [
    "Run_ID",
    "Run_Date",
    "Dataset_ID",
    "Face_Model_ID",
    "Age_Model_ID",
    "Gender_Model_ID",
    "Sample_N",
    "Random_Seed",
    "Selection_Method",
    "Execution_Provider",
    "Warmup_Runs",
    "Batch_Size",
    "Device",
    "OS",
    "ORT_Version",
    "Git_Commit",
    "Config_ID",
    "W&B_Run",
    "Result_JSON",
    "Run_Status",
    "Notes",
]


def create_report(
    *,
    result_path: Path,
    output_dir: Path,
    dataset_name: str,
    dataset_license: str,
    dataset_source_url: str,
    dataset_local_path: str,
    face_model_id: str,
    age_model_id: str,
    gender_model_id: str,
    wandb_run: str,
) -> dict[str, Path]:
    result_file = result_path.expanduser().resolve()
    result = json.loads(result_file.read_text(encoding="utf-8"))

    report_dir = output_dir.expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    validation = _load_validation_report(result)
    paths = {
        "datasets": report_dir / "01_Datasets.tsv",
        "models": report_dir / "02_Models.tsv",
        "runs": report_dir / "03_Experiment_Runs.tsv",
        "headline": report_dir / "04_Headline_Metrics.tsv",
        "slices": report_dir / "05_Slice_Metrics.tsv",
        "summary": report_dir / "README.md",
    }

    _write_tsv(paths["datasets"], DATASET_COLUMNS, [_dataset_row(
        result=result,
        validation=validation,
        dataset_name=dataset_name,
        dataset_license=dataset_license,
        dataset_source_url=dataset_source_url,
        dataset_local_path=dataset_local_path,
    )])
    _write_tsv(paths["models"], MODEL_COLUMNS, _model_rows(
        result=result,
        face_model_id=face_model_id,
        age_model_id=age_model_id,
        gender_model_id=gender_model_id,
    ))
    _write_tsv(paths["runs"], RUN_COLUMNS, [_run_row(
        result=result,
        result_file=result_file,
        face_model_id=face_model_id,
        age_model_id=age_model_id,
        gender_model_id=gender_model_id,
        wandb_run=wandb_run,
    )])

    exports = result.get("exports", {})
    _copy_export(exports.get("headline_tsv"), paths["headline"])
    _copy_export(exports.get("slices_tsv"), paths["slices"])
    _copy_optional_export(exports.get("samples_csv"), report_dir)
    _copy_optional_export(exports.get("failures_csv"), report_dir)

    paths["summary"].write_text(_summary_markdown(result, validation, paths), encoding="utf-8")
    return paths


def _dataset_row(
    *,
    result: dict[str, Any],
    validation: dict[str, Any],
    dataset_name: str,
    dataset_license: str,
    dataset_source_url: str,
    dataset_local_path: str,
) -> dict[str, Any]:
    dataset = result["dataset"]
    dataset_id = result["run"]["dataset_id"]
    version = _manifest_version(result)
    invalid_count = len(validation.get("invalid_filenames", []))
    decode_failures = len(validation.get("decode_failures", []))
    notes = [
        "UTKFace aligned chip filenames are supported.",
        f"Malformed filenames: {invalid_count}.",
        f"Decode failures: {decode_failures}.",
        "No verified identity IDs or bounding-box ground truth in manifest.",
    ]
    return {
        "Dataset_ID": dataset_id,
        "Dataset_Name": dataset_name,
        "Version": version,
        "Image_Count": dataset.get("record_count", validation.get("valid_records", "")),
        "Age_Label": "Yes",
        "Gender_Label": "Yes",
        "Face_BBox": "No",
        "Landmarks": "No",
        "Identity_ID": "No",
        "Age_Range": f"{dataset.get('age_min', '')}-{dataset.get('age_max', '')}",
        "Split_Method": "all; benchmark sample selected from saved sample list",
        "License": dataset_license,
        "Source_URL": dataset_source_url,
        "Local_Path": dataset_local_path,
        "Status": "Completed",
        "Notes": " ".join(notes),
    }


def _model_rows(
    *,
    result: dict[str, Any],
    face_model_id: str,
    age_model_id: str,
    gender_model_id: str,
) -> list[dict[str, Any]]:
    models = result["models"]
    return [
        _model_row(face_model_id, "Face Detection", "face_det_lite", models["face"], "Input size handled by runtime preprocessing."),
        _model_row(age_model_id, "Age Prediction", "age.onnx", models["age"], "Age output interpreted by current runtime pipeline."),
        _model_row(gender_model_id, "Gender Prediction", "AdaFace IR50 MS1MV2 Gender", models["gender"], "Female probability threshold is 0.5."),
    ]


def _model_row(
    model_id: str,
    task: str,
    model_name: str,
    model: dict[str, Any],
    notes: str,
) -> dict[str, Any]:
    return {
        "Model_ID": model_id,
        "Task": task,
        "Model_Name": model_name,
        "Version": "current local ONNX export",
        "Format": "ONNX",
        "Input_Size": "",
        "Precision": "",
        "Parameters": "",
        "Training_Dataset": "",
        "File_Path": model.get("path", ""),
        "File_Hash": model.get("sha256", ""),
        "Status": "Active",
        "Baseline?": "Yes",
        "Notes": notes,
    }


def _run_row(
    *,
    result: dict[str, Any],
    result_file: Path,
    face_model_id: str,
    age_model_id: str,
    gender_model_id: str,
    wandb_run: str,
) -> dict[str, Any]:
    run = result["run"]
    config = result["config"]
    providers = result.get("providers", {})
    environment = result.get("environment", {})
    pipeline = result["pipeline"]
    failed = pipeline.get("failed_images", 0)
    status = "Completed" if failed == 0 else "Completed with runtime failures"
    return {
        "Run_ID": run["run_id"],
        "Run_Date": _date_only(result.get("created_at")),
        "Dataset_ID": run["dataset_id"],
        "Face_Model_ID": face_model_id,
        "Age_Model_ID": age_model_id,
        "Gender_Model_ID": gender_model_id,
        "Sample_N": run.get("sample_count", ""),
        "Random_Seed": run.get("random_seed", ""),
        "Selection_Method": _selection_method(run),
        "Execution_Provider": " -> ".join(providers.get("age") or config.get("providers_requested", [])),
        "Warmup_Runs": config.get("warmup_runs", ""),
        "Batch_Size": 1,
        "Device": platform.machine(),
        "OS": environment.get("platform", platform.platform()),
        "ORT_Version": environment.get("onnxruntime", ""),
        "Git_Commit": run.get("git_commit", ""),
        "Config_ID": run.get("config_id", ""),
        "W&B_Run": wandb_run,
        "Result_JSON": str(result_file),
        "Run_Status": status,
        "Notes": (
            f"{pipeline.get('successful_images', '')}/{pipeline.get('processed_images', '')} "
            "images completed. Manifest SHA-256: "
            f"{run.get('manifest_sha256', '')}."
        ),
    }


def _load_validation_report(result: dict[str, Any]) -> dict[str, Any]:
    dataset_id = result["run"]["dataset_id"]
    report = REPO_ROOT / "output" / "validation" / f"{dataset_id}_validation.json"
    if not report.exists():
        return {}
    return json.loads(report.read_text(encoding="utf-8"))


def _manifest_version(result: dict[str, Any]) -> str:
    manifest_path = Path(result["run"]["manifest_path"])
    if not manifest_path.exists():
        return ""
    with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
        reader = csv.DictReader(manifest_file)
        first = next(reader, None)
    return first.get("dataset_version", "") if first else ""


def _selection_method(run: dict[str, Any]) -> str:
    strategy = run.get("sampling_strategy", "")
    sample_list = run.get("sample_list_path")
    if sample_list:
        return f"{strategy}; saved sample list"
    return strategy


def _date_only(created_at: str | None) -> str:
    if not created_at:
        return ""
    return datetime.fromisoformat(created_at).date().isoformat()


def _write_tsv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _copy_export(source: str | None, destination: Path) -> None:
    if not source:
        raise ValueError(f"Missing export path for {destination.name}")
    shutil.copyfile(Path(source).expanduser().resolve(), destination)


def _copy_optional_export(source: str | None, output_dir: Path) -> None:
    if source:
        source_path = Path(source).expanduser().resolve()
        shutil.copyfile(source_path, output_dir / source_path.name)


def _summary_markdown(
    result: dict[str, Any],
    validation: dict[str, Any],
    paths: dict[str, Path],
) -> str:
    run = result["run"]
    face = result["face"]
    age = result["age"]
    gender = result["gender"]
    pipeline = result["pipeline"]
    invalid_count = len(validation.get("invalid_filenames", []))
    failures = result.get("failures", [])
    return "\n".join(
        [
            f"# Workbook Report: {run['run_id']}",
            "",
            "## Sheet Files",
            "",
            f"- `01_Datasets`: `{paths['datasets']}`",
            f"- `02_Models`: `{paths['models']}`",
            f"- `03_Experiment_Runs`: `{paths['runs']}`",
            f"- `04_Headline_Metrics`: `{paths['headline']}`",
            f"- `05_Slice_Metrics`: `{paths['slices']}`",
            "",
            "## Headline",
            "",
            f"- Dataset: `{run['dataset_id']}` with `{result['dataset']['record_count']}` validated records",
            f"- Sample: `{run['sample_count']}` images via `{run['sampling_strategy']}`",
            f"- Face coverage: `{face['detection_coverage']}`",
            f"- Age MAE/RMSE: `{age['mae']}` / `{age['rmse']}` years",
            f"- Gender accuracy / macro F1: `{gender['accuracy']}` / `{gender['macro_f1']}`",
            f"- Pipeline success: `{pipeline['successful_images']}/{pipeline['processed_images']}`",
            f"- Throughput: `{pipeline['throughput_images_per_second']}` images/s",
            "",
            "## Checks And Notes",
            "",
            f"- Dataset malformed filenames: `{invalid_count}`",
            f"- Runtime failures in selected sample: `{len(failures)}`",
            "- Percent/rate metrics are decimal proportions for workbook import.",
            "- Latency metrics are milliseconds.",
            "- UTKFace race labels are preserved in manifest metadata but are not primary headline metrics.",
            "",
        ]
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default="UTKFace")
    parser.add_argument("--dataset-license", default="")
    parser.add_argument("--dataset-source-url", default="https://susanqq.github.io/UTKFace/")
    parser.add_argument("--dataset-local-path", default="~/Downloads/archive/UTKFace")
    parser.add_argument("--face-model-id", default="FD001")
    parser.add_argument("--age-model-id", default="AGE001")
    parser.add_argument("--gender-model-id", default="GEN001")
    parser.add_argument("--wandb-run", default="")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    paths = create_report(
        result_path=args.result,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        dataset_license=args.dataset_license,
        dataset_source_url=args.dataset_source_url,
        dataset_local_path=args.dataset_local_path,
        face_model_id=args.face_model_id,
        age_model_id=args.age_model_id,
        gender_model_id=args.gender_model_id,
        wandb_run=args.wandb_run,
    )
    print(f"[workbook-report] wrote {paths['summary'].parent}")


if __name__ == "__main__":
    main()
