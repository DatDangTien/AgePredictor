# Runtime Benchmark Metrics

This benchmark evaluates the production ONNX pipeline used by the app:

1. `face_det_lite-onnx-w8a8/face_det_lite.onnx`
2. `age.onnx`
3. `adaface_ir50_ms1mv2_gender.onnx`

## Current Component Structure

Core folder meanings:

| Folder | Meaning |
|---|---|
| `app/` | Production-facing inference code that the benchmark should stay aligned with. |
| `benchmark/` | Dataset-agnostic benchmark runtime, manifest loading, sampling, metric calculation, exports, and workbook-report tooling. |
| `scripts/` | Tailored dataset/model utility scripts. Each dataset gets its own `prepare_<dataset>.py` parser because filename conventions and labels differ by dataset. |
| `data/` | Optional in-repo local dataset location. Large datasets may also live outside the repo and be passed with `--data-dir`. |
| `models/` | Exported ONNX runtime artifacts used by the face, age, and gender pipeline. |
| `manifests/` | Canonical dataset manifests and reusable sample lists generated before benchmarking. |
| `output/` | Validation reports, benchmark JSON files, TSV/CSV exports, W&B artifacts, and workbook-report packages. |
| `tests/` | Focused unit tests for parsers, manifests, sampling, metrics, W&B flattening, and report behavior. |

```text
AgePredictor/
├── app/
│   └── inference.py
│
├── benchmark/
│   ├── benchmark_runtime.py
│   ├── records.py
│   ├── manifest.py
│   ├── metrics.py
│   ├── sampling.py
│   ├── exporters.py
│   ├── create_sample_list.py
│   └── create_workbook_report.py
│
├── scripts/
│   ├── prepare_all_age_faces.py
│   ├── prepare_utkface.py
│   ├── prepare_fairface.py
│   └── prepare_<future_dataset>.py
│
├── data/
│   └── <optional_local_dataset_files>/
│
├── models/
│   ├── face_det_lite-onnx-w8a8/
│   ├── age.onnx
│   └── adaface_ir50_ms1mv2_gender.onnx
│
├── manifests/
│   ├── DS001_all_age_faces.csv
│   ├── DS002_utkface.csv
│   ├── DS003_fairface.csv
│   ├── DSXXX_<future_dataset>.csv
│   └── samples/
│       └── DSXXX_sample_300_seed42.txt
│
├── output/
│   ├── benchmark_<dataset>_<run_id>.json
│   ├── benchmark_<dataset>_<run_id>_headline.tsv
│   ├── benchmark_<dataset>_<run_id>_slices.tsv
│   ├── benchmark_<dataset>_<run_id>_samples.csv
│   ├── benchmark_<dataset>_<run_id>_failures.csv
│   ├── validation/
│   │   └── <dataset>_validation.json
│   └── workbook_reports/
│       └── <run_id>/
│           ├── 01_Datasets.tsv
│           ├── 02_Models.tsv
│           ├── 03_Experiment_Runs.tsv
│           ├── 04_Headline_Metrics.tsv
│           └── 05_Slice_Metrics.tsv
│
└── tests/
    ├── test_manifest.py
    ├── test_prepare_all_age_faces.py
    ├── test_prepare_utkface.py
    ├── test_prepare_fairface.py
    └── test_benchmark_runtime.py
```

The benchmark now uses a manifest-first structure. Dataset-specific parsing
happens once in a preparation script, then the runtime benchmark consumes a
canonical CSV manifest instead of re-parsing dataset filenames.

Flow:

```text
raw dataset
  -> dataset-specific prepare script
  -> canonical manifest
  -> reusable sample list
  -> runtime benchmark
  -> JSON/TSV/CSV outputs
  -> workbook report TSV package
```

Main benchmark components:

| Path | Role |
|---|---|
| `benchmark/benchmark_runtime.py` | Runs the ONNX face, age, and gender pipeline from a canonical manifest. |
| `benchmark/records.py` | Defines the canonical benchmark record shape shared across loaders, samplers, metrics, and exports. |
| `benchmark/manifest.py` | Loads and validates manifest CSV files. |
| `benchmark/metrics.py` | Computes latency, age, gender, confusion matrix, AUC, and age-interval metrics. |
| `benchmark/sampling.py` | Selects records using `all`, `evenly_spaced`, `random`, `stratified_age_gender`, or a sample list. |
| `benchmark/exporters.py` | Writes companion result exports such as headline TSV, slice TSV, sample CSV, and failure CSV. |
| `benchmark/create_sample_list.py` | Creates reusable sample-id files for deterministic benchmark subsets. |
| `benchmark/create_workbook_report.py` | Packages completed benchmark results into workbook-import TSV files. |
| `scripts/prepare_all_age_faces.py` | Builds the DS001 All-Age-Faces manifest from local images. |
| `scripts/prepare_utkface.py` | Builds the DS002 UTKFace manifest, including `.jpg.chip.jpg` images. |
| `scripts/prepare_fairface.py` | Builds one DS003 FairFace manifest from the official train and validation label CSVs and image folders. |
| `scripts/run_fairface_margin025_benchmark.sh` | Prepares and benchmarks the complete FairFace margin 0.25 dataset on CUDA with W&B logging. |
| `scripts/run_fairface_margin125_benchmark.sh` | Prepares and benchmarks the complete FairFace margin 1.25 dataset on CUDA with W&B logging. |
| `scripts/prepare_adience.py` | Builds separate DS004 manifests for the Adience aligned and cropped-face variants. |
| `scripts/run_adience_benchmarks.sh` | Prepares and benchmarks both Adience variants with progress logs and separate W&B runs. |
| `scripts/prepare_<future_dataset>.py` | Placeholder pattern for future dataset-specific manifest builders. |
| `manifests/` | Stores canonical dataset manifests and reusable sample lists. |
| `output/` | Stores benchmark JSON results, validation reports, TSV summaries, CSV sample rows, and failure reports. |
| `output/workbook_reports/` | Stores sheet-ready TSV packages copied into the benchmark workbook. |
| `data/` | Recommended in-repo location for local datasets; external dataset paths also work with `--data-dir`. |

The runtime is intentionally dataset-agnostic. If a new dataset has different
filename conventions or labels, add or adjust a `scripts/prepare_*.py` file and
keep `benchmark/benchmark_runtime.py` focused on inference and metrics.

## Command Flow

Run commands from the repo root. Use the project virtual environment if it
exists:

```bash
cd /Users/trananhchuong/Documents/workspace/AgePredictor
source .venv/bin/activate
```

If you prefer not to activate the environment, replace `python` below with
`.venv/bin/python`.

Set these variables for the dataset/run you want to benchmark:

```bash
DATASET_ID=DSXXX
DATASET_NAME=your_dataset
DATASET_VERSION=local
DATA_DIR=/path/to/your_dataset
PREPARE_SCRIPT=scripts/prepare_your_dataset.py
MANIFEST=manifests/${DATASET_ID}_${DATASET_NAME}.csv
VALIDATION_REPORT=output/validation/${DATASET_ID}_validation.json
SAMPLE_N=300
SEED=42
SAMPLING_STRATEGY=stratified_age_gender
SAMPLE_LIST=manifests/samples/${DATASET_ID}_sample_${SAMPLE_N}_seed${SEED}.txt
RUN_ID=RUNXXX_${DATASET_NAME}_WANDB
CONFIG_ID=CFG001
PROVIDER=coreml
RESULT_JSON=output/benchmark_${DATASET_NAME}_${RUN_ID}.json
REPORT_DIR=output/workbook_reports/${RUN_ID}
WANDB_RUN_URL=https://wandb.ai/<entity>/<project>/runs/<run-id>
```

Current examples:

| Dataset | Suggested values |
|---|---|
| All-Age-Faces | `DATASET_ID=DS001`, `DATASET_NAME=all_age_faces`, `PREPARE_SCRIPT=scripts/prepare_all_age_faces.py`, `SAMPLING_STRATEGY=evenly_spaced` |
| UTKFace | `DATASET_ID=DS002`, `DATASET_NAME=utkface`, `PREPARE_SCRIPT=scripts/prepare_utkface.py`, `DATA_DIR=~/Downloads/archive/UTKFace`, `SAMPLING_STRATEGY=stratified_age_gender` |
| FairFace | `DATASET_ID=DS003`, `DATASET_NAME=fairface`, `PREPARE_SCRIPT=scripts/prepare_fairface.py`, `DATA_DIR=data/FairFace/fairface-img-margin025-trainval`, `SAMPLING_STRATEGY=stratified_age_gender` |
| Future dataset | Add `scripts/prepare_your_dataset.py`, then set `DATASET_ID`, `DATASET_NAME`, `DATA_DIR`, and `PREPARE_SCRIPT` to match it. |

### 1. Confirm Runtime Models

The benchmark expects exported ONNX runtime models under `models/`:

```text
models/face_det_lite-onnx-w8a8/face_det_lite.onnx
models/age.onnx
models/adaface_ir50_ms1mv2_gender.onnx
```

If the source model artifacts are present but ONNX files are missing, export
them first:

```bash
python scripts/onnx_export.py
```

### 2. Prepare The Dataset Manifest

Each dataset needs a tailored prepare script because filename conventions,
labels, metadata, and validation rules differ across datasets. The prepare
script writes one canonical manifest for the benchmark runtime:

```bash
python "$PREPARE_SCRIPT" \
  --data-dir "$DATA_DIR" \
  --dataset-id "$DATASET_ID" \
  --dataset-version "$DATASET_VERSION" \
  --output "$MANIFEST" \
  --validation-report "$VALIDATION_REPORT"
```

FairFace requires the official train and validation label CSVs in addition to
the `train/` and `val/` image folders. `prepare_fairface.py` looks for
`fairface_label_train.csv` and `fairface_label_val.csv` inside `--data-dir` or
its parent. Use `--train-labels` and `--val-labels` when they live elsewhere:

```bash
python scripts/prepare_fairface.py \
  --data-dir data/FairFace/fairface-img-margin025-trainval \
  --train-labels data/FairFace/fairface_label_train.csv \
  --val-labels data/FairFace/fairface_label_val.csv
```

FairFace provides age intervals rather than exact ages. The source interval is
stored in `metadata_json`, and the runtime reports interval-aware error as the
distance to the nearest valid interval boundary. The default also stores a
representative proxy age for compatibility with the exact-age metrics. Use
`--age-label-policy blank` to disable those proxy-age metrics; interval-aware
accuracy and age inference latency still run.

The complete margin-specific pipelines can be run directly. Both use DS003 and
distinguish the image variants through `dataset_version`, manifest names, run
names, and W&B tags:

```bash
./scripts/run_fairface_margin025_benchmark.sh
./scripts/run_fairface_margin125_benchmark.sh
```

Manifest preparation prints a start message, periodic progress every 1,000
label rows, a summary for each split, and a final validation scan message.

Run both complete Adience variants sequentially with:

```bash
./scripts/run_adience_benchmarks.sh
```

The runner defaults to CUDA, the full five-fold label set, blank proxy ages,
and W&B online mode. Override settings with environment variables, for example:

```bash
DATA_DIR=/path/to/adience \
WANDB_ENTITY=my-team \
PREPARE_PROGRESS_EVERY=500 \
./scripts/run_adience_benchmarks.sh
```

Console output is also saved under `output/logs/`. Each variant receives its
own manifest, validation report, benchmark JSON, and W&B run.

### 3. Create A Reusable Sample List

This creates a deterministic 300-image sample for repeated benchmark runs:

```bash
python benchmark/create_sample_list.py \
  --manifest "$MANIFEST" \
  --strategy "$SAMPLING_STRATEGY" \
  --limit "$SAMPLE_N" \
  --seed "$SEED" \
  --output "$SAMPLE_LIST"
```

Useful sampling strategies:

- `evenly_spaced`: deterministic coverage over manifest order.
- `random`: deterministic random sample using `--seed`.
- `stratified_age`: balanced by true-age interval when age labels exist.
- `stratified_age_gender`: balanced by true-age interval and gender when both labels exist.

### 4. Run The Benchmark

Quick smoke test:

```bash
python benchmark/benchmark_runtime.py \
  --manifest "$MANIFEST" \
  --dataset-id "$DATASET_ID" \
  --run-id "${RUN_ID}_SMOKE" \
  --config-id "$CONFIG_ID" \
  --limit 5 \
  --provider cpu \
  --warmup-runs 0 \
  --output "output/benchmark_${DATASET_NAME}_${RUN_ID}_smoke.json"
```

Full repeatable run:

```bash
python benchmark/benchmark_runtime.py \
  --manifest "$MANIFEST" \
  --sample-list "$SAMPLE_LIST" \
  --dataset-id "$DATASET_ID" \
  --run-id "$RUN_ID" \
  --config-id "$CONFIG_ID" \
  --provider "$PROVIDER" \
  --warmup-runs 10 \
  --output "$RESULT_JSON"
```

CUDA machines can use `--provider cuda`; macOS should usually use
`--provider coreml` or `--provider cpu`.

### 5. Optional W&B Logging

Add `--wandb` to log scalar metrics and save the JSON result as a compact W&B
artifact:

```bash
python benchmark/benchmark_runtime.py \
  --manifest "$MANIFEST" \
  --sample-list "$SAMPLE_LIST" \
  --dataset-id "$DATASET_ID" \
  --run-id "$RUN_ID" \
  --config-id "$CONFIG_ID" \
  --provider "$PROVIDER" \
  --warmup-runs 10 \
  --wandb \
  --output "$RESULT_JSON"
```

### 6. Review Outputs

Each benchmark run writes the main JSON output plus companion files next to it:

```text
output/benchmark_<dataset>_<run_id>.json
output/benchmark_<dataset>_<run_id>_headline.tsv
output/benchmark_<dataset>_<run_id>_slices.tsv
output/benchmark_<dataset>_<run_id>_samples.csv
output/benchmark_<dataset>_<run_id>_failures.csv
```

Use the JSON for complete metrics and per-sample details. Use the TSV/CSV files
for quick spreadsheet review and reporting.

### 7. Create Workbook Import Files

To package a completed run for `AgePredictor_Benchmarking_Ablation_Template.xlsx`,
create sheet-ready TSV files:

```bash
python benchmark/create_workbook_report.py \
  --result "$RESULT_JSON" \
  --output-dir "$REPORT_DIR" \
  --dataset-name "$DATASET_NAME" \
  --dataset-local-path "$DATA_DIR" \
  --wandb-run "$WANDB_RUN_URL"
```

Copy or import the sheet rows from the run-specific subfolder under
`output/workbook_reports/`. The report folder contains import-ready rows for:

```text
output/workbook_reports/<run_id>/01_Datasets.tsv
output/workbook_reports/<run_id>/02_Models.tsv
output/workbook_reports/<run_id>/03_Experiment_Runs.tsv
output/workbook_reports/<run_id>/04_Headline_Metrics.tsv
output/workbook_reports/<run_id>/05_Slice_Metrics.tsv
```

It also copies the per-sample CSV and failure CSV for verification.

### 8. Run Tests

Run the focused benchmark tests after changing benchmark logic:

```bash
python -m unittest \
  tests/test_benchmark_runtime.py \
  tests/test_manifest.py \
  tests/test_prepare_all_age_faces.py \
  tests/test_prepare_utkface.py \
  tests/test_prepare_fairface.py
```

For a syntax-only check:

```bash
python -m py_compile \
  benchmark/records.py \
  benchmark/manifest.py \
  benchmark/metrics.py \
  benchmark/sampling.py \
  benchmark/exporters.py \
  benchmark/benchmark_runtime.py \
  benchmark/create_sample_list.py \
  benchmark/create_workbook_report.py \
  scripts/prepare_all_age_faces.py \
  scripts/prepare_utkface.py \
  scripts/prepare_fairface.py
```

## Latest Reference Run

Latest local reference result: `output/benchmark_aaf_RUN001_MANIFEST.json`

Run configuration:

- Images evaluated: `300`
- Selection: deterministic, evenly spaced sample over filename order
- Runtime providers: `CoreMLExecutionProvider`, then `CPUExecutionProvider`
- Warmup runs: `10`
- Schema version: `2.0`

Headline results:

| Area | Metric | Result |
|---|---:|---:|
| Face detector | Detection coverage | `300/300 (100.00%)` |
| Face detector | Stage latency mean / p50 / p95 | `10.59 / 9.41 / 21.01 ms` |
| Age | MAE / RMSE | `10.14 / 13.41 years` |
| Age | Median / p90 absolute error | `7.73 / 21.80 years` |
| Age | Within 5 / 10 years | `37.00% / 58.33%` |
| Gender | Accuracy | `96.67%` |
| Gender | Balanced accuracy | `96.77%` |
| Gender | Macro F1 | `0.966` |
| Gender | Female ROC-AUC / PR-AUC | `0.997 / 0.997` |
| Pipeline | Success rate | `300/300 (100.00%)` |
| Pipeline | End-to-end latency mean / p50 / p95 | `488.09 / 489.99 / 575.29 ms` |
| Pipeline | Wall throughput | `2.05 images/s` |

## Dataset Metrics

`dataset.image_count` is the full validated corpus size. `selected_images` is
the number of images actually benchmarked after applying `--limit`.

`age_bins` counts all validated images by true-age interval:

| Interval | Full Dataset Count |
|---|---:|
| `0-12` | `1,373` |
| `13-19` | `911` |
| `20-29` | `3,144` |
| `30-39` | `3,397` |
| `40-49` | `2,071` |
| `50-59` | `1,160` |
| `60-69` | `750` |
| `70-79` | `464` |
| `80-89` | `52` |
| `90-99` | `0` |
| `100-109` | `0` |
| `110-119` | `0` |

`gender_counts` and `gender_rates` summarize inferred All-Age-Faces gender
labels across the full corpus:

| Gender | Count | Rate |
|---|---:|---:|
| Female | `7,381` | `55.40%` |
| Male | `5,941` | `44.60%` |

## Face Metrics

Face detection is evaluated as operational coverage and latency:

- `processed_images`: images sent into face detection.
- `images_with_detection`: images where at least one face was found.
- `detection_coverage`: `images_with_detection / processed_images`.
- `mean_faces_per_processed_image`: average number of detected faces.
- `model_latency`: ONNX Runtime session latency for the face detector.
- `stage_latency`: total face-detection stage latency, including preprocessing
  and postprocessing.

Because the flat All-Age-Faces data has no bounding boxes, the benchmark does
not compute face AP, IoU, precision, or recall.

## Age Metrics

Age predictions are compared against the true age encoded in each filename.

- `evaluated_images`: images with successful face, age, and gender inference.
- `mae`: mean absolute error in years.
- `rmse`: root mean squared error in years; penalizes large mistakes more than
  MAE.
- `median_absolute_error`: median age error in years.
- `p90_absolute_error`: 90th percentile age error in years.
- `within_5_years`: fraction of predictions with absolute error at most 5 years.
- `within_10_years`: fraction of predictions with absolute error at most 10
  years.
- `interval_aware`: metrics computed against interval-labelled ages. Error is
  zero inside the interval, distance to the lower bound below it, and distance
  to the upper bound above it. Open-ended intervals such as FairFace `70+` have
  zero error for every prediction at or above 70.
- `model_latency`: ONNX Runtime session latency for the age model.

`by_true_age_interval` repeats the main age-error metrics inside each true-age
bucket. In the latest 300-image run:

| True Age Interval | Samples | MAE | RMSE | Within 5y | Within 10y |
|---|---:|---:|---:|---:|---:|
| `0-12` | `32` | `19.27` | `20.19` | `0.00%` | `3.12%` |
| `13-19` | `20` | `13.19` | `13.69` | `0.00%` | `20.00%` |
| `20-29` | `70` | `4.73` | `6.02` | `65.71%` | `87.14%` |
| `30-39` | `76` | `4.01` | `5.15` | `68.42%` | `94.74%` |
| `40-49` | `47` | `8.92` | `10.38` | `27.66%` | `57.45%` |
| `50-59` | `26` | `15.09` | `16.24` | `0.00%` | `26.92%` |
| `60-69` | `17` | `22.66` | `24.22` | `0.00%` | `5.88%` |
| `70-79` | `11` | `27.23` | `30.01` | `0.00%` | `18.18%` |
| `80-89` | `1` | `29.07` | `29.07` | `0.00%` | `0.00%` |
| `90-99` | `0` | `n/a` | `n/a` | `n/a` | `n/a` |
| `100-109` | `0` | `n/a` | `n/a` | `n/a` | `n/a` |
| `110-119` | `0` | `n/a` | `n/a` | `n/a` | `n/a` |

This interval view shows that the current age model performs best around
`20-39`, and struggles most on children and older adults.

## Gender Metrics

Gender predictions are compared against inferred All-Age-Faces gender labels.
The positive class for ROC-AUC and PR-AUC is Female, using `female_probability`
as the score.

- `accuracy`: overall correct gender predictions.
- `balanced_accuracy`: average of Female recall and Male recall.
- `macro_f1`: average of Female F1 and Male F1, weighting both classes equally.
- `female_accuracy`: Female recall, or true Female predicted Female over all
  true Female samples.
- `male_accuracy`: Male recall, or true Male predicted Male over all true Male
  samples.
- `female_precision`: predicted Female samples that were truly Female.
- `female_recall`: same value as `female_accuracy`.
- `female_f1`: harmonic mean of Female precision and recall.
- `male_precision`: predicted Male samples that were truly Male.
- `male_recall`: same value as `male_accuracy`.
- `male_f1`: harmonic mean of Male precision and recall.
- `roc_auc_female`: threshold-independent ranking quality for Female vs Male.
- `pr_auc_female`: area under the precision-recall curve for Female.
- `confusion_matrix`: named confusion matrix counts.
- `prediction_counts`: number of Female and Male predictions.
- `prediction_rates`: prediction counts divided by evaluated images.
- `confidence`: summary of the selected-class confidence.
- `female_probability`: summary of raw Female probability.
- `model_latency`: ONNX Runtime session latency for the gender model.

Latest confusion matrix:

|  | Pred Female | Pred Male |
|---|---:|---:|
| True Female | `159` | `7` |
| True Male | `3` | `131` |

Latest gender result:

- Accuracy: `96.67%`
- Balanced accuracy: `96.77%`
- Macro F1: `0.966`
- Female precision / recall / F1: `98.15% / 95.78% / 0.970`
- Male precision / recall / F1: `94.93% / 97.76% / 0.963`
- Female ROC-AUC / PR-AUC: `0.997 / 0.997`

## Pipeline Metrics

Pipeline metrics describe end-to-end behavior across the full benchmark loop:

- `processed_images`: selected images processed by the benchmark.
- `successful_images`: images where face detection, age, and gender all
  completed.
- `failed_images`: images that failed decode, detection, crop, or inference.
- `success_rate`: `successful_images / processed_images`.
- `latency`: end-to-end per-image latency.
- `wall_seconds`: total benchmark wall-clock time.
- `throughput_images_per_second`: selected image count divided by wall time.

Failures are preserved in `failures`, and per-image timings and predictions are
preserved in `samples`.

## W&B Logging

W&B receives every populated aggregate scalar from the benchmark result. Numeric
values are logged as metrics, while numeric, boolean, and text values are also
written to the run summary so they appear as run-table columns. This includes
run identity, environment, configuration, provider, model, dataset, face, age,
gender, and pipeline fields. Empty and `null` values are omitted. For example:

- `age/mae`
- `age/interval_aware/mae`
- `age/interval_aware/within_interval`
- `age/by_true_age_interval/0-12/mae`
- `gender/accuracy`
- `gender/confusion_matrix/true_female_pred_female`
- `pipeline/throughput_images_per_second`

Large row collections (`samples` and `failures`) and repeated sample-ID lists
are not expanded into thousands of W&B summary columns. They remain available
in the complete JSON result artifact and the generated CSV exports. Artifact
metadata itself is intentionally compact to stay below W&B's 100-key metadata
limit.
