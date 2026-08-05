# Runtime Benchmark Metrics

This benchmark evaluates the production ONNX pipeline used by the app:

1. `face_det_lite-onnx-w8a8/face_det_lite.onnx`
2. `age.onnx`
3. `adaface_ir50_ms1mv2_gender.onnx`

The current dataset is All-Age-Faces in a flat `data/` directory. Filenames
encode age as `NNNNNAxx.jpg`. Gender labels are inferred from the official
All-Age-Faces id split: `00000`-`07380` is Female and `07381`-`13321` is Male.
The flat dataset does not include bounding boxes, so face detection is measured
by coverage rather than AP/IoU.

## Current Component Structure

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
│   └── exporters.py
│
├── scripts/
│   ├── prepare_all_age_faces.py
│   ├── prepare_utkface.py
│   ├── prepare_imdb_wiki.py
│   └── prepare_megaface.py
│
├── manifests/
│   ├── DS001_all_age_faces.csv
│   ├── DS002_utkface.csv
│   ├── DS003_megaface.csv
│   └── DS004_imdb_wiki.csv
│
├── output/
│   ├── benchmark_<dataset>_<run_id>.json
│   ├── benchmark_<dataset>_<run_id>_headline.tsv
│   ├── benchmark_<dataset>_<run_id>_slices.tsv
│   ├── benchmark_<dataset>_<run_id>_samples.csv
│   ├── benchmark_<dataset>_<run_id>_failures.csv
│   └── validation/
│       └── <dataset>_validation.json
│
└── tests/
    ├── test_manifest.py
    ├── test_prepare_all_age_faces.py
    ├── test_prepare_utkface.py
    └── test_benchmark_runtime.py
```

The benchmark now uses a manifest-first structure. Dataset-specific parsing
happens once in a preparation script, then the runtime benchmark consumes a
canonical CSV manifest instead of re-parsing dataset filenames.

Flow:

```text
raw dataset -> prepare script -> canonical manifest -> optional sample list -> runtime benchmark -> JSON/TSV/CSV outputs
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
| `scripts/prepare_all_age_faces.py` | Builds the DS001 All-Age-Faces manifest from local images. |
| `scripts/prepare_utkface.py` | Builds a UTKFace-style manifest for future DS002 runs. |
| `manifests/` | Stores canonical dataset manifests and reusable sample lists. |
| `output/` | Stores benchmark JSON results, validation reports, TSV summaries, CSV sample rows, and failure reports. |
| `data/` | Recommended in-repo location for local datasets; external dataset paths also work with `--data-dir`. |

The runtime is intentionally dataset-agnostic. If a new dataset has different
filename conventions or labels, add or adjust a `scripts/prepare_*.py` file and
keep `benchmark/benchmark_runtime.py` focused on inference and metrics.

## Command Flow

Run commands from the repo root:

```bash
cd /Users/trananhchuong/Documents/workspace/AgePredictor
```

Use the project virtual environment if it exists:

```bash
source .venv/bin/activate
```

If you prefer not to activate the environment, replace `python` below with
`.venv/bin/python`.

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

For All-Age-Faces in the repo's `data/` directory:

```bash
python scripts/prepare_all_age_faces.py \
  --data-dir data \
  --dataset-id DS001 \
  --dataset-version official \
  --output manifests/DS001_all_age_faces.csv \
  --validation-report output/validation/DS001_validation.json
```

If the dataset lives elsewhere, keep the same command and point `--data-dir` to
that folder:

```bash
python scripts/prepare_all_age_faces.py \
  --data-dir /path/to/All-Age-Faces \
  --dataset-id DS001 \
  --dataset-version official \
  --output manifests/DS001_all_age_faces.csv \
  --validation-report output/validation/DS001_validation.json
```

For a future UTKFace-style dataset:

```bash
python scripts/prepare_utkface.py \
  --data-dir /path/to/UTKFace \
  --dataset-id DS002 \
  --dataset-version local \
  --output manifests/DS002_utkface.csv \
  --validation-report output/validation/DS002_validation.json
```

### 3. Create A Reusable Sample List

This creates a deterministic 300-image sample for repeated benchmark runs:

```bash
python benchmark/create_sample_list.py \
  --manifest manifests/DS001_all_age_faces.csv \
  --strategy evenly_spaced \
  --limit 300 \
  --seed 42 \
  --output manifests/samples/DS001_sample_300_seed42.txt
```

For label-balanced sampling, use:

```bash
python benchmark/create_sample_list.py \
  --manifest manifests/DS001_all_age_faces.csv \
  --strategy stratified_age_gender \
  --limit 300 \
  --seed 42 \
  --output manifests/samples/DS001_sample_300_seed42.txt
```

### 4. Run The Benchmark

Recommended repeatable All-Age-Faces run:

```bash
python benchmark/benchmark_runtime.py \
  --manifest manifests/DS001_all_age_faces.csv \
  --sample-list manifests/samples/DS001_sample_300_seed42.txt \
  --dataset-id DS001 \
  --run-id RUN001 \
  --config-id CFG001 \
  --provider coreml \
  --warmup-runs 10 \
  --output output/benchmark_aaf_RUN001.json
```

Quick smoke test:

```bash
python benchmark/benchmark_runtime.py \
  --manifest manifests/DS001_all_age_faces.csv \
  --dataset-id DS001 \
  --run-id SMOKE001 \
  --config-id CFG001 \
  --limit 5 \
  --provider cpu \
  --warmup-runs 0 \
  --output output/benchmark_smoke.json
```

CUDA machines can use `--provider cuda`; macOS should usually use
`--provider coreml` or `--provider cpu`.

### 5. Optional W&B Logging

Add `--wandb` to log scalar metrics and save the JSON result as a compact W&B
artifact:

```bash
python benchmark/benchmark_runtime.py \
  --manifest manifests/DS001_all_age_faces.csv \
  --sample-list manifests/samples/DS001_sample_300_seed42.txt \
  --dataset-id DS001 \
  --run-id RUN001_WANDB \
  --config-id CFG001 \
  --provider coreml \
  --warmup-runs 10 \
  --wandb \
  --output output/benchmark_aaf_RUN001_WANDB.json
```

### 6. Review Outputs

Each benchmark run writes the main JSON output plus companion files next to it:

```text
output/benchmark_aaf_RUN001.json
output/benchmark_aaf_RUN001_headline.tsv
output/benchmark_aaf_RUN001_slices.tsv
output/benchmark_aaf_RUN001_samples.csv
output/benchmark_aaf_RUN001_failures.csv
```

Use the JSON for complete metrics and per-sample details. Use the TSV/CSV files
for quick spreadsheet review and reporting.

### 7. Run Tests

Run the focused benchmark tests after changing benchmark logic:

```bash
python -m unittest \
  tests/test_benchmark_runtime.py \
  tests/test_manifest.py \
  tests/test_prepare_all_age_faces.py \
  tests/test_prepare_utkface.py
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
  scripts/prepare_all_age_faces.py \
  scripts/prepare_utkface.py
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

`wandb_metrics()` recursively flattens numeric values under `dataset`, `face`,
`age`, `gender`, and `pipeline`. For example:

- `age/mae`
- `age/by_true_age_interval/0-12/mae`
- `gender/accuracy`
- `gender/confusion_matrix/true_female_pred_female`
- `pipeline/throughput_images_per_second`

All scalar metrics are logged with `run.log(metrics)`. The JSON result is saved
as a W&B artifact, while artifact metadata is intentionally compact to stay
below W&B's 100-key metadata limit.
