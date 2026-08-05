# AgePredictor Benchmarking Template — Agent Integration Guide

## 1. Purpose

This document explains how benchmark results must be structured so they can be inserted into the workbook:

`AgePredictor_Benchmarking_Ablation_Template.xlsx`

The benchmark scope contains three pipeline components:

1. **Face detection**
2. **Age prediction**
3. **Gender prediction**

The workbook is designed for:

- evaluating the same model pipeline on multiple datasets;
- comparing alternative face, age, and gender models;
- recording runtime and predictive performance;
- storing detailed age-group and class-level metrics;
- performing controlled ablation studies;
- tracing every reported result back to a reproducible benchmark run.

The agent should treat the workbook as a reporting schema. It does not need to edit the workbook directly. It should produce structured result files and rows that can be pasted or imported into the relevant sheets.

---

## 2. Workbook structure

The workbook contains eight sheets:

| Sheet | Purpose |
|---|---|
| `00_Dashboard` | Displays selected-run metadata, headline KPIs, and age-group MAE chart |
| `01_Datasets` | Registry containing one row per dataset |
| `02_Models` | Registry containing one row per model |
| `03_Experiment_Runs` | Central run registry; one row per benchmark execution |
| `04_Headline_Metrics` | One wide row of major metrics per run |
| `05_Slice_Metrics` | Long-format detailed metrics for age groups, classes, and other subsets |
| `06_Ablation_Studies` | Baseline-versus-variant comparisons |
| `07_Metric_Dictionary` | Canonical metric names, units, directions, and definitions |

The key relationship is:

```text
Dataset + Face Model + Age Model + Gender Model + Configuration
                              |
                              v
                         Experiment Run
                              |
               +--------------+--------------+
               |                             |
               v                             v
        Headline Metrics               Slice Metrics
               |
               v
        Ablation Comparison
```

`Run_ID` is the main key connecting benchmark configuration and results.

---

## 3. Identifier conventions

Use stable IDs instead of model or dataset names in result tables.

| Entity | Format | Example |
|---|---|---|
| Dataset | `DS###` | `DS002` |
| Face detector | `FD###` | `FD001` |
| Age model | `AGE###` | `AGE001` |
| Gender model | `GEN###` | `GEN001` |
| Experiment run | `RUN###` | `RUN002` |
| Configuration | `CFG###` | `CFG002` |
| Ablation study | `ABL###` | `ABL001` |

Current baseline identifiers:

```text
DS001   All-Age-Faces
DS002   UTKFace

FD001   face_det_lite
AGE001  age.onnx
GEN001  AdaFace IR50 MS1MV2 Gender
```

Do not silently reuse an existing ID for a different dataset, model file, preprocessing method, or configuration.

---

## 4. Sheet schemas

## 4.1 `01_Datasets`

One row per dataset.

```text
Dataset_ID
Dataset_Name
Version
Image_Count
Age_Label
Gender_Label
Face_BBox
Landmarks
Identity_ID
Age_Range
Split_Method
License
Source_URL
Local_Path
Status
Notes
```

### Field guidance

| Field | Expected content |
|---|---|
| `Dataset_ID` | Stable dataset identifier |
| `Version` | Official release name, revision, archive version, or retrieval date |
| `Image_Count` | Full validated image count, not only benchmark subset size |
| `Age_Label` | `Yes`, `No`, `Derived`, or explanation |
| `Gender_Label` | `Yes`, `No`, `Inferred`, or explanation |
| `Face_BBox` | Whether reference face boxes are available |
| `Landmarks` | Whether facial key-point coordinates are available |
| `Identity_ID` | Whether images can be grouped by person identity |
| `Split_Method` | Official split or custom reproducible split strategy |
| `Status` | `Planned`, `Active`, `Completed`, or `Archived` |
| `Notes` | Label caveats, file organization, annotation quality, access restrictions |

Unavailable annotations must be recorded explicitly. Do not assume that aligned or cropped faces imply that bounding-box ground truth is available.

---

## 4.2 `02_Models`

One row per model variant.

```text
Model_ID
Task
Model_Name
Version
Format
Input_Size
Precision
Parameters
Training_Dataset
File_Path
File_Hash
Status
Baseline?
Notes
```

Valid `Task` values:

```text
Face Detection
Age Prediction
Gender Prediction
```

A new model file, quantized version, input resolution, output interpretation, or materially different preprocessing method should receive a new model ID or configuration ID.

Record a SHA-256 hash when possible:

```bash
sha256sum path/to/model.onnx
```

---

## 4.3 `03_Experiment_Runs`

One row per complete benchmark execution.

```text
Run_ID
Run_Date
Dataset_ID
Face_Model_ID
Age_Model_ID
Gender_Model_ID
Sample_N
Random_Seed
Selection_Method
Execution_Provider
Warmup_Runs
Batch_Size
Device
OS
ORT_Version
Git_Commit
Config_ID
W&B_Run
Result_JSON
Run_Status
Notes
```

### Required run metadata

The agent must record at least:

- dataset ID and version;
- all three model IDs;
- evaluated sample count;
- sample-selection method;
- random seed, when randomness is used;
- execution providers in priority order;
- warm-up count;
- batch size;
- device and operating system;
- ONNX Runtime version;
- Git commit;
- result JSON path;
- run completion status.

Example:

```tsv
RUN002	2026-08-05	DS002	FD001	AGE001	GEN001	300	42	Deterministic stratified sampling by age and gender	CUDAExecutionProvider → CPUExecutionProvider	10	1	NVIDIA GPU	 Linux	1.x.x	<git-commit>	CFG002		output/benchmark_utkface_RUN002.json	Completed	UTKFace baseline evaluation
```

Do not compare latency values from different hardware or execution providers without clearly labeling the environment difference.

---

## 4.4 `04_Headline_Metrics`

One wide row per run.

```text
Run_ID
Face_Coverage
Face_Model_Mean_ms
Face_Model_P50_ms
Face_Model_P95_ms
Face_Stage_Mean_ms
Face_Stage_P50_ms
Face_Stage_P95_ms
AGE_MAE
AGE_RMSE
AGE_Median_AE
AGE_P90_AE
AGE_Within5
AGE_Within10
GEN_Accuracy
GEN_Balanced_Accuracy
GEN_Macro_F1
GEN_Female_ROC_AUC
GEN_Female_PR_AUC
PIPE_Success_Rate
PIPE_Mean_ms
PIPE_P50_ms
PIPE_P95_ms
PIPE_Throughput_img_s
```

### Units and value conventions

| Metric type | Stored value |
|---|---|
| Percentages/rates | Decimal proportion from `0.0` to `1.0` |
| Latency | Milliseconds |
| Age error | Years |
| AUC/F1 | Decimal score from `0.0` to `1.0` |
| Throughput | Images per second |
| Counts | Integer |

Examples:

```text
96.67% accuracy  -> 0.9667
100% coverage    -> 1.0
10.10-year MAE   -> 10.10
84.39 ms latency -> 84.39
```

Use a blank or `null` for an unavailable metric. Never use `0` to represent missing data.

---

## 4.5 `05_Slice_Metrics`

Detailed metrics use long format.

```text
Run_ID
Task
Slice_Type
Slice_Value
Metric
Value
Unit
Sample_N
```

Each row represents one metric for one subset.

Example:

```tsv
RUN002	Age Prediction	True Age Interval	20-29	MAE	6.42	years	74
RUN002	Age Prediction	True Age Interval	20-29	RMSE	8.11	years	74
RUN002	Age Prediction	True Age Interval	20-29	Within 5 Years	0.5270	proportion	74
RUN002	Gender Prediction	True Class	Female	Recall	0.9512	proportion	123
```

`Sample_N` is mandatory for slice metrics. A metric based on one sample must not appear equivalent to a metric based on hundreds of samples.

### Current age intervals

The template currently uses:

```text
0-12
13-19
20-29
30-39
40-49
50-59
60-69
70-79
80-89
```

For datasets containing older ages, extend the same ten-year pattern:

```text
90-99
100-109
110-119
```

UTKFace contains ages up to 116, so the UTKFace benchmark should prepare rows through `110-119` when those groups contain valid samples.

For every age group, prepare at least:

```text
Samples
MAE
RMSE
Within 5 Years
Within 10 Years
Mean Signed Error
Median Absolute Error
```

The workbook currently visualizes MAE by age group, but additional metrics should still be saved in `05_Slice_Metrics`.

### Recommended additional slices

When labels and sample sizes permit:

- gender class;
- age interval;
- race category;
- image resolution;
- face size;
- detector confidence;
- pose or occlusion;
- dataset source or split.

Sensitive subgroup labels must only be used when they are officially supplied and relevant to responsible model analysis.

---

## 4.6 `06_Ablation_Studies`

Ablation comparisons use:

```text
Study_ID
Baseline_Run
Variant_Run
Metric_ID
Changed_Component
Baseline_Value
Variant_Value
Raw_Delta
Improvement_%
Direction
Better?
Notes
```

Recommended `Changed_Component` values:

```text
Dataset
Face Model
Age Model
Gender Model
Input Resolution
Quantization
Face Alignment
Crop Margin
Confidence Threshold
Execution Provider
Sampling Strategy
Other
```

For a controlled ablation, change one major factor while keeping the dataset, sample list, hardware, runtime provider, and other model components constant.

Changing the dataset is primarily a cross-dataset generalization comparison, not a pure model ablation. It may still be stored in this sheet when clearly labeled as `Dataset`.

---

## 4.7 `07_Metric_Dictionary`

Canonical headline metric IDs include:

```text
Face_Coverage
Face_Model_Mean_ms
Face_Model_P95_ms
Face_Stage_Mean_ms
AGE_MAE
AGE_RMSE
AGE_Median_AE
AGE_P90_AE
AGE_Within5
AGE_Within10
GEN_Accuracy
GEN_Balanced_Accuracy
GEN_Macro_F1
PIPE_Success_Rate
PIPE_Mean_ms
PIPE_P95_ms
PIPE_Throughput_img_s
```

Use the exact spelling and capitalization when preparing headline metrics or ablation rows.

---

## 5. Required benchmark outputs

For every completed run, create the following files:

```text
output/
├── benchmark_<dataset>_<run_id>.json
├── benchmark_<dataset>_<run_id>_headline.tsv
├── benchmark_<dataset>_<run_id>_slices.tsv
├── benchmark_<dataset>_<run_id>_failures.csv
└── benchmark_<dataset>_<run_id>_samples.csv
```

### 5.1 Main JSON

Recommended structure:

```json
{
  "schema_version": "1.0",
  "run": {
    "run_id": "RUN002",
    "run_date": "2026-08-05",
    "dataset_id": "DS002",
    "face_model_id": "FD001",
    "age_model_id": "AGE001",
    "gender_model_id": "GEN001",
    "config_id": "CFG002",
    "sample_n": 300,
    "random_seed": 42,
    "selection_method": "deterministic stratified sampling",
    "execution_provider": [
      "CUDAExecutionProvider",
      "CPUExecutionProvider"
    ],
    "warmup_runs": 10,
    "batch_size": 1,
    "device": "NVIDIA GPU",
    "os": "Linux",
    "onnxruntime_version": "",
    "git_commit": ""
  },
  "dataset": {
    "validated_image_count": 0,
    "selected_images": 300,
    "age_counts": {},
    "gender_counts": {},
    "invalid_files": 0
  },
  "face": {
    "processed_images": 0,
    "images_with_detection": 0,
    "detection_coverage": null,
    "mean_faces_per_processed_image": null,
    "model_latency": {
      "mean_ms": null,
      "p50_ms": null,
      "p95_ms": null
    },
    "stage_latency": {
      "mean_ms": null,
      "p50_ms": null,
      "p95_ms": null
    }
  },
  "age": {
    "evaluated_images": 0,
    "mae": null,
    "rmse": null,
    "median_absolute_error": null,
    "p90_absolute_error": null,
    "within_5_years": null,
    "within_10_years": null,
    "mean_signed_error": null,
    "by_true_age_interval": {}
  },
  "gender": {
    "evaluated_images": 0,
    "accuracy": null,
    "balanced_accuracy": null,
    "macro_f1": null,
    "roc_auc_female": null,
    "pr_auc_female": null,
    "confusion_matrix": {
      "true_female_pred_female": 0,
      "true_female_pred_male": 0,
      "true_male_pred_female": 0,
      "true_male_pred_male": 0
    },
    "by_true_class": {}
  },
  "pipeline": {
    "processed_images": 0,
    "successful_images": 0,
    "failed_images": 0,
    "success_rate": null,
    "latency": {
      "mean_ms": null,
      "p50_ms": null,
      "p95_ms": null
    },
    "wall_seconds": null,
    "throughput_images_per_second": null
  },
  "failures": [],
  "samples": []
}
```

### 5.2 Headline TSV

The file must contain exactly one header row and one result row using the `04_Headline_Metrics` column order.

### 5.3 Slice TSV

The file must use the exact `05_Slice_Metrics` long-format schema and may contain any number of rows.

### 5.4 Per-image samples

Recommended columns:

```text
Run_ID
Dataset_ID
Image_Path
True_Age
Predicted_Age
Absolute_Age_Error
Signed_Age_Error
True_Gender
Predicted_Gender
Female_Probability
Gender_Confidence
Face_Detected
Face_Count
Detector_Confidence
Face_Model_ms
Face_Stage_ms
Age_Model_ms
Gender_Model_ms
Pipeline_ms
Failure_Stage
Failure_Reason
```

Per-image outputs are not pasted into the main workbook, but they are required for verification, error analysis, and future slices.

---

## 6. UTKFace-specific preparation

## 6.1 Filename interpretation

UTKFace filenames generally follow:

```text
[age]_[gender]_[race]_[date&time].jpg
```

Interpretation:

| Token | Meaning |
|---|---|
| `age` | Age in years, normally `0-116` |
| `gender` | `0 = Male`, `1 = Female` |
| `race` | `0 = White`, `1 = Black`, `2 = Asian`, `3 = Indian`, `4 = Others` |
| `date&time` | Collection timestamp |

Example:

```text
25_1_2_20170116174525125.jpg
```

Expected parsed labels:

```text
age    = 25
gender = Female
race   = Asian
```

The parser must validate every filename and log malformed or unsupported entries rather than silently assigning labels.

Recommended parser output:

```text
image_path
age
gender_code
gender_label
race_code
race_label
timestamp
parse_status
parse_error
```

## 6.2 UTKFace benchmark rules

1. Validate all image files before sampling.
2. Report the full validated corpus count.
3. Record age and gender distributions before sampling.
4. Prefer a deterministic stratified sample by age interval and gender.
5. Save the exact selected-image list.
6. Use the same selected-image list across model ablations.
7. Extend age intervals through `110-119`.
8. Log every failed decode, detection, crop, or inference.
9. Keep race labels optional and separate from the primary age/gender benchmark.
10. Do not infer identity-disjoint splitting from filenames; no verified person identity ID is encoded.

---

## 7. Metric definitions

## 7.1 Age metrics

For true age \(y_i\) and prediction \(\hat{y}_i\):

### Mean Absolute Error

```text
MAE = mean(abs(predicted_age - true_age))
```

### Root Mean Squared Error

```text
RMSE = sqrt(mean((predicted_age - true_age)^2))
```

### Mean Signed Error

```text
Bias = mean(predicted_age - true_age)
```

Interpretation:

```text
positive bias -> model predicts older
negative bias -> model predicts younger
```

Report MAE and RMSE globally and for every true-age interval.

## 7.2 Gender metrics

Report at least:

- Accuracy
- Balanced Accuracy
- Macro F1
- Confusion Matrix
- Female ROC-AUC
- Female PR-AUC
- Female and Male precision, recall, and F1

The current convention uses **Female as the positive class**:

```text
Female = 1
Male   = 0
```

The score supplied to ROC-AUC and PR-AUC must therefore be `female_probability`.

Do not calculate AUC from hard class predictions.

## 7.3 Face metrics

When reference bounding boxes are unavailable, report:

- processed images;
- images with at least one detection;
- detection coverage;
- mean faces per image;
- model latency;
- complete detection-stage latency.

Do not report AP, IoU, detector precision, or detector recall unless trustworthy reference boxes are available.

## 7.4 Pipeline metrics

Report:

- processed images;
- successful images;
- failed images;
- success rate;
- end-to-end mean, p50, and p95 latency;
- total wall time;
- throughput in images per second.

A successful image must complete face detection, valid crop creation, age inference, and gender inference.

---

## 8. Reproducibility requirements

Before each benchmark, record:

```text
Dataset version
Dataset archive hash, if available
Selected file list
Random seed
Model hashes
Git commit
Python version
ONNX Runtime version
Execution providers
Hardware
Operating system
Thread settings
Warm-up count
Batch size
Model input sizes
Detection threshold
NMS threshold
Face crop margin
Face alignment setting
Age output interpretation
Gender decision threshold
```

Run warm-up inference before timed measurements.

Recommended minimum:

```text
Warm-up runs: 10
Timed repetitions: 3 or more
```

When several repetitions are used, preserve each repetition and report the aggregate mean and variability.

---

## 9. Validation checklist

A benchmark is ready for workbook insertion only when all checks pass.

### Dataset checks

- [ ] Dataset ID and version are recorded.
- [ ] Full validated image count is recorded.
- [ ] Label distributions are recorded.
- [ ] Malformed filenames are counted and logged.
- [ ] Exact benchmark sample list is saved.
- [ ] Sampling method and seed are recorded.

### Model checks

- [ ] All three model IDs are recorded.
- [ ] Model file paths and hashes are recorded.
- [ ] Preprocessing and thresholds are recorded.
- [ ] Execution providers and device are recorded.

### Result checks

- [ ] Headline row contains one `Run_ID`.
- [ ] Percentages are stored as decimal proportions.
- [ ] Missing values are blank or `null`, not zero.
- [ ] Age metrics include sample counts for every interval.
- [ ] Gender confusion-matrix counts sum to evaluated gender samples.
- [ ] Pipeline successes plus failures equal processed images.
- [ ] Throughput is consistent with image count and wall time.
- [ ] Per-image results and failure logs are saved.
- [ ] Raw JSON and import-ready TSV files are generated.

---

## 10. Recommended command behavior for the benchmark agent

The benchmark script should support arguments similar to:

```bash
python benchmark.py \
  --dataset utkface \
  --dataset-id DS002 \
  --data-dir /path/to/UTKFace \
  --face-model /path/to/face_det_lite.onnx \
  --age-model /path/to/age.onnx \
  --gender-model /path/to/gender.onnx \
  --run-id RUN002 \
  --config-id CFG002 \
  --limit 300 \
  --seed 42 \
  --sampling stratified \
  --warmup 10 \
  --output-dir output/
```

The exact CLI may differ, but the result package and field meanings must follow this document.

---

## 11. Final agent deliverables

At the end of a benchmark, the agent should report:

```text
1. Run status
2. Dataset and models used
3. Sample count and selection strategy
4. Headline face, age, gender, and pipeline metrics
5. Age metrics by age interval
6. Gender confusion matrix and class metrics
7. Failure summary
8. Paths to JSON, TSV, sample, and failure files
9. Reproducibility metadata
10. Any data-quality or comparability warnings
```

The agent must not claim that a metric improved unless the baseline and variant runs are comparable in dataset, sample list, hardware, runtime provider, and unchanged pipeline components.
