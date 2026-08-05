# AgePredictor Repository Refactor Plan

## Goal

Refactor the repository so the benchmark pipeline can evaluate multiple datasets with different annotation formats while keeping one shared runtime benchmark implementation.

The target architecture is:

```text
Raw dataset
    ↓
Dataset-specific preparation script
    ↓
Canonical manifest
    ↓
Generic benchmark_runtime.py
    ↓
Benchmark outputs
```

The benchmark runtime must no longer contain dataset-specific filename parsing or label rules.

---

# 1. Final repository structure

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

The `benchmark/datasets/` folder is not required for the current design because dataset-specific logic is handled by the preparation scripts. Also take this structure as reference only not to remove other existing files, only take this structure as the core components that should appear in this repo. 

---

# 2. Canonical manifest schema

All preparation scripts must output the same schema.

Recommended CSV columns:

```text
dataset_id
dataset_version
sample_id
image_path
age
gender
identity_id
bbox_x1
bbox_y1
bbox_x2
bbox_y2
landmarks_path
dataset_split
source
is_valid
validation_error
metadata_json
```

## Field conventions

| Field | Rule |
|---|---|
| `dataset_id` | Stable ID such as `DS001`, `DS002` |
| `dataset_version` | Official version or release identifier |
| `sample_id` | Unique inside the dataset |
| `image_path` | Absolute or repository-resolved image path |
| `age` | Integer years or blank |
| `gender` | `Female`, `Male`, or blank |
| `identity_id` | Person identity ID when available |
| `bbox_*` | Reference bounding box in `xyxy` format or blank |
| `landmarks_path` | Path to landmark annotation or blank |
| `dataset_split` | `train`, `validation`, `test`, `all`, or custom split name |
| `source` | Dataset source such as `UTKFace`, `IMDb`, `Wikipedia` |
| `is_valid` | `true` or `false` |
| `validation_error` | Error reason for invalid records |
| `metadata_json` | Extra dataset-specific metadata as valid JSON |

Missing values must be blank in CSV and converted to `None` by the loader.

Do not use `0` to represent a missing age, bounding box, metric, or label.

---

# 3. Step-by-step implementation order

## Step 1 — Create `benchmark/records.py`

Create a common benchmark record.

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkRecord:
    dataset_id: str
    dataset_version: str
    sample_id: str
    path: Path

    age: int | None = None
    gender_label: str | None = None
    identity_id: str | None = None

    bbox_xyxy: tuple[float, float, float, float] | None = None
    landmarks_path: Path | None = None

    dataset_split: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

Validation requirements:

- `sample_id` must not be empty.
- `gender_label` must be `Female`, `Male`, or `None`.
- `age` must be non-negative when present.
- bounding boxes must contain four numeric values when present.
- image path existence should be checked while loading the manifest.

---

## Step 2 — Create `benchmark/manifest.py`

Responsibilities:

1. Read canonical CSV.
2. Validate the header.
3. Parse optional values.
4. Resolve image paths.
5. Validate gender labels.
6. Parse `metadata_json`.
7. Return `list[BenchmarkRecord]`.
8. Detect duplicate `sample_id` values.
9. Reject an empty manifest.

Recommended public API:

```python
def load_manifest(path: Path) -> list[BenchmarkRecord]:
    ...
```

Optional helper functions:

```python
def optional_int(value: str) -> int | None:
    ...

def optional_float(value: str) -> float | None:
    ...

def parse_bbox(row: dict[str, str]) -> tuple[float, float, float, float] | None:
    ...
```

Acceptance criteria:

- valid manifest loads successfully;
- missing image path raises a clear error;
- invalid JSON includes row number in the error;
- unsupported gender includes row number in the error;
- duplicate sample IDs are reported;
- blank optional fields become `None`.

---

## Step 3 — Move reusable metrics into `benchmark/metrics.py`

Move or preserve the following shared metric functions from the current runtime file:

```text
latency_summary
value_summary
age_metrics
age_metrics_by_interval
gender_metrics
_binary_confusion
_roc_auc_binary
_pr_auc_binary
_safe_divide
_f1
```

Add mean signed age error:

```python
mean_signed_error = mean(predicted_age - true_age)
```

The age metric output should include:

```text
evaluated_images
mae
rmse
mean_signed_error
median_absolute_error
p90_absolute_error
within_5_years
within_10_years
```

The per-age-group output should include the same metrics plus:

```text
samples
```

---

## Step 4 — Extend age intervals

The current benchmark only reports through `80-89`.

Use:

```python
AGE_INTERVALS = (
    ("0-12", 0, 12),
    ("13-19", 13, 19),
    ("20-29", 20, 29),
    ("30-39", 30, 39),
    ("40-49", 40, 49),
    ("50-59", 50, 59),
    ("60-69", 60, 69),
    ("70-79", 70, 79),
    ("80-89", 80, 89),
    ("90-99", 90, 99),
    ("100-109", 100, 109),
    ("110-119", 110, 119),
)
```

This is necessary for UTKFace because its age range extends to 116.

---

## Step 5 — Create `benchmark/sampling.py`

Move dataset-independent sampling logic here.

Required strategies:

```text
all
evenly_spaced
random
stratified_age
stratified_age_gender
from_sample_list
```

Recommended API:

```python
def select_records(
    records: Sequence[BenchmarkRecord],
    *,
    limit: int | None,
    strategy: str,
    seed: int,
) -> list[BenchmarkRecord]:
    ...
```

Requirements:

- deterministic when a seed is provided;
- selected sample IDs must be saved;
- the same sample list must be reusable across model ablations;
- stratified sampling must record the requested and actual counts per group.

For reproducible ablation studies, support:

```bash
--sample-list manifests/samples/DS002_sample_300_seed42.txt
```

---

## Step 6 — Refactor `benchmark_runtime.py`

Remove all All-Age-Faces-specific logic:

```text
AAF_FILENAME_PATTERN
AAF_LAST_FEMALE_ID
AAFRecord
load_aaf_records
infer_aaf_gender_label
DEFAULT_EXPECTED_COUNT
```

The runtime should no longer know how labels are represented in raw datasets.

Replace:

```python
all_records = load_aaf_records(...)
```

with:

```python
all_records = load_manifest(config.manifest_path)
```

### New runtime configuration

Recommended fields:

```python
@dataclass(frozen=True)
class BenchmarkConfig:
    manifest_path: Path
    output_path: Path

    dataset_id: str
    run_id: str
    config_id: str

    face_model_path: Path
    age_model_path: Path
    gender_model_path: Path

    providers: tuple[str, ...] = ("CPUExecutionProvider",)
    limit: int | None = None
    sampling_strategy: str = "all"
    random_seed: int = 42
    sample_list_path: Path | None = None

    warmup_runs: int = 10
    progress_every: int = 100
```

### New CLI arguments

Add:

```text
--manifest
--dataset-id
--run-id
--config-id
--sampling
--seed
--sample-list
```

Remove or deprecate:

```text
--data-dir
--expected-count
```

The dataset preparation scripts should handle raw dataset validation and expected file counts.

---

## Step 7 — Support optional ground-truth labels

The runtime must run inference even when age or gender ground truth is unavailable.

### Age inference

Always produce:

```text
age_predicted
age_model_ms
```

Only calculate evaluation fields when `record.age is not None`:

```text
age_absolute_error
age_signed_error
```

Only append to metric arrays when true age exists.

### Gender inference

Always produce:

```text
gender_predicted
gender_confidence
female_probability
gender_model_ms
```

Only calculate:

```text
gender_correct
```

and metric arrays when `record.gender_label is not None`.

### Required distinction

Track these separately:

```text
age_inference_images
age_evaluated_images
gender_inference_images
gender_evaluated_images
pipeline_successful_images
```

Do not define pipeline success using the number of age labels or evaluated age samples.

Pipeline success means:

1. image decoded;
2. face detected;
3. face crop created;
4. age inference completed;
5. gender inference completed.

---

## Step 8 — Update measurement storage

Refactor `BenchmarkMeasurements` to include:

```python
@dataclass
class BenchmarkMeasurements:
    face_stage_ms: list[float]
    face_model_ms: list[float]
    face_counts: list[int]

    age_model_ms: list[float]
    gender_model_ms: list[float]
    pipeline_ms: list[float]

    age_predictions: list[float]
    age_labels: list[int]

    gender_predictions: list[str]
    gender_labels: list[str]
    gender_confidences: list[float]
    female_probabilities: list[float]

    age_inference_count: int
    gender_inference_count: int
    pipeline_success_count: int

    samples: list[dict[str, Any]]
    failures: list[dict[str, Any]]
```

Failure records should include:

```text
sample_id
filename
stage
reason
```

Recommended failure stages:

```text
manifest
decode
face_detection
face_crop
age_inference
gender_inference
unknown
```

---

## Step 9 — Update result JSON schema

Recommended result structure:

```json
{
  "schema_version": "2.0",
  "created_at": "",
  "run": {
    "run_id": "",
    "config_id": "",
    "dataset_id": "",
    "manifest_path": "",
    "manifest_sha256": "",
    "sample_count": 0,
    "sampling_strategy": "",
    "random_seed": 42,
    "sample_list_path": ""
  },
  "environment": {},
  "providers": {},
  "models": {},
  "dataset": {},
  "face": {},
  "age": {},
  "gender": {},
  "pipeline": {},
  "failures": [],
  "samples": []
}
```

Include:

- manifest hash;
- model hashes;
- exact selected sample IDs;
- Git commit when available;
- environment versions;
- requested and active execution providers.

---

## Step 10 — Create `benchmark/exporters.py`

Generate workbook-compatible outputs.

Required functions:

```python
def export_headline_tsv(result: dict, path: Path) -> Path:
    ...

def export_slice_tsv(result: dict, path: Path) -> Path:
    ...

def export_samples_csv(result: dict, path: Path) -> Path:
    ...

def export_failures_csv(result: dict, path: Path) -> Path:
    ...
```

### Headline TSV column order

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

### Slice TSV column order

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

Store rates as decimal proportions:

```text
96.67% -> 0.9667
```

---

# 4. Dataset preparation scripts

## Step 11 — Create `scripts/prepare_all_age_faces.py`

Purpose:

- preserve the original benchmark behavior;
- create `DS001_all_age_faces.csv`;
- provide a regression-test dataset for the refactored runtime.

Required logic:

```text
filename pattern: NNNNNAxx.jpg
age: xx
image ID: NNNNN
gender:
    Female when ID <= 7380
    Male otherwise
```

Required validation report:

```text
files_discovered
valid_records
invalid_filenames
decode_failures
age_min
age_max
age_bin_counts
gender_counts
duplicate_sample_ids
```

Example command:

```bash
python scripts/prepare_all_age_faces.py \
  --data-dir /path/to/All-Age-Faces \
  --dataset-id DS001 \
  --dataset-version official \
  --output manifests/DS001_all_age_faces.csv \
  --validation-report output/validation/DS001_validation.json
```

---

## Step 12 — Regression-test the new pipeline with All-Age-Faces

Run the manifest-based benchmark using the same:

- 300 images;
- selection method;
- model files;
- execution provider;
- warm-up count.

Example:

```bash
python benchmark/benchmark_runtime.py \
  --manifest manifests/DS001_all_age_faces.csv \
  --dataset-id DS001 \
  --run-id RUN001_MANIFEST \
  --config-id CFG001 \
  --limit 300 \
  --sampling evenly_spaced \
  --provider coreml \
  --warmup-runs 10 \
  --output output/benchmark_aaf_RUN001_MANIFEST.json
```

Expected predictive results should be close to the previous reference:

```text
Age MAE: approximately 10.10
Age RMSE: approximately 13.38
Gender Accuracy: approximately 96.67%
Gender Macro F1: approximately 0.966
```

Latency may vary slightly across runs.

Do not continue to UTKFace until:

- selected filenames match the old benchmark selection;
- age and gender labels match;
- predictive metrics match within a small tolerance;
- no new pipeline failures appear.

---

## Step 13 — Create `scripts/prepare_utkface.py`

UTKFace filename format:

```text
[age]_[gender]_[race]_[timestamp].jpg
```

Mappings:

```text
gender:
0 = Male
1 = Female

race:
0 = White
1 = Black
2 = Asian
3 = Indian
4 = Others
```

Required parser checks:

- filename matches expected structure;
- age is between 0 and 116;
- gender code is `0` or `1`;
- race code is between `0` and `4`;
- image exists;
- image can be decoded;
- sample ID is unique.

Store race and timestamp inside `metadata_json`.

Example metadata:

```json
{
  "gender_code": 1,
  "race_code": 2,
  "race_label": "Asian",
  "collection_timestamp": "20170116174525125"
}
```

Example command:

```bash
python scripts/prepare_utkface.py \
  --data-dir /datasets/UTKFace \
  --dataset-id DS002 \
  --dataset-version official \
  --output manifests/DS002_utkface.csv \
  --validation-report output/validation/DS002_validation.json
```

Required validation summary:

```text
files_discovered
valid_records
invalid_filenames
decode_failures
age_min
age_max
unique_ages
age_bin_counts
gender_counts
race_counts
duplicate_sample_ids
```

---

## Step 14 — Validate the UTKFace manifest before inference

Check:

- age range is `0-116`;
- only `Female` and `Male` appear in normalized gender;
- age-bin counts include `90-99`, `100-109`, and `110-119`;
- all image paths resolve;
- no duplicate sample IDs;
- invalid filenames are listed;
- decoded image count matches valid record count.

Save the validation report before running GPU inference.

---

## Step 15 — Create a fixed UTKFace sample list

Recommended first benchmark:

```text
300 samples
stratified by true-age interval and gender
seed = 42
```

Example:

```bash
python benchmark/create_sample_list.py \
  --manifest manifests/DS002_utkface.csv \
  --strategy stratified_age_gender \
  --limit 300 \
  --seed 42 \
  --output manifests/samples/DS002_sample_300_seed42.txt
```

Use the same list for every model comparison.

If a standalone sample-list script is not created, add equivalent functionality to `benchmark/sampling.py` and expose it through the benchmark CLI.

---

## Step 16 — Run the UTKFace baseline benchmark

Example:

```bash
python benchmark/benchmark_runtime.py \
  --manifest manifests/DS002_utkface.csv \
  --sample-list manifests/samples/DS002_sample_300_seed42.txt \
  --dataset-id DS002 \
  --run-id RUN002 \
  --config-id CFG002 \
  --provider cuda \
  --warmup-runs 10 \
  --output output/benchmark_utkface_RUN002.json
```

Expected outputs:

```text
output/benchmark_utkface_RUN002.json
output/benchmark_utkface_RUN002_headline.tsv
output/benchmark_utkface_RUN002_slices.tsv
output/benchmark_utkface_RUN002_samples.csv
output/benchmark_utkface_RUN002_failures.csv
```

---

# 5. Later dataset scripts

## Step 17 — Create `scripts/prepare_imdb_wiki.py`

Read labels from the official `.mat` metadata.

Required source fields:

```text
full_path
dob
photo_taken
gender
name
face_location
face_score
second_face_score
celeb_id
```

Preparation logic:

- calculate age from `dob` and `photo_taken`;
- normalize gender;
- skip or flag `NaN` gender;
- reject invalid or implausible ages;
- preserve celebrity identity;
- convert `face_location` to canonical `xyxy`;
- record IMDb or Wikipedia source;
- log missing or corrupt images.

Identity-disjoint splitting is possible using celebrity name or `celeb_id`.

---

## Step 18 — Create `scripts/prepare_megaface.py`

MegaFace does not provide suitable age and gender ground truth.

Set:

```text
age = blank
gender = blank
```

Preserve when available:

```text
bounding box
landmarks
Flickr account ID
source image ID
face index
rotation
detector confidence
```

The runtime should still report:

```text
face detection metrics
age inference latency
gender inference latency
pipeline latency
```

It should not calculate:

```text
Age MAE/RMSE
Gender Accuracy/F1/AUC
```

Those metrics must be `null` or blank, not zero.

---

# 6. Testing requirements

## Unit tests

### `test_manifest.py`

Test:

- valid row;
- blank optional labels;
- unsupported gender;
- missing image;
- malformed JSON;
- duplicate sample ID;
- malformed bounding box.

### `test_prepare_all_age_faces.py`

Test:

- valid filename;
- invalid filename;
- female boundary ID `07380`;
- male boundary ID `07381`;
- age extraction.

### `test_prepare_utkface.py`

Test:

- valid filename;
- age `0`;
- age `116`;
- invalid age;
- invalid gender code;
- invalid race code;
- extra underscores;
- unsupported extension.

### `test_benchmark_runtime.py`

Use mocked inference outputs to verify:

- labeled samples contribute to metrics;
- unlabeled samples still contribute to inference and latency counts;
- pipeline success is independent of label availability;
- failures include stage and sample ID;
- export files use expected columns.

---

# 7. Backward compatibility

During migration, optionally keep the old CLI behavior temporarily:

```bash
--data-dir
```

but print a deprecation warning:

```text
Direct All-Age-Faces loading is deprecated.
Generate a manifest with scripts/prepare_all_age_faces.py and use --manifest.
```

Remove direct dataset parsing after the manifest pipeline passes regression testing.

---

# 8. Documentation updates

Update the repository README with:

1. architecture diagram;
2. canonical manifest schema;
3. dataset preparation commands;
4. benchmark command;
5. output file descriptions;
6. metric definitions;
7. supported datasets;
8. current model IDs;
9. reproducibility requirements;
10. example workflow for UTKFace.

Add links to:

```text
AgePredictor_Benchmark_Template_Agent_Guide.md
AgePredictor_Repo_Refactor_Next_Steps.md
```

---

# 9. Recommended implementation checkpoints

## Checkpoint A — Core manifest support

Complete when:

- `BenchmarkRecord` exists;
- manifest loader passes tests;
- runtime accepts `--manifest`;
- All-Age-Faces-specific parsing is removed from the runtime.

## Checkpoint B — All-Age-Faces regression

Complete when:

- DS001 manifest is generated;
- same 300-image sample is selected;
- age and gender results match the old benchmark;
- output JSON remains valid;
- headline and slice TSV exports work.

## Checkpoint C — UTKFace readiness

Complete when:

- DS002 manifest is generated;
- validation report is reviewed;
- age intervals reach `110-119`;
- fixed sample list is saved;
- benchmark runs successfully on the SSH server.

## Checkpoint D — Multi-dataset support

Complete when:

- IMDB-WIKI metadata can be converted;
- MegaFace unlabeled records run without metric errors;
- workbook-compatible rows can be generated for every run.

---

# 10. Final execution order for the agent

Follow this order exactly:

```text
1. Create benchmark/records.py
2. Create benchmark/manifest.py
3. Move shared metrics into benchmark/metrics.py
4. Extend age intervals to 110-119
5. Create benchmark/sampling.py
6. Refactor benchmark_runtime.py to load manifests
7. Add optional-label handling
8. Separate inference counts from evaluation counts
9. Update result JSON to schema version 2.0
10. Create benchmark/exporters.py
11. Create scripts/prepare_all_age_faces.py
12. Generate DS001 manifest
13. Regression-test against the old All-Age-Faces result
14. Fix any result mismatch
15. Create scripts/prepare_utkface.py
16. Generate and validate DS002 manifest
17. Save a fixed UTKFace sample list
18. Run the UTKFace baseline benchmark
19. Export headline and slice metrics
20. Update README and benchmark documentation
21. Add IMDB-WIKI preparation
22. Add MegaFace preparation
23. Add complete unit tests
24. Remove deprecated direct-dataset parsing
```

---

# 11. Definition of done

The refactor is complete when:

- one generic runtime supports all datasets through manifests;
- no raw dataset filename convention exists inside `benchmark_runtime.py`;
- every dataset preparation script outputs the same canonical schema;
- missing labels are supported safely;
- every run creates JSON, headline TSV, slice TSV, samples CSV, and failures CSV;
- All-Age-Faces reproduces the existing baseline;
- UTKFace runs on the SSH server using a fixed sample list;
- age results are reported globally and by age interval;
- gender results include accuracy, balanced accuracy, macro F1, confusion matrix, ROC-AUC, and PR-AUC;
- all model files, manifests, providers, sample lists, and run settings are traceable;
- output rows can be inserted into the benchmarking workbook without manual restructuring.
