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

## Latest Reference Run

Latest local reference result: `output/benchmark_runtime.json`

Run configuration:

- Images evaluated: `300`
- Selection: deterministic, evenly spaced sample over filename order
- Runtime providers: `CoreMLExecutionProvider`, then `CPUExecutionProvider`
- Warmup runs: `10`
- W&B metrics logged: `104`

Headline results:

| Area | Metric | Result |
|---|---:|---:|
| Face detector | Detection coverage | `300/300 (100.00%)` |
| Face detector | Stage latency mean / p50 / p95 | `70.81 / 68.98 / 102.46 ms` |
| Age | MAE / RMSE | `10.10 / 13.38 years` |
| Age | Median / p90 absolute error | `7.88 / 22.01 years` |
| Age | Within 5 / 10 years | `37.00% / 59.33%` |
| Gender | Accuracy | `96.67%` |
| Gender | Balanced accuracy | `96.70%` |
| Gender | Macro F1 | `0.966` |
| Gender | Female ROC-AUC / PR-AUC | `0.997 / 0.997` |
| Pipeline | Success rate | `300/300 (100.00%)` |
| Pipeline | End-to-end latency mean / p50 / p95 | `84.39 / 82.48 / 120.76 ms` |
| Pipeline | Wall throughput | `11.85 images/s` |

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
| `0-12` | `32` | `19.09` | `20.02` | `0.00%` | `6.25%` |
| `13-19` | `20` | `13.05` | `13.54` | `0.00%` | `15.00%` |
| `20-29` | `70` | `4.64` | `5.93` | `68.57%` | `87.14%` |
| `30-39` | `76` | `4.16` | `5.34` | `65.79%` | `94.74%` |
| `40-49` | `47` | `8.89` | `10.47` | `27.66%` | `61.70%` |
| `50-59` | `26` | `14.83` | `16.02` | `0.00%` | `30.77%` |
| `60-69` | `17` | `22.59` | `24.16` | `0.00%` | `5.88%` |
| `70-79` | `11` | `27.38` | `30.18` | `0.00%` | `18.18%` |
| `80-89` | `1` | `28.11` | `28.11` | `0.00%` | `0.00%` |

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
| True Female | `160` | `6` |
| True Male | `4` | `130` |

Latest gender result:

- Accuracy: `96.67%`
- Balanced accuracy: `96.70%`
- Macro F1: `0.966`
- Female precision / recall / F1: `97.56% / 96.39% / 0.970`
- Male precision / recall / F1: `95.59% / 97.01% / 0.963`
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
