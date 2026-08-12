#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-data/FairFace/fairface-img-margin025-trainval}"
TRAIN_LABELS="${TRAIN_LABELS:-data/FairFace/fairface_label_train.csv}"
VAL_LABELS="${VAL_LABELS:-data/FairFace/fairface_label_val.csv}"
DATASET_ID="${DATASET_ID:-DS003}"
CONFIG_ID="${CONFIG_ID:-CFG001}"
PROVIDER="${PROVIDER:-cuda}"
AGE_LABEL_POLICY="${AGE_LABEL_POLICY:-blank}"
WANDB_PROJECT="${WANDB_PROJECT:-AgeGenderPredictor}"
WANDB_MODE="${WANDB_MODE:-online}"
RUN_ID="${RUN_ID:-RUN_$(date -u +%Y%m%d_%H%M%S)_FAIRFACE_MARGIN025_FULL}"

MANIFEST="${MANIFEST:-manifests/${DATASET_ID}_fairface_margin025.csv}"
VALIDATION_REPORT="${VALIDATION_REPORT:-output/validation/${DATASET_ID}_fairface_margin025_validation.json}"
RESULT_JSON="${RESULT_JSON:-output/${RUN_ID}.json}"

for required_path in "${DATA_DIR}/train" "${DATA_DIR}/val" "${TRAIN_LABELS}" "${VAL_LABELS}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Missing required FairFace path: ${required_path}" >&2
        exit 1
    fi
done

if [[ "${PROVIDER}" == "cuda" ]]; then
    "${PYTHON_BIN}" -c 'import sys; import onnxruntime as ort; providers = ort.get_available_providers(); print(f"[fairface] ONNX Runtime providers: {providers}"); sys.exit("CUDAExecutionProvider is unavailable") if "CUDAExecutionProvider" not in providers else None'
fi

echo "[1/2] Preparing FairFace margin 0.25 manifest"
"${PYTHON_BIN}" -u scripts/prepare_fairface.py \
    --data-dir "${DATA_DIR}" \
    --train-labels "${TRAIN_LABELS}" \
    --val-labels "${VAL_LABELS}" \
    --dataset-id "${DATASET_ID}" \
    --dataset-version margin025 \
    --image-padding 0.25 \
    --age-label-policy "${AGE_LABEL_POLICY}" \
    --progress-every "${PREPARE_PROGRESS_EVERY:-1000}" \
    --output "${MANIFEST}" \
    --validation-report "${VALIDATION_REPORT}"

WANDB_ARGS=(
    --wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-mode "${WANDB_MODE}"
    --wandb-run-name "${RUN_ID}"
    --wandb-tags fairface margin025 full-dataset "${PROVIDER}" interval-aware
)
if [[ -n "${WANDB_ENTITY:-}" ]]; then
    WANDB_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

echo "[2/2] Benchmarking the complete FairFace margin 0.25 dataset"
"${PYTHON_BIN}" -u benchmark/benchmark_runtime.py \
    --manifest "${MANIFEST}" \
    --dataset-id "${DATASET_ID}" \
    --run-id "${RUN_ID}" \
    --config-id "${CONFIG_ID}" \
    --provider "${PROVIDER}" \
    --sampling all \
    --limit 0 \
    --warmup-runs "${WARMUP_RUNS:-10}" \
    --progress-every "${BENCHMARK_PROGRESS_EVERY:-100}" \
    --output "${RESULT_JSON}" \
    "${WANDB_ARGS[@]}"

echo "[fairface] complete: ${RESULT_JSON}"
