#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-data/adience}"
DATASET_ID="${DATASET_ID:-DS004}"
CONFIG_ID="${CONFIG_ID:-CFG001}"
PROVIDER="${PROVIDER:-cuda}"
FOLD_SET="${FOLD_SET:-all}"
AGE_LABEL_POLICY="${AGE_LABEL_POLICY:-blank}"
WANDB_PROJECT="${WANDB_PROJECT:-AgeGenderPredictor}"
WANDB_MODE="${WANDB_MODE:-online}"

if [[ "${FOLD_SET}" == "all" ]]; then
    LABEL_PREFIX="fold"
    FOLD_SUFFIX=""
elif [[ "${FOLD_SET}" == "frontal" ]]; then
    LABEL_PREFIX="fold_frontal"
    FOLD_SUFFIX="_frontal"
else
    echo "[adience-faces] FOLD_SET must be all or frontal: ${FOLD_SET}" >&2
    exit 1
fi

RUN_STAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_ID:-RUN_${RUN_STAMP}_ADIENCE_FACES_FULL}"
MANIFEST="${MANIFEST:-manifests/${DATASET_ID}_adience_faces${FOLD_SUFFIX}.csv}"
VALIDATION_NAME="${DATASET_ID}_adience_faces${FOLD_SUFFIX}_validation.json"
VALIDATION_REPORT="${VALIDATION_REPORT:-output/validation/${VALIDATION_NAME}}"
RESULT_JSON="${RESULT_JSON:-output/${RUN_ID}.json}"
LOG_PATH="${LOG_PATH:-output/logs/${RUN_ID}.log}"

mkdir -p "$(dirname -- "${LOG_PATH}")"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "[adience-faces] started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[adience-faces] repository: ${REPO_ROOT}"
echo "[adience-faces] data: ${DATA_DIR}/faces"
echo "[adience-faces] fold set: ${FOLD_SET} | age policy: ${AGE_LABEL_POLICY}"
echo "[adience-faces] provider: ${PROVIDER} | W&B: ${WANDB_PROJECT}"
echo "[adience-faces] persistent log: ${LOG_PATH}"

for required_path in "${DATA_DIR}/faces" "${DATA_DIR}/labels"; do
    if [[ ! -d "${required_path}" ]]; then
        echo "[adience-faces] missing directory: ${required_path}" >&2
        exit 1
    fi
done

for fold_index in 0 1 2 3 4; do
    label_path="${DATA_DIR}/labels/${LABEL_PREFIX}_${fold_index}_data.txt"
    if [[ ! -f "${label_path}" ]]; then
        echo "[adience-faces] missing label file: ${label_path}" >&2
        exit 1
    fi
done

if [[ "${PROVIDER}" == "cuda" ]]; then
    "${PYTHON_BIN}" -c '
import sys
import onnxruntime as ort
providers = ort.get_available_providers()
print(f"[adience-faces] ONNX Runtime providers: {providers}")
if "CUDAExecutionProvider" not in providers:
    sys.exit("CUDAExecutionProvider is unavailable")
'
fi

echo "[1/2] Preparing Adience faces manifest"
"${PYTHON_BIN}" -u scripts/prepare_adience.py \
    --data-dir "${DATA_DIR}" \
    --image-variant faces \
    --fold-set "${FOLD_SET}" \
    --dataset-id "${DATASET_ID}" \
    --dataset-version official-faces \
    --age-label-policy "${AGE_LABEL_POLICY}" \
    --progress-every "${PREPARE_PROGRESS_EVERY:-1000}" \
    --output "${MANIFEST}" \
    --validation-report "${VALIDATION_REPORT}"

WANDB_ARGS=(
    --wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-mode "${WANDB_MODE}"
    --wandb-run-name "${RUN_ID}"
    --wandb-tags adience faces full-dataset "${PROVIDER}" \
        "${FOLD_SET}" interval-aware
)
if [[ -n "${WANDB_ENTITY:-}" ]]; then
    WANDB_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

echo "[2/2] Benchmarking the complete Adience faces dataset"
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

echo "[adience-faces] complete: ${RESULT_JSON}"
echo "[adience-faces] log: ${LOG_PATH}"
