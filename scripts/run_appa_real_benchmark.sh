#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-data/APPA-REAL/appa-real-release}"
DATASET_ID="${DATASET_ID:-DS005}"
DATASET_VERSION="${DATASET_VERSION:-official}"
CONFIG_ID="${CONFIG_ID:-CFG001}"
PROVIDER="${PROVIDER:-cuda}"
WANDB_PROJECT="${WANDB_PROJECT:-AgeGenderPredictor}"
WANDB_MODE="${WANDB_MODE:-online}"
RUN_ID="${RUN_ID:-RUN_$(date -u +%Y%m%d_%H%M%S)_APPA_REAL_FULL}"

MANIFEST="${MANIFEST:-manifests/${DATASET_ID}_appa_real.csv}"
VALIDATION_REPORT="${VALIDATION_REPORT:-output/validation/${DATASET_ID}_appa_real_validation.json}"
RESULT_JSON="${RESULT_JSON:-output/${RUN_ID}.json}"
LOG_PATH="${LOG_PATH:-output/logs/${RUN_ID}.log}"

mkdir -p "$(dirname -- "${LOG_PATH}")"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "[appa-real] started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[appa-real] repository: ${REPO_ROOT}"
echo "[appa-real] data: ${DATA_DIR}"
echo "[appa-real] provider: ${PROVIDER} | W&B: ${WANDB_PROJECT} (${WANDB_MODE})"
echo "[appa-real] persistent log: ${LOG_PATH}"

for required_path in \
    "${DATA_DIR}/train" \
    "${DATA_DIR}/valid" \
    "${DATA_DIR}/test" \
    "${DATA_DIR}/gt_train.csv" \
    "${DATA_DIR}/gt_valid.csv" \
    "${DATA_DIR}/gt_test.csv"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "[appa-real] missing required path: ${required_path}" >&2
        exit 1
    fi
done

if [[ "${PROVIDER}" == "cuda" ]]; then
    "${PYTHON_BIN}" -c '
import sys
import onnxruntime as ort
providers = ort.get_available_providers()
print(f"[appa-real] ONNX Runtime providers: {providers}")
if "CUDAExecutionProvider" not in providers:
    sys.exit("CUDAExecutionProvider is unavailable")
'
fi

echo "[1/2] Preparing the complete APPA-REAL manifest"
"${PYTHON_BIN}" -u scripts/prepare_appa_real.py \
    --data-dir "${DATA_DIR}" \
    --dataset-id "${DATASET_ID}" \
    --dataset-version "${DATASET_VERSION}" \
    --progress-every "${PREPARE_PROGRESS_EVERY:-1000}" \
    --output "${MANIFEST}" \
    --validation-report "${VALIDATION_REPORT}"

WANDB_ARGS=(
    --wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-mode "${WANDB_MODE}"
    --wandb-run-name "${RUN_ID}"
    --wandb-tags appa-real exact-age full-dataset no-gender-labels "${PROVIDER}"
)
if [[ -n "${WANDB_ENTITY:-}" ]]; then
    WANDB_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

echo "[2/2] Benchmarking all APPA-REAL train, validation, and test images"
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

echo "[appa-real] complete: ${RESULT_JSON}"
echo "[appa-real] validation: ${VALIDATION_REPORT}"
echo "[appa-real] log: ${LOG_PATH}"
