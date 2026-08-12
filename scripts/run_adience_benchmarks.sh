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
    echo "[adience] FOLD_SET must be all or frontal, got: ${FOLD_SET}" >&2
    exit 1
fi

RUN_STAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_GROUP_ID="${RUN_GROUP_ID:-RUN_${RUN_STAMP}_ADIENCE}"
ALIGNED_RUN_ID="${ALIGNED_RUN_ID:-${RUN_GROUP_ID}_ALIGNED_FULL}"
FACES_RUN_ID="${FACES_RUN_ID:-${RUN_GROUP_ID}_FACES_FULL}"

ALIGNED_MANIFEST_DEFAULT="manifests/${DATASET_ID}_adience_aligned${FOLD_SUFFIX}.csv"
FACES_MANIFEST_DEFAULT="manifests/${DATASET_ID}_adience_faces${FOLD_SUFFIX}.csv"
VALIDATION_DIR="output/validation"
ALIGNED_REPORT_NAME="${DATASET_ID}_adience_aligned${FOLD_SUFFIX}_validation.json"
FACES_REPORT_NAME="${DATASET_ID}_adience_faces${FOLD_SUFFIX}_validation.json"
ALIGNED_REPORT_DEFAULT="${VALIDATION_DIR}/${ALIGNED_REPORT_NAME}"
FACES_REPORT_DEFAULT="${VALIDATION_DIR}/${FACES_REPORT_NAME}"
ALIGNED_MANIFEST="${ALIGNED_MANIFEST:-${ALIGNED_MANIFEST_DEFAULT}}"
FACES_MANIFEST="${FACES_MANIFEST:-${FACES_MANIFEST_DEFAULT}}"
ALIGNED_VALIDATION_REPORT="${ALIGNED_VALIDATION_REPORT:-${ALIGNED_REPORT_DEFAULT}}"
FACES_VALIDATION_REPORT="${FACES_VALIDATION_REPORT:-${FACES_REPORT_DEFAULT}}"
ALIGNED_RESULT_JSON="${ALIGNED_RESULT_JSON:-output/${ALIGNED_RUN_ID}.json}"
FACES_RESULT_JSON="${FACES_RESULT_JSON:-output/${FACES_RUN_ID}.json}"

LOG_PATH="${LOG_PATH:-output/logs/${RUN_GROUP_ID}.log}"
mkdir -p "$(dirname -- "${LOG_PATH}")"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "[adience] started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[adience] repository: ${REPO_ROOT}"
echo "[adience] data: ${DATA_DIR}"
echo "[adience] fold set: ${FOLD_SET} | age policy: ${AGE_LABEL_POLICY}"
echo "[adience] provider: ${PROVIDER} | W&B project: ${WANDB_PROJECT}"
echo "[adience] persistent log: ${LOG_PATH}"

for required_path in "${DATA_DIR}/aligned" "${DATA_DIR}/faces" "${DATA_DIR}/labels"; do
    if [[ ! -d "${required_path}" ]]; then
        echo "[adience] missing required directory: ${required_path}" >&2
        exit 1
    fi
done

for fold_index in 0 1 2 3 4; do
    label_path="${DATA_DIR}/labels/${LABEL_PREFIX}_${fold_index}_data.txt"
    if [[ ! -f "${label_path}" ]]; then
        echo "[adience] missing required label file: ${label_path}" >&2
        exit 1
    fi
done

if [[ "${PROVIDER}" == "cuda" ]]; then
    "${PYTHON_BIN}" -c '
import sys
import onnxruntime as ort
providers = ort.get_available_providers()
print(f"[adience] ONNX Runtime providers: {providers}")
if "CUDAExecutionProvider" not in providers:
    sys.exit("CUDAExecutionProvider is unavailable")
'
fi

step=1
for variant in aligned faces; do
    if [[ "${variant}" == "aligned" ]]; then
        manifest="${ALIGNED_MANIFEST}"
        validation_report="${ALIGNED_VALIDATION_REPORT}"
        result_json="${ALIGNED_RESULT_JSON}"
        run_id="${ALIGNED_RUN_ID}"
    else
        manifest="${FACES_MANIFEST}"
        validation_report="${FACES_VALIDATION_REPORT}"
        result_json="${FACES_RESULT_JSON}"
        run_id="${FACES_RUN_ID}"
    fi

    echo "[${step}/4] Preparing Adience ${variant} manifest"
    "${PYTHON_BIN}" -u scripts/prepare_adience.py \
        --data-dir "${DATA_DIR}" \
        --image-variant "${variant}" \
        --fold-set "${FOLD_SET}" \
        --dataset-id "${DATASET_ID}" \
        --dataset-version "official-${variant}" \
        --age-label-policy "${AGE_LABEL_POLICY}" \
        --progress-every "${PREPARE_PROGRESS_EVERY:-1000}" \
        --output "${manifest}" \
        --validation-report "${validation_report}"
    step=$((step + 1))

    WANDB_ARGS=(
        --wandb
        --wandb-project "${WANDB_PROJECT}"
        --wandb-mode "${WANDB_MODE}"
        --wandb-run-name "${run_id}"
        --wandb-tags adience "${variant}" full-dataset "${PROVIDER}" \
            "${FOLD_SET}" interval-aware
    )
    if [[ -n "${WANDB_ENTITY:-}" ]]; then
        WANDB_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
    fi

    echo "[${step}/4] Benchmarking the complete Adience ${variant} dataset"
    "${PYTHON_BIN}" -u benchmark/benchmark_runtime.py \
        --manifest "${manifest}" \
        --dataset-id "${DATASET_ID}" \
        --run-id "${run_id}" \
        --config-id "${CONFIG_ID}" \
        --provider "${PROVIDER}" \
        --sampling all \
        --limit 0 \
        --warmup-runs "${WARMUP_RUNS:-10}" \
        --progress-every "${BENCHMARK_PROGRESS_EVERY:-100}" \
        --output "${result_json}" \
        "${WANDB_ARGS[@]}"
    step=$((step + 1))

    echo "[adience] ${variant} complete: ${result_json}"
done

echo "[adience] all runs complete: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[adience] aligned result: ${ALIGNED_RESULT_JSON}"
echo "[adience] faces result: ${FACES_RESULT_JSON}"
echo "[adience] log: ${LOG_PATH}"
