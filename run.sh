#!/usr/bin/env bash
# Local dev launcher (conda env: vision_worker).
#
# PYTHONNOUSERSITE=1 is required: ~/.local/.../onnxruntime shadows the env's
# real onnxruntime as a namespace package, breaking `import onnxruntime`.
set -euo pipefail

PY=/home/naiscorp/miniconda3/envs/vision_worker/bin/python
export PYTHONNOUSERSITE=1
export MODELS_DIR="${MODELS_DIR:-models}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

exec "$PY" -m uvicorn app.main:app --host "$HOST" --port "$PORT" "$@"
