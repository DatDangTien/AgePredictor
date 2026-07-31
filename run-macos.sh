#!/usr/bin/env bash
# Local macOS launcher. Activate the Conda environment before running.

set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "Activate Conda first: conda activate agepredictor"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PY="$CONDA_PREFIX/bin/python"

export PYTHONNOUSERSITE=1
export MODELS_DIR="${MODELS_DIR:-$SCRIPT_DIR/models}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

exec "$PY" -m uvicorn app.main:app --host "$HOST" --port "$PORT" "$@"