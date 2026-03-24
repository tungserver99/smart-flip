#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-}"
RUN_NAME="${RUN_NAME:-float_eval}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -z "$MODEL_PATH" ]]; then
  echo "MODEL_PATH is required. Example: MODEL_PATH=mistralai/Mistral-7B-v0.3 bash scripts/bash/run_float_model.sh"
  exit 1
fi

"$PYTHON_BIN" main.py float_model \
  --model-path "$MODEL_PATH" \
  --run-name "$RUN_NAME"
