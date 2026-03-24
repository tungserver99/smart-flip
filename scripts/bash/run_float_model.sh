#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-Mistral-7B-v0.3}"
MODELS_ROOT="${MODELS_ROOT:-/models}"
RUN_NAME="${RUN_NAME:-float_eval}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" main.py float_model \
  --model-path "$MODEL_PATH" \
  --models-root "$MODELS_ROOT" \
  --run-name "$RUN_NAME"
