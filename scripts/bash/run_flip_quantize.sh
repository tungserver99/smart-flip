#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-Mistral-7B-v0.3}"
MODELS_ROOT="${MODELS_ROOT:-/models}"
ORIGIN_METHOD="${ORIGIN_METHOD:-awq}"
RUN_NAME="${RUN_NAME:-awq_flip_run}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" main.py flip_quantize \
  --model-path "$MODEL_PATH" \
  --models-root "$MODELS_ROOT" \
  --origin-method "$ORIGIN_METHOD" \
  --run-name "$RUN_NAME"
