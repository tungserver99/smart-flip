#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-./models/Mistral-7B-v0.3}"
RUN_NAME="${RUN_NAME:-awq_raw_run}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" main.py raw_quantize \
  --model-path "$MODEL_PATH" \
  --run-name "$RUN_NAME"
