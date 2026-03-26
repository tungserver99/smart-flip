#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-Mistral-7B-v0.3}"
MODELS_ROOT="${MODELS_ROOT:-/models}"
RUN_NAME="${RUN_NAME:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LM_EVAL_TASK_PRESET="${LM_EVAL_TASK_PRESET:-extended}"
INCLUDE_LM_EVAL="${INCLUDE_LM_EVAL:-1}"
INCLUDE_C4="${INCLUDE_C4:-1}"
USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-smartflip}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

EXTRA_ARGS=(
  --lm-eval-task-preset "$LM_EVAL_TASK_PRESET"
)

if [ -n "$RUN_NAME" ]; then
  EXTRA_ARGS+=(--run-name "$RUN_NAME")
fi

if [ "$INCLUDE_LM_EVAL" != "1" ]; then
  EXTRA_ARGS+=(--no-lm-eval)
fi

if [ "$INCLUDE_C4" != "1" ]; then
  EXTRA_ARGS+=(--no-c4)
fi

if [ "$USE_WANDB" = "1" ]; then
  EXTRA_ARGS+=(--use-wandb --wandb-project "$WANDB_PROJECT")
  if [ -n "$WANDB_ENTITY" ]; then
    EXTRA_ARGS+=(--wandb-entity "$WANDB_ENTITY")
  fi
fi

"$PYTHON_BIN" main.py float_model   --model-path "$MODEL_PATH"   --models-root "$MODELS_ROOT"   "${EXTRA_ARGS[@]}"
