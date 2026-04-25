#!/usr/bin/env bash
# set -euo pipefail

# MODELS_ROOT="${MODELS_ROOT:-/models}"
# PYTHON_BIN="${PYTHON_BIN:-python}"
# RESULTS_MODELS_DIR="${RESULTS_MODELS_DIR:-./results/models}"
# RESULTS_EVAL_DIR="${RESULTS_EVAL_DIR:-./results/eval}"
# CALIBRATION_CACHE_DIR="${CALIBRATION_CACHE_DIR:-./data/cache/calibration}"
# EVAL_CACHE_DIR="${EVAL_CACHE_DIR:-./data/cache/eval}"
# CALIB_DATASET="${CALIB_DATASET:-c4}"
# N_CALIB="${N_CALIB:-128}"
# CALIB_SEQLEN="${CALIB_SEQLEN:-2048}"
# GROUP_SIZE="${GROUP_SIZE:--1}"
# SEED="${SEED:-42}"
# STRIDE="${STRIDE:-512}"
# MAX_LENGTH="${MAX_LENGTH:-2048}"
# C4_SAMPLES="${C4_SAMPLES:-500}"
# LM_EVAL_TASK_PRESET="${LM_EVAL_TASK_PRESET:-extended}"
# INCLUDE_LM_EVAL="${INCLUDE_LM_EVAL:-1}"
# INCLUDE_C4="${INCLUDE_C4:-1}"
# USE_WANDB="${USE_WANDB:-1}"
# WANDB_PROJECT="${WANDB_PROJECT:-egbc}"
# WANDB_ENTITY="${WANDB_ENTITY:-}"
# RUN_FLOAT_MODEL="${RUN_FLOAT_MODEL:-0}"
# BITS="${BITS:-4}"
# KNEE_TOLERANCE="${KNEE_TOLERANCE:-0.0}"
# MAX_FLIP_PERCENT="${MAX_FLIP_PERCENT:-0.05}"

# GPTQ_PERCDAMP="${GPTQ_PERCDAMP:-0.01}"
# GPTQ_SYM="${GPTQ_SYM:-0}"
# GPTQ_ACT_ORDER="${GPTQ_ACT_ORDER:-1}"
# GPTQ_TRUE_SEQUENTIAL="${GPTQ_TRUE_SEQUENTIAL:-1}"
# GPTQ_STATIC_GROUPS="${GPTQ_STATIC_GROUPS:-0}"
# GPTQ_MSE="${GPTQ_MSE:-0}"

# ORIGIN_METHOD="gptq"
# POST_CORRECTION="smart_flip"

# MODEL_PATHS=(
#   # "meta-llama/Meta-Llama-3-8B"  # Already completed; keep commented to skip rerunning it.
#   "meta-llama/Llama-3.1-8B"
#   "mistralai/Mistral-7B-v0.3"
#   "Qwen/Qwen2.5-7B"
# )

# add_gptq_args() {
#   local -n args_ref=$1
#   args_ref+=(--gptq-percdamp "$GPTQ_PERCDAMP")
#   if [ "$GPTQ_SYM" = "1" ]; then
#     args_ref+=(--gptq-sym)
#   fi
#   if [ "$GPTQ_ACT_ORDER" = "1" ]; then
#     args_ref+=(--gptq-act-order)
#   fi
#   if [ "$GPTQ_TRUE_SEQUENTIAL" = "1" ]; then
#     args_ref+=(--gptq-true-sequential)
#   else
#     args_ref+=(--no-gptq-true-sequential)
#   fi
#   if [ "$GPTQ_STATIC_GROUPS" = "1" ]; then
#     args_ref+=(--gptq-static-groups)
#   fi
#   if [ "$GPTQ_MSE" = "1" ]; then
#     args_ref+=(--gptq-mse)
#   fi
# }

# for MODEL_PATH in "${MODEL_PATHS[@]}"; do
#   MODEL_SLUG="${MODEL_PATH##*/}"
#   FLOAT_RUN_NAME="${ORIGIN_METHOD}_float_${MODEL_SLUG}"
#   SMART_FLIP_RUN_NAME="${ORIGIN_METHOD}_smart_flip_${MODEL_SLUG}_b${BITS}_k${KNEE_TOLERANCE}_f${MAX_FLIP_PERCENT}"

#   FLOAT_ARGS=(
#     --model-path "$MODEL_PATH"
#     --models-root "$MODELS_ROOT"
#     --results-eval-dir "$RESULTS_EVAL_DIR"
#     --eval-cache-dir "$EVAL_CACHE_DIR"
#     --seed "$SEED"
#     --stride "$STRIDE"
#     --max-length "$MAX_LENGTH"
#     --c4-samples "$C4_SAMPLES"
#     --lm-eval-task-preset "$LM_EVAL_TASK_PRESET"
#   )

#   QUANT_BASE_ARGS=(
#     --model-path "$MODEL_PATH"
#     --models-root "$MODELS_ROOT"
#     --results-models-dir "$RESULTS_MODELS_DIR"
#     --results-eval-dir "$RESULTS_EVAL_DIR"
#     --calibration-cache-dir "$CALIBRATION_CACHE_DIR"
#     --eval-cache-dir "$EVAL_CACHE_DIR"
#     --calib-dataset "$CALIB_DATASET"
#     --n-calib "$N_CALIB"
#     --calib-seqlen "$CALIB_SEQLEN"
#     --group-size "$GROUP_SIZE"
#     --seed "$SEED"
#     --stride "$STRIDE"
#     --max-length "$MAX_LENGTH"
#     --c4-samples "$C4_SAMPLES"
#     --lm-eval-task-preset "$LM_EVAL_TASK_PRESET"
#   )

#   if [ "$INCLUDE_LM_EVAL" != "1" ]; then
#     FLOAT_ARGS+=(--no-lm-eval)
#     QUANT_BASE_ARGS+=(--no-lm-eval)
#   fi

#   if [ "$INCLUDE_C4" != "1" ]; then
#     FLOAT_ARGS+=(--no-c4)
#     QUANT_BASE_ARGS+=(--no-c4)
#   fi

#   if [ "$USE_WANDB" = "1" ]; then
#     FLOAT_ARGS+=(--use-wandb --wandb-project "$WANDB_PROJECT")
#     QUANT_BASE_ARGS+=(--use-wandb --wandb-project "$WANDB_PROJECT")
#     if [ -n "$WANDB_ENTITY" ]; then
#       FLOAT_ARGS+=(--wandb-entity "$WANDB_ENTITY")
#       QUANT_BASE_ARGS+=(--wandb-entity "$WANDB_ENTITY")
#     fi
#   fi

#   if [ "$RUN_FLOAT_MODEL" = "1" ]; then
#     echo "==> float_model :: ${MODEL_PATH}"
#     "$PYTHON_BIN" main.py float_model "${FLOAT_ARGS[@]}" --run-name "$FLOAT_RUN_NAME"
#   else
#     echo "==> skipping float_model :: ${MODEL_PATH}"
#   fi

#   echo "==> smart_flip :: ${MODEL_PATH} :: origin=${ORIGIN_METHOD} :: bits=${BITS} :: knee=${KNEE_TOLERANCE} :: max_flip=${MAX_FLIP_PERCENT}"
#   QUANT_ARGS=(
#     "${QUANT_BASE_ARGS[@]}"
#     --origin-method "$ORIGIN_METHOD"
#     --post-correction "$POST_CORRECTION"
#     --bits "$BITS"
#     --knee-tolerance "$KNEE_TOLERANCE"
#     --max-flip-percent "$MAX_FLIP_PERCENT"
#     --run-name "$SMART_FLIP_RUN_NAME"
#   )
#   add_gptq_args QUANT_ARGS
#   "$PYTHON_BIN" main.py quantize "${QUANT_ARGS[@]}"
# done

set -euo pipefail

MODELS_ROOT="${MODELS_ROOT:-/models}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESULTS_MODELS_DIR="${RESULTS_MODELS_DIR:-./results/models}"
RESULTS_EVAL_DIR="${RESULTS_EVAL_DIR:-./results/eval}"
CALIBRATION_CACHE_DIR="${CALIBRATION_CACHE_DIR:-./data/cache/calibration}"
EVAL_CACHE_DIR="${EVAL_CACHE_DIR:-./data/cache/eval}"
CALIB_DATASET="${CALIB_DATASET:-c4}"
N_CALIB="${N_CALIB:-128}"
CALIB_SEQLEN="${CALIB_SEQLEN:-2048}"
GROUP_SIZE="${GROUP_SIZE:--1}"
SEED="${SEED:-42}"
STRIDE="${STRIDE:-512}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
C4_SAMPLES="${C4_SAMPLES:-500}"
LM_EVAL_TASK_PRESET="${LM_EVAL_TASK_PRESET:-extended}"
INCLUDE_LM_EVAL="${INCLUDE_LM_EVAL:-1}"
INCLUDE_C4="${INCLUDE_C4:-1}"
USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-egbc}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
RUN_FLOAT_MODEL="${RUN_FLOAT_MODEL:-0}"
BITS="${BITS:-4}"
KNEE_TOLERANCE="${KNEE_TOLERANCE:-0.0}"
MAX_FLIP_PERCENT="${MAX_FLIP_PERCENT:-0.05}"

GPTQ_PERCDAMP="${GPTQ_PERCDAMP:-0.01}"
GPTQ_SYM="${GPTQ_SYM:-0}"
GPTQ_ACT_ORDER="${GPTQ_ACT_ORDER:-1}"
GPTQ_TRUE_SEQUENTIAL="${GPTQ_TRUE_SEQUENTIAL:-1}"
GPTQ_STATIC_GROUPS="${GPTQ_STATIC_GROUPS:-0}"
GPTQ_MSE="${GPTQ_MSE:-0}"

ORIGIN_METHOD="gptq"
SMART_FLIP_POST_CORRECTION="smart_flip"

MODEL_PATHS=(
  # "meta-llama/Meta-Llama-3-8B"
  # "meta-llama/Llama-3.1-8B"
  "mistralai/Mistral-7B-v0.3"
  "Qwen/Qwen2.5-7B"
)

add_gptq_args() {
  local -n args_ref=$1
  args_ref+=(--gptq-percdamp "$GPTQ_PERCDAMP")

  if [ "$GPTQ_SYM" = "1" ]; then
    args_ref+=(--gptq-sym)
  fi
  if [ "$GPTQ_ACT_ORDER" = "1" ]; then
    args_ref+=(--gptq-act-order)
  fi
  if [ "$GPTQ_TRUE_SEQUENTIAL" = "1" ]; then
    args_ref+=(--gptq-true-sequential)
  else
    args_ref+=(--no-gptq-true-sequential)
  fi
  if [ "$GPTQ_STATIC_GROUPS" = "1" ]; then
    args_ref+=(--gptq-static-groups)
  fi
  if [ "$GPTQ_MSE" = "1" ]; then
    args_ref+=(--gptq-mse)
  fi
}

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
  MODEL_SLUG="${MODEL_PATH##*/}"

  FLOAT_RUN_NAME="${ORIGIN_METHOD}_float_${MODEL_SLUG}"
  RAW_RUN_NAME="${ORIGIN_METHOD}_raw_${MODEL_SLUG}_b${BITS}"
  SMART_FLIP_RUN_NAME="${ORIGIN_METHOD}_smart_flip_${MODEL_SLUG}_b${BITS}_k${KNEE_TOLERANCE}_f${MAX_FLIP_PERCENT}"

  FLOAT_ARGS=(
    --model-path "$MODEL_PATH"
    --models-root "$MODELS_ROOT"
    --results-eval-dir "$RESULTS_EVAL_DIR"
    --eval-cache-dir "$EVAL_CACHE_DIR"
    --seed "$SEED"
    --stride "$STRIDE"
    --max-length "$MAX_LENGTH"
    --c4-samples "$C4_SAMPLES"
    --lm-eval-task-preset "$LM_EVAL_TASK_PRESET"
  )

  QUANT_BASE_ARGS=(
    --model-path "$MODEL_PATH"
    --models-root "$MODELS_ROOT"
    --results-models-dir "$RESULTS_MODELS_DIR"
    --results-eval-dir "$RESULTS_EVAL_DIR"
    --calibration-cache-dir "$CALIBRATION_CACHE_DIR"
    --eval-cache-dir "$EVAL_CACHE_DIR"
    --calib-dataset "$CALIB_DATASET"
    --n-calib "$N_CALIB"
    --calib-seqlen "$CALIB_SEQLEN"
    --group-size "$GROUP_SIZE"
    --seed "$SEED"
    --stride "$STRIDE"
    --max-length "$MAX_LENGTH"
    --c4-samples "$C4_SAMPLES"
    --lm-eval-task-preset "$LM_EVAL_TASK_PRESET"
  )

  if [ "$INCLUDE_LM_EVAL" != "1" ]; then
    FLOAT_ARGS+=(--no-lm-eval)
    QUANT_BASE_ARGS+=(--no-lm-eval)
  fi

  if [ "$INCLUDE_C4" != "1" ]; then
    FLOAT_ARGS+=(--no-c4)
    QUANT_BASE_ARGS+=(--no-c4)
  fi

  if [ "$USE_WANDB" = "1" ]; then
    FLOAT_ARGS+=(--use-wandb --wandb-project "$WANDB_PROJECT")
    QUANT_BASE_ARGS+=(--use-wandb --wandb-project "$WANDB_PROJECT")

    if [ -n "$WANDB_ENTITY" ]; then
      FLOAT_ARGS+=(--wandb-entity "$WANDB_ENTITY")
      QUANT_BASE_ARGS+=(--wandb-entity "$WANDB_ENTITY")
    fi
  fi

  if [ "$RUN_FLOAT_MODEL" = "1" ]; then
    echo "==> float_model :: ${MODEL_PATH}"
    "$PYTHON_BIN" main.py float_model "${FLOAT_ARGS[@]}" --run-name "$FLOAT_RUN_NAME"
  else
    echo "==> skipping float_model :: ${MODEL_PATH}"
  fi

  echo "==> gptq_raw :: ${MODEL_PATH} :: bits=${BITS}"
  RAW_ARGS=(
    "${QUANT_BASE_ARGS[@]}"
    --origin-method "$ORIGIN_METHOD"
    --bits "$BITS"
    --run-name "$RAW_RUN_NAME"
  )
  add_gptq_args RAW_ARGS
  "$PYTHON_BIN" main.py quantize "${RAW_ARGS[@]}"

  echo "==> smart_flip :: ${MODEL_PATH} :: origin=${ORIGIN_METHOD} :: bits=${BITS} :: knee=${KNEE_TOLERANCE} :: max_flip=${MAX_FLIP_PERCENT}"
  SMART_FLIP_ARGS=(
    "${QUANT_BASE_ARGS[@]}"
    --origin-method "$ORIGIN_METHOD"
    --post-correction "$SMART_FLIP_POST_CORRECTION"
    --bits "$BITS"
    --knee-tolerance "$KNEE_TOLERANCE"
    --max-flip-percent "$MAX_FLIP_PERCENT"
    --run-name "$SMART_FLIP_RUN_NAME"
  )
  add_gptq_args SMART_FLIP_ARGS
  "$PYTHON_BIN" main.py quantize "${SMART_FLIP_ARGS[@]}"
done