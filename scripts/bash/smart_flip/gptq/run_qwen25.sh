#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B}"
MODELS_ROOT="${MODELS_ROOT:-/models}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESULTS_MODELS_DIR="${RESULTS_MODELS_DIR:-./results/models}"
RESULTS_EVAL_DIR="${RESULTS_EVAL_DIR:-./results/eval}"
CALIBRATION_CACHE_DIR="${CALIBRATION_CACHE_DIR:-./data/cache/calibration}"
EVAL_CACHE_DIR="${EVAL_CACHE_DIR:-./data/cache/eval}"
CALIB_DATASET="${CALIB_DATASET:-c4}"
N_CALIB="${N_CALIB:-128}"
CALIB_SEQLEN="${CALIB_SEQLEN:-2048}"
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
RUN_FLOAT_MODEL="${RUN_FLOAT_MODEL:-1}"
RUN_RAW_QUANTIZE="${RUN_RAW_QUANTIZE:-1}"

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

GPTQ_PERCDAMP="${GPTQ_PERCDAMP:-0.01}"
GPTQ_SYM="${GPTQ_SYM:-1}"
GPTQ_ACT_ORDER="${GPTQ_ACT_ORDER:-0}"
GPTQ_TRUE_SEQUENTIAL="${GPTQ_TRUE_SEQUENTIAL:-1}"
GPTQ_STATIC_GROUPS="${GPTQ_STATIC_GROUPS:-0}"
GPTQ_MSE="${GPTQ_MSE:-0}"

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

ORIGIN_METHOD="gptq"
POST_CORRECTION="smart_flip"
MODEL_SLUG="${MODEL_PATH##*/}"
FLOAT_RUN_NAME="${FLOAT_RUN_NAME:-${ORIGIN_METHOD}_float_${MODEL_SLUG}}"
RAW_RUN_NAME="${RAW_RUN_NAME:-${ORIGIN_METHOD}_raw_${MODEL_SLUG}}"
RAW_MODEL_DIR="${RAW_MODEL_DIR:-${RESULTS_MODELS_DIR}/${ORIGIN_METHOD}_raw/${RAW_RUN_NAME}}"
BITS_VALUES=(4)
KNEE_VALUES=(0.0 0.01 0.02 0.03 0.04 0.05)
MAX_FLIP_VALUES=(0.01 0.02 0.03 0.04 0.05)

if [ "$RUN_FLOAT_MODEL" = "1" ]; then
  echo "==> float_model :: ${MODEL_PATH}"
  "$PYTHON_BIN" main.py float_model "${FLOAT_ARGS[@]}" --run-name "$FLOAT_RUN_NAME"
else
  echo "==> skipping float_model :: ${MODEL_PATH}"
fi

if [ "$RUN_RAW_QUANTIZE" = "1" ]; then
  echo "==> raw_quantize :: ${MODEL_PATH} :: origin=${ORIGIN_METHOD}"
  RAW_ARGS=(
    "${QUANT_BASE_ARGS[@]}"
    --origin-method "$ORIGIN_METHOD"
    --post-correction none
    --run-name "$RAW_RUN_NAME"
    --bits "4"
  )
  add_gptq_args RAW_ARGS
  "$PYTHON_BIN" main.py quantize "${RAW_ARGS[@]}"
else
  if [ ! -f "$RAW_MODEL_DIR/gptq_raw_artifacts.pt" ]; then
    echo "Missing raw GPTQ artifacts at ${RAW_MODEL_DIR}/gptq_raw_artifacts.pt" >&2
    exit 1
  fi
  echo "==> skipping raw_quantize :: ${MODEL_PATH} :: using existing raw model at ${RAW_MODEL_DIR}"
fi

for bits in "${BITS_VALUES[@]}"; do
  for knee in "${KNEE_VALUES[@]}"; do
    for max_flip in "${MAX_FLIP_VALUES[@]}"; do
      run_name="${ORIGIN_METHOD}_smart_flip_${MODEL_SLUG}_b${bits}_k${knee}_f${max_flip}"
      echo "==> smart_flip :: ${MODEL_PATH} :: origin=${ORIGIN_METHOD} :: bits=${bits} :: knee=${knee} :: max_flip=${max_flip}"
      QUANT_ARGS=(
        "${QUANT_BASE_ARGS[@]}"
        --origin-method "$ORIGIN_METHOD"
        --post-correction "$POST_CORRECTION"
        --bits "$bits"
        --knee-tolerance "$knee"
        --max-flip-percent "$max_flip"
        --gptq-raw-path "$RAW_MODEL_DIR"
        --run-name "$run_name"
      )
      add_gptq_args QUANT_ARGS
      "$PYTHON_BIN" main.py quantize "${QUANT_ARGS[@]}"
    done
  done
done
