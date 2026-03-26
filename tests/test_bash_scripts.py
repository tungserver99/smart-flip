import unittest
from pathlib import Path


class BashScriptTests(unittest.TestCase):
    def test_bash_wrappers_expose_current_eval_flags(self):
        scripts = {
            "run_float_model.sh": [
                'LM_EVAL_TASK_PRESET="${LM_EVAL_TASK_PRESET:-extended}"',
                'INCLUDE_LM_EVAL="${INCLUDE_LM_EVAL:-1}"',
                'INCLUDE_C4="${INCLUDE_C4:-1}"',
                'USE_WANDB="${USE_WANDB:-1}"',
                'WANDB_PROJECT="${WANDB_PROJECT:-smartflip}"',
                '--lm-eval-task-preset "$LM_EVAL_TASK_PRESET"',
                'EXTRA_ARGS+=(--no-lm-eval)',
                'EXTRA_ARGS+=(--no-c4)',
                'EXTRA_ARGS+=(--use-wandb --wandb-project "$WANDB_PROJECT")',
            ],
            "run_raw_quantize.sh": [
                'LM_EVAL_TASK_PRESET="${LM_EVAL_TASK_PRESET:-extended}"',
                'INCLUDE_LM_EVAL="${INCLUDE_LM_EVAL:-1}"',
                'INCLUDE_C4="${INCLUDE_C4:-1}"',
                'USE_WANDB="${USE_WANDB:-1}"',
                'WANDB_PROJECT="${WANDB_PROJECT:-smartflip}"',
                '--lm-eval-task-preset "$LM_EVAL_TASK_PRESET"',
                'EXTRA_ARGS+=(--use-wandb --wandb-project "$WANDB_PROJECT")',
            ],
            "run_flip_quantize.sh": [
                'LM_EVAL_TASK_PRESET="${LM_EVAL_TASK_PRESET:-extended}"',
                'INCLUDE_LM_EVAL="${INCLUDE_LM_EVAL:-1}"',
                'INCLUDE_C4="${INCLUDE_C4:-1}"',
                'USE_WANDB="${USE_WANDB:-1}"',
                'WANDB_PROJECT="${WANDB_PROJECT:-smartflip}"',
                'KNEE_TOLERANCE="${KNEE_TOLERANCE:-0.0}"',
                'MAX_FLIP_PERCENT="${MAX_FLIP_PERCENT:-0.05}"',
                '--knee-tolerance "$KNEE_TOLERANCE"',
                '--max-flip-percent "$MAX_FLIP_PERCENT"',
                'EXTRA_ARGS+=(--use-wandb --wandb-project "$WANDB_PROJECT")',
            ],
            "run_mistral_grid.sh": [
                'MODEL_PATH="${MODEL_PATH:-mistralai/Mistral-7B-v0.3}"',
                'KNEE_VALUES=(0.0 0.01 0.02 0.03 0.04 0.05)',
                'MAX_FLIP_VALUES=(0.01 0.02 0.03 0.04 0.05)',
                'bash scripts/bash/run_float_model.sh',
                'bash scripts/bash/run_raw_quantize.sh',
                'bash scripts/bash/run_flip_quantize.sh',
            ],
            "run_llama_grid.sh": [
                'MODEL_PATH="${MODEL_PATH:-meta-llama/Meta-Llama-3.1-8B}"',
                'KNEE_VALUES=(0.0 0.01 0.02 0.03 0.04 0.05)',
                'MAX_FLIP_VALUES=(0.01 0.02 0.03 0.04 0.05)',
                'bash scripts/bash/run_float_model.sh',
                'bash scripts/bash/run_raw_quantize.sh',
                'bash scripts/bash/run_flip_quantize.sh',
            ],
            "run_llama3_grid.sh": [
                'MODEL_PATH="${MODEL_PATH:-meta-llama/Meta-Llama-3-8B}"',
                'KNEE_VALUES=(0.0 0.01 0.02 0.03 0.04 0.05)',
                'MAX_FLIP_VALUES=(0.01 0.02 0.03 0.04 0.05)',
                'bash scripts/bash/run_float_model.sh',
                'bash scripts/bash/run_raw_quantize.sh',
                'bash scripts/bash/run_flip_quantize.sh',
            ],            "run_qwen_grid.sh": [
                'MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B}"',
                'KNEE_VALUES=(0.0 0.01 0.02 0.03 0.04 0.05)',
                'MAX_FLIP_VALUES=(0.01 0.02 0.03 0.04 0.05)',
                'bash scripts/bash/run_float_model.sh',
                'bash scripts/bash/run_raw_quantize.sh',
                'bash scripts/bash/run_flip_quantize.sh',
            ],
        }

        for script_name, snippets in scripts.items():
            content = Path("scripts/bash", script_name).read_text(encoding="utf-8")
            for snippet in snippets:
                self.assertIn(snippet, content, f"Missing '{snippet}' in {script_name}")


if __name__ == "__main__":
    unittest.main()
