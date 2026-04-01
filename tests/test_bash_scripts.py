import unittest
from pathlib import Path


class BashScriptTests(unittest.TestCase):
    def test_root_bash_wrappers_expose_current_eval_flags(self):
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
            "run_quantize.sh": [
                'POST_CORRECTION="${POST_CORRECTION:-none}"',
                '--post-correction "$POST_CORRECTION"',
                '--bias-correction-samples "$BIAS_CORRECTION_SAMPLES"',
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
        }

        for script_name, snippets in scripts.items():
            content = Path("scripts/bash", script_name).read_text(encoding="utf-8")
            for snippet in snippets:
                self.assertIn(snippet, content, f"Missing '{snippet}' in {script_name}")

    def test_grouped_model_launchers_cover_smart_flip_and_bias_correction(self):
        scripts = {
            "smart_flip/run_llama3_grid.sh": [
                'MODEL_PATH="${MODEL_PATH:-meta-llama/Meta-Llama-3-8B}"',
                'KNEE_VALUES=(0.0 0.01 0.02 0.03 0.04 0.05)',
                'MAX_FLIP_VALUES=(0.01 0.02 0.03 0.04 0.05)',
                'bash scripts/bash/run_float_model.sh',
                'bash scripts/bash/run_raw_quantize.sh',
                'bash scripts/bash/run_flip_quantize.sh',
            ],
            "smart_flip/run_llama31_grid.sh": [
                'MODEL_PATH="${MODEL_PATH:-meta-llama/Meta-Llama-3.1-8B}"',
                'KNEE_VALUES=(0.0 0.01 0.02 0.03 0.04 0.05)',
                'MAX_FLIP_VALUES=(0.01 0.02 0.03 0.04 0.05)',
                'bash scripts/bash/run_float_model.sh',
                'bash scripts/bash/run_raw_quantize.sh',
                'bash scripts/bash/run_flip_quantize.sh',
            ],
            "smart_flip/run_mistral_grid.sh": [
                'MODEL_PATH="${MODEL_PATH:-mistralai/Mistral-7B-v0.3}"',
                'KNEE_VALUES=(0.0 0.01 0.02 0.03 0.04 0.05)',
                'MAX_FLIP_VALUES=(0.01 0.02 0.03 0.04 0.05)',
                'bash scripts/bash/run_float_model.sh',
                'bash scripts/bash/run_raw_quantize.sh',
                'bash scripts/bash/run_flip_quantize.sh',
            ],
            "smart_flip/run_qwen25_grid.sh": [
                'MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B}"',
                'KNEE_VALUES=(0.0 0.01 0.02 0.03 0.04 0.05)',
                'MAX_FLIP_VALUES=(0.01 0.02 0.03 0.04 0.05)',
                'bash scripts/bash/run_float_model.sh',
                'bash scripts/bash/run_raw_quantize.sh',
                'bash scripts/bash/run_flip_quantize.sh',
            ],
            "bias_correction/run_llama3.sh": [
                'MODEL_PATH="${MODEL_PATH:-meta-llama/Meta-Llama-3-8B}"',
                'POST_CORRECTION="${POST_CORRECTION:-bias_correction}"',
                'BIAS_CORRECTION_SAMPLES="${BIAS_CORRECTION_SAMPLES:-4096}"',
                'bash scripts/bash/run_float_model.sh',
                'bash scripts/bash/run_raw_quantize.sh',
                'bash scripts/bash/run_quantize.sh',
            ],
            "bias_correction/run_llama31.sh": [
                'MODEL_PATH="${MODEL_PATH:-meta-llama/Meta-Llama-3.1-8B}"',
                'POST_CORRECTION="${POST_CORRECTION:-bias_correction}"',
                'BIAS_CORRECTION_SAMPLES="${BIAS_CORRECTION_SAMPLES:-4096}"',
                'bash scripts/bash/run_float_model.sh',
                'bash scripts/bash/run_raw_quantize.sh',
                'bash scripts/bash/run_quantize.sh',
            ],
            "bias_correction/run_mistral.sh": [
                'MODEL_PATH="${MODEL_PATH:-mistralai/Mistral-7B-v0.3}"',
                'POST_CORRECTION="${POST_CORRECTION:-bias_correction}"',
                'BIAS_CORRECTION_SAMPLES="${BIAS_CORRECTION_SAMPLES:-4096}"',
                'bash scripts/bash/run_float_model.sh',
                'bash scripts/bash/run_raw_quantize.sh',
                'bash scripts/bash/run_quantize.sh',
            ],
            "bias_correction/run_qwen25.sh": [
                'MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B}"',
                'POST_CORRECTION="${POST_CORRECTION:-bias_correction}"',
                'BIAS_CORRECTION_SAMPLES="${BIAS_CORRECTION_SAMPLES:-4096}"',
                'bash scripts/bash/run_float_model.sh',
                'bash scripts/bash/run_raw_quantize.sh',
                'bash scripts/bash/run_quantize.sh',
            ],
        }

        for relative_path, snippets in scripts.items():
            content = Path("scripts/bash", relative_path).read_text(encoding="utf-8")
            for snippet in snippets:
                self.assertIn(snippet, content, f"Missing '{snippet}' in {relative_path}")


if __name__ == "__main__":
    unittest.main()
