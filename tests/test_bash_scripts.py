import unittest
from pathlib import Path


class BashScriptTests(unittest.TestCase):
    def test_bash_scripts_follow_new_correction_origin_layout(self):
        base = Path("scripts/bash")
        expected = {
            Path("smart_flip/awq/run_llama3.sh"),
            Path("smart_flip/awq/run_llama31.sh"),
            Path("smart_flip/awq/run_mistral.sh"),
            Path("smart_flip/awq/run_qwen25.sh"),
            Path("smart_flip/flatquant/run_llama3.sh"),
            Path("smart_flip/flatquant/run_llama31.sh"),
            Path("smart_flip/flatquant/run_mistral.sh"),
            Path("smart_flip/flatquant/run_qwen25.sh"),
            Path("bias_correction/awq/run_llama3.sh"),
            Path("bias_correction/awq/run_llama31.sh"),
            Path("bias_correction/awq/run_mistral.sh"),
            Path("bias_correction/awq/run_qwen25.sh"),
            Path("bias_correction/flatquant/run_llama3.sh"),
            Path("bias_correction/flatquant/run_llama31.sh"),
            Path("bias_correction/flatquant/run_mistral.sh"),
            Path("bias_correction/flatquant/run_qwen25.sh"),
        }

        actual = {path.relative_to(base) for path in base.rglob("*.sh")}
        self.assertEqual(actual, expected)

    def test_smart_flip_scripts_define_grid_search_per_origin_method(self):
        scripts = {
            "smart_flip/awq/run_llama3.sh": [
                'FLOAT_ARGS=(',
                'QUANT_BASE_ARGS=(',
                'ORIGIN_METHOD="awq"',
                'POST_CORRECTION="smart_flip"',
                'KNEE_VALUES=(0.0 0.01 0.02 0.03 0.04 0.05)',
                'MAX_FLIP_VALUES=(0.01 0.02 0.03 0.04 0.05)',
                'for knee in "${KNEE_VALUES[@]}"; do',
                'for max_flip in "${MAX_FLIP_VALUES[@]}"; do',
                'main.py float_model',
                'main.py quantize',
                '--origin-method "$ORIGIN_METHOD"',
                '--post-correction "$POST_CORRECTION"',
            ],
            "smart_flip/flatquant/run_llama3.sh": [
                'FLOAT_ARGS=(',
                'QUANT_BASE_ARGS=(',
                'ORIGIN_METHOD="flatquant"',
                'POST_CORRECTION="smart_flip"',
                'BITS_VALUES=(4)',
                'KNEE_VALUES=(0.0 0.01 0.02 0.03 0.04 0.05)',
                'MAX_FLIP_VALUES=(0.01 0.02 0.03 0.04 0.05)',
                'for bits in "${BITS_VALUES[@]}"; do',
                'main.py float_model',
                'main.py quantize',
                '--origin-method "$ORIGIN_METHOD"',
                '--post-correction "$POST_CORRECTION"',
                '--bits "$bits"',
                '--flatquant-epochs "$FLATQUANT_EPOCHS"',
            ],
        }

        for relative_path, snippets in scripts.items():
            content = Path("scripts/bash", relative_path).read_text(encoding="utf-8")
            for snippet in snippets:
                self.assertIn(snippet, content, f"Missing {snippet!r} in {relative_path}")

    def test_bias_correction_scripts_use_single_run_defaults(self):
        scripts = {
            "bias_correction/awq/run_llama3.sh": [
                'ORIGIN_METHOD="awq"',
                'POST_CORRECTION="bias_correction"',
                'BIAS_CORRECTION_SAMPLES="${BIAS_CORRECTION_SAMPLES:-4096}"',
                '--origin-method "$ORIGIN_METHOD"',
                '--post-correction "$POST_CORRECTION"',
                '--bias-correction-samples "$BIAS_CORRECTION_SAMPLES"',
            ],
            "bias_correction/flatquant/run_llama3.sh": [
                'ORIGIN_METHOD="flatquant"',
                'POST_CORRECTION="bias_correction"',
                'BIAS_CORRECTION_SAMPLES="${BIAS_CORRECTION_SAMPLES:-4096}"',
                '--origin-method "$ORIGIN_METHOD"',
                '--post-correction "$POST_CORRECTION"',
                '--flatquant-epochs "$FLATQUANT_EPOCHS"',
            ],
        }

        for relative_path, snippets in scripts.items():
            content = Path("scripts/bash", relative_path).read_text(encoding="utf-8")
            for snippet in snippets:
                self.assertIn(snippet, content, f"Missing {snippet!r} in {relative_path}")


    def test_flatquant_mistral_script_can_skip_float_model(self):
        content = Path("scripts/bash/smart_flip/flatquant/run_mistral.sh").read_text(encoding="utf-8")
        self.assertIn('RUN_FLOAT_MODEL="${RUN_FLOAT_MODEL:-1}"', content)
        self.assertIn('if [ "$RUN_FLOAT_MODEL" = "1" ]; then', content)
        self.assertIn('echo "==> skipping float_model :: ${MODEL_PATH}"', content)
        self.assertIn('RAW_MODEL_DIR="${RAW_MODEL_DIR:-${RESULTS_MODELS_DIR}/${ORIGIN_METHOD}_raw/${RAW_RUN_NAME}}"', content)
        self.assertIn('--flatquant-raw-path "$RAW_MODEL_DIR"', content)
        self.assertIn('"$PYTHON_BIN" main.py float_model', content)


if __name__ == "__main__":
    unittest.main()
