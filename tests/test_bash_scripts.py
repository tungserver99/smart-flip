#!/usr/bin/env python
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
            Path("smart_flip/gptq/run_all_models_single_setting.sh"),
            Path("smart_flip/gptq/run_llama3.sh"),
            Path("smart_flip/gptq/run_llama31.sh"),
            Path("smart_flip/gptq/run_mistral.sh"),
            Path("smart_flip/gptq/run_qwen25.sh"),
            Path("smart_flip/flatquant/run_llama3.sh"),
            Path("smart_flip/flatquant/run_llama31.sh"),
            Path("smart_flip/flatquant/run_mistral.sh"),
            Path("smart_flip/flatquant/run_qwen25.sh"),
            Path("bias_correction/awq/run_llama3.sh"),
            Path("bias_correction/awq/run_llama31.sh"),
            Path("bias_correction/awq/run_mistral.sh"),
            Path("bias_correction/awq/run_qwen25.sh"),
            Path("bias_correction/gptq/run_llama3.sh"),
            Path("bias_correction/gptq/run_llama31.sh"),
            Path("bias_correction/gptq/run_mistral.sh"),
            Path("bias_correction/gptq/run_qwen25.sh"),
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
                'MAX_FLIP_VALUES=(0.05 0.02 0.03 0.04 0.01)',
                'for bits in "${BITS_VALUES[@]}"; do',
                'main.py float_model',
                'main.py quantize',
                '--origin-method "$ORIGIN_METHOD"',
                '--post-correction "$POST_CORRECTION"',
                '--bits "$bits"',
                '--flatquant-epochs "$FLATQUANT_EPOCHS"',
            ],
            "smart_flip/gptq/run_llama3.sh": [
                'FLOAT_ARGS=(',
                'QUANT_BASE_ARGS=(',
                'ORIGIN_METHOD="gptq"',
                'POST_CORRECTION="smart_flip"',
                'GPTQ_PERCDAMP="${GPTQ_PERCDAMP:-0.01}"',
                'KNEE_VALUES=(0.0 0.01 0.02 0.03 0.04 0.05)',
                'MAX_FLIP_VALUES=(0.01 0.02 0.03 0.04 0.05)',
                'for knee in "${KNEE_VALUES[@]}"; do',
                'for max_flip in "${MAX_FLIP_VALUES[@]}"; do',
                '--origin-method "$ORIGIN_METHOD"',
                '--post-correction "$POST_CORRECTION"',
                '--gptq-percdamp "$GPTQ_PERCDAMP"',
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
            "bias_correction/gptq/run_llama3.sh": [
                'ORIGIN_METHOD="gptq"',
                'POST_CORRECTION="bias_correction"',
                'BIAS_CORRECTION_SAMPLES="${BIAS_CORRECTION_SAMPLES:-4096}"',
                'GPTQ_PERCDAMP="${GPTQ_PERCDAMP:-0.01}"',
                '--origin-method "$ORIGIN_METHOD"',
                '--post-correction "$POST_CORRECTION"',
                '--bias-correction-samples "$BIAS_CORRECTION_SAMPLES"',
                '--gptq-percdamp "$GPTQ_PERCDAMP"',
            ],
        }

        for relative_path, snippets in scripts.items():
            content = Path("scripts/bash", relative_path).read_text(encoding="utf-8")
            for snippet in snippets:
                self.assertIn(snippet, content, f"Missing {snippet!r} in {relative_path}")


    def test_flatquant_smart_flip_scripts_can_skip_float_model(self):
        for relative_path in [
            "scripts/bash/smart_flip/flatquant/run_llama3.sh",
            "scripts/bash/smart_flip/flatquant/run_llama31.sh",
            "scripts/bash/smart_flip/flatquant/run_mistral.sh",
            "scripts/bash/smart_flip/flatquant/run_qwen25.sh",
        ]:
            content = Path(relative_path).read_text(encoding="utf-8")
            self.assertIn('RUN_FLOAT_MODEL="${RUN_FLOAT_MODEL:-1}"', content)
            self.assertIn('RUN_RAW_QUANTIZE="${RUN_RAW_QUANTIZE:-1}"', content)
            self.assertIn('if [ "$RUN_FLOAT_MODEL" = "1" ]; then', content)
            self.assertIn('echo "==> skipping float_model :: ${MODEL_PATH}"', content)
            self.assertIn('if [ "$RUN_RAW_QUANTIZE" = "1" ]; then', content)
            self.assertIn('echo "==> skipping raw_quantize :: ${MODEL_PATH} :: using existing raw model at ${RAW_MODEL_DIR}"', content)
            self.assertIn('RAW_MODEL_DIR="${RAW_MODEL_DIR:-${RESULTS_MODELS_DIR}/${ORIGIN_METHOD}_raw/${RAW_RUN_NAME}}"', content)
            self.assertIn('--flatquant-raw-path "$RAW_MODEL_DIR"', content)
            self.assertIn('"$PYTHON_BIN" main.py float_model', content)



    def test_flatquant_bias_correction_scripts_can_skip_float_model(self):
        for relative_path in [
            "scripts/bash/bias_correction/flatquant/run_llama3.sh",
            "scripts/bash/bias_correction/flatquant/run_llama31.sh",
            "scripts/bash/bias_correction/flatquant/run_mistral.sh",
            "scripts/bash/bias_correction/flatquant/run_qwen25.sh",
        ]:
            content = Path(relative_path).read_text(encoding="utf-8")
            self.assertIn("RUN_FLOAT_MODEL=\"${RUN_FLOAT_MODEL:-1}\"", content)
            self.assertIn("RUN_RAW_QUANTIZE=\"${RUN_RAW_QUANTIZE:-1}\"", content)
            self.assertIn("if [ \"$RUN_FLOAT_MODEL\" = \"1\" ]; then", content)
            self.assertIn("echo \"==> skipping float_model :: ${MODEL_PATH}\"", content)
            self.assertIn("if [ \"$RUN_RAW_QUANTIZE\" = \"1\" ]; then", content)
            self.assertIn("echo \"==> skipping raw_quantize :: ${MODEL_PATH} :: using existing raw model at ${RAW_MODEL_DIR}\"", content)
            self.assertIn("RAW_MODEL_DIR=\"${RAW_MODEL_DIR:-${RESULTS_MODELS_DIR}/${ORIGIN_METHOD}_raw/${RAW_RUN_NAME}}\"", content)
            self.assertIn("--flatquant-raw-path \"$RAW_MODEL_DIR\"", content)
            self.assertIn("\"$PYTHON_BIN\" main.py float_model", content)


    def test_flatquant_smart_flip_scripts_include_model_slug_in_run_names(self):
        for relative_path in [
            "scripts/bash/smart_flip/flatquant/run_llama3.sh",
            "scripts/bash/smart_flip/flatquant/run_llama31.sh",
            "scripts/bash/smart_flip/flatquant/run_mistral.sh",
            "scripts/bash/smart_flip/flatquant/run_qwen25.sh",
        ]:
            content = Path(relative_path).read_text(encoding="utf-8")
            self.assertIn('MODEL_SLUG="${MODEL_PATH##*/}"', content)
            self.assertIn('run_name="${ORIGIN_METHOD}_smart_flip_${MODEL_SLUG}_b${bits}_k${knee}_f${max_flip}"', content)

    def test_flatquant_bias_correction_scripts_include_model_slug_in_run_names(self):
        for relative_path in [
            "scripts/bash/bias_correction/flatquant/run_llama3.sh",
            "scripts/bash/bias_correction/flatquant/run_llama31.sh",
            "scripts/bash/bias_correction/flatquant/run_mistral.sh",
            "scripts/bash/bias_correction/flatquant/run_qwen25.sh",
        ]:
            content = Path(relative_path).read_text(encoding="utf-8")
            self.assertIn('MODEL_SLUG="${MODEL_PATH##*/}"', content)
            self.assertIn('FLOAT_RUN_NAME="${FLOAT_RUN_NAME:-${ORIGIN_METHOD}_float_${MODEL_SLUG}}"', content)
            self.assertIn('RAW_RUN_NAME="${RAW_RUN_NAME:-${ORIGIN_METHOD}_raw_${MODEL_SLUG}}"', content)
            self.assertIn('CORR_RUN_NAME="${CORR_RUN_NAME:-${ORIGIN_METHOD}_bias_correction_${MODEL_SLUG}}"', content)

    def test_awq_bias_correction_scripts_can_skip_float_model(self):
        for relative_path in [
            "scripts/bash/bias_correction/awq/run_llama3.sh",
            "scripts/bash/bias_correction/awq/run_llama31.sh",
            "scripts/bash/bias_correction/awq/run_mistral.sh",
            "scripts/bash/bias_correction/awq/run_qwen25.sh",
        ]:
            content = Path(relative_path).read_text(encoding="utf-8")
            self.assertIn('RUN_FLOAT_MODEL="${RUN_FLOAT_MODEL:-1}"', content)
            self.assertIn('if [ "$RUN_FLOAT_MODEL" = "1" ]; then', content)
            self.assertIn('echo "==> skipping float_model :: ${MODEL_PATH}"', content)
            self.assertIn('"$PYTHON_BIN" main.py float_model', content)

    def test_awq_bias_correction_scripts_include_model_slug_in_run_names(self):
        for relative_path in [
            "scripts/bash/bias_correction/awq/run_llama3.sh",
            "scripts/bash/bias_correction/awq/run_llama31.sh",
            "scripts/bash/bias_correction/awq/run_mistral.sh",
            "scripts/bash/bias_correction/awq/run_qwen25.sh",
        ]:
            content = Path(relative_path).read_text(encoding="utf-8")
            self.assertIn('MODEL_SLUG="${MODEL_PATH##*/}"', content)
            self.assertIn('FLOAT_RUN_NAME="${FLOAT_RUN_NAME:-${ORIGIN_METHOD}_float_${MODEL_SLUG}}"', content)
            self.assertIn('RAW_RUN_NAME="${RAW_RUN_NAME:-${ORIGIN_METHOD}_raw_${MODEL_SLUG}}"', content)
            self.assertIn('CORR_RUN_NAME="${CORR_RUN_NAME:-${ORIGIN_METHOD}_bias_correction_${MODEL_SLUG}}"', content)

    def test_awq_qwen_smart_flip_script_can_skip_float_model(self):
        content = Path("scripts/bash/smart_flip/awq/run_qwen25.sh").read_text(encoding="utf-8")
        self.assertIn('RUN_FLOAT_MODEL="${RUN_FLOAT_MODEL:-1}"', content)
        self.assertIn('if [ "$RUN_FLOAT_MODEL" = "1" ]; then', content)
        self.assertIn('echo "==> skipping float_model :: ${MODEL_PATH}"', content)
        self.assertIn('"$PYTHON_BIN" main.py float_model', content)

    def test_awq_qwen_smart_flip_script_includes_model_slug_in_run_names(self):
        content = Path("scripts/bash/smart_flip/awq/run_qwen25.sh").read_text(encoding="utf-8")
        self.assertIn('MODEL_SLUG="${MODEL_PATH##*/}"', content)
        self.assertIn('FLOAT_RUN_NAME="${FLOAT_RUN_NAME:-${ORIGIN_METHOD}_float_${MODEL_SLUG}}"', content)
        self.assertIn('RAW_RUN_NAME="${RAW_RUN_NAME:-${ORIGIN_METHOD}_raw_${MODEL_SLUG}}"', content)
        self.assertIn('run_name="${ORIGIN_METHOD}_smart_flip_${MODEL_SLUG}_b${bits}_k${knee}_f${max_flip}"', content)

    def test_gptq_smart_flip_scripts_include_model_slug_and_gptq_flags(self):
        for relative_path in [
            "scripts/bash/smart_flip/gptq/run_llama3.sh",
            "scripts/bash/smart_flip/gptq/run_llama31.sh",
            "scripts/bash/smart_flip/gptq/run_mistral.sh",
            "scripts/bash/smart_flip/gptq/run_qwen25.sh",
        ]:
            content = Path(relative_path).read_text(encoding="utf-8")
            self.assertIn('MODEL_SLUG="${MODEL_PATH##*/}"', content)
            self.assertIn('RUN_FLOAT_MODEL="${RUN_FLOAT_MODEL:-1}"', content)
            self.assertIn('RUN_RAW_QUANTIZE="${RUN_RAW_QUANTIZE:-1}"', content)
            self.assertIn('FLOAT_RUN_NAME="${FLOAT_RUN_NAME:-${ORIGIN_METHOD}_float_${MODEL_SLUG}}"', content)
            self.assertIn('RAW_RUN_NAME="${RAW_RUN_NAME:-${ORIGIN_METHOD}_raw_${MODEL_SLUG}}"', content)
            self.assertIn('RAW_MODEL_DIR="${RAW_MODEL_DIR:-${RESULTS_MODELS_DIR}/${ORIGIN_METHOD}_raw/${RAW_RUN_NAME}}"', content)
            self.assertIn('run_name="${ORIGIN_METHOD}_smart_flip_${MODEL_SLUG}_b${bits}_k${knee}_f${max_flip}"', content)
            self.assertIn('GPTQ_PERCDAMP="${GPTQ_PERCDAMP:-0.01}"', content)
            self.assertIn('if [ "$RUN_RAW_QUANTIZE" = "1" ]; then', content)
            self.assertIn('echo "==> skipping raw_quantize :: ${MODEL_PATH} :: using existing raw model at ${RAW_MODEL_DIR}"', content)
            self.assertIn('--gptq-raw-path "$RAW_MODEL_DIR"', content)
            self.assertIn('--gptq-percdamp "$GPTQ_PERCDAMP"', content)

    def test_gptq_bias_correction_scripts_include_model_slug_and_gptq_flags(self):
        for relative_path in [
            "scripts/bash/bias_correction/gptq/run_llama3.sh",
            "scripts/bash/bias_correction/gptq/run_llama31.sh",
            "scripts/bash/bias_correction/gptq/run_mistral.sh",
            "scripts/bash/bias_correction/gptq/run_qwen25.sh",
        ]:
            content = Path(relative_path).read_text(encoding="utf-8")
            self.assertIn('MODEL_SLUG="${MODEL_PATH##*/}"', content)
            self.assertIn('RUN_FLOAT_MODEL="${RUN_FLOAT_MODEL:-1}"', content)
            self.assertIn('RUN_RAW_QUANTIZE="${RUN_RAW_QUANTIZE:-1}"', content)
            self.assertIn('FLOAT_RUN_NAME="${FLOAT_RUN_NAME:-${ORIGIN_METHOD}_float_${MODEL_SLUG}}"', content)
            self.assertIn('RAW_RUN_NAME="${RAW_RUN_NAME:-${ORIGIN_METHOD}_raw_${MODEL_SLUG}}"', content)
            self.assertIn('RAW_MODEL_DIR="${RAW_MODEL_DIR:-${RESULTS_MODELS_DIR}/${ORIGIN_METHOD}_raw/${RAW_RUN_NAME}}"', content)
            self.assertIn('CORR_RUN_NAME="${CORR_RUN_NAME:-${ORIGIN_METHOD}_bias_correction_${MODEL_SLUG}}"', content)
            self.assertIn('GPTQ_PERCDAMP="${GPTQ_PERCDAMP:-0.01}"', content)
            self.assertIn('if [ "$RUN_RAW_QUANTIZE" = "1" ]; then', content)
            self.assertIn('echo "==> skipping raw_quantize :: ${MODEL_PATH} :: using existing raw model at ${RAW_MODEL_DIR}"', content)
            self.assertIn('--gptq-raw-path "$RAW_MODEL_DIR"', content)
            self.assertIn('--gptq-percdamp "$GPTQ_PERCDAMP"', content)

    def test_gptq_all_models_single_setting_script_runs_fixed_smart_flip_config(self):
        content = Path("scripts/bash/smart_flip/gptq/run_all_models_single_setting.sh").read_text(encoding="utf-8")
        for snippet in [
            'MODEL_PATHS=(',
            '"meta-llama/Meta-Llama-3-8B"',
            '"meta-llama/Llama-3.1-8B"',
            '"mistralai/Mistral-7B-v0.3"',
            '"Qwen/Qwen2.5-7B"',
            'RUN_RAW_QUANTIZE="${RUN_RAW_QUANTIZE:-1}"',
            'KNEE_TOLERANCE="${KNEE_TOLERANCE:-0.0}"',
            'MAX_FLIP_PERCENT="${MAX_FLIP_PERCENT:-0.05}"',
            '--post-correction none',
            '--post-correction "$POST_CORRECTION"',
            '--gptq-raw-path "$RAW_MODEL_DIR"',
            '--knee-tolerance "$KNEE_TOLERANCE"',
            '--max-flip-percent "$MAX_FLIP_PERCENT"',
            'for MODEL_PATH in "${MODEL_PATHS[@]}"; do',
            '"$PYTHON_BIN" main.py quantize',
        ]:
            self.assertIn(snippet, content)

if __name__ == "__main__":
    unittest.main()
