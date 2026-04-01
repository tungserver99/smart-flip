import json
import unittest
from pathlib import Path

from src.evaluation.lm_eval import LMEvalHarnessRunner


class LMEvalSerializationTests(unittest.TestCase):
    def test_runner_converts_non_json_payload_fields_before_writing(self):
        tmpdir = Path('data/cache/test_lm_eval_runner')
        tmpdir.mkdir(parents=True, exist_ok=True)
        runner = LMEvalHarnessRunner(tasks=["arc_easy"], output_dir=str(tmpdir), run_name="demo")
        payload = {
            "results": {"arc_easy": {"acc,none": 0.5}},
            "config": {"formatter": lambda value: value},
        }

        runner._write_raw_results("float_model", payload)

        output_path = tmpdir / "demo_float_model.json"
        saved = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(saved["results"]["arc_easy"]["acc,none"], 0.5)
        self.assertEqual(saved["config"]["formatter"], "<callable <lambda>>")

    def test_runner_returns_json_safe_raw_payload(self):
        runner = LMEvalHarnessRunner(tasks=["arc_easy"], output_dir="./results/eval/lm_eval", run_name="demo")
        payload = {
            "results": {"arc_easy": {"acc,none": 0.5}},
            "config": {"formatter": lambda value: value},
        }

        safe_payload = runner._make_json_safe(payload)
        encoded = json.dumps(safe_payload)

        self.assertIsInstance(encoded, str)
        self.assertEqual(safe_payload["config"]["formatter"], "<callable <lambda>>")

    def test_model_args_include_hf_token_when_provided(self):
        runner = LMEvalHarnessRunner(tasks=["arc_easy"], hf_token="hf-secret")

        self.assertIn('token=hf-secret', runner._model_args('dummy-model'))


if __name__ == "__main__":
    unittest.main()

