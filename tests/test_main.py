import unittest
from types import SimpleNamespace
from unittest.mock import patch

import main


class ParserModeTests(unittest.TestCase):
    def test_parser_accepts_single_model_modes(self):
        parser = main.build_parser()
        mode_args = {
            "float_model": ["float_model", "--model-path", "dummy-model"],
            "raw_quantize": ["raw_quantize", "--model-path", "dummy-model"],
            "flip_quantize": ["flip_quantize", "--model-path", "dummy-model"],
            "compare_all": [
                "compare_all",
                "--model-path",
                "dummy-model",
                "--awq-raw-path",
                "raw-model",
                "--awq-flip-path",
                "flip-model",
            ],
        }

        for mode, argv in mode_args.items():
            args = parser.parse_args(argv)
            self.assertEqual(args.mode, mode)

    def test_compare_all_dispatches_all_three_model_paths(self):
        args = SimpleNamespace(
            model_path="fp-model",
            awq_raw_path="raw-model",
            awq_flip_path="flip-model",
            run_name="cmp",
            results_eval_dir="./results/eval",
            eval_cache_dir="./data/cache/eval",
            seed=42,
            stride=512,
            max_length=2048,
            include_c4=False,
            c4_samples=10,
        )

        with patch("main.evaluate_model_paths") as evaluate_model_paths:
            main.run_compare_all(args)

        evaluate_model_paths.assert_called_once()
        called_args = evaluate_model_paths.call_args[0]
        self.assertIs(called_args[0], args)
        self.assertEqual(
            called_args[1],
            {"float_model": "fp-model", "raw_quantize": "raw-model", "flip_quantize": "flip-model"},
        )


if __name__ == "__main__":
    unittest.main()
