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
                "--raw-path",
                "raw-model",
                "--flip-path",
                "flip-model",
            ],
        }

        for mode, argv in mode_args.items():
            args = parser.parse_args(argv)
            self.assertEqual(args.mode, mode)

    def test_quantize_modes_set_origin_and_post_correction_defaults(self):
        parser = main.build_parser()

        raw_args = parser.parse_args(["raw_quantize", "--model-path", "dummy-model"])
        self.assertEqual(raw_args.origin_method, "awq")
        self.assertEqual(raw_args.post_correction, "none")

        flip_args = parser.parse_args(["flip_quantize", "--model-path", "dummy-model"])
        self.assertEqual(flip_args.origin_method, "awq")
        self.assertEqual(flip_args.post_correction, "smart_flip")

    def test_compare_all_dispatches_all_three_model_paths(self):
        args = SimpleNamespace(
            model_path="fp-model",
            raw_path="raw-model",
            flip_path="flip-model",
            models_root="/models",
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

    def test_resolve_model_reference_prefers_existing_path(self):
        with patch("main.Path.exists", side_effect=[True]):
            resolved = main.resolve_model_reference("existing-model", models_root="/models")

        self.assertEqual(resolved, "existing-model")

    def test_resolve_model_reference_finds_model_under_models_root(self):
        with patch("main.Path.exists", side_effect=[False, True]):
            resolved = main.resolve_model_reference("Mistral-7B-v0.3", models_root="/models")

        self.assertEqual(resolved, str(main.Path("/models") / "Mistral-7B-v0.3"))

    def test_resolve_model_reference_leaves_huggingface_id_untouched(self):
        with patch("main.Path.exists", side_effect=[False, False]):
            resolved = main.resolve_model_reference("mistralai/Mistral-7B-v0.3", models_root="/models")

        self.assertEqual(resolved, "mistralai/Mistral-7B-v0.3")


if __name__ == "__main__":
    unittest.main()
