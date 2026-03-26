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
        self.assertTrue(raw_args.include_lm_eval)
        self.assertEqual(raw_args.lm_eval_task_preset, "extended")
        self.assertIn("lambada_openai", raw_args.lm_eval_tasks)

        flip_args = parser.parse_args(["flip_quantize", "--model-path", "dummy-model"])
        self.assertEqual(flip_args.origin_method, "awq")
        self.assertEqual(flip_args.post_correction, "smart_flip")
        self.assertTrue(flip_args.include_lm_eval)
        self.assertEqual(flip_args.lm_eval_task_preset, "extended")
        self.assertIn("lambada_openai", flip_args.lm_eval_tasks)

    def test_eval_modes_enable_full_results_by_default(self):
        parser = main.build_parser()

        args = parser.parse_args(["float_model", "--model-path", "dummy-model"])

        self.assertTrue(args.include_c4)
        self.assertTrue(args.include_lm_eval)
        self.assertEqual(args.lm_eval_task_preset, "extended")
        self.assertEqual(args.lm_eval_tasks, main.DEFAULT_LM_EVAL_TASKS["extended"])
        self.assertIn("lambada_openai", args.lm_eval_tasks)
        self.assertNotIn("lambada_openai", main.DEFAULT_LM_EVAL_TASKS["core"])

    def test_eval_modes_can_disable_lm_eval(self):
        parser = main.build_parser()

        args = parser.parse_args(["float_model", "--model-path", "dummy-model", "--no-lm-eval"])

        self.assertFalse(args.include_lm_eval)

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
            include_lm_eval=True,
            lm_eval_task_preset="extended",
            lm_eval_tasks=main.DEFAULT_LM_EVAL_TASKS["extended"],
            lm_eval_num_fewshot=None,
            lm_eval_batch_size="auto",
            lm_eval_output_dir="./results/eval/lm_eval",
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

    def test_evaluate_model_paths_runs_both_evaluators_by_default(self):
        args = SimpleNamespace(
            seed=42,
            stride=512,
            max_length=2048,
            eval_cache_dir="./data/cache/eval",
            include_c4=True,
            c4_samples=10,
            include_lm_eval=True,
            lm_eval_task_preset="extended",
            lm_eval_tasks=main.DEFAULT_LM_EVAL_TASKS["extended"],
            lm_eval_num_fewshot=None,
            lm_eval_batch_size="auto",
            lm_eval_output_dir="./results/eval/lm_eval",
            run_name="demo",
            results_eval_dir="./results/eval",
        )
        model_paths = {"float_model": "dummy-model"}

        with patch("main.run_perplexity_evaluation", return_value={"WikiText-2": {"float_model": {"perplexity": 12.3}}}) as run_ppl:
            with patch("main.run_lm_eval", return_value={"float_model": {"results": {"arc_easy": {"acc,none": 0.5}}}}) as run_lm_eval:
                output_path = main.evaluate_model_paths(args, model_paths)

        run_ppl.assert_called_once_with(args, model_paths)
        run_lm_eval.assert_called_once_with(args, model_paths)
        self.assertEqual(output_path, main.Path("./results/eval/demo.json"))

    def test_evaluate_model_paths_can_skip_lm_eval(self):
        args = SimpleNamespace(
            seed=42,
            stride=512,
            max_length=2048,
            eval_cache_dir="./data/cache/eval",
            include_c4=True,
            c4_samples=10,
            include_lm_eval=False,
            lm_eval_task_preset="extended",
            lm_eval_tasks=main.DEFAULT_LM_EVAL_TASKS["extended"],
            lm_eval_num_fewshot=None,
            lm_eval_batch_size="auto",
            lm_eval_output_dir="./results/eval/lm_eval",
            run_name="demo",
            results_eval_dir="./results/eval",
        )
        model_paths = {"float_model": "dummy-model"}

        with patch("main.run_perplexity_evaluation", return_value={"WikiText-2": {"float_model": {"perplexity": 12.3}}}) as run_ppl:
            with patch("main.run_lm_eval") as run_lm_eval:
                main.evaluate_model_paths(args, model_paths)

        run_ppl.assert_called_once_with(args, model_paths)
        run_lm_eval.assert_not_called()


if __name__ == "__main__":
    unittest.main()
