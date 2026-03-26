import unittest
from types import SimpleNamespace
from unittest.mock import patch

import main


class RunTrackingTests(unittest.TestCase):
    def test_build_model_slug_prefers_last_path_segment_and_sanitizes(self):
        self.assertEqual(main.build_model_slug("mistralai/Mistral-7B-v0.3"), "Mistral-7B-v0.3")
        self.assertEqual(main.build_model_slug("/models/Qwen2.5-7B"), "Qwen2p5-7B")

    def test_build_auto_run_name_includes_model_tuning_seed_and_timestamp(self):
        run_name = main.build_auto_run_name(
            variant="awq_flip",
            args=SimpleNamespace(
                model_path="mistralai/Mistral-7B-v0.3",
                bits=4,
                group_size=128,
                knee_tolerance=0.1,
                max_flip_percent=0.05,
                seed=7,
            ),
            timestamp="20260326-101530",
        )

        self.assertEqual(run_name, "awq_flip_Mistral-7B-v0.3_b4_g128_k0p1_f0p05_s7_20260326-101530")

    def test_build_wandb_tags_adds_model_and_variant_tags(self):
        tags = main.build_wandb_tags(
            SimpleNamespace(wandb_tags=["manual"]),
            variant="awq_flip",
            model_slug="Mistral-7B-v0.3",
        )

        self.assertEqual(tags[:3], ["manual", "model:Mistral-7B-v0.3", "variant:awq_flip"])

    def test_evaluate_model_paths_logs_to_wandb_when_enabled(self):
        args = SimpleNamespace(
            model_path="mistralai/Mistral-7B-v0.3",
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
            use_wandb=True,
            wandb_project="smart-flip",
            wandb_entity=None,
            wandb_tags=[],
        )
        model_paths = {"float_model": "dummy-model"}

        with patch("main.run_perplexity_evaluation", return_value={"WikiText-2": {"float_model": {"perplexity": 12.3}}}) as run_ppl:
            with patch("main.run_lm_eval", return_value={"float_model": {"results": {"arc_easy": {"acc,none": 0.5}}}}) as run_lm_eval:
                with patch("main.log_results_to_wandb") as log_results_to_wandb:
                    output_path = main.evaluate_model_paths(args, model_paths)

        run_ppl.assert_called_once_with(args, model_paths)
        run_lm_eval.assert_called_once_with(args, model_paths)
        log_results_to_wandb.assert_called_once()
        _, logged_run_name, logged_variant, _, _, logged_model_slug = log_results_to_wandb.call_args[0]
        self.assertEqual(logged_run_name, "demo")
        self.assertEqual(logged_variant, "evaluation")
        self.assertEqual(logged_model_slug, "Mistral-7B-v0.3")
        self.assertEqual(output_path, main.Path("./results/eval/demo.json"))


if __name__ == "__main__":
    unittest.main()
