import builtins
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import main


class ParserModeTests(unittest.TestCase):
    def test_parser_accepts_single_model_modes(self):
        parser = main.build_parser()
        mode_args = {
            "float_model": ["float_model", "--model-path", "dummy-model"],
            "quantize": ["quantize", "--model-path", "dummy-model"],
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

    def test_quantize_mode_accepts_post_correction_parameter(self):
        parser = main.build_parser()

        args = parser.parse_args([
            "quantize",
            "--model-path",
            "dummy-model",
            "--origin-method",
            "flatquant",
            "--post-correction",
            "bias_correction",
        ])

        self.assertEqual(args.origin_method, "flatquant")
        self.assertEqual(args.post_correction, "bias_correction")

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


    def test_flatquant_origin_method_accepts_flatquant_defaults(self):
        parser = main.build_parser()

        args = parser.parse_args([
            "raw_quantize",
            "--model-path",
            "dummy-model",
            "--origin-method",
            "flatquant",
        ])

        self.assertEqual(args.origin_method, "flatquant")
        self.assertEqual(args.bits, 4)
        self.assertEqual(args.flatquant_epochs, 15)
        self.assertEqual(args.flatquant_cali_bsz, 4)

    def test_flatquant_origin_method_accepts_debug_diagnostics_flags(self):
        parser = main.build_parser()

        args = parser.parse_args([
            "raw_quantize",
            "--model-path",
            "dummy-model",
            "--origin-method",
            "flatquant",
            "--flatquant-debug-diagnostics",
            "--flatquant-debug-sample-limit",
            "64",
        ])

        self.assertTrue(args.flatquant_debug_diagnostics)
        self.assertEqual(args.flatquant_debug_sample_limit, 64)

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


    def test_build_auto_run_name_includes_model_slug_for_smart_flip_variants(self):
        args = SimpleNamespace(
            model_path="meta-llama/Meta-Llama-3-8B",
            resolved_source_model="meta-llama/Meta-Llama-3-8B",
            bits=4,
            group_size=128,
            knee_tolerance=0.0,
            max_flip_percent=0.05,
            seed=42,
        )

        run_name = main.build_auto_run_name("flatquant_smart_flip", args, timestamp="20260410-120000")

        self.assertEqual(run_name, "flatquant_smart_flip_Meta-Llama-3-8B_b4_g128_k0_f0p05_s42_20260410-120000")

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

    def test_save_evaluation_results_falls_back_when_temp_write_fails(self):
        results = {"perplexity": {"WikiText-2": {"raw_quantize": {"perplexity": 12.3}}}}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = main.Path(tmpdir) / "flatquant_raw.json"
            real_open = builtins.open

            def flaky_open(path, *args, **kwargs):
                candidate = main.Path(path)
                if candidate == output_path.with_suffix(output_path.suffix + ".tmp"):
                    raise OSError(5, "Input/output error")
                return real_open(path, *args, **kwargs)

            with patch("builtins.open", side_effect=flaky_open):
                main.save_evaluation_results(results, output_path)

            self.assertTrue(output_path.exists())
            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                '{\n'
                '  "perplexity": {\n'
                '    "WikiText-2": {\n'
                '      "raw_quantize": {\n'
                '        "perplexity": 12.3\n'
                "      }\n"
                "    }\n"
                "  }\n"
                "}",
            )

    def test_run_quantize_requests_tensor_calibration_for_flatquant(self):
        parser = main.build_parser()
        with tempfile.TemporaryDirectory() as tmpdir:
            args = parser.parse_args([
                "raw_quantize",
                "--model-path",
                "dummy-model",
                "--origin-method",
                "flatquant",
                "--results-models-dir",
                tmpdir,
                "--results-eval-dir",
                tmpdir,
            ])

            tokenizer = SimpleNamespace(pad_token=None, eos_token="</s>", save_pretrained=lambda _path: None)
            model = SimpleNamespace(
                eval=lambda: None,
                save_pretrained=lambda _path, **_kwargs: None,
            )
            quantizer = SimpleNamespace(
                set_artifact_dir=lambda _path: None,
                quantize_model_sequential=lambda *args, **kwargs: None,
                build_evaluation_target=lambda: {"model": "in-memory", "tokenizer": "tok", "evaluation_target": {"kind": "in_memory_model"}},
                describe_evaluation_target=lambda: {"kind": "in_memory_model"},
                layer_stats={},
            )

            with patch('main.AutoTokenizer.from_pretrained', return_value=tokenizer):
                with patch('main.AutoModelForCausalLM.from_pretrained', return_value=model):
                    with patch('src.calibration.load_calibration_data', return_value=[torch.tensor([[1, 2, 3]])]) as load_calibration_data:
                        with patch('src.quantization.pipeline.create_quantizer', return_value=(quantizer, SimpleNamespace(__dict__={}), None)):
                            main.run_quantize(args)

            self.assertTrue(load_calibration_data.call_args.kwargs.get('return_tensors'))

    def test_run_quantize_disables_safe_serialization_for_flatquant(self):
        parser = main.build_parser()
        with tempfile.TemporaryDirectory() as tmpdir:
            args = parser.parse_args([
                "raw_quantize",
                "--model-path",
                "dummy-model",
                "--origin-method",
                "flatquant",
                "--results-models-dir",
                tmpdir,
                "--results-eval-dir",
                tmpdir,
            ])

            tokenizer = SimpleNamespace(pad_token=None, eos_token="</s>", save_pretrained=lambda _path: None)
            observed = {}
            model = SimpleNamespace(
                eval=lambda: None,
                save_pretrained=lambda path, **kwargs: observed.update({"path": path, "kwargs": kwargs}),
            )
            quantizer = SimpleNamespace(
                set_artifact_dir=lambda _path: None,
                quantize_model_sequential=lambda *args, **kwargs: None,
                build_evaluation_target=lambda: {"model": "in-memory", "tokenizer": "tok", "evaluation_target": {"kind": "in_memory_model"}},
                describe_evaluation_target=lambda: {"kind": "in_memory_model"},
                layer_stats={},
            )

            with patch('main.AutoTokenizer.from_pretrained', return_value=tokenizer):
                with patch('main.AutoModelForCausalLM.from_pretrained', return_value=model):
                    with patch('src.calibration.load_calibration_data', return_value=[torch.tensor([[1, 2, 3]])]):
                        with patch('src.quantization.pipeline.create_quantizer', return_value=(quantizer, SimpleNamespace(__dict__={}), None)):
                            main.run_quantize(args)

            self.assertFalse(observed["kwargs"].get("safe_serialization", True))

    def test_run_raw_quantize_evaluates_then_deletes_temporary_model_dir(self):
        args = SimpleNamespace()
        output_dir = main.Path('./results/models/.tmp/raw-run')

        with patch('main.run_quantize', return_value=output_dir) as run_quantize:
            with patch('main.evaluate_model_paths') as evaluate_model_paths:
                with patch('main.shutil.rmtree') as rmtree:
                    main.run_raw_quantize(args)

        run_quantize.assert_called_once_with(args)
        evaluate_model_paths.assert_called_once_with(args, {'raw_quantize': str(output_dir)}, variant='awq_raw')
        rmtree.assert_called_once_with(output_dir, ignore_errors=True)

    def test_run_flip_quantize_deletes_temporary_model_dir_even_if_eval_fails(self):
        args = SimpleNamespace()
        output_dir = main.Path('./results/models/.tmp/flip-run')

        with patch('main.run_quantize', return_value=output_dir) as run_quantize:
            with patch('main.evaluate_model_paths', side_effect=RuntimeError('eval failed')):
                with patch('main.shutil.rmtree') as rmtree:
                    with self.assertRaisesRegex(RuntimeError, 'eval failed'):
                        main.run_flip_quantize(args)

        run_quantize.assert_called_once_with(args)
        rmtree.assert_called_once_with(output_dir, ignore_errors=True)



    def test_quantize_mode_accepts_flatquant_raw_path(self):
        parser = main.build_parser()

        args = parser.parse_args([
            "quantize",
            "--model-path",
            "dummy-model",
            "--origin-method",
            "flatquant",
            "--post-correction",
            "smart_flip",
            "--flatquant-raw-path",
            "./results/models/flatquant_raw/raw-run",
        ])

        self.assertEqual(args.flatquant_raw_path, "./results/models/flatquant_raw/raw-run")

    def test_run_raw_quantize_keeps_flatquant_raw_model_dir(self):
        args = SimpleNamespace(origin_method="flatquant", post_correction="none")
        output_dir = main.Path('./results/models/flatquant_raw/raw-run')

        with patch('main.run_quantize', return_value=output_dir) as run_quantize:
            with patch('main.evaluate_model_paths') as evaluate_model_paths:
                with patch('main.shutil.rmtree') as rmtree:
                    main.run_raw_quantize(args)

        run_quantize.assert_called_once_with(args)
        evaluate_model_paths.assert_called_once_with(args, {'raw_quantize': str(output_dir)}, variant='flatquant_raw')
        rmtree.assert_not_called()

if __name__ == "__main__":
    unittest.main()




