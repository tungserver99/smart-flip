import sys
import importlib
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import src.calibration as calibration
from src.post_correction.bias_correction import BiasCorrectionConfig, BiasCorrectionCorrection
from src.post_correction.smart_flip import SmartFlipConfig, SmartFlipCorrection
from src.quantization.awq import AWQConfig, AWQQuantizerXL
from src.quantization.flatquant import FlatQuantConfig, FlatQuantRTNQuantizer, _ActivationCollector
from src.quantization.gptq import GPTQ
from src.quantization.pipeline import QuantizationRecipe, create_quantizer


class QuantizationAssemblyTests(unittest.TestCase):
    def make_args(self):
        return SimpleNamespace(
            bits=4,
            n_grid=20,
            group_size=128,
            max_tokens_per_sample=2048,
            layer_batch_size=16,
            lmhead_chunks=4,
            knee_tolerance=0.0,
            max_flip_percent=0.05,
            use_james_stein=True,
            bias_correction_samples=4096,
            flatquant_epochs=15,
            flatquant_cali_bsz=4,
            flatquant_lr=5e-3,
            flatquant_cali_trans=True,
            flatquant_add_diag=True,
            flatquant_lwc=True,
            flatquant_lac=True,
            flatquant_diag_init="sq_style",
            flatquant_diag_alpha=0.3,
            flatquant_debug_diagnostics=False,
            flatquant_debug_sample_limit=256,
        )

    def test_recipe_variant_name_is_generic(self):
        self.assertEqual(QuantizationRecipe(origin_method="awq", post_correction="none").variant_name, "awq_raw")
        self.assertEqual(
            QuantizationRecipe(origin_method="flatquant", post_correction="none").variant_name,
            "flatquant_raw",
        )
        self.assertEqual(
            QuantizationRecipe(origin_method="awq", post_correction="smart_flip").variant_name,
            "awq_smart_flip",
        )
        self.assertEqual(
            QuantizationRecipe(origin_method="flatquant", post_correction="smart_flip").variant_name,
            "flatquant_smart_flip",
        )
        self.assertEqual(
            QuantizationRecipe(origin_method="awq", post_correction="bias_correction").variant_name,
            "awq_bias_correction",
        )

    def test_create_quantizer_builds_awq_without_post_correction(self):
        recipe = QuantizationRecipe(origin_method="awq", post_correction="none")
        quantizer, base_config, correction = create_quantizer(
            model=object(),
            tokenizer=object(),
            device="cpu",
            args=self.make_args(),
            recipe=recipe,
        )

        self.assertIsInstance(quantizer, AWQQuantizerXL)
        self.assertIsInstance(base_config, AWQConfig)
        self.assertIsNone(correction)
        self.assertIsNone(quantizer.post_correction)

    def test_create_quantizer_builds_awq_with_smart_flip_post_correction(self):
        recipe = QuantizationRecipe(origin_method="awq", post_correction="smart_flip")
        quantizer, base_config, correction = create_quantizer(
            model=object(),
            tokenizer=object(),
            device="cpu",
            args=self.make_args(),
            recipe=recipe,
        )

        self.assertIsInstance(quantizer, AWQQuantizerXL)
        self.assertIsInstance(base_config, AWQConfig)
        self.assertIsInstance(correction, SmartFlipCorrection)
        self.assertIsInstance(correction.config, SmartFlipConfig)
        self.assertIs(quantizer.post_correction, correction)

    def test_create_quantizer_builds_awq_with_bias_correction_post_correction(self):
        recipe = QuantizationRecipe(origin_method="awq", post_correction="bias_correction")
        quantizer, base_config, correction = create_quantizer(
            model=object(),
            tokenizer=object(),
            device="cpu",
            args=self.make_args(),
            recipe=recipe,
        )

        self.assertIsInstance(quantizer, AWQQuantizerXL)
        self.assertIsInstance(base_config, AWQConfig)
        self.assertIsInstance(correction, BiasCorrectionCorrection)
        self.assertIsInstance(correction.config, BiasCorrectionConfig)
        self.assertIs(quantizer.post_correction, correction)

    def test_create_quantizer_builds_gptq_with_smart_flip(self):
        recipe = QuantizationRecipe(origin_method="gptq", post_correction="smart_flip")
        args = self.make_args()
        args.gptq_percdamp = 0.01
        args.gptq_sym = False
        args.gptq_act_order = False
        args.gptq_true_sequential = True
        args.gptq_static_groups = False
        args.gptq_mse = False
        quantizer, base_config, correction = create_quantizer(
            model=SimpleNamespace(config=SimpleNamespace(model_type="llama", _name_or_path="meta-llama/Llama-3-8B")),
            tokenizer=object(),
            device="cpu",
            args=args,
            recipe=recipe,
        )

        self.assertEqual(type(quantizer).__name__, "GPTQQuantizer")
        self.assertEqual(base_config.bits, 4)
        self.assertIsInstance(correction, SmartFlipCorrection)


    def test_create_quantizer_builds_flatquant_with_smart_flip_post_correction(self):
        recipe = QuantizationRecipe(origin_method="flatquant", post_correction="smart_flip")
        quantizer, base_config, correction = create_quantizer(
            model=SimpleNamespace(config=SimpleNamespace(model_type="llama", _name_or_path="meta-llama/Llama-3-8B")),
            tokenizer=object(),
            device="cpu",
            args=self.make_args(),
            recipe=recipe,
        )

        self.assertIsInstance(quantizer, FlatQuantRTNQuantizer)
        self.assertIsInstance(base_config, FlatQuantConfig)
        self.assertIsInstance(correction, SmartFlipCorrection)
        self.assertEqual(base_config.w_bits, 4)
        self.assertEqual(base_config.a_bits, 4)
        self.assertEqual(base_config.k_bits, 16)
        self.assertEqual(base_config.v_bits, 16)
        self.assertIs(quantizer.post_correction, correction)


    def test_create_quantizer_builds_flatquant_with_debug_diagnostics_config(self):
        recipe = QuantizationRecipe(origin_method="flatquant", post_correction="none")
        args = self.make_args()
        args.flatquant_debug_diagnostics = True
        args.flatquant_debug_sample_limit = 64

        quantizer, base_config, correction = create_quantizer(
            model=SimpleNamespace(config=SimpleNamespace(model_type="llama", _name_or_path="meta-llama/Llama-3-8B")),
            tokenizer=object(),
            device="cpu",
            args=args,
            recipe=recipe,
        )

        self.assertIsInstance(quantizer, FlatQuantRTNQuantizer)
        self.assertIsInstance(base_config, FlatQuantConfig)
        self.assertIsNone(correction)
        self.assertTrue(base_config.debug_diagnostics)
        self.assertEqual(base_config.debug_sample_limit, 64)


class BiasCorrectionTests(unittest.TestCase):
    make_args = QuantizationAssemblyTests.make_args

    def test_apply_bias_delta_adds_delta_when_module_has_no_bias(self):
        correction = BiasCorrectionCorrection()
        module = torch.nn.Linear(2, 2, bias=False)
        bias_delta = torch.tensor([0.25, -0.5], dtype=torch.float32)

        correction.apply_bias_delta(module, bias_delta, device="cpu", dtype=torch.float32)

        self.assertIsNotNone(module.bias)
        self.assertTrue(torch.allclose(module.bias.detach(), bias_delta))

    def test_apply_bias_delta_adds_delta_to_existing_bias(self):
        correction = BiasCorrectionCorrection()
        module = torch.nn.Linear(2, 2, bias=True)
        module.bias.data = torch.tensor([1.0, -2.0], dtype=torch.float32)
        bias_delta = torch.tensor([0.25, -0.5], dtype=torch.float32)

        correction.apply_bias_delta(module, bias_delta, device="cpu", dtype=torch.float32)

        self.assertTrue(torch.allclose(module.bias.detach(), torch.tensor([1.25, -2.5], dtype=torch.float32)))

    def test_compute_bias_delta_accepts_activation_collector(self):
        correction = BiasCorrectionCorrection(BiasCorrectionConfig(max_samples=32))
        module = torch.nn.Linear(2, 2, bias=False)
        module.weight.data = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        quant_weight = torch.zeros_like(module.weight.data)
        collector = _ActivationCollector(sample_limit=16, track_mean=False, storage_dtype=torch.float32)
        collector.add(torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32))

        bias_delta = correction.compute_bias_delta(module, quant_weight, collector, device="cpu")

        self.assertTrue(torch.allclose(bias_delta, torch.tensor([2.0, 3.0], dtype=torch.float32)))


class GPTQPostCorrectionTests(unittest.TestCase):
    def test_dead_columns_are_not_reintroduced_by_smart_flip(self):
        layer = torch.nn.Linear(2, 2, bias=False)
        layer.weight.data = torch.tensor([[1.25, -0.75], [0.5, 0.125]], dtype=torch.float32)
        gptq = GPTQ(layer, smart_flip=SmartFlipCorrection())
        gptq.quantizer = SimpleNamespace(maxq=torch.tensor(15))
        gptq.activation_sums = torch.tensor([10.0, 0.0], dtype=torch.float64)
        gptq.activation_count = 1

        original = torch.tensor([[1.25, -0.75], [0.5, 0.125]], dtype=torch.float32)
        quant = torch.tensor([[1.0, 0.0], [0.5, 0.0]], dtype=torch.float32)
        pre_round = torch.tensor([[8.9, 3.8], [7.9, 4.2]], dtype=torch.float32)
        integer = torch.tensor([[8.0, 4.0], [8.0, 4.0]], dtype=torch.float32)
        scale = torch.ones_like(original) * 0.25
        zero = torch.ones_like(original) * 4.0

        corrected = gptq._run_post_correction(
            original,
            quant,
            pre_round,
            integer,
            scale,
            zero,
            dead_mask=torch.tensor([False, True]),
        )

        self.assertTrue(torch.allclose(corrected[:, 1], torch.zeros(2, dtype=torch.float32)))

    def test_gptq_grouping_falls_back_to_full_subset_for_wrapped_qwen_style_names(self):
        from src.quantization.gptq import GPTQConfig, GPTQQuantizer

        quantizer = GPTQQuantizer(
            model=SimpleNamespace(config=SimpleNamespace(model_type="qwen2", _name_or_path="Qwen/Qwen2.5-7B")),
            tokenizer=object(),
            device="cpu",
            config=GPTQConfig(),
        )
        full = {
            "self_attn.q_proj.linear": object(),
            "self_attn.k_proj.linear": object(),
            "self_attn.v_proj.linear": object(),
            "self_attn.o_proj.linear": object(),
            "mlp.gate_proj.linear": object(),
            "mlp.up_proj.linear": object(),
            "mlp.down_proj.linear": object(),
        }

        groups = quantizer._get_sequential_groups(full)

        self.assertEqual(groups, [list(full.keys())])

    def test_gptq_rejects_calibration_tokens_outside_model_vocab(self):
        from src.quantization.gptq import GPTQConfig, GPTQQuantizer

        model = SimpleNamespace(
            config=SimpleNamespace(use_cache=False, vocab_size=32768),
            model=SimpleNamespace(layers=[torch.nn.Identity()]),
        )
        quantizer = GPTQQuantizer(
            model=model,
            tokenizer=object(),
            device="cpu",
            config=GPTQConfig(),
        )

        with self.assertRaisesRegex(ValueError, "outside the model vocabulary range"):
            quantizer.quantize_model_sequential([torch.tensor([[0, 32768]])])


class CalibrationCacheTests(unittest.TestCase):
    make_args = QuantizationAssemblyTests.make_args

    def test_c4_tensor_cache_is_scoped_by_tokenizer_identity(self):
        class FakeTokenizer:
            def __init__(self, name_or_path, vocab_size, tokens):
                self.name_or_path = name_or_path
                self.vocab_size = vocab_size
                self._tokens = tokens

            def __call__(self, text, return_tensors="pt"):
                del text, return_tensors
                return SimpleNamespace(input_ids=self._tokens)

            def decode(self, tokens, skip_special_tokens=True):
                del tokens, skip_special_tokens
                return self.name_or_path

        texts = [{"text": "x" * 40}]
        tokenizer_a = FakeTokenizer("model-a", 32768, torch.tensor([[1, 2, 3, 4]]))
        tokenizer_b = FakeTokenizer("model-b", 128256, torch.tensor([[11, 12, 13, 14]]))

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.calibration.load_dataset", return_value=texts):
                samples_a = calibration.get_c4_calibration_data(
                    tokenizer_a,
                    n_samples=1,
                    seqlen=4,
                    seed=42,
                    return_tensors=True,
                    cache_dir=tmpdir,
                )
                samples_b = calibration.get_c4_calibration_data(
                    tokenizer_b,
                    n_samples=1,
                    seqlen=4,
                    seed=42,
                    return_tensors=True,
                    cache_dir=tmpdir,
                )

            cache_files = sorted(path.name for path in Path(tmpdir).glob("*.pkl"))

        self.assertEqual(int(samples_a[0].max().item()), 4)
        self.assertEqual(int(samples_b[0].max().item()), 14)
        self.assertEqual(len(cache_files), 2)
        self.assertNotEqual(cache_files[0], cache_files[1])

    def test_legacy_tensor_cache_with_out_of_range_tokens_is_regenerated(self):
        class FakeTokenizer:
            def __init__(self, name_or_path, vocab_size, tokens):
                self.name_or_path = name_or_path
                self.vocab_size = vocab_size
                self._tokens = tokens

            def __call__(self, text, return_tensors="pt"):
                del text, return_tensors
                return SimpleNamespace(input_ids=self._tokens)

        legacy_sample = [torch.tensor([[0, 127799]])]
        tokenizer = FakeTokenizer(
            "mistralai/Mistral-7B-v0.3",
            32768,
            torch.tensor([[11, 12, 13, 14]]),
        )
        texts = [{"text": "x" * 40}]

        with tempfile.TemporaryDirectory() as tmpdir:
            legacy_path = Path(tmpdir) / "c4_calib_n1_len2_seed42_tensorsTrue.pkl"
            with open(legacy_path, "wb") as handle:
                import pickle

                pickle.dump(legacy_sample, handle)

            with patch("src.calibration.load_dataset", return_value=texts):
                samples = calibration.get_c4_calibration_data(
                    tokenizer,
                    n_samples=1,
                    seqlen=2,
                    seed=42,
                    return_tensors=True,
                    cache_dir=tmpdir,
                )

            self.assertEqual(len(samples), 1)
            self.assertLess(int(samples[0].max().item()), tokenizer.vocab_size)
            self.assertFalse(legacy_path.exists())
            migrated = list(Path(tmpdir).glob("c4_calib_n1_len2_seed42_tensorsTrue_tok*.pkl"))
            self.assertEqual(len(migrated), 1)

    def test_gptq_quantizer_save_and_load_raw_artifacts_round_trip(self):
        from src.quantization.gptq import GPTQConfig, GPTQQuantizer

        quantizer = GPTQQuantizer(
            model=SimpleNamespace(config=SimpleNamespace(model_type="llama", _name_or_path="meta-llama/Llama-3-8B")),
            tokenizer=object(),
            device="cpu",
            config=GPTQConfig(),
        )
        quantizer.raw_artifacts = {
            "model.layers.0.self_attn.q_proj": {
                "activation_count": 1,
                "activation_sums": torch.tensor([1.0, 2.0], dtype=torch.float64),
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            quantizer.save_raw_artifacts(tmpdir)
            loaded = quantizer.load_raw_artifacts(tmpdir)

        self.assertEqual(loaded["model.layers.0.self_attn.q_proj"]["activation_count"], 1)
        self.assertTrue(
            torch.allclose(
                loaded["model.layers.0.self_attn.q_proj"]["activation_sums"],
                torch.tensor([1.0, 2.0], dtype=torch.float64),
            )
        )

    def test_apply_saved_raw_artifact_matches_direct_smart_flip_post_correction(self):
        layer = torch.nn.Linear(2, 2, bias=False)
        layer.weight.data = torch.tensor([[1.25, -0.75], [0.5, 0.125]], dtype=torch.float32)

        base_gptq = GPTQ(layer)
        base_gptq.quantizer = SimpleNamespace(maxq=torch.tensor(15))
        original = torch.tensor([[1.25, -0.75], [0.5, 0.125]], dtype=torch.float32)
        quant = torch.tensor([[1.0, 0.0], [0.5, 0.0]], dtype=torch.float32)
        pre_round = torch.tensor([[8.9, 3.8], [7.9, 4.2]], dtype=torch.float32)
        integer = torch.tensor([[8.0, 4.0], [8.0, 4.0]], dtype=torch.float32)
        scale = torch.ones_like(original) * 0.25
        zero = torch.ones_like(original) * 4.0
        dead_mask = torch.tensor([False, True])

        quant_state = base_gptq._build_quant_state(original, pre_round, integer, scale, zero)
        artifact = {
            "quant_state": base_gptq._serialize_quant_state(quant_state),
            "activation_sums": torch.tensor([10.0, 0.0], dtype=torch.float64),
            "activation_count": 1,
            "activation_samples": None,
            "dead": dead_mask,
        }

        expected_layer = torch.nn.Linear(2, 2, bias=False)
        expected_layer.weight.data = quant.clone()
        expected_gptq = GPTQ(expected_layer, smart_flip=SmartFlipCorrection())
        expected_gptq.quantizer = SimpleNamespace(maxq=torch.tensor(15))
        expected_gptq.activation_sums = artifact["activation_sums"]
        expected_gptq.activation_count = artifact["activation_count"]
        expected = expected_gptq._run_post_correction(
            original,
            quant,
            pre_round,
            integer,
            scale,
            zero,
            dead_mask=dead_mask,
        )

        actual_layer = torch.nn.Linear(2, 2, bias=False)
        actual_layer.weight.data = quant.clone()
        actual_gptq = GPTQ(actual_layer, smart_flip=SmartFlipCorrection())
        actual_gptq.apply_saved_raw_artifact(artifact)

        self.assertTrue(torch.allclose(actual_layer.weight.data, expected))

    def test_apply_saved_raw_artifact_applies_bias_correction_without_requantizing(self):
        layer = torch.nn.Linear(2, 2, bias=False)
        original = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        quant = torch.zeros_like(original)
        pre_round = torch.zeros_like(original)
        integer = torch.zeros_like(original)
        scale = torch.ones_like(original)
        zero = torch.zeros_like(original)

        base_gptq = GPTQ(layer)
        base_gptq.quantizer = SimpleNamespace(maxq=torch.tensor(15))
        quant_state = base_gptq._build_quant_state(original, pre_round, integer, scale, zero)
        activation_samples = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        artifact = {
            "quant_state": base_gptq._serialize_quant_state(quant_state),
            "activation_sums": activation_samples.sum(dim=0, dtype=torch.float64),
            "activation_count": activation_samples.shape[0],
            "activation_samples": activation_samples,
            "dead": torch.tensor([False, False]),
        }

        gptq = GPTQ(layer, bias_correction=BiasCorrectionCorrection(BiasCorrectionConfig(max_samples=32)))
        gptq.apply_saved_raw_artifact(artifact)

        self.assertTrue(torch.allclose(layer.weight.data, quant))
        self.assertIsNotNone(layer.bias)
        self.assertTrue(torch.allclose(layer.bias.detach(), torch.tensor([2.0, 3.0], dtype=torch.float32)))


    def test_flatquant_model_utils_supports_mistral_repo_names(self):
        mistral_apply = object()

        class MistralConfig:
            @classmethod
            def from_pretrained(cls, _name):
                return SimpleNamespace()

        class MistralForCausalLM:
            @classmethod
            def from_pretrained(cls, _name, torch_dtype=None, config=None, use_auth_token=None, low_cpu_mem_usage=None):
                model = SimpleNamespace(config=config)
                return model

        transformers_stub = types.ModuleType("transformers")
        transformers_stub.MistralConfig = MistralConfig
        transformers_stub.MistralForCausalLM = MistralForCausalLM

        llama_utils = types.ModuleType("flatquant.model_tools.llama_utils")
        llama_utils.apply_flatquant_to_llama = object()
        llama31_utils = types.ModuleType("flatquant.model_tools.llama31_utils")
        llama31_utils.apply_flatquant_to_llama_31 = object()

        mistral_utils = types.ModuleType("src.quantization.flatquant_mistral")
        mistral_utils.apply_flatquant_to_mistral = mistral_apply

        backups = {name: sys.modules.get(name) for name in [
            "transformers",
            "flatquant.model_tools.llama_utils",
            "flatquant.model_tools.llama31_utils",
            "src.quantization.flatquant_mistral",
        ]}
        try:
            sys.modules["transformers"] = transformers_stub
            sys.modules["flatquant.model_tools.llama_utils"] = llama_utils
            sys.modules["flatquant.model_tools.llama31_utils"] = llama31_utils
            sys.modules["src.quantization.flatquant_mistral"] = mistral_utils

            import flatquant.model_utils as model_utils
            importlib.reload(model_utils)

            _model, apply_fn = model_utils.get_model("mistralai/Mistral-7B-v0.3", hf_token=None)
            self.assertIs(apply_fn, mistral_apply)
        finally:
            for name, module in backups.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module


    def test_flatquant_select_apply_fn_for_llama_does_not_require_qwen_imports(self):
        quantizer = FlatQuantRTNQuantizer(
            model=SimpleNamespace(config=SimpleNamespace(model_type="llama", _name_or_path="meta-llama/Llama-3-8B")),
            tokenizer=object(),
            device="cpu",
            config=FlatQuantConfig(),
        )

        flat_utils = types.ModuleType("flatquant.flat_utils")
        flat_utils.load_flat_parameters = lambda *args, **kwargs: None
        flat_utils.reparameterize_model = lambda *args, **kwargs: None
        llama_utils = types.ModuleType("flatquant.model_tools.llama_utils")
        llama_utils.apply_flatquant_to_llama = object()
        llama31_utils = types.ModuleType("flatquant.model_tools.llama31_utils")
        llama31_utils.apply_flatquant_to_llama_31 = object()
        train_utils = types.ModuleType("flatquant.train_utils")
        train_utils.cali_flat_quant = lambda *args, **kwargs: None
        mistral_utils = types.ModuleType("src.quantization.flatquant_mistral")
        mistral_utils.apply_flatquant_to_mistral = object()

        backups = {name: sys.modules.get(name) for name in [
            "flatquant.flat_utils",
            "flatquant.model_tools.llama_utils",
            "flatquant.model_tools.llama31_utils",
            "flatquant.train_utils",
            "src.quantization.flatquant_mistral",
            "flatquant.model_tools.qwen_utils",
        ]}
        try:
            sys.modules["flatquant.flat_utils"] = flat_utils
            sys.modules["flatquant.model_tools.llama_utils"] = llama_utils
            sys.modules["flatquant.model_tools.llama31_utils"] = llama31_utils
            sys.modules["flatquant.train_utils"] = train_utils
            sys.modules["src.quantization.flatquant_mistral"] = mistral_utils
            sys.modules.pop("flatquant.model_tools.qwen_utils", None)

            apply_fn = quantizer._select_apply_fn()

            self.assertIs(apply_fn, llama_utils.apply_flatquant_to_llama)
        finally:
            for name, module in backups.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

    def test_flatquant_model_utils_recognizes_qwen25_hf_repo_names(self):
        import flatquant.model_utils as model_utils

        original_get_qwen2 = model_utils.get_qwen2
        sentinel = object()
        model_utils.get_qwen2 = lambda model_name, hf_token: (sentinel, hf_token)
        try:
            model, token = model_utils.get_model("Qwen/Qwen2.5-7B", hf_token="abc")
            self.assertIs(model, sentinel)
            self.assertEqual(token, "abc")
        finally:
            model_utils.get_qwen2 = original_get_qwen2

    def test_flatquant_selector_supports_mistral_models(self):
        recipe = QuantizationRecipe(origin_method="flatquant", post_correction="none")
        quantizer, base_config, correction = create_quantizer(
            model=SimpleNamespace(config=SimpleNamespace(model_type="mistral", _name_or_path="mistralai/Mistral-7B-v0.3")),
            tokenizer=object(),
            device="cpu",
            args=self.make_args(),
            recipe=recipe,
        )

        self.assertIsInstance(quantizer, FlatQuantRTNQuantizer)
        self.assertIsInstance(base_config, FlatQuantConfig)
        self.assertIsNone(correction)
        self.assertEqual(quantizer.model.config.model_type, "mistral")

    def test_flatquant_extracts_hidden_states_from_tensor_or_tuple_outputs(self):
        hidden_states = torch.randn(1, 2, 3)
        self.assertIs(FlatQuantRTNQuantizer._extract_hidden_states(hidden_states), hidden_states)
        self.assertIs(FlatQuantRTNQuantizer._extract_hidden_states((hidden_states, None)), hidden_states)

    def test_flatquant_build_calibration_loader_accepts_token_tensors_without_retokenizing(self):
        class TokenizerStub:
            pad_token_id = 99
            eos_token_id = 100

            def __init__(self):
                self.calls = 0

            def __call__(self, *_args, **_kwargs):
                self.calls += 1
                raise AssertionError("tokenizer should not be called for tensor calibration samples")

        tokenizer = TokenizerStub()
        quantizer = FlatQuantRTNQuantizer(
            model=SimpleNamespace(config=SimpleNamespace(model_type="llama", _name_or_path="meta-llama/Llama-3-8B"), seqlen=6),
            tokenizer=tokenizer,
            device="cpu",
            config=FlatQuantConfig(),
        )

        token_sample = torch.tensor([[11, 12, 13, 14]], dtype=torch.long)
        loader = quantizer._build_calibration_loader([token_sample])

        self.assertEqual(tokenizer.calls, 0)
        self.assertEqual(len(loader), 1)
        input_ids, targets = loader[0]
        self.assertTrue(torch.equal(input_ids[:, :4], token_sample))
        self.assertTrue(torch.equal(input_ids[:, 4:], torch.tensor([[99, 99]], dtype=torch.long)))
        self.assertTrue(torch.equal(targets, input_ids))


    def test_run_flatquant_raw_creates_exp_dir(self):
        recipe = QuantizationRecipe(origin_method="flatquant", post_correction="none")
        model = SimpleNamespace(config=SimpleNamespace(model_type="llama", _name_or_path="meta-llama/Llama-3-8B"), seqlen=16)
        model.eval = lambda: model
        model.to = lambda _device: model
        quantizer, _base_config, _correction = create_quantizer(
            model=model,
            tokenizer=object(),
            device="cpu",
            args=self.make_args(),
            recipe=recipe,
        )

        flatquant_pkg = types.ModuleType("flatquant")
        data_utils = types.ModuleType("flatquant.data_utils")
        flat_utils = types.ModuleType("flatquant.flat_utils")
        train_utils = types.ModuleType("flatquant.train_utils")
        utils = types.ModuleType("flatquant.utils")
        gptq_utils = types.ModuleType("gptq_utils")

        data_utils.get_loaders = lambda *args, **kwargs: []
        flat_utils.load_flat_parameters = lambda *args, **kwargs: None
        flat_utils.load_flat_matrices = lambda *args, **kwargs: None
        flat_utils.save_flat_matrices = lambda *args, **kwargs: None
        flat_utils.reparameterize_model = lambda _model: None
        flat_utils.save_quantized_weights_with_safetensors = lambda *args, **kwargs: None
        gptq_utils.gptq_fwrd = lambda *args, **kwargs: {}
        gptq_utils.rtn_fwrd = lambda *args, **kwargs: {}
        utils.DEV = "cpu"
        utils.distribute_model = lambda _model: None

        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir) / "nested" / "smart-flip-flatquant"
            flatquant_args = quantizer._build_flatquant_args(0)
            flatquant_args.exp_dir = str(exp_dir)
            quantizer._build_flatquant_args = lambda _nsamples: flatquant_args
            quantizer._select_apply_fn = lambda: (lambda _args, model_obj: model_obj)

            observed = {}

            def fake_cali(args, model_obj, loader, device, logger=None):
                observed["exists"] = Path(args.exp_dir).exists()
                observed["dir"] = args.exp_dir

            train_utils.cali_flat_quant = fake_cali

            backups = {
                name: sys.modules.get(name)
                for name in [
                    "flatquant",
                    "flatquant.data_utils",
                    "flatquant.flat_utils",
                    "flatquant.train_utils",
                    "flatquant.utils",
                    "gptq_utils",
                ]
            }
            try:
                sys.modules["flatquant"] = flatquant_pkg
                sys.modules["flatquant.data_utils"] = data_utils
                sys.modules["flatquant.flat_utils"] = flat_utils
                sys.modules["flatquant.train_utils"] = train_utils
                sys.modules["flatquant.utils"] = utils
                sys.modules["gptq_utils"] = gptq_utils

                quantizer._run_flatquant_raw(n_samples=0)

                self.assertEqual(observed["dir"], str(exp_dir))
                self.assertTrue(observed["exists"])
            finally:
                for name, module in backups.items():
                    if module is None:
                        sys.modules.pop(name, None)
                    else:
                        sys.modules[name] = module

    def test_flatquant_quantize_model_sequential_calls_run_flatquant_raw_with_grad_enabled(self):
        recipe = QuantizationRecipe(origin_method="flatquant", post_correction="none")
        quantizer, _base_config, _correction = create_quantizer(
            model=SimpleNamespace(config=SimpleNamespace(model_type="llama", _name_or_path="meta-llama/Llama-3-8B"), seqlen=16),
            tokenizer=object(),
            device="cpu",
            args=self.make_args(),
            recipe=recipe,
        )

        seen = {}

        def fake_run_flatquant_raw(_n_samples, reuse_flat_parameters_path=None):
            seen["grad_enabled"] = torch.is_grad_enabled()
            seen["reuse_flat_parameters_path"] = reuse_flat_parameters_path
            raise RuntimeError("stop-after-run-flatquant-raw")

        quantizer._run_flatquant_raw = fake_run_flatquant_raw

        with self.assertRaisesRegex(RuntimeError, "stop-after-run-flatquant-raw"):
            quantizer.quantize_model_sequential(["sample"], n_samples=1)

        self.assertTrue(seen["grad_enabled"])

    def test_run_flatquant_raw_returns_trainloader_and_args(self):
        recipe = QuantizationRecipe(origin_method="flatquant", post_correction="smart_flip")
        model = SimpleNamespace(config=SimpleNamespace(model_type="llama", _name_or_path="meta-llama/Llama-3-8B"), seqlen=16)
        model.eval = lambda: model
        model.to = lambda _device: model
        quantizer, _base_config, _correction = create_quantizer(
            model=model,
            tokenizer=object(),
            device="cpu",
            args=self.make_args(),
            recipe=recipe,
        )

        flatquant_pkg = types.ModuleType("flatquant")
        data_utils = types.ModuleType("flatquant.data_utils")
        flat_utils = types.ModuleType("flatquant.flat_utils")
        train_utils = types.ModuleType("flatquant.train_utils")
        utils = types.ModuleType("flatquant.utils")
        gptq_utils = types.ModuleType("gptq_utils")

        expected_loader = [("tokens", "targets")]
        data_utils.get_loaders = lambda *args, **kwargs: expected_loader
        flat_utils.load_flat_parameters = lambda *args, **kwargs: None
        flat_utils.load_flat_matrices = lambda *args, **kwargs: None
        flat_utils.save_flat_matrices = lambda *args, **kwargs: None
        flat_utils.reparameterize_model = lambda _model: None
        flat_utils.save_quantized_weights_with_safetensors = lambda *args, **kwargs: None
        train_utils.cali_flat_quant = lambda *args, **kwargs: None
        utils.DEV = "cpu"
        utils.distribute_model = lambda _model: None
        gptq_utils.gptq_fwrd = lambda *args, **kwargs: {}
        gptq_utils.rtn_fwrd = lambda *args, **kwargs: {}

        quantizer._select_apply_fn = lambda: (lambda _args, model_obj: model_obj)

        backups = {
            name: sys.modules.get(name)
            for name in [
                "flatquant",
                "flatquant.data_utils",
                "flatquant.flat_utils",
                "flatquant.train_utils",
                "flatquant.utils",
                "gptq_utils",
            ]
        }
        try:
            sys.modules["flatquant"] = flatquant_pkg
            sys.modules["flatquant.data_utils"] = data_utils
            sys.modules["flatquant.flat_utils"] = flat_utils
            sys.modules["flatquant.train_utils"] = train_utils
            sys.modules["flatquant.utils"] = utils
            sys.modules["gptq_utils"] = gptq_utils

            trainloader, flatquant_args = quantizer._run_flatquant_raw(n_samples=1)

            self.assertIs(trainloader, expected_loader)
            self.assertEqual(flatquant_args.nsamples, 1)
        finally:
            for name, module in backups.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

    def test_run_flatquant_raw_reuses_saved_flat_parameters_without_recalibration(self):
        recipe = QuantizationRecipe(origin_method="flatquant", post_correction="smart_flip")
        model = SimpleNamespace(config=SimpleNamespace(model_type="llama", _name_or_path="meta-llama/Llama-3-8B"), seqlen=16)
        model.eval = lambda: model
        model.to = lambda _device: model
        quantizer, _base_config, _correction = create_quantizer(
            model=model,
            tokenizer=object(),
            device="cpu",
            args=self.make_args(),
            recipe=recipe,
        )

        flatquant_pkg = types.ModuleType("flatquant")
        data_utils = types.ModuleType("flatquant.data_utils")
        flat_utils = types.ModuleType("flatquant.flat_utils")
        train_utils = types.ModuleType("flatquant.train_utils")
        utils = types.ModuleType("flatquant.utils")
        gptq_utils = types.ModuleType("gptq_utils")

        expected_loader = [("tokens", "targets")]
        data_utils.get_loaders = lambda *args, **kwargs: expected_loader
        observed = {"load_path": None, "cali_called": False}
        flat_utils.load_flat_parameters = lambda _args, _model, path=None: observed.__setitem__("load_path", path)
        flat_utils.load_flat_matrices = lambda *args, **kwargs: None
        flat_utils.save_flat_matrices = lambda *args, **kwargs: None
        flat_utils.reparameterize_model = lambda _model: None
        flat_utils.save_quantized_weights_with_safetensors = lambda *args, **kwargs: None
        train_utils.cali_flat_quant = lambda *args, **kwargs: observed.__setitem__("cali_called", True)
        utils.DEV = "cpu"
        utils.distribute_model = lambda _model: None
        gptq_utils.gptq_fwrd = lambda *args, **kwargs: {}
        gptq_utils.rtn_fwrd = lambda *args, **kwargs: {}

        quantizer._select_apply_fn = lambda: (lambda _args, model_obj: model_obj)

        backups = {
            name: sys.modules.get(name)
            for name in [
                "flatquant",
                "flatquant.data_utils",
                "flatquant.flat_utils",
                "flatquant.train_utils",
                "flatquant.utils",
                "gptq_utils",
            ]
        }
        try:
            sys.modules["flatquant"] = flatquant_pkg
            sys.modules["flatquant.data_utils"] = data_utils
            sys.modules["flatquant.flat_utils"] = flat_utils
            sys.modules["flatquant.train_utils"] = train_utils
            sys.modules["flatquant.utils"] = utils
            sys.modules["gptq_utils"] = gptq_utils

            trainloader, flatquant_args = quantizer._run_flatquant_raw(n_samples=1, reuse_flat_parameters_path="/tmp/raw-flatquant")

            self.assertIs(trainloader, expected_loader)
            self.assertEqual(flatquant_args.nsamples, 1)
            self.assertEqual(observed["load_path"], "/tmp/raw-flatquant")
            self.assertFalse(observed["cali_called"])
        finally:
            for name, module in backups.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

    def test_flatquant_root_points_to_vendored_package(self):
        expected_root = Path(__file__).resolve().parents[1] / "flatquant"
        self.assertEqual(FlatQuantRTNQuantizer._flatquant_root(), expected_root)
        self.assertTrue((expected_root / "__init__.py").exists())

    def test_flatquant_aligns_reused_layer_modules_to_wrapped_weight_devices(self):
        class FakeMover:
            def __init__(self):
                self.to_calls = []

            def to(self, device):
                self.to_calls.append(device)
                return self

        class FakeFlatLinear:
            def __init__(self, device):
                self.linear = SimpleNamespace(weight=SimpleNamespace(device=device))

        layer = SimpleNamespace(
            self_attn=FakeMover(),
            mlp=FakeMover(),
            input_layernorm=FakeMover(),
            post_attention_layernorm=FakeMover(),
        )
        layer.self_attn.q_proj = FakeFlatLinear(torch.device("meta"))
        layer.mlp.up_proj = FakeFlatLinear(torch.device("cpu"))
        model = SimpleNamespace(model=SimpleNamespace(layers=[layer]))

        FlatQuantRTNQuantizer._align_reused_flatquant_module_devices(model)

        self.assertEqual(layer.self_attn.to_calls, [torch.device("meta")])
        self.assertEqual(layer.input_layernorm.to_calls, [torch.device("meta")])
        self.assertEqual(layer.mlp.to_calls, [torch.device("cpu")])
        self.assertEqual(layer.post_attention_layernorm.to_calls, [torch.device("cpu")])

    def test_flatquant_quantize_module_records_stage_diagnostics_for_raw_rtn(self):
        quantizer = FlatQuantRTNQuantizer(
            model=SimpleNamespace(config=SimpleNamespace(model_type="llama", _name_or_path="meta-llama/Llama-3-8B"), seqlen=16),
            tokenizer=object(),
            device="cpu",
            config=FlatQuantConfig(debug_diagnostics=True),
        )
        module = torch.nn.Linear(2, 2, bias=False)
        module.weight.data = torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=torch.float32)
        activation_batches = [
            torch.tensor([[1.0, 0.5], [-1.0, 2.0]], dtype=torch.float32),
        ]

        quantizer._quantize_module("model.layers.0.self_attn.q_proj.linear", module, activation_batches)

        stats = quantizer.layer_stats["model.layers.0.self_attn.q_proj.linear"]
        self.assertIn("stage_diagnostics", stats)
        self.assertIn("post_flatquant", stats["stage_diagnostics"])
        self.assertIn("post_rtn", stats["stage_diagnostics"])
        self.assertIn("evaluation_target", quantizer.build_evaluation_target())
        self.assertEqual(quantizer.build_evaluation_target()["evaluation_target"]["kind"], "in_memory_model")
        self.assertGreaterEqual(stats["stage_diagnostics"]["post_rtn"]["output_mse"], 0.0)

    def test_flatquant_quantize_module_records_post_correction_stage_diagnostics(self):
        class FakeCorrection:
            def prepare_activation_means(self, mean):
                return mean

            def apply(self, quant_state, _post_mean):
                corrected_weight = quant_state.dequantize_truncated() + 0.125
                return corrected_weight, 12.5, {"total": 3}

        quantizer = FlatQuantRTNQuantizer(
            model=SimpleNamespace(config=SimpleNamespace(model_type="llama", _name_or_path="meta-llama/Llama-3-8B"), seqlen=16),
            tokenizer=object(),
            device="cpu",
            config=FlatQuantConfig(debug_diagnostics=True),
            post_correction=FakeCorrection(),
        )
        module = torch.nn.Linear(2, 2, bias=False)
        module.weight.data = torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=torch.float32)
        activation_batches = [
            torch.tensor([[1.0, 0.5], [-1.0, 2.0]], dtype=torch.float32),
        ]

        quantizer._quantize_module("model.layers.0.mlp.down_proj.linear", module, activation_batches)

        stats = quantizer.layer_stats["model.layers.0.mlp.down_proj.linear"]
        self.assertIn("post_correction", stats["stage_diagnostics"])
        self.assertIn("post_correction_error", stats)
        self.assertEqual(stats["flip_stats"]["total"], 3)

if __name__ == "__main__":
    unittest.main()
