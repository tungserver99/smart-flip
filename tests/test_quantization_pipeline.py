import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from src.post_correction.bias_correction import BiasCorrectionConfig, BiasCorrectionCorrection
from src.post_correction.smart_flip import SmartFlipConfig, SmartFlipCorrection
from src.quantization.awq import AWQConfig, AWQQuantizerXL
from src.quantization.flatquant import FlatQuantConfig, FlatQuantRTNQuantizer
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


    def test_flatquant_prepare_model_creates_exp_dir(self):
        recipe = QuantizationRecipe(origin_method="flatquant", post_correction="none")
        model = SimpleNamespace(config=SimpleNamespace(model_type="llama", _name_or_path="meta-llama/Llama-3-8B"), seqlen=16)
        model.eval = lambda: model
        quantizer, _base_config, _correction = create_quantizer(
            model=model,
            tokenizer=object(),
            device="cpu",
            args=self.make_args(),
            recipe=recipe,
        )

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

            quantizer._imports = {
                "cali_flat_quant": fake_cali,
                "reparameterize_model": lambda _model: None,
            }

            quantizer._prepare_model([])

            self.assertEqual(observed["dir"], str(exp_dir))
            self.assertTrue(observed["exists"])

    def test_flatquant_prepare_model_runs_with_grad_enabled(self):
        recipe = QuantizationRecipe(origin_method="flatquant", post_correction="none")
        quantizer, _base_config, _correction = create_quantizer(
            model=SimpleNamespace(config=SimpleNamespace(model_type="llama", _name_or_path="meta-llama/Llama-3-8B"), seqlen=16),
            tokenizer=object(),
            device="cpu",
            args=self.make_args(),
            recipe=recipe,
        )

        seen = {}

        def fake_build_loader(_calibration_data):
            return []

        def fake_prepare_model(_loader, reuse_flat_parameters_path=None):
            seen["grad_enabled"] = torch.is_grad_enabled()
            seen["reuse_flat_parameters_path"] = reuse_flat_parameters_path
            raise RuntimeError("stop-after-prepare")

        quantizer._build_calibration_loader = fake_build_loader
        quantizer._prepare_model = fake_prepare_model

        with self.assertRaisesRegex(RuntimeError, "stop-after-prepare"):
            quantizer.quantize_model_sequential(["sample"], n_samples=1)

        self.assertTrue(seen["grad_enabled"])



    def test_flatquant_prepare_model_can_load_saved_parameters_without_recalibration(self):
        recipe = QuantizationRecipe(origin_method="flatquant", post_correction="smart_flip")
        model = SimpleNamespace(config=SimpleNamespace(model_type="llama", _name_or_path="meta-llama/Llama-3-8B"), seqlen=16)
        model.eval = lambda: model
        quantizer, _base_config, _correction = create_quantizer(
            model=model,
            tokenizer=object(),
            device="cpu",
            args=self.make_args(),
            recipe=recipe,
        )

        flatquant_args = quantizer._build_flatquant_args(0)
        quantizer._build_flatquant_args = lambda _nsamples: flatquant_args
        quantizer._select_apply_fn = lambda: (lambda _args, model_obj: model_obj)

        observed = {"load_path": None, "cali_called": False, "reparameterized": False}

        def fake_load(args, model_obj, path=None):
            observed["load_path"] = path
            return model_obj

        def fake_cali(args, model_obj, loader, device, logger=None):
            observed["cali_called"] = True

        def fake_reparameterize(model_obj):
            observed["reparameterized"] = True
            return model_obj

        quantizer._imports = {
            "load_flat_parameters": fake_load,
            "cali_flat_quant": fake_cali,
            "reparameterize_model": fake_reparameterize,
        }

        quantizer._prepare_model([], reuse_flat_parameters_path="/tmp/raw-flatquant")

        self.assertEqual(observed["load_path"], "/tmp/raw-flatquant")
        self.assertFalse(observed["cali_called"])
        self.assertTrue(observed["reparameterized"])

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

if __name__ == "__main__":
    unittest.main()
