import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from src.quantization.flatquant import FlatQuantConfig, FlatQuantRTNQuantizer


class DummyCorrection:
    def __init__(self):
        self.config = SimpleNamespace(max_samples=16)


class FlatQuantPostCorrectionTests(unittest.TestCase):
    def test_post_correction_runs_raw_flatquant_first(self):
        model = SimpleNamespace(
            seqlen=2048,
            config=SimpleNamespace(use_cache=False, hidden_size=4),
            model=SimpleNamespace(layers=[]),
            eval=lambda: None,
        )
        tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=0)
        quantizer = FlatQuantRTNQuantizer(
            model=model,
            tokenizer=tokenizer,
            device="cpu",
            config=FlatQuantConfig(),
            post_correction=DummyCorrection(),
        )

        calls = {"raw": 0}

        def fake_run_raw(_n_samples, reuse_flat_parameters_path=None):
            calls["raw"] += 1
            return [], SimpleNamespace()

        quantizer._run_flatquant_raw = fake_run_raw
        quantizer._build_calibration_loader = lambda _samples: []
        quantizer._get_layer_inputs = lambda _loader: (torch.zeros((0,)), None, None)

        quantizer.quantize_model_sequential([], n_samples=0, reuse_flat_parameters_path=None)

        self.assertEqual(calls["raw"], 1)


    def test_post_correction_uses_post_flatquant_weights_for_flip_stage(self):
        class DummyLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(4, 4, bias=False)

        class DummyModel:
            def __init__(self, layer):
                self.seqlen = 2048
                self.config = SimpleNamespace(use_cache=False, hidden_size=4)
                self.model = SimpleNamespace(layers=[layer])

            def eval(self):
                return self

        layer = DummyLayer()
        layer.proj.weight.data.fill_(1.0)
        model = DummyModel(layer)
        tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=0)
        quantizer = FlatQuantRTNQuantizer(
            model=model,
            tokenizer=tokenizer,
            device="cpu",
            config=FlatQuantConfig(),
            post_correction=DummyCorrection(),
        )

        captured = {}

        def fake_run_raw(_n_samples, reuse_flat_parameters_path=None):
            layer.proj.weight.data.fill_(2.0)
            quantizer._rtn_quantizers = {"model.layers.0.proj.linear": object()}
            return [], SimpleNamespace()

        quantizer._run_flatquant_raw = fake_run_raw
        quantizer._build_calibration_loader = lambda _samples: []
        quantizer._get_layer_inputs = lambda _loader: (torch.zeros((0, 1, 4)), None, None)
        quantizer._find_linear_modules = lambda _layer, name="": {"proj.linear": layer.proj}
        quantizer._collect_subset_activations = lambda _layer, subset, _inps, _mask, _pos: (
            {name: [] for name in subset},
            torch.zeros((0, 1, 4)),
        )

        def fake_quantize_module(name, module, activation_batches, float_weight_override=None, quantizer_override=None):
            captured["name"] = name
            captured["float_weight_override"] = float_weight_override.clone()
            captured["live_weight"] = module.weight.detach().clone()
            captured["quantizer_override"] = quantizer_override

        quantizer._quantize_module = fake_quantize_module

        quantizer.quantize_model_sequential([], n_samples=0, reuse_flat_parameters_path=None)

        self.assertEqual(captured["name"], "model.layers.0.proj.linear")
        self.assertTrue(torch.allclose(captured["live_weight"], torch.full((4, 4), 2.0)))
        self.assertTrue(torch.allclose(captured["float_weight_override"], captured["live_weight"]))
        self.assertIsNotNone(captured["quantizer_override"])


    def test_build_quant_state_from_rtn_matches_flatquant_symmetric_quantizer(self):
        from flatquant.quant_utils import WeightQuantizer

        model = SimpleNamespace(
            seqlen=2048,
            config=SimpleNamespace(use_cache=False, hidden_size=4),
            model=SimpleNamespace(layers=[]),
            eval=lambda: None,
        )
        tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=0)
        quantizer = FlatQuantRTNQuantizer(
            model=model,
            tokenizer=tokenizer,
            device="cpu",
            config=FlatQuantConfig(),
            post_correction=None,
        )

        float_weight = torch.tensor([[-1.2, -0.3, 0.2, 1.1]], dtype=torch.float32)
        rtn_quantizer = WeightQuantizer()
        rtn_quantizer.configure(4, perchannel=True, sym=True, mse=False)
        rtn_quantizer.find_params(float_weight)

        quant_state = quantizer._build_quant_state_from_rtn(float_weight, rtn_quantizer)
        reconstructed = quant_state.dequantize_truncated()
        expected = rtn_quantizer.quantize(float_weight)

        self.assertTrue(torch.allclose(reconstructed, expected))

if __name__ == "__main__":
    unittest.main()
