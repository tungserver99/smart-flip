import unittest
from types import SimpleNamespace

from src.smart_flip.quantization.awq import AWQConfig, AWQQuantizerXL
from src.smart_flip.quantization.pipeline import QuantizationRecipe, create_quantizer
from src.smart_flip.post_correction.smart_flip import SmartFlipConfig, SmartFlipCorrection


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
        )

    def test_recipe_variant_name_is_generic(self):
        self.assertEqual(QuantizationRecipe(origin_method="awq", post_correction="none").variant_name, "awq_raw")
        self.assertEqual(
            QuantizationRecipe(origin_method="awq", post_correction="smart_flip").variant_name,
            "awq_flip",
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


if __name__ == "__main__":
    unittest.main()
