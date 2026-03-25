"""Generic quantization recipe and backend selection."""

from __future__ import annotations

from dataclasses import dataclass

from src.smart_flip.quantization.awq import AWQQuantizationConfig, AWQQuantizerXL


@dataclass(frozen=True)
class QuantizationRecipe:
    origin_method: str = "awq"
    post_correction: str = "none"

    @property
    def use_flip(self) -> bool:
        return self.post_correction == "smart_flip"

    @property
    def variant_name(self) -> str:
        if self.post_correction == "smart_flip":
            return f"{self.origin_method}_flip"
        return f"{self.origin_method}_raw"


def build_awq_config(args, recipe: QuantizationRecipe) -> AWQQuantizationConfig:
    return AWQQuantizationConfig(
        bits=args.bits,
        n_grid=args.n_grid,
        group_size=args.group_size,
        use_flip=recipe.use_flip,
        knee_tolerance=args.knee_tolerance,
        max_tokens_per_sample=args.max_tokens_per_sample,
        layer_batch_size=args.layer_batch_size,
        lmhead_chunks=args.lmhead_chunks,
        max_flip_percent=args.max_flip_percent,
        use_james_stein=args.use_james_stein,
    )


def create_quantizer(model, tokenizer, device: str, args, recipe: QuantizationRecipe):
    if recipe.origin_method == "awq":
        config = build_awq_config(args, recipe)
        return AWQQuantizerXL(model=model, tokenizer=tokenizer, device=device, config=config), config

    raise NotImplementedError(f"Unsupported origin method: {recipe.origin_method}")
