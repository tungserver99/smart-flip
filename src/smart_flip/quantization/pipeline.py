"""Generic quantization recipe and backend selection."""

from __future__ import annotations

from dataclasses import dataclass

from src.smart_flip.post_correction.smart_flip import SmartFlipConfig, SmartFlipCorrection
from src.smart_flip.quantization.awq import AWQConfig, AWQQuantizerXL


@dataclass(frozen=True)
class QuantizationRecipe:
    origin_method: str = "awq"
    post_correction: str = "none"

    @property
    def variant_name(self) -> str:
        if self.post_correction == "smart_flip":
            return f"{self.origin_method}_flip"
        return f"{self.origin_method}_raw"


def build_awq_config(args) -> AWQConfig:
    return AWQConfig(
        bits=args.bits,
        n_grid=args.n_grid,
        group_size=args.group_size,
        max_tokens_per_sample=args.max_tokens_per_sample,
        layer_batch_size=args.layer_batch_size,
        lmhead_chunks=args.lmhead_chunks,
    )


def build_post_correction(args, recipe: QuantizationRecipe):
    if recipe.post_correction == "none":
        return None
    if recipe.post_correction == "smart_flip":
        return SmartFlipCorrection(
            SmartFlipConfig(
                knee_tolerance=args.knee_tolerance,
                max_flip_percent=args.max_flip_percent,
                use_james_stein=args.use_james_stein,
            )
        )
    raise NotImplementedError(f"Unsupported post correction: {recipe.post_correction}")


def create_quantizer(model, tokenizer, device: str, args, recipe: QuantizationRecipe):
    correction = build_post_correction(args, recipe)

    if recipe.origin_method == "awq":
        config = build_awq_config(args)
        quantizer = AWQQuantizerXL(
            model=model,
            tokenizer=tokenizer,
            device=device,
            config=config,
            post_correction=correction,
        )
        return quantizer, config, correction

    raise NotImplementedError(f"Unsupported origin method: {recipe.origin_method}")
