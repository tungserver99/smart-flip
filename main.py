from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_output_dir(base_dir: str, variant: str, run_name: str | None) -> Path:
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(base_dir) / variant / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_quantize(args):
    from src.smart_flip.calibration import load_calibration_data
    from src.smart_flip.quantization.quantizer import QuantizationConfig, SmartFlipAWQQuantizerXL

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    calibration_data = load_calibration_data(
        args.calib_dataset,
        tokenizer,
        n_samples=args.n_calib,
        seqlen=args.calib_seqlen,
        seed=args.seed,
        cache_dir=args.calibration_cache_dir,
    )

    config = QuantizationConfig(
        bits=args.bits,
        n_grid=args.n_grid,
        group_size=args.group_size,
        use_flip=(args.variant == "awq_flip"),
        knee_tolerance=args.knee_tolerance,
        max_tokens_per_sample=args.max_tokens_per_sample,
        layer_batch_size=args.layer_batch_size,
        lmhead_chunks=args.lmhead_chunks,
        max_flip_percent=args.max_flip_percent,
        use_james_stein=args.use_james_stein,
    )

    quantizer = SmartFlipAWQQuantizerXL(model=model, tokenizer=tokenizer, device=device, config=config)
    quantizer.quantize_model_sequential(calibration_data, n_samples=args.n_calib)

    output_dir = build_output_dir(args.results_models_dir, args.variant, args.run_name)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "variant": args.variant,
        "source_model": args.model_path,
        "config": vars(args),
        "quantizer_config": config.__dict__,
        "layer_stats": quantizer.layer_stats,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"\nSaved {args.variant} model to {output_dir}")


def run_evaluate(args):
    from src.smart_flip.evaluation.sliding_window import SlidingWindowEvaluator

    evaluator = SlidingWindowEvaluator(
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=args.seed,
        stride=args.stride,
        max_length=args.max_length,
        cache_dir=args.eval_cache_dir,
    )

    model_paths = {"fp": args.fp_model_path}
    if args.awq_raw_path:
        model_paths["awq_raw"] = args.awq_raw_path
    if args.awq_flip_path:
        model_paths["awq_flip"] = args.awq_flip_path

    evaluator.run(model_paths, include_c4=args.include_c4, c4_samples=args.c4_samples)

    output_path = Path(args.results_eval_dir) / f"{args.run_name or datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    evaluator.save_results(output_path)
    print(f"\nSaved evaluation results to {output_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="Smart Flip AWQ project entrypoint")
    subparsers = parser.add_subparsers(dest="command", required=True)

    quantize = subparsers.add_parser("quantize", help="Create awq_raw or awq_flip model")
    quantize.add_argument("--model-path", required=True, help="HF model name or local model path")
    quantize.add_argument("--variant", choices=["awq_raw", "awq_flip"], required=True)
    quantize.add_argument("--run-name", default=None)
    quantize.add_argument("--results-models-dir", default="./results/models")
    quantize.add_argument("--calibration-cache-dir", default="./data/cache/calibration")
    quantize.add_argument("--calib-dataset", choices=["c4", "wikitext2", "wikitext2-simple"], default="c4")
    quantize.add_argument("--n-calib", type=int, default=128)
    quantize.add_argument("--calib-seqlen", type=int, default=2048)
    quantize.add_argument("--seed", type=int, default=42)
    quantize.add_argument("--bits", type=int, default=4, choices=[3, 4])
    quantize.add_argument("--n-grid", type=int, default=20)
    quantize.add_argument("--group-size", type=int, default=128)
    quantize.add_argument("--max-tokens-per-sample", type=int, default=2048)
    quantize.add_argument("--layer-batch-size", type=int, default=16)
    quantize.add_argument("--lmhead-chunks", type=int, default=4)
    quantize.add_argument("--use-james-stein", action="store_true", default=True)
    quantize.add_argument("--no-james-stein", dest="use_james_stein", action="store_false")
    quantize.add_argument("--knee-tolerance", type=float, default=0.0)
    quantize.add_argument("--max-flip-percent", type=float, default=0.05)
    quantize.set_defaults(func=run_quantize)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate fp, awq_raw, and awq_flip")
    evaluate.add_argument("--fp-model-path", required=True)
    evaluate.add_argument("--awq-raw-path")
    evaluate.add_argument("--awq-flip-path")
    evaluate.add_argument("--run-name", default=None)
    evaluate.add_argument("--results-eval-dir", default="./results/eval")
    evaluate.add_argument("--eval-cache-dir", default="./data/cache/eval")
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--stride", type=int, default=512)
    evaluate.add_argument("--max-length", type=int, default=2048)
    evaluate.add_argument("--include-c4", action="store_true", default=True)
    evaluate.add_argument("--no-c4", dest="include_c4", action="store_false")
    evaluate.add_argument("--c4-samples", type=int, default=500)
    evaluate.set_defaults(func=run_evaluate)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
