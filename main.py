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


def evaluate_model_paths(args, model_paths: dict[str, str]):
    from src.smart_flip.evaluation.sliding_window import SlidingWindowEvaluator

    evaluator = SlidingWindowEvaluator(
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=args.seed,
        stride=args.stride,
        max_length=args.max_length,
        cache_dir=args.eval_cache_dir,
    )
    evaluator.run(model_paths, include_c4=args.include_c4, c4_samples=args.c4_samples)

    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = Path(args.results_eval_dir) / f"{run_name}.json"
    evaluator.save_results(output_path)
    print(f"\nSaved evaluation results to {output_path}")
    return output_path


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
    return output_dir


def run_float_model(args):
    evaluate_model_paths(args, {"float_model": args.model_path})


def run_raw_quantize(args):
    output_dir = run_quantize(args)
    evaluate_model_paths(args, {"raw_quantize": str(output_dir)})


def run_flip_quantize(args):
    output_dir = run_quantize(args)
    evaluate_model_paths(args, {"flip_quantize": str(output_dir)})


def run_compare_all(args):
    model_paths = {
        "float_model": args.model_path,
        "raw_quantize": args.awq_raw_path,
        "flip_quantize": args.awq_flip_path,
    }
    evaluate_model_paths(args, model_paths)


def build_parser():
    parser = argparse.ArgumentParser(description="Smart Flip AWQ project entrypoint")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def add_eval_args(cmd):
        cmd.add_argument("--run-name", default=None)
        cmd.add_argument("--results-eval-dir", default="./results/eval")
        cmd.add_argument("--eval-cache-dir", default="./data/cache/eval")
        cmd.add_argument("--seed", type=int, default=42)
        cmd.add_argument("--stride", type=int, default=512)
        cmd.add_argument("--max-length", type=int, default=2048)
        cmd.add_argument("--include-c4", action="store_true", default=True)
        cmd.add_argument("--no-c4", dest="include_c4", action="store_false")
        cmd.add_argument("--c4-samples", type=int, default=500)

    def add_quant_args(cmd):
        cmd.add_argument("--model-path", required=True, help="HF model name or local model path")
        cmd.add_argument("--results-models-dir", default="./results/models")
        cmd.add_argument("--calibration-cache-dir", default="./data/cache/calibration")
        cmd.add_argument("--calib-dataset", choices=["c4", "wikitext2", "wikitext2-simple"], default="c4")
        cmd.add_argument("--n-calib", type=int, default=128)
        cmd.add_argument("--calib-seqlen", type=int, default=2048)
        cmd.add_argument("--bits", type=int, default=4, choices=[3, 4])
        cmd.add_argument("--n-grid", type=int, default=20)
        cmd.add_argument("--group-size", type=int, default=128)
        cmd.add_argument("--max-tokens-per-sample", type=int, default=2048)
        cmd.add_argument("--layer-batch-size", type=int, default=16)
        cmd.add_argument("--lmhead-chunks", type=int, default=4)
        cmd.add_argument("--use-james-stein", action="store_true", default=True)
        cmd.add_argument("--no-james-stein", dest="use_james_stein", action="store_false")
        cmd.add_argument("--knee-tolerance", type=float, default=0.0)
        cmd.add_argument("--max-flip-percent", type=float, default=0.05)
        add_eval_args(cmd)

    float_model = subparsers.add_parser("float_model", help="Evaluate the original float model only")
    float_model.add_argument("--model-path", required=True, help="HF model name or local model path")
    add_eval_args(float_model)
    float_model.set_defaults(func=run_float_model)

    raw_quantize = subparsers.add_parser("raw_quantize", help="Quantize with raw AWQ, then evaluate that model")
    add_quant_args(raw_quantize)
    raw_quantize.set_defaults(func=run_raw_quantize, variant="awq_raw")

    flip_quantize = subparsers.add_parser("flip_quantize", help="Quantize with AWQ plus smart flip, then evaluate that model")
    add_quant_args(flip_quantize)
    flip_quantize.set_defaults(func=run_flip_quantize, variant="awq_flip")

    compare_all = subparsers.add_parser("compare_all", help="Evaluate float_model, raw_quantize, and flip_quantize together")
    compare_all.add_argument("--model-path", required=True, help="HF model name or local model path for the float model")
    compare_all.add_argument("--awq-raw-path", required=True)
    compare_all.add_argument("--awq-flip-path", required=True)
    add_eval_args(compare_all)
    compare_all.set_defaults(func=run_compare_all)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
