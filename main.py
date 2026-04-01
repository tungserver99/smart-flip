from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_LM_EVAL_TASKS = {
    "core": [
        "arc_easy",
        "arc_challenge",
        "hellaswag",
        "piqa",
        "winogrande",
    ],
    "extended": [
        "arc_easy",
        "arc_challenge",
        "hellaswag",
        "piqa",
        "winogrande",
        "boolq",
        "rte",
        "openbookqa",
        "lambada_openai",
    ],
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_runtime_env(env_path: str | Path = ".env"):
    env_file = Path(env_path)
    if not env_file.exists():
        return

    try:
        from dotenv import load_dotenv

        load_dotenv(env_file, override=False)
        return
    except ImportError:
        pass

    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def resolve_hf_token() -> str | None:
    return (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
    )


def resolve_wandb_api_key() -> str | None:
    return os.getenv("WANDB_API_KEY")


def normalize_run_value(value) -> str:
    text = f"{value:g}" if isinstance(value, float) else str(value)
    return text.replace("-", "m").replace(".", "p").replace("/", "_")


def build_model_slug(model_ref: str) -> str:
    candidate = str(model_ref).rstrip("/\\").split("/")[-1].split("\\")[-1]
    def replace_numeric_dot(match: re.Match[str]) -> str:
        start = match.start()
        if start > 0 and candidate[start - 1].lower() == "v":
            return match.group(0)
        return match.group(0).replace(".", "p")

    candidate = re.sub(r"\d+\.\d+", replace_numeric_dot, candidate)
    candidate = candidate.replace(" ", "_").replace("/", "_").replace("\\", "_")
    return candidate


def build_auto_run_name(variant: str, args, timestamp: str | None = None) -> str:
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    parts = [variant]
    model_ref = getattr(args, "resolved_source_model", None) or getattr(args, "model_path", None)
    if model_ref:
        parts.append(build_model_slug(model_ref))
    if hasattr(args, "bits"):
        parts.append(f"b{args.bits}")
    if hasattr(args, "group_size"):
        parts.append(f"g{args.group_size}")
    if variant.endswith("_flip") or "smart_flip" in variant:
        if hasattr(args, "knee_tolerance"):
            parts.append(f"k{normalize_run_value(args.knee_tolerance)}")
        if hasattr(args, "max_flip_percent"):
            parts.append(f"f{normalize_run_value(args.max_flip_percent)}")
    if hasattr(args, "seed"):
        parts.append(f"s{args.seed}")
    parts.append(timestamp)
    return "_".join(parts)


def resolve_run_name(args, variant: str) -> str:
    existing = getattr(args, "resolved_run_name", None)
    if existing:
        return existing

    run_name = args.run_name or build_auto_run_name(variant, args)
    args.resolved_run_name = run_name
    return run_name


def build_output_dir(base_dir: str, variant: str, run_name: str | None) -> Path:
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(base_dir) / variant / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_metadata_config(args) -> dict:
    return {key: value for key, value in vars(args).items() if not callable(value)}


def resolve_model_reference(model_ref: str, models_root: str = "/models") -> str:
    model_path = Path(model_ref)
    if model_path.exists():
        return str(model_path)

    candidate = Path(models_root) / model_ref
    if candidate.exists():
        return str(candidate)

    return model_ref


def run_perplexity_evaluation(args, model_paths: dict[str, str]) -> dict:
    from src.evaluation.sliding_window import SlidingWindowEvaluator

    evaluator = SlidingWindowEvaluator(
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=args.seed,
        stride=args.stride,
        max_length=args.max_length,
        cache_dir=args.eval_cache_dir,
        hf_token=resolve_hf_token(),
    )
    return evaluator.run(model_paths, include_c4=args.include_c4, c4_samples=args.c4_samples)


def run_lm_eval(args, model_paths: dict[str, str]) -> dict:
    from src.evaluation.lm_eval import LMEvalHarnessRunner

    runner = LMEvalHarnessRunner(
        tasks=args.lm_eval_tasks,
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=args.lm_eval_batch_size,
        num_fewshot=args.lm_eval_num_fewshot,
        output_dir=args.lm_eval_output_dir,
        run_name=getattr(args, "resolved_run_name", args.run_name),
        hf_token=resolve_hf_token(),
    )
    return runner.run(model_paths)


def save_evaluation_results(results: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    temp_path.replace(output_path)


def flatten_numeric_metrics(prefix: str, payload, flat: dict[str, float]):
    if isinstance(payload, dict):
        for key, value in payload.items():
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            flatten_numeric_metrics(child_prefix, value, flat)
    elif isinstance(payload, (int, float)) and not isinstance(payload, bool):
        flat[prefix] = payload


def collect_wandb_metrics(results: dict) -> dict[str, float]:
    flat: dict[str, float] = {}

    perplexity_results = results.get("perplexity", {})
    if isinstance(perplexity_results, dict):
        for dataset_name, variants in perplexity_results.items():
            if not isinstance(variants, dict):
                continue
            for metrics in variants.values():
                if not isinstance(metrics, dict):
                    continue
                perplexity = metrics.get("perplexity")
                if isinstance(perplexity, (int, float)) and not isinstance(perplexity, bool):
                    flat[f"perplexity/{dataset_name}"] = perplexity
                    break

    lm_eval_results = results.get("lm_eval", {})
    if isinstance(lm_eval_results, dict):
        for variant_payload in lm_eval_results.values():
            if not isinstance(variant_payload, dict):
                continue
            summary = variant_payload.get("summary")
            raw_results = variant_payload.get("raw", {}).get("results")
            task_results = summary if isinstance(summary, dict) else raw_results if isinstance(raw_results, dict) else None
            if not isinstance(task_results, dict):
                continue
            for task_name, metrics in task_results.items():
                if f"lm_eval/{task_name}" in flat or not isinstance(metrics, dict):
                    continue
                accuracy = metrics.get("acc,none")
                if isinstance(accuracy, (int, float)) and not isinstance(accuracy, bool):
                    flat[f"lm_eval/{task_name}"] = accuracy

    return flat


def build_wandb_tags(args, variant: str, model_slug: str) -> list[str]:
    tags = list(getattr(args, "wandb_tags", []))
    auto_tags = [
        f"model:{model_slug}",
        f"variant:{variant}",
    ]
    origin_method = getattr(args, "origin_method", None)
    if origin_method:
        auto_tags.append(f"origin:{origin_method}")
    post_correction = getattr(args, "post_correction", None)
    if post_correction:
        auto_tags.append(f"correction:{post_correction}")

    for tag in auto_tags:
        if tag not in tags:
            tags.append(tag)
    return tags


def log_results_to_wandb(args, run_name: str, variant: str, model_paths: dict[str, str], results: dict, model_slug: str):
    try:
        import wandb
    except ImportError:
        print("\nWarning: wandb is not installed. Install 'wandb' or disable logging with --no-wandb.")
        return

    api_key = resolve_wandb_api_key()
    if api_key:
        wandb.login(key=api_key, relogin=True)

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        job_type=variant,
        tags=build_wandb_tags(args, variant, model_slug),
        config=build_metadata_config(args),
        reinit=True,
    )
    if run is None:
        return

    flat_metrics = collect_wandb_metrics(results)
    if flat_metrics:
        wandb.log(flat_metrics)

    wandb.summary["variant"] = variant
    wandb.summary["model_slug"] = model_slug
    wandb.summary["model_paths"] = model_paths
    wandb.summary["source_model"] = getattr(args, "model_path", None)
    wandb.finish()


def evaluate_model_paths(args, model_paths: dict[str, str], variant: str = "evaluation"):
    run_name = resolve_run_name(args, variant)
    model_ref = getattr(args, "resolved_source_model", None) or getattr(args, "model_path", None) or next(iter(model_paths.values()))
    model_slug = build_model_slug(model_ref)
    combined_results = {
        "perplexity": run_perplexity_evaluation(args, model_paths),
    }

    if args.include_lm_eval:
        combined_results["lm_eval"] = run_lm_eval(args, model_paths)

    output_path = Path(args.results_eval_dir) / f"{run_name}.json"
    save_evaluation_results(combined_results, output_path)
    if getattr(args, "use_wandb", False):
        log_results_to_wandb(args, run_name, variant, model_paths, combined_results, model_slug)
    print(f"\nSaved evaluation results to {output_path}")
    return output_path


def run_quantize(args):
    from src.calibration import load_calibration_data
    from src.quantization.pipeline import QuantizationRecipe, create_quantizer

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    resolved_model = resolve_model_reference(args.model_path, models_root=args.models_root)
    args.resolved_source_model = resolved_model
    hf_token = resolve_hf_token()

    tokenizer = AutoTokenizer.from_pretrained(resolved_model, trust_remote_code=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        resolved_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token,
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

    recipe = QuantizationRecipe(
        origin_method=args.origin_method,
        post_correction=args.post_correction,
    )
    quantizer, base_config, correction = create_quantizer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        args=args,
        recipe=recipe,
    )
    quantizer.quantize_model_sequential(calibration_data, n_samples=args.n_calib)

    run_name = resolve_run_name(args, recipe.variant_name)
    output_dir = build_output_dir(args.results_models_dir, recipe.variant_name, run_name)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "variant": recipe.variant_name,
        "origin_method": recipe.origin_method,
        "post_correction": recipe.post_correction,
        "source_model": args.model_path,
        "resolved_source_model": resolved_model,
        "config": build_metadata_config(args),
        "base_config": base_config.__dict__,
        "post_correction_config": correction.config.__dict__ if correction is not None else None,
        "layer_stats": quantizer.layer_stats,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"\nSaved {recipe.variant_name} model to {output_dir}")
    return output_dir


def run_float_model(args):
    evaluate_model_paths(
        args,
        {"float_model": resolve_model_reference(args.model_path, models_root=args.models_root)},
        variant="float_model",
    )


def build_quantized_model_key(post_correction: str) -> str:
    if post_correction == "none":
        return "raw_quantize"
    return f"{post_correction}_quantize"


def run_quantize_with_evaluation(args):
    from src.quantization.pipeline import QuantizationRecipe

    recipe = QuantizationRecipe(
        origin_method=getattr(args, "origin_method", "awq"),
        post_correction=getattr(args, "post_correction", "none"),
    )
    output_dir = run_quantize(args)
    try:
        evaluate_model_paths(
            args,
            {build_quantized_model_key(recipe.post_correction): str(output_dir)},
            variant=recipe.variant_name,
        )
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def run_raw_quantize(args):
    args.post_correction = "none"
    run_quantize_with_evaluation(args)


def run_flip_quantize(args):
    args.post_correction = "smart_flip"
    run_quantize_with_evaluation(args)

def run_compare_all(args):
    model_paths = {
        "float_model": resolve_model_reference(args.model_path, models_root=args.models_root),
        "raw_quantize": resolve_model_reference(args.raw_path, models_root=args.models_root),
        "flip_quantize": resolve_model_reference(args.flip_path, models_root=args.models_root),
    }
    evaluate_model_paths(args, model_paths, variant="compare_all")


def build_parser():
    parser = argparse.ArgumentParser(description="Smart Flip quantization project entrypoint")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def add_eval_args(cmd):
        cmd.add_argument("--models-root", default="/models")
        cmd.add_argument("--run-name", default=None)
        cmd.add_argument("--results-eval-dir", default="./results/eval")
        cmd.add_argument("--eval-cache-dir", default="./data/cache/eval")
        cmd.add_argument("--seed", type=int, default=42)
        cmd.add_argument("--stride", type=int, default=512)
        cmd.add_argument("--max-length", type=int, default=2048)
        cmd.add_argument("--include-c4", action="store_true", default=True)
        cmd.add_argument("--no-c4", dest="include_c4", action="store_false")
        cmd.add_argument("--c4-samples", type=int, default=500)
        cmd.add_argument("--include-lm-eval", action="store_true", default=True)
        cmd.add_argument("--no-lm-eval", dest="include_lm_eval", action="store_false")
        cmd.add_argument("--lm-eval-task-preset", choices=sorted(DEFAULT_LM_EVAL_TASKS), default="extended")
        cmd.add_argument("--lm-eval-tasks", nargs="+", default=list(DEFAULT_LM_EVAL_TASKS["extended"]))
        cmd.add_argument("--lm-eval-num-fewshot", type=int, default=None)
        cmd.add_argument("--lm-eval-batch-size", default="auto")
        cmd.add_argument("--lm-eval-output-dir", default="./results/eval/lm_eval")
        cmd.add_argument("--use-wandb", action="store_true", default=False)
        cmd.add_argument("--no-wandb", dest="use_wandb", action="store_false")
        cmd.add_argument("--wandb-project", default="smartflip")
        cmd.add_argument("--wandb-entity", default=None)
        cmd.add_argument("--wandb-tags", nargs="*", default=[])

    def add_quant_args(cmd):
        cmd.add_argument("--model-path", required=True, help="HF model name or local model path")
        cmd.add_argument("--origin-method", choices=["awq"], default="awq")
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
        cmd.add_argument("--bias-correction-samples", type=int, default=4096)
        add_eval_args(cmd)

    float_model = subparsers.add_parser("float_model", help="Evaluate the original float model only")
    float_model.add_argument("--model-path", required=True, help="HF model name or local model path")
    add_eval_args(float_model)
    float_model.set_defaults(func=run_float_model)

    quantize = subparsers.add_parser("quantize", help="Quantize with AWQ and an optional post-correction, then evaluate that model")
    add_quant_args(quantize)
    quantize.add_argument("--post-correction", choices=["none", "smart_flip", "bias_correction"], default="none")
    quantize.set_defaults(func=run_quantize_with_evaluation)

    raw_quantize = subparsers.add_parser("raw_quantize", help="Quantize with raw AWQ, then evaluate that model")
    add_quant_args(raw_quantize)
    raw_quantize.set_defaults(func=run_raw_quantize, post_correction="none")

    flip_quantize = subparsers.add_parser("flip_quantize", help="Quantize with AWQ plus smart flip, then evaluate that model")
    add_quant_args(flip_quantize)
    flip_quantize.set_defaults(func=run_flip_quantize, post_correction="smart_flip")

    compare_all = subparsers.add_parser("compare_all", help="Evaluate float_model, raw_quantize, and flip_quantize together")
    compare_all.add_argument("--model-path", required=True, help="HF model name or local model path for the float model")
    compare_all.add_argument("--raw-path", required=True)
    compare_all.add_argument("--flip-path", required=True)
    add_eval_args(compare_all)
    compare_all.set_defaults(func=run_compare_all)

    return parser


def main():
    load_runtime_env()
    parser = build_parser()
    args = parser.parse_args()
    if args.lm_eval_tasks == list(DEFAULT_LM_EVAL_TASKS["extended"]) and args.lm_eval_task_preset in DEFAULT_LM_EVAL_TASKS:
        args.lm_eval_tasks = list(DEFAULT_LM_EVAL_TASKS[args.lm_eval_task_preset])
    args.func(args)


if __name__ == "__main__":
    main()


