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

from src.evaluation.flatquant_runner import run_flatquant_perplexity_evaluation
from src.evaluation.sliding_window import SlidingWindowEvaluator
from src.io_utils import dump_json

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
    return {
        key: value
        for key, value in vars(args).items()
        if not key.startswith("_") and not callable(value)
    }


def resolve_model_reference(model_ref: str, models_root: str = "/models") -> str:
    model_path = Path(model_ref)
    if model_path.exists():
        return str(model_path)

    candidate = Path(models_root) / model_ref
    if candidate.exists():
        return str(candidate)

    return model_ref


def run_perplexity_evaluation(args, model_paths: dict[str, str]) -> dict:
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
    dump_json(output_path, results, indent=2)


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
    from src.quantization.gptq import GPTQ_RAW_ARTIFACT_FILENAME

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    resolved_source_model = resolve_model_reference(args.model_path, models_root=args.models_root)
    args.resolved_source_model = resolved_source_model
    recipe = QuantizationRecipe(
        origin_method=args.origin_method,
        post_correction=args.post_correction,
    )
    run_name = resolve_run_name(args, recipe.variant_name)
    output_dir = build_output_dir(args.results_models_dir, recipe.variant_name, run_name)
    flatquant_raw_path = resolve_flatquant_raw_path(args, recipe)
    gptq_raw_path = resolve_gptq_raw_path(args, recipe)
    hf_token = resolve_hf_token()
    resolved_model = gptq_raw_path if gptq_raw_path is not None else resolved_source_model

    if recipe.origin_method == "flatquant":
        from flatquant import model_utils as fq_model_utils
        model, _apply_fn = fq_model_utils.get_model(resolved_model, hf_token)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(resolved_model, use_fast=False, use_auth_token=hf_token)
    else:
        tokenizer = AutoTokenizer.from_pretrained(resolved_model, trust_remote_code=True, token=hf_token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "token": hf_token,
        }
        if recipe.origin_method != "gptq":
            model_kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(
            resolved_model,
            **model_kwargs,
        )
        model.eval()

    calibration_data = []
    if gptq_raw_path is None:
        calibration_data = load_calibration_data(
            args.calib_dataset,
            tokenizer,
            n_samples=args.n_calib,
            seqlen=args.calib_seqlen,
            seed=args.seed,
            return_tensors=recipe.origin_method in {"flatquant", "gptq"},
            cache_dir=args.calibration_cache_dir,
        )

    quantizer, base_config, correction = create_quantizer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        args=args,
        recipe=recipe,
    )
    if recipe.origin_method == "flatquant":
        quantizer.set_artifact_dir(output_dir)
        quantizer.quantize_model_sequential(
            calibration_data,
            n_samples=args.n_calib,
            reuse_flat_parameters_path=flatquant_raw_path,
        )
    elif recipe.origin_method == "gptq" and gptq_raw_path is not None:
        quantizer.apply_post_correction_from_raw_artifacts(gptq_raw_path)
    else:
        quantizer.quantize_model_sequential(calibration_data, n_samples=args.n_calib)
    if recipe.origin_method == "flatquant":
        args._evaluation_model_spec = quantizer.build_evaluation_target()

    model.save_pretrained(output_dir, safe_serialization=recipe.origin_method != "flatquant")
    tokenizer.save_pretrained(output_dir)
    # if recipe.origin_method == "gptq" and recipe.post_correction == "none":
    #     quantizer.save_raw_artifacts(output_dir)

    metadata = {
        "variant": recipe.variant_name,
        "origin_method": recipe.origin_method,
        "post_correction": recipe.post_correction,
        "source_model": args.model_path,
        "resolved_source_model": resolved_source_model,
        "flatquant_raw_path": flatquant_raw_path,
        "gptq_raw_path": gptq_raw_path,
        "gptq_raw_artifact_file": GPTQ_RAW_ARTIFACT_FILENAME if recipe.origin_method == "gptq" and recipe.post_correction == "none" else None,
        "config": build_metadata_config(args),
        "base_config": base_config.__dict__,
        "post_correction_config": correction.config.__dict__ if correction is not None else None,
        "layer_stats": quantizer.layer_stats,
        "evaluation_target": quantizer.describe_evaluation_target() if hasattr(quantizer, "describe_evaluation_target") else {"kind": "saved_model_dir", "path": str(output_dir)},
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


def should_preserve_quantized_output(recipe) -> bool:
    return recipe.origin_method in {"flatquant", "gptq"} and recipe.post_correction == "none"


def resolve_flatquant_raw_path(args, recipe):
    if recipe.origin_method != "flatquant" or recipe.post_correction == "none":
        return None
    return getattr(args, "flatquant_raw_path", None)


def resolve_gptq_raw_path(args, recipe):
    if recipe.origin_method != "gptq" or recipe.post_correction == "none":
        return None
    raw_path = getattr(args, "gptq_raw_path", None)
    if raw_path is None:
        return None
    return resolve_model_reference(raw_path, models_root=args.models_root)

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
    evaluation_target = getattr(args, "_evaluation_model_spec", None) or str(output_dir)
    try:
        evaluate_model_paths(
            args,
            {build_quantized_model_key(recipe.post_correction): evaluation_target},
            variant=recipe.variant_name,
        )
    finally:
        if not should_preserve_quantized_output(recipe):
            shutil.rmtree(output_dir, ignore_errors=True)
        if hasattr(args, "_evaluation_model_spec"):
            delattr(args, "_evaluation_model_spec")


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
        cmd.add_argument("--origin-method", choices=["awq", "flatquant", "gptq"], default="awq")
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
        cmd.add_argument("--gptq-percdamp", type=float, default=0.01)
        cmd.add_argument("--gptq-sym", action="store_true", default=False)
        cmd.add_argument("--gptq-act-order", action="store_true", default=False)
        cmd.add_argument("--gptq-true-sequential", action="store_true", default=True)
        cmd.add_argument("--no-gptq-true-sequential", dest="gptq_true_sequential", action="store_false")
        cmd.add_argument("--gptq-static-groups", action="store_true", default=False)
        cmd.add_argument("--gptq-mse", action="store_true", default=False)
        cmd.add_argument("--flatquant-epochs", type=int, default=15)
        cmd.add_argument("--flatquant-cali-bsz", type=int, default=4)
        cmd.add_argument("--flatquant-lr", type=float, default=5e-3)
        cmd.add_argument("--flatquant-cali-trans", action="store_true", default=True)
        cmd.add_argument("--no-flatquant-cali-trans", dest="flatquant_cali_trans", action="store_false")
        cmd.add_argument("--flatquant-add-diag", action="store_true", default=True)
        cmd.add_argument("--no-flatquant-add-diag", dest="flatquant_add_diag", action="store_false")
        cmd.add_argument("--flatquant-lwc", action="store_true", default=True)
        cmd.add_argument("--no-flatquant-lwc", dest="flatquant_lwc", action="store_false")
        cmd.add_argument("--flatquant-lac", action="store_true", default=True)
        cmd.add_argument("--no-flatquant-lac", dest="flatquant_lac", action="store_false")
        cmd.add_argument("--flatquant-diag-init", choices=["sq_style", "one_style"], default="sq_style")
        cmd.add_argument("--flatquant-diag-alpha", type=float, default=0.3)
        cmd.add_argument("--flatquant-debug-diagnostics", action="store_true", default=False)
        cmd.add_argument("--flatquant-debug-sample-limit", type=int, default=256)
        add_eval_args(cmd)

    float_model = subparsers.add_parser("float_model", help="Evaluate the original float model only")
    float_model.add_argument("--model-path", required=True, help="HF model name or local model path")
    add_eval_args(float_model)
    float_model.set_defaults(func=run_float_model)

    quantize = subparsers.add_parser("quantize", help="Quantize with AWQ and an optional post-correction, then evaluate that model")
    add_quant_args(quantize)
    quantize.add_argument("--post-correction", choices=["none", "smart_flip", "bias_correction"], default="none")
    quantize.add_argument("--flatquant-raw-path", default=None)
    quantize.add_argument("--gptq-raw-path", default=None)
    quantize.set_defaults(func=run_quantize_with_evaluation)

    raw_quantize = subparsers.add_parser("raw_quantize", help="Quantize with raw AWQ, then evaluate that model")
    add_quant_args(raw_quantize)
    raw_quantize.add_argument("--flatquant-raw-path", default=None)
    raw_quantize.add_argument("--gptq-raw-path", default=None)
    raw_quantize.set_defaults(func=run_raw_quantize, post_correction="none")

    flip_quantize = subparsers.add_parser("flip_quantize", help="Quantize with AWQ plus smart flip, then evaluate that model")
    add_quant_args(flip_quantize)
    flip_quantize.add_argument("--flatquant-raw-path", default=None)
    flip_quantize.add_argument("--gptq-raw-path", default=None)
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
