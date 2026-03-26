# Smart Flip AWQ

Project structure refactored from the original experimental scripts, with the algorithm still anchored to `legacy/scripts/awq_js_xl.py` and the method description in `docs/reference/post-correction-quantization-fix.md`.

## Layout

- `main.py`: single entrypoint for quantization and evaluation
- `src/smart_flip/`: project code
- `data/`: local caches and dataset artifacts
- `results/models/`: quantized model outputs
- `results/eval/`: evaluation JSON results
- `docs/reference/`: reference docs used to anchor the implementation
- `legacy/`: archived experimental scripts kept for traceability

## Quantization Architecture

The quantization flow is now split conceptually into:
- `origin_method`: the base quantizer, currently `awq`
- `post_correction`: the correction stage, currently `none` or `smart_flip`

Code ownership:
- `src/smart_flip/quantization/awq.py`: AWQ raw backend
- `src/smart_flip/quantization/state.py`: quantized tensor state shared across stages
- `src/smart_flip/post_correction/smart_flip.py`: Smart Flip post-correction
- `src/smart_flip/quantization/pipeline.py`: assembly of backend + correction
- `src/smart_flip/evaluation/sliding_window.py`: perplexity evaluation on WikiText-2 and optional C4
- `src/smart_flip/evaluation/lm_eval.py`: downstream benchmark evaluation through `lm-evaluation-harness`

Current CLI recipes:
- `raw_quantize` = `origin_method=awq` + `post_correction=none`
- `flip_quantize` = `origin_method=awq` + `post_correction=smart_flip`

This keeps today's behavior unchanged while making the pipeline ready for future backends such as GPTQ.

## Model Resolution

`--model-path` now supports both:
- a local path that already exists
- a short model name under `--models-root` (default: `/models`)
- a Hugging Face model id

Resolution order:
1. use `--model-path` directly if it exists
2. otherwise try `<models_root>/<model_path>`
3. otherwise treat it as a Hugging Face model id

This means on your server you can pass `--model-path Mistral-7B-v0.3` and it will resolve to `/models/Mistral-7B-v0.3` automatically.

## Evaluation

Every CLI mode now runs the full evaluation stack by default:
- perplexity on `WikiText-2`
- optional perplexity on `C4`
- `lm_eval` downstream tasks

Default `lm_eval` preset: `extended`
- `arc_easy`
- `arc_challenge`
- `hellaswag`
- `piqa`
- `winogrande`
- `boolq`
- `rte`
- `openbookqa`
- `lambada_openai`

Available presets:
- `core`: `arc_easy`, `arc_challenge`, `hellaswag`, `piqa`, `winogrande`
- `extended`: `core` plus `boolq`, `rte`, `openbookqa`, `lambada_openai`

Useful switches:
- `--no-lm-eval`: skip downstream benchmark evaluation
- `--lm-eval-task-preset core|extended`: choose a preset
- `--lm-eval-tasks ...`: override the preset with an explicit task list
- `--no-c4`: skip the C4 perplexity pass
- `--use-wandb`: log evaluation metrics to Weights & Biases
- `--wandb-project` / `--wandb-entity`: choose the W&B destination

The combined JSON written under `results/eval/` now contains both `perplexity` and `lm_eval` sections.

## Modes

- `float_model`: evaluate the original float model only
- `raw_quantize`: build the chosen origin method without correction, then evaluate that quantized model
- `flip_quantize`: build the chosen origin method and then apply smart flip, then evaluate that quantized model
- `compare_all`: evaluate float, raw, and corrected models together

## Examples

Evaluate the float model from `/models` with the default full evaluation suite:

```bash
python main.py float_model \
  --model-path Mistral-7B-v0.3
```

Evaluate the float model from Hugging Face:

```bash
python main.py float_model \
  --model-path mistralai/Mistral-7B-v0.3
```

Run only the core downstream benchmark preset:

```bash
python main.py float_model \
  --model-path Mistral-7B-v0.3 \
  --lm-eval-task-preset core
```

Skip `lm_eval` and keep perplexity only:

```bash
python main.py float_model \
  --model-path Mistral-7B-v0.3 \
  --no-lm-eval
```

Build and evaluate standard AWQ:

```bash
python main.py raw_quantize \
  --model-path Mistral-7B-v0.3 \
  --origin-method awq \
  --run-name <run_name>
```

Build and evaluate AWQ + smart flip:

```bash
python main.py flip_quantize \
  --model-path Mistral-7B-v0.3 \
  --origin-method awq \
  --run-name <run_name>
```

Compare all three milestones:

```bash
python main.py compare_all \
  --model-path Mistral-7B-v0.3 \
  --raw-path <results/models/awq_raw/...> \
  --flip-path <results/models/awq_flip/...>
```

## Shell Scripts

The scripts in `scripts/bash/` default to:
- `MODEL_PATH=Mistral-7B-v0.3`
- `MODELS_ROOT=/models`
- `ORIGIN_METHOD=awq`
- `USE_WANDB=1`

So on your server they will use `/models/Mistral-7B-v0.3` automatically unless you override them.

## Notes

- Install runtime dependencies from `requirements.txt`, which now includes `lm-eval`.
- If `--run-name` is omitted, the CLI now auto-generates one from the variant, tuning hyperparameters, seed, and timestamp.
- AWQ alpha search is intentionally fixed to run on raw AWQ, not on the flipped variant.
- Flip behavior is controlled by dedicated arguments such as `--knee-tolerance` and `--max-flip-percent`.
- Old scripts are archived in `legacy/` for comparison and recovery.
