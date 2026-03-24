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

## Modes

- `float_model`: evaluate the original float model only
- `raw_quantize`: build standard AWQ, then evaluate that quantized model
- `flip_quantize`: build AWQ + smart flip, then evaluate that quantized model
- `compare_all`: evaluate float, AWQ raw, and AWQ + flip together

## Examples

Evaluate the float model from `/models`:

```bash
python main.py float_model \
  --model-path Mistral-7B-v0.3
```

Evaluate the float model from Hugging Face:

```bash
python main.py float_model \
  --model-path mistralai/Mistral-7B-v0.3
```

Build and evaluate standard AWQ:

```bash
python main.py raw_quantize \
  --model-path Mistral-7B-v0.3 \
  --run-name <run_name>
```

Build and evaluate AWQ + smart flip:

```bash
python main.py flip_quantize \
  --model-path Mistral-7B-v0.3 \
  --run-name <run_name>
```

Compare all three milestones:

```bash
python main.py compare_all \
  --model-path Mistral-7B-v0.3 \
  --awq-raw-path <results/models/awq_raw/...> \
  --awq-flip-path <results/models/awq_flip/...>
```

## Shell Scripts

The scripts in `scripts/bash/` default to:
- `MODEL_PATH=Mistral-7B-v0.3`
- `MODELS_ROOT=/models`

So on your server they will use `/models/Mistral-7B-v0.3` automatically unless you override them.

## Notes

- AWQ alpha search is intentionally fixed to run on raw AWQ, not on the flipped variant.
- Flip behavior is controlled by dedicated arguments such as `--knee-tolerance` and `--max-flip-percent`.
- Evaluation currently reports perplexity only.
- Old scripts are archived in `legacy/` for comparison and recovery.
