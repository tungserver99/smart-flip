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

## Modes

- `float_model`: evaluate the original float model only
- `raw_quantize`: build standard AWQ, then evaluate that quantized model
- `flip_quantize`: build AWQ + smart flip, then evaluate that quantized model
- `compare_all`: evaluate float, AWQ raw, and AWQ + flip together

## Examples

Evaluate the float model:

```bash
python main.py float_model \
  --model-path <model_path>
```

Build and evaluate standard AWQ:

```bash
python main.py raw_quantize \
  --model-path <model_path> \
  --run-name <run_name>
```

Build and evaluate AWQ + smart flip:

```bash
python main.py flip_quantize \
  --model-path <model_path> \
  --run-name <run_name>
```

Compare all three milestones:

```bash
python main.py compare_all \
  --model-path <float_model_path_or_hf_name> \
  --awq-raw-path <results/models/awq_raw/...> \
  --awq-flip-path <results/models/awq_flip/...>
```

## Notes

- AWQ alpha search is intentionally fixed to run on raw AWQ, not on the flipped variant.
- Flip behavior is controlled by dedicated arguments such as `--knee-tolerance` and `--max-flip-percent`.
- Evaluation currently reports perplexity only.
- Old scripts are archived in `legacy/` for comparison and recovery.
