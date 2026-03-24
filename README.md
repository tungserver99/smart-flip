# Smart Flip AWQ

Project structure refactored from the original experimental scripts, with the algorithm still anchored to `awq_js_xl.py` and the method description in `post-correction-quantization-fix.md`.

## Layout

- `main.py`: single entrypoint for quantization and evaluation
- `src/smart_flip/`: project code
- `data/`: local caches and dataset artifacts
- `results/models/`: quantized model outputs
- `results/eval/`: evaluation JSON results
- `docs/reference/`: reference docs used to anchor the implementation
- `legacy/`: archived experimental scripts kept for traceability

## Model Milestones

- `fp`: original float model from `--model-path`
- `awq_raw`: standard AWQ using alpha searched on AWQ raw only
- `awq_flip`: AWQ raw plus smart flip using the same AWQ scaling logic

## Quantize

Create standard AWQ:

```bash
python main.py quantize \
  --model-path <model_path> \
  --variant awq_raw \
  --run-name <run_name>
```

Create AWQ + smart flip:

```bash
python main.py quantize \
  --model-path <model_path> \
  --variant awq_flip \
  --run-name <run_name>
```

## Evaluate

```bash
python main.py evaluate \
  --fp-model-path <fp_model_or_hf_name> \
  --awq-raw-path <results/models/awq_raw/...> \
  --awq-flip-path <results/models/awq_flip/...>
```

## Notes

- AWQ alpha search is intentionally fixed to run on raw AWQ, not on the flipped variant.
- Flip behavior is controlled by dedicated arguments such as `--knee-tolerance` and `--max-flip-percent`.
- Old scripts are archived in `legacy/` for comparison and recovery.
