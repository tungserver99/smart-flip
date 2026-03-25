# Origin Method And Post Correction Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Separate base quantization backends from post-correction logic so Smart Flip can be mounted on top of AWQ today and GPTQ later.

**Architecture:** Move AWQ raw quantization into its own backend module, move Smart Flip into a dedicated post-correction module, and let the pipeline assemble `origin_method + post_correction` into the runnable quantizer. Keep the current AWQ behavior anchored to the old implementation while changing module ownership.

**Tech Stack:** Python, PyTorch, argparse, unittest

### Task 1: Lock the assembly contract with tests

**Files:**
- Create: `tests/test_quantization_pipeline.py`

**Step 1: Write the failing test**
Define tests that require:
- `QuantizationRecipe` to expose generic variant names
- `create_quantizer(...)` to return an AWQ backend without correction for `post_correction=none`
- `create_quantizer(...)` to return an AWQ backend plus `SmartFlipCorrection` for `post_correction=smart_flip`

**Step 2: Run test to verify it fails**
Run: `D:\miniconda3\envs\topmost\python.exe -m unittest tests.test_quantization_pipeline -v`
Expected: import or contract failure because the new API does not exist yet.

**Step 3: Write minimal implementation**
Add the new backend/correction factories and classes needed to satisfy the tests.

**Step 4: Run test to verify it passes**
Run: `D:\miniconda3\envs\topmost\python.exe -m unittest tests.test_quantization_pipeline -v`
Expected: PASS.

**Step 5: Commit**
Commit after the full refactor is verified, not after this isolated task.

### Task 2: Move AWQ raw quantization into the base backend

**Files:**
- Modify: `src/smart_flip/quantization/awq.py`
- Create: `src/smart_flip/quantization/state.py`
- Modify: `src/smart_flip/quantization/quantizer.py`

**Step 1: Move the AWQ config and raw groupwise quantization state into dedicated backend files**
Implement `AWQConfig`, `AWQQuantizerXL`, and `IntegerQuantizedTensorState`.

**Step 2: Preserve the original AWQ search and sequential quantization flow**
Keep alpha search raw-only and keep lm_head chunking behavior.

**Step 3: Leave a compatibility shim for old imports**
Turn `quantizer.py` into a thin wrapper so old import paths do not explode during transition.

### Task 3: Move Smart Flip into post-correction

**Files:**
- Create: `src/smart_flip/post_correction/__init__.py`
- Create: `src/smart_flip/post_correction/smart_flip.py`

**Step 1: Move Smart Flip config and helpers out of AWQ**
Implement `SmartFlipConfig`, `SmartFlipCorrection`, `find_knee_point`, and `compute_james_stein_mean`.

**Step 2: Make correction consume quantized state instead of raw AWQ internals**
Use `IntegerQuantizedTensorState` as the contract between base quantization and post-correction.

### Task 4: Rewire pipeline and CLI

**Files:**
- Modify: `src/smart_flip/quantization/pipeline.py`
- Modify: `src/smart_flip/quantization/__init__.py`
- Modify: `main.py`
- Modify: `README.md`
- Modify: `scripts/bash/run_raw_quantize.sh`
- Modify: `scripts/bash/run_flip_quantize.sh`

**Step 1: Build `origin_method` and `post_correction` independently**
`create_quantizer(...)` should assemble the selected base quantizer with the selected correction.

**Step 2: Update metadata**
Write `base_config` and `post_correction_config` separately in run metadata.

**Step 3: Keep user-facing modes stable**
`raw_quantize` and `flip_quantize` should still work, but now as recipes assembled by the pipeline.

### Task 5: Verify and commit

**Files:**
- Modify: `tests/test_main.py`
- Run: `tests/test_metadata.py`
- Run: `tests/test_requirements.py`

**Step 1: Run the full test suite**
Run: `D:\miniconda3\envs\topmost\python.exe -m unittest tests.test_main tests.test_metadata tests.test_quantization_pipeline tests.test_requirements -v`
Expected: PASS.

**Step 2: Run CLI smoke checks**
Run:
- `D:\miniconda3\envs\topmost\python.exe main.py raw_quantize --help`
- `D:\miniconda3\envs\topmost\python.exe main.py flip_quantize --help`
- `D:\miniconda3\envs\topmost\python.exe main.py compare_all --help`
Expected: all commands print usage successfully.

**Step 3: Commit only the refactor files**
Do not include unrelated notebook changes.
