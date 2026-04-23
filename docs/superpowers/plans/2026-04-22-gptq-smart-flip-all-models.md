# GPTQ Smart Flip All Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one GPTQ smart-flip bash script that runs the raw GPTQ step and a single smart-flip setting for four supported LLMs.

**Architecture:** Extend the existing `scripts/bash/smart_flip/gptq` pattern with a new aggregate runner. Keep the current per-model scripts unchanged, and add a focused test that asserts the new file exists and uses the fixed `knee=0.0` and `max_flip=0.05` settings.

**Tech Stack:** Bash, Python unittest/pytest

---

### Task 1: Add red test coverage for the aggregate script

**Files:**
- Modify: `tests/test_bash_scripts.py`
- Test: `tests/test_bash_scripts.py`

- [ ] **Step 1: Write the failing test**

```python
expected = {
    Path("smart_flip/gptq/run_all_models_single_setting.sh"),
}
content = Path("scripts/bash/smart_flip/gptq/run_all_models_single_setting.sh").read_text(encoding="utf-8")
assert 'KNEE_TOLERANCE="${KNEE_TOLERANCE:-0.0}"' in content
assert 'MAX_FLIP_PERCENT="${MAX_FLIP_PERCENT:-0.05}"' in content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_bash_scripts.py -q`
Expected: FAIL because `scripts/bash/smart_flip/gptq/run_all_models_single_setting.sh` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
# No production code in this task.
```

- [ ] **Step 4: Run test to verify it still reflects the missing script**

Run: `python3 -m pytest tests/test_bash_scripts.py -q`
Expected: FAIL with missing-file assertions for the new script.

- [ ] **Step 5: Commit**

```bash
git add tests/test_bash_scripts.py
git commit -m "test: cover aggregate gptq smart flip runner"
```

### Task 2: Add the aggregate GPTQ smart-flip runner

**Files:**
- Create: `scripts/bash/smart_flip/gptq/run_all_models_single_setting.sh`
- Test: `tests/test_bash_scripts.py`

- [ ] **Step 1: Write the failing test**

```python
assert '"meta-llama/Meta-Llama-3-8B"' in content
assert '"meta-llama/Llama-3.1-8B"' in content
assert '"mistralai/Mistral-7B-v0.3"' in content
assert '"Qwen/Qwen2.5-7B"' in content
assert '--knee-tolerance "$KNEE_TOLERANCE"' in content
assert '--max-flip-percent "$MAX_FLIP_PERCENT"' in content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_bash_scripts.py -q`
Expected: FAIL because the script content is still missing.

- [ ] **Step 3: Write minimal implementation**

```bash
MODEL_PATHS=(
  "meta-llama/Meta-Llama-3-8B"
  "meta-llama/Llama-3.1-8B"
  "mistralai/Mistral-7B-v0.3"
  "Qwen/Qwen2.5-7B"
)
KNEE_TOLERANCE="${KNEE_TOLERANCE:-0.0}"
MAX_FLIP_PERCENT="${MAX_FLIP_PERCENT:-0.05}"
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
  "$PYTHON_BIN" main.py quantize ...
done
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_bash_scripts.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/bash/smart_flip/gptq/run_all_models_single_setting.sh tests/test_bash_scripts.py docs/superpowers/plans/2026-04-22-gptq-smart-flip-all-models.md
git commit -m "feat: add aggregate gptq smart flip runner"
```
