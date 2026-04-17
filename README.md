# smart-flip

`smart-flip` la repo nghien cuu va van hanh cho cac thu nghiem quantization xoay quanh:

- `awq`
- `flatquant`
- cac hau xu ly `smart_flip` va `bias_correction`
- danh gia perplexity va `lm-evaluation-harness`

Entrypoint chinh cua repo la `main.py`. Cac script trong `scripts/bash/` chi la wrapper de chay nhanh cac recipe pho bien.

## Cau truc repo

- `main.py`: CLI chinh cho quantization va evaluation
- `src/quantization/`: pipeline quantization, AWQ, FlatQuant adapter, bias correction
- `src/post_correction/`: `smart_flip` va cac correction stage
- `src/evaluation/`: evaluation thong thuong va evaluation cho FlatQuant
- `flatquant/`: phan code FlatQuant goc duoc repo nay tai su dung
- `scripts/bash/`: cac wrapper `.sh` de chay nhanh theo model family va recipe
- `datasets/`: dataset local ma FlatQuant loader can su dung
- `data/cache/`: cache calibration/evaluation runtime
- `results/models/`: artifact model sau quantization
- `results/eval/`: ket qua evaluation JSON
- `legacy/`: script va tai lieu cu de doi chieu

## Cai dat

```bash
pip install -r requirements.txt
```

Neu dung model private tren Hugging Face hoac can log W&B, tao file `.env` o root repo:

```bash
HF_TOKEN=...
WANDB_API_KEY=...
```

`main.py` tu dong nap `.env` neu file ton tai.

## Cach repo hoat dong

Repo co 2 nhom luong chinh:

1. `float_model`
   Danh gia model float goc.
2. `quantize`
   Quantize roi danh gia ngay sau do.

`quantize` duoc cau hinh boi:

- `--origin-method awq|flatquant`
- `--post-correction none|smart_flip|bias_correction`

Nhung mode con lai thuc chat la shortcut:

- `raw_quantize` = `post_correction=none`
- `flip_quantize` = `post_correction=smart_flip`
- `compare_all` = danh gia dong thoi `float`, `raw`, `flip`

## Model path resolution

`--model-path` duoc resolve theo thu tu:

1. dung truc tiep neu la local path ton tai
2. thu `<models_root>/<model_path>` voi `--models-root` mac dinh la `/models`
3. neu khong tim thay thi xem nhu Hugging Face model id

Vi du:

- `--model-path /models/Mistral-7B-v0.3`
- `--model-path Mistral-7B-v0.3 --models-root /models`
- `--model-path mistralai/Mistral-7B-v0.3`

## CLI co san

Xem help:

```bash
python main.py -h
python main.py quantize -h
```

### 1. Danh gia float model

```bash
python main.py float_model \
  --model-path mistralai/Mistral-7B-v0.3
```

### 2. AWQ raw

```bash
python main.py quantize \
  --model-path mistralai/Mistral-7B-v0.3 \
  --origin-method awq \
  --post-correction none \
  --bits 4 \
  --run-name awq_raw_mistral
```

### 3. AWQ + smart_flip

```bash
python main.py quantize \
  --model-path mistralai/Mistral-7B-v0.3 \
  --origin-method awq \
  --post-correction smart_flip \
  --bits 4 \
  --knee-tolerance 0.02 \
  --max-flip-percent 0.03 \
  --run-name awq_smart_flip_mistral
```

### 4. AWQ + bias_correction

```bash
python main.py quantize \
  --model-path mistralai/Mistral-7B-v0.3 \
  --origin-method awq \
  --post-correction bias_correction \
  --bits 4 \
  --bias-correction-samples 4096 \
  --run-name awq_bias_correction_mistral
```

### 5. FlatQuant raw

```bash
python main.py quantize \
  --model-path mistralai/Mistral-7B-v0.3 \
  --origin-method flatquant \
  --post-correction none \
  --bits 4 \
  --flatquant-epochs 15 \
  --flatquant-cali-bsz 4 \
  --flatquant-lr 5e-3 \
  --run-name flatquant_raw_mistral
```

### 6. FlatQuant + smart_flip

Voi `flatquant`, cac recipe co correction can tro den artifact raw truoc do bang `--flatquant-raw-path`.

```bash
python main.py quantize \
  --model-path mistralai/Mistral-7B-v0.3 \
  --origin-method flatquant \
  --post-correction smart_flip \
  --bits 4 \
  --knee-tolerance 0.02 \
  --max-flip-percent 0.03 \
  --flatquant-raw-path ./results/models/flatquant_raw/flatquant_raw_mistral \
  --run-name flatquant_smart_flip_mistral
```

### 7. FlatQuant + bias_correction

```bash
python main.py quantize \
  --model-path mistralai/Mistral-7B-v0.3 \
  --origin-method flatquant \
  --post-correction bias_correction \
  --bits 4 \
  --bias-correction-samples 4096 \
  --flatquant-raw-path ./results/models/flatquant_raw/flatquant_raw_mistral \
  --run-name flatquant_bias_correction_mistral
```

### 8. Compare all

```bash
python main.py compare_all \
  --model-path mistralai/Mistral-7B-v0.3 \
  --raw-path ./results/models/awq_raw/awq_raw_mistral \
  --flip-path ./results/models/awq_smart_flip/awq_smart_flip_mistral
```

## Evaluation

Moi lan chay evaluation se ghi JSON vao `results/eval/`.

Repo co 2 luong evaluation:

- luong mac dinh trong `src/evaluation/sliding_window.py`
  - tai WikiText-2 va C4 qua Hugging Face
  - cache vao `data/cache/eval`
- luong FlatQuant trong `src/evaluation/flatquant_runner.py`
  - dung local dataset script trong `datasets/`
  - can co du lieu local o `smart-flip/datasets`

Mac dinh:

- co WikiText-2
- co C4
- co `lm_eval`
- preset `lm_eval` mac dinh la `extended`

Mot so tuy chon huu ich:

- `--no-c4`
- `--no-lm-eval`
- `--lm-eval-task-preset core|extended`
- `--lm-eval-tasks ...`
- `--use-wandb`

## Dataset va cache

Can phan biet 2 khai niem:

1. `data/cache/...`
   Day la cache runtime do repo tu tao khi chay calibration/evaluation.
2. `datasets/...`
   Day la dataset local ma mot so FlatQuant loader can doc truc tiep.

### Dataset mac dinh cua `main.py`

Khi chay cac luong AWQ thong thuong va sliding-window evaluation, repo se tu tai du lieu qua Hugging Face:

- calibration `c4` trong `src/calibration.py`
- WikiText-2 test trong `src/evaluation/sliding_window.py`
- C4 validation trong `src/evaluation/sliding_window.py`

Ban khong can tai tay vao `smart-flip/datasets` chi de dung cac luong nay.

### Dataset local can cho FlatQuant

FlatQuant loader trong cac file sau doc dataset local:

- `flatquant/data_utils.py`
- `src/evaluation/flatquant_data_utils.py`

No tim du lieu trong:

- `datasets/wikitext`
- `datasets/allenai/c4`
- `datasets/ptb_text_only`
- `datasets/pile-val-backup`

Trong thuc te, toi thieu ban nen chuan bi `datasets/wikitext`, vi repo dang kem san dataset script va day la phan de gap nhat khi chay evaluation theo luong FlatQuant.

### Cach tai WikiText-2 vao `smart-flip/datasets`

Repo da kem san script `datasets/wikitext/wikitext.py`, script nay se doc file zip local:

- `datasets/wikitext/wikitext-2-raw-v1.zip`

Cach tai:

```bash
mkdir -p datasets/wikitext
cd datasets/wikitext
wget -O wikitext-2-raw-v1.zip \
  "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip?download=true"
cd /workspace/smart-flip
```

Sau khi tai xong, duong dan can co dang:

```text
smart-flip/
  datasets/
    wikitext/
      wikitext.py
      wikitext-2-raw-v1.zip
```

### Dataset local khac cho FlatQuant

Neu ban su dung dung cac loader tuong ung trong FlatQuant, hay dat du lieu nhu sau:

- `datasets/allenai/c4/en/c4-train.00000-of-01024.json.gz`
- `datasets/allenai/c4/en/c4-validation.00000-of-00008.json.gz`
- `datasets/ptb_text_only/...`
- `datasets/pile-val-backup/...`

Luu y:

- README nay khong kem script tai tu dong cho `c4`, `ptb_text_only`, `pile-val-backup`
- neu ban khong goi cac loader do thi khong can tai truoc
- luong `src/calibration.py` va `src/evaluation/sliding_window.py` da co the tu tai du lieu tu Hugging Face

## Chay nhanh bang file `.sh`

Repo co 16 wrapper script, chia theo:

- `scripts/bash/smart_flip/awq/`
- `scripts/bash/smart_flip/flatquant/`
- `scripts/bash/bias_correction/awq/`
- `scripts/bash/bias_correction/flatquant/`

Moi nhom co 4 script cho:

- `run_mistral.sh`
- `run_llama3.sh`
- `run_llama31.sh`
- `run_qwen25.sh`

### Cach dung co ban

Vi du voi Mistral:

```bash
bash scripts/bash/smart_flip/awq/run_mistral.sh
```

Hoac override model:

```bash
MODEL_PATH=meta-llama/Meta-Llama-3-8B \
MODELS_ROOT=/models \
bash scripts/bash/smart_flip/awq/run_llama3.sh
```

### Script `smart_flip/awq`

Script se:

1. chay `float_model`
2. chay `awq raw`
3. quet grid `knee_tolerance` x `max_flip_percent` cho `smart_flip`

Vi du:

```bash
MODEL_PATH=mistralai/Mistral-7B-v0.3 \
bash scripts/bash/smart_flip/awq/run_mistral.sh
```

### Script `bias_correction/awq`

Script se:

1. tuy chon chay `float_model`
2. chay `awq raw`
3. chay `awq + bias_correction`

Vi du:

```bash
MODEL_PATH=mistralai/Mistral-7B-v0.3 \
BIAS_CORRECTION_SAMPLES=4096 \
bash scripts/bash/bias_correction/awq/run_mistral.sh
```

### Script `smart_flip/flatquant`

Script se:

1. tuy chon chay `float_model`
2. tuy chon chay `flatquant raw`
3. neu bo qua raw thi doc lai `RAW_MODEL_DIR`
4. chay `flatquant + smart_flip` voi `--flatquant-raw-path "$RAW_MODEL_DIR"`

Vi du:

```bash
MODEL_PATH=mistralai/Mistral-7B-v0.3 \
bash scripts/bash/smart_flip/flatquant/run_mistral.sh
```

Neu da co raw artifact roi:

```bash
MODEL_PATH=mistralai/Mistral-7B-v0.3 \
RUN_RAW_QUANTIZE=0 \
RAW_MODEL_DIR=./results/models/flatquant_raw/flatquant_raw_Mistral-7B-v0.3 \
bash scripts/bash/smart_flip/flatquant/run_mistral.sh
```

### Script `bias_correction/flatquant`

Tuong tu, script nay dung raw FlatQuant truoc roi moi chay correction:

```bash
MODEL_PATH=mistralai/Mistral-7B-v0.3 \
bash scripts/bash/bias_correction/flatquant/run_mistral.sh
```

Hoac tai su dung raw artifact:

```bash
MODEL_PATH=mistralai/Mistral-7B-v0.3 \
RUN_RAW_QUANTIZE=0 \
RAW_MODEL_DIR=./results/models/flatquant_raw/flatquant_raw_Mistral-7B-v0.3 \
bash scripts/bash/bias_correction/flatquant/run_mistral.sh
```

### Bien moi truong pho bien cho `.sh`

Tat ca wrapper script deu ho tro cac bien moi truong nay:

- `MODEL_PATH`
- `MODELS_ROOT`
- `PYTHON_BIN`
- `RESULTS_MODELS_DIR`
- `RESULTS_EVAL_DIR`
- `CALIBRATION_CACHE_DIR`
- `EVAL_CACHE_DIR`
- `CALIB_DATASET`
- `N_CALIB`
- `CALIB_SEQLEN`
- `SEED`
- `STRIDE`
- `MAX_LENGTH`
- `C4_SAMPLES`
- `LM_EVAL_TASK_PRESET`
- `INCLUDE_LM_EVAL`
- `INCLUDE_C4`
- `USE_WANDB`
- `WANDB_PROJECT`
- `WANDB_ENTITY`

FlatQuant wrapper con co them:

- `RUN_FLOAT_MODEL`
- `RUN_RAW_QUANTIZE`
- `RAW_MODEL_DIR`
- `FLATQUANT_EPOCHS`
- `FLATQUANT_CALI_BSZ`
- `FLATQUANT_LR`
- `FLATQUANT_DIAG_INIT`
- `FLATQUANT_DIAG_ALPHA`
- `FLATQUANT_CALI_TRANS`
- `FLATQUANT_ADD_DIAG`
- `FLATQUANT_LWC`
- `FLATQUANT_LAC`

## Output duoc tao ra o dau

- model artifact: `results/models/<variant>/<run_name>/`
- evaluation JSON: `results/eval/<run_name>.json`
- metadata model: `results/models/<variant>/<run_name>/metadata.json`

Luu y:

- voi `flatquant` + `post_correction=none`, raw output duoc giu lai de cac correction stage co the tai su dung
- mot so quantized output tam co the bi xoa sau evaluation neu pipeline khong can giu no lai

## Ghi chu

- `smart_flip` va `bias_correction` deu la post-correction stage.
- `flatquant` can dataset local day du hon `awq`, nhat la khi chay qua luong evaluation/loader cua FlatQuant.
- Neu ban gap loi dataset khi chay FlatQuant, hay kiem tra truoc `smart-flip/datasets`.
- Cac script trong `legacy/` duoc giu lai de doi chieu, khong phai la luong van hanh chinh nua.
