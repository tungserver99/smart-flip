"""
Calibration data loaders reused from the original experimental scripts.
"""

from __future__ import annotations

import pickle
import random
from pathlib import Path

import torch
from datasets import load_dataset


def get_c4_calibration_data(
    tokenizer,
    n_samples: int = 128,
    seqlen: int = 2048,
    seed: int = 42,
    return_tensors: bool = False,
    cache_dir: str = "./data/cache/calibration",
):
    print("\n[C4 Calibration Data - Optimized]")
    print(f"  Samples: {n_samples}")
    print(f"  Sequence length: {seqlen} tokens")
    print(f"  Seed: {seed}")

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    cache_file = cache_path / f"c4_calib_n{n_samples}_len{seqlen}_seed{seed}_tensors{return_tensors}.pkl"
    if cache_file.exists():
        print(f"  Loading from cache: {cache_file}")
        with open(cache_file, "rb") as handle:
            return pickle.load(handle)

    random.seed(seed)
    url = "https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.00000-of-01024.json.gz"
    traindata = load_dataset(
        "json",
        data_files={"train": url},
        split="train",
        streaming=True,
    )

    dataset = []
    skipped = 0
    char_threshold = seqlen * 3

    for data in traindata:
        text = data["text"]
        if len(text) < char_threshold:
            skipped += 1
            continue

        trainenc = tokenizer(text, return_tensors="pt")
        if trainenc.input_ids.shape[1] < seqlen:
            skipped += 1
            continue

        max_start = trainenc.input_ids.shape[1] - seqlen
        start_idx = random.randint(0, max_start)
        end_idx = start_idx + seqlen
        inp = trainenc.input_ids[:, start_idx:end_idx]

        if return_tensors:
            dataset.append(inp)
        else:
            dataset.append(tokenizer.decode(inp[0], skip_special_tokens=True))

        if len(dataset) == n_samples:
            break

    print(f"  Collected {len(dataset)} samples from C4")
    print(f"  Skipped {skipped} documents")

    with open(cache_file, "wb") as handle:
        pickle.dump(dataset, handle)

    return dataset


def get_wikitext2_calibration_data(
    tokenizer,
    n_samples: int = 128,
    seqlen: int = 2048,
    seed: int = 42,
    split: str = "train",
    return_tensors: bool = False,
    cache_dir: str = "./data/cache/calibration",
):
    print("\n[WikiText-2 Calibration Data]")
    print(f"  Samples: {n_samples}")
    print(f"  Sequence length: {seqlen} tokens")
    print(f"  Split: {split}")
    print(f"  Seed: {seed}")

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    cache_file = cache_path / f"wikitext2_calib_n{n_samples}_len{seqlen}_seed{seed}_split{split}_tensors{return_tensors}.pkl"
    if cache_file.exists():
        print(f"  Loading from cache: {cache_file}")
        with open(cache_file, "rb") as handle:
            return pickle.load(handle)

    random.seed(seed)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    texts = [item["text"] for item in dataset if item["text"].strip()]

    all_tokens = []
    for text in texts:
        tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        all_tokens.append(tokens)

    all_tokens = torch.cat(all_tokens, dim=0)
    num_chunks = len(all_tokens) // seqlen
    if num_chunks < n_samples:
        n_samples = num_chunks

    chunk_indices = random.sample(range(num_chunks), n_samples)
    calibration_samples = []
    for idx in chunk_indices:
        start = idx * seqlen
        end = start + seqlen
        chunk_tokens = all_tokens[start:end]
        if return_tensors:
            calibration_samples.append(chunk_tokens.unsqueeze(0).clone())
        else:
            calibration_samples.append(tokenizer.decode(chunk_tokens, skip_special_tokens=True))

    with open(cache_file, "wb") as handle:
        pickle.dump(calibration_samples, handle)

    return calibration_samples

def load_calibration_data(
    dataset_name: str,
    tokenizer,
    n_samples: int = 128,
    seqlen: int = 2048,
    seed: int = 42,
    return_tensors: bool = False,
    cache_dir: str = "./data/cache/calibration",
):
    dataset_name = dataset_name.lower()

    if dataset_name == "c4":
        return get_c4_calibration_data(
            tokenizer,
            n_samples=n_samples,
            seqlen=seqlen,
            seed=seed,
            return_tensors=return_tensors,
            cache_dir=cache_dir,
        )
    if dataset_name == "wikitext2":
        return get_wikitext2_calibration_data(
            tokenizer,
            n_samples=n_samples,
            seqlen=seqlen,
            seed=seed,
            split="train",
            return_tensors=return_tensors,
            cache_dir=cache_dir,
        )
    if dataset_name == "wikitext2-simple":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        return [item["text"] for item in dataset if len(item["text"].strip()) > 100][:n_samples]

    raise ValueError(f"Unknown calibration dataset: {dataset_name}")
