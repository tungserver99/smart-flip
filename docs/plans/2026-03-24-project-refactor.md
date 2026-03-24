# Smart Flip AWQ Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reorganize the project around one clean AWQ baseline plus one smart-flip variant while preserving the logic of `awq_js_xl.py` and `post-correction-quantization-fix.md`.

**Architecture:** Keep the original quantization algorithm close to the experimental script, but move it into a package under `src/`. Split the flow into raw AWQ alpha search, raw AWQ quantization, and optional smart-flip post-correction. Add one evaluation pipeline that compares `fp`, `awq_raw`, and `awq_flip`.

**Tech Stack:** Python, PyTorch, Hugging Face Transformers, Datasets
