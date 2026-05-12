# NDPA Notes

## 2026-05-12 18:47:46 EDT - LMSYS validation blocked

LMSYS-Chat-1M validation could not run automatically. Reason: Python package `datasets` is not installed. The dataset is gated on HuggingFace and distributed as parquet; this repo intentionally does not add datasets/pyarrow as required dependencies. Provide --input with an accepted/exported JSON or JSONL sample, or install datasets after accepting the license.

## 2026-05-12 18:48:12 EDT - LoCoMo public dataset size

Requested LoCoMo runner expected 50 conversations, but the maintained Snap/USC public file at https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json contains 10 conversations. The runner scores the official public locomo10.json file and records the actual dataset size in eval/locomo_results.json.
