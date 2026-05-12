# NDPA Notes

## 2026-05-12 18:47:46 EDT - LMSYS validation blocked

LMSYS-Chat-1M validation could not run automatically. Reason: Python package `datasets` is not installed. The dataset is gated on HuggingFace and distributed as parquet; this repo intentionally does not add datasets/pyarrow as required dependencies. Provide --input with an accepted/exported JSON or JSONL sample, or install datasets after accepting the license.
