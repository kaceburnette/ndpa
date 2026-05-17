#!/usr/bin/env python3
"""Local filesystem Hydration microbenchmark.

Measures raw local blob reads through server.hydration.hydrate_local_blob.
This is not a hosted API benchmark and makes no network, database, Supabase,
LLM, or embedding calls.

Usage:
    python3 -m eval.hydration_microbench
    python3 -m eval.hydration_microbench --iterations 500
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from server.hydration import hydrate_local_blob

DEFAULT_SIZES = {
    "1KB": 1024,
    "10KB": 10 * 1024,
    "100KB": 100 * 1024,
    "1MB": 1024 * 1024,
}


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int((len(ordered) - 1) * pct)
    return ordered[idx]


def write_blob(root: Path, label: str, size_bytes: int) -> str:
    storage_key = f"bench/{label}.txt"
    path = root / storage_key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x" * size_bytes, encoding="utf-8")
    return storage_key


def measure(root: Path, storage_key: str, iterations: int, warmup: int) -> dict:
    for _ in range(warmup):
        result = hydrate_local_blob(storage_key, root)
        if not result["found"]:
            raise RuntimeError(f"warmup hydrate failed: {result}")

    samples_ms: list[float] = []
    content_bytes = 0
    for _ in range(iterations):
        start = time.perf_counter()
        result = hydrate_local_blob(storage_key, root)
        samples_ms.append((time.perf_counter() - start) * 1000.0)
        if not result["found"]:
            raise RuntimeError(f"hydrate failed: {result}")
        content_bytes = result["content_bytes"]

    return {
        "content_bytes": content_bytes,
        "iterations": iterations,
        "p50_ms": round(statistics.median(samples_ms), 4),
        "p95_ms": round(percentile(samples_ms, 0.95), 4),
        "min_ms": round(min(samples_ms), 4),
        "max_ms": round(max(samples_ms), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--json-out",
        default="eval/hydration_microbench_results.json",
        help="Write results JSON. Use '' to disable.",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="ndpa_hydration_bench_") as tmp:
        root = Path(tmp)
        results = {
            "benchmark": "local_fs_hydration_microbench",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "notes": (
                "Local filesystem read through hydrate_local_blob; not hosted "
                "API latency and not object-storage latency."
            ),
            "iterations": args.iterations,
            "warmup": args.warmup,
            "sizes": {},
        }
        for label, size in DEFAULT_SIZES.items():
            storage_key = write_blob(root, label, size)
            results["sizes"][label] = measure(root, storage_key, args.iterations, args.warmup)

    print(json.dumps(results, indent=2))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
