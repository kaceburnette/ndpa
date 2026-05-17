#!/usr/bin/env python3
"""
Measure NDPA FastAPI latency.

Usage:
    export NDPA_API_KEY="ndpa_..."  # or NDPA_TEST_API_KEY
    export NEW_BASE_URL="http://localhost:8000"
    python3 -m eval.compare_latency --requests 50
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

# Load .env
ENV_FILE = Path("/Users/kaceburnette/Desktop/ndp/.env")
if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, _, v = line.partition("=")
            os.environ[k.strip()] = v.strip().strip("\"'")

sys.path.insert(0, "/Users/kaceburnette/Desktop/ndp/sdk/python")
from ndpa import Client  # noqa: E402

OLD_BASE = os.environ.get("OLD_BASE_URL", "")
NEW_BASE = os.environ.get("NEW_BASE_URL", os.environ.get("NDPA_BASE_URL", "http://localhost:8000"))
API_KEY = os.environ.get("NDPA_API_KEY") or os.environ.get("NDPA_TEST_API_KEY", "")


def measure(base_url: str, n: int, query: str, end_user_id: str | None) -> dict:
    if not API_KEY:
        raise SystemExit("Set NDPA_API_KEY")
    c = Client(api_key=API_KEY, base_url=base_url, timeout=20.0, async_send=False)
    # Warm 3
    for _ in range(3):
        c.get_predictions(query=query, k=5, end_user_id=end_user_id)
    samples = []
    for _ in range(n):
        t0 = time.monotonic()
        c.get_predictions(query=query, k=5, end_user_id=end_user_id)
        samples.append((time.monotonic() - t0) * 1000.0)
    samples.sort()
    return {
        "n": n,
        "p50_ms": round(statistics.median(samples), 1),
        "p95_ms": round(samples[int(0.95 * n)], 1),
        "p99_ms": round(samples[int(0.99 * n)], 1),
        "mean_ms": round(statistics.mean(samples), 1),
    }


def measure_cold(base_url: str, n: int, query: str, end_user_id: str | None) -> dict:
    if not API_KEY:
        raise SystemExit("Set NDPA_API_KEY")
    c = Client(api_key=API_KEY, base_url=base_url, timeout=20.0, async_send=False)
    samples = []
    timings = []
    for i in range(n):
        # Different cache key, same stripped query text inside the prediction kernel.
        cold_query = query + (" " * (i + 1))
        t0 = time.monotonic()
        resp = c.get_predictions(query=cold_query, k=5, end_user_id=end_user_id)
        samples.append((time.monotonic() - t0) * 1000.0)
        if resp.get("timing"):
            timings.append(resp["timing"])
    samples.sort()
    result = {
        "n": n,
        "p50_ms": round(statistics.median(samples), 1),
        "p95_ms": round(samples[int(0.95 * n)], 1),
        "p99_ms": round(samples[int(0.99 * n)], 1),
        "mean_ms": round(statistics.mean(samples), 1),
    }
    if timings:
        for key in ("auth_ms", "db_candidate_ms", "rank_ms", "total_ms"):
            vals = [float(t.get(key, 0)) for t in timings]
            result[f"server_{key}_mean"] = round(statistics.mean(vals), 1)
    return result


def measure_staged(base_url: str, n: int, query: str, end_user_id: str | None) -> dict:
    if not API_KEY:
        raise SystemExit("Set NDPA_API_KEY")
    c = Client(api_key=API_KEY, base_url=base_url, timeout=20.0, async_send=False)
    stage_samples = []
    warm_samples = []
    warm_timings = []
    for i in range(n):
        staged_query = query + (" " * (i + 1))
        t0 = time.monotonic()
        c.stage(query=staged_query, k=5, end_user_id=end_user_id)
        stage_samples.append((time.monotonic() - t0) * 1000.0)

        t1 = time.monotonic()
        resp = c.get_predictions(query=staged_query, k=5, end_user_id=end_user_id)
        warm_samples.append((time.monotonic() - t1) * 1000.0)
        if resp.get("timing"):
            warm_timings.append(resp["timing"])

    stage_samples.sort()
    warm_samples.sort()
    result = {
        "n": n,
        "stage": {
            "p50_ms": round(statistics.median(stage_samples), 1),
            "p95_ms": round(stage_samples[int(0.95 * n)], 1),
            "mean_ms": round(statistics.mean(stage_samples), 1),
        },
        "warm_prediction": {
            "p50_ms": round(statistics.median(warm_samples), 1),
            "p95_ms": round(warm_samples[int(0.95 * n)], 1),
            "mean_ms": round(statistics.mean(warm_samples), 1),
        },
    }
    if warm_timings:
        result["warm_prediction"]["cache_hits"] = sum(1 for t in warm_timings if t.get("cache_hit"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--query", default="rust async tokio borrow checker")
    parser.add_argument("--end-user-id", default=None)
    parser.add_argument(
        "--profile",
        choices=("warm", "cold", "staged", "all"),
        default="warm",
        help="warm preserves legacy behavior; staged measures explicit /stage then cache hit.",
    )
    args = parser.parse_args()

    if OLD_BASE:
        print(f"Old endpoint: {OLD_BASE}")
    print(f"New endpoint: {NEW_BASE}")
    print(f"Query:        {args.query!r}")
    print(f"Requests:     {args.requests}")
    print()

    old = None
    if OLD_BASE and args.profile == "warm":
        old = measure(OLD_BASE, args.requests, args.query, args.end_user_id)
        print("Old endpoint:")
        print(json.dumps(old, indent=2))

    if args.profile == "cold":
        new = measure_cold(NEW_BASE, args.requests, args.query, args.end_user_id)
    elif args.profile == "staged":
        new = measure_staged(NEW_BASE, args.requests, args.query, args.end_user_id)
    elif args.profile == "all":
        new = {
            "cold": measure_cold(NEW_BASE, args.requests, args.query, args.end_user_id),
            "staged": measure_staged(NEW_BASE, args.requests, args.query, args.end_user_id),
            "legacy_warm": measure(NEW_BASE, args.requests, args.query, args.end_user_id),
        }
    else:
        new = measure(NEW_BASE, args.requests, args.query, args.end_user_id)
    print("\nNew/FastAPI endpoint:")
    print(json.dumps(new, indent=2))
    if old:
        print("\nDelta:")
        for k in ("p50_ms", "p95_ms", "p99_ms", "mean_ms"):
            delta = old[k] - new[k]
            pct = (delta / old[k]) * 100 if old[k] else 0
            print(f"  {k}: {old[k]} → {new[k]} ({delta:+.1f}ms, {pct:+.1f}%)")


if __name__ == "__main__":
    main()
