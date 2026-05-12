"""
Scaling benchmark for the Predictions API.

Measures predictions latency (P50/P95/P99) at multiple corpus sizes.
Generates synthetic conversations with a synthetic end_user_id so this
doesn't pollute production user data, then deletes them after.

Usage:
    python3 -m eval.scaling_benchmark
    python3 -m eval.scaling_benchmark --max 10000

Honest caveat:
    Edge function cold starts add ~200-400ms to the first request after idle.
    The benchmark warms the function with 3 dummy calls before measuring.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "sdk" / "python"))

from ndp.config import load_config  # noqa: E402
from ndpa import Client  # noqa: E402

# Synthetic topic vocabulary — keeps BoW realistic
TOPICS = {
    "fitness": "workout squat deadlift bench protein gym recovery sleep",
    "cooking": "recipe pasta garlic olive oil tomato basil oven temperature",
    "api":     "fastapi pydantic uvicorn route middleware async http",
    "react":   "react hooks useEffect useState component props lifecycle",
    "db":      "postgres index query vacuum sequence wal replication",
    "rust":    "rust borrow checker lifetime trait impl async tokio cargo",
    "ml":      "neural network gradient descent backprop overfitting",
    "biz":     "founders fundraising valuation cap table seed series",
}


def gen_text(rng: random.Random) -> tuple[str, list[str]]:
    """Pick 1-2 topics, generate plausible-looking conversation text."""
    topics = list(TOPICS.keys())
    picked = [rng.choice(topics)] if rng.random() < 0.7 else rng.sample(topics, 2)
    body = " ".join(TOPICS[t] for t in picked)
    turns = []
    for _ in range(rng.randint(3, 8)):
        turns.append(f"[user] question about {rng.choice(picked)} something specific")
        turns.append(f"[assistant] {body} — detailed answer with technical info")
    return "\n\n".join(turns), picked


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    k = int(len(sorted_v) * p)
    return sorted_v[min(k, len(sorted_v) - 1)]


def populate(client: Client, end_user_id: str, n: int):
    """Bulk insert N synthetic conversations under a benchmark end_user_id."""
    rng = random.Random(42)
    print(f"  Inserting {n} synthetic conversations...", flush=True)
    t0 = time.time()
    for i in range(n):
        content, _topics = gen_text(rng)
        session_id = f"bench_{i:06d}"
        events = []
        for chunk in content.split("\n\n"):
            role = "user" if chunk.startswith("[user]") else "assistant"
            text = chunk.split("] ", 1)[-1]
            events.append({
                "role": role,
                "content": text,
                "ts": time.time() - rng.uniform(0, 365 * 86400),
            })
        try:
            client.log_events(session_id, events, end_user_id=end_user_id)
        except Exception as e:
            print(f"  insert failed at {i}: {e}")
            return
        if (i + 1) % 200 == 0:
            print(f"    {i + 1}/{n}", flush=True)
    print(f"  Populated in {time.time() - t0:.1f}s")


def warm_up(client: Client, end_user_id: str, n: int = 3):
    print(f"  Warming Edge Function ({n} calls)...", flush=True)
    for _ in range(n):
        client.get_predictions("", query="warmup", k=3, end_user_id=end_user_id)


def benchmark_at_scale(client: Client, end_user_id: str, n_queries: int = 30) -> dict:
    """Run n_queries varied predictions, measure latencies."""
    queries = [
        "workout protein recovery gym",
        "fastapi pydantic async route",
        "react useEffect useState hooks",
        "postgres index query plan",
        "rust borrow checker lifetime",
        "neural network gradient descent",
        "founders seed series valuation",
        "pasta garlic olive oil",
    ]
    latencies_ms: list[float] = []
    server_ms: list[float] = []
    rng = random.Random(7)
    for i in range(n_queries):
        q = rng.choice(queries)
        t0 = time.time()
        result = client.get_predictions("", query=q, k=5, end_user_id=end_user_id)
        latencies_ms.append((time.time() - t0) * 1000)
        sm = result.get("elapsed_ms")
        if sm is not None:
            server_ms.append(float(sm))

    return {
        "n_queries": n_queries,
        "wall_p50": percentile(latencies_ms, 0.5),
        "wall_p95": percentile(latencies_ms, 0.95),
        "wall_p99": percentile(latencies_ms, 0.99),
        "server_p50": percentile(server_ms, 0.5) if server_ms else 0,
        "server_p95": percentile(server_ms, 0.95) if server_ms else 0,
    }


def cleanup(end_user_id: str):
    """Delete the synthetic benchmark data."""
    cfg = load_config()
    from supabase import create_client
    sb = create_client(cfg["supabase_url"], cfg["supabase_key"])
    print(f"  Cleaning up benchmark data for {end_user_id}...")
    sb.table("ndp_events").delete().eq("end_user_id", end_user_id).execute()
    sb.table("ndp_conversations").delete().eq("end_user_id", end_user_id).execute()
    sb.table("ndp_sessions").delete().eq("end_user_id", end_user_id).execute()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=1000,
                        help="Max corpus size to benchmark (default 1000; use 10000 for full curve)")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Leave benchmark data in DB after run")
    args = parser.parse_args()

    cfg = load_config()
    client = Client(api_key=cfg["api_key"], async_send=False, timeout=20.0)
    end_user_id = f"benchmark_{int(time.time())}"

    print("=" * 72)
    print(" NDPA Predictions API — Scaling Benchmark")
    print("=" * 72)
    print(f" benchmark end_user_id: {end_user_id}")
    print(f" max corpus size: {args.max}")
    print()

    sizes = [50, 200, 500, 1000]
    if args.max >= 5000:
        sizes.extend([2500, 5000])
    if args.max >= 10000:
        sizes.append(10000)
    sizes = [s for s in sizes if s <= args.max]

    results = []
    populated = 0

    for size in sizes:
        delta = size - populated
        if delta > 0:
            populate(client, end_user_id, delta)
            populated = size
        warm_up(client, end_user_id)
        print(f"  Benchmarking @ {size} conversations...")
        r = benchmark_at_scale(client, end_user_id)
        r["corpus_size"] = size
        results.append(r)
        print(f"    wall:   P50={r['wall_p50']:6.0f}ms  P95={r['wall_p95']:6.0f}ms  P99={r['wall_p99']:6.0f}ms")
        if r["server_p50"]:
            print(f"    server: P50={r['server_p50']:6.0f}ms  P95={r['server_p95']:6.0f}ms")
        print()

    print("Summary (predictions API latency):")
    print(f"  {'corpus':>10}  {'P50':>8}  {'P95':>8}  {'P99':>8}  {'server P50':>11}")
    for r in results:
        print(f"  {r['corpus_size']:>10}  {r['wall_p50']:>6.0f}ms  {r['wall_p95']:>6.0f}ms  {r['wall_p99']:>6.0f}ms  {r['server_p50']:>9.0f}ms")

    if not args.no_cleanup:
        cleanup(end_user_id)


if __name__ == "__main__":
    main()
