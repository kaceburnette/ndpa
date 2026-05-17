#!/usr/bin/env python3
"""
Run PQHR baselines and competitor adapters.

Default mode is no-cost and local-only:
    python3 -m eval.competitor_pqhr --system all-local --max-users 0

Paid/reactive memory adapters are opt-in:
    python3 -m eval.competitor_pqhr --system mem0 --allow-paid-api

The Mem0/Zep/Letta/Pinecone adapters are category-fit tests. They are reactive
retrievers, so the PQHR trajectory is adapted into a query. Do not report them
as native predict-forward systems.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from eval.pre_query_hit_rate import (  # noqa: E402
    Conversation,
    TRAJECTORY_WINDOW_DEFAULT,
    baseline_random,
    baseline_recency,
    bm25_score,
    build_pqhr_samples,
    compute_idf,
    load_public_conversations,
    score,
    tfidf_bow,
)

RESULTS_PATH = Path(__file__).with_name("competitor_pqhr_results.json")
TOP_K = 5


def load_dotenv() -> None:
    env_file = Path(__file__).resolve().parents[1] / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        os.environ[key.strip()] = value.strip().strip("\"'")


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def limited_users(
    users: dict[str, list[Conversation]], max_users: int
) -> dict[str, list[Conversation]]:
    if max_users == 0:
        return users
    return dict(list(users.items())[:max_users])


def build_samples_by_user(
    users: dict[str, list[Conversation]], trajectory_window: int
) -> tuple[dict[str, list], list]:
    samples_by_user: dict[str, list] = {}
    all_samples = []
    for qid, convs in users.items():
        samples = build_pqhr_samples(convs, trajectory_window)
        samples_by_user[qid] = samples
        all_samples.extend(samples)
    return samples_by_user, all_samples


def metric_result(
    system: str,
    version: str,
    status: str,
    metrics: dict | None,
    *,
    n_users: int,
    n_samples: int,
    elapsed_sec: float,
    notes: str,
    trajectory_window: int,
    dataset: str = "LongMemEval-S public synthetic users",
) -> dict:
    result = {
        "system": system,
        "version": version,
        "status": status,
        "dataset": dataset,
        "trajectory_window": trajectory_window,
        "ground_truth": "BM25 top-5 past conversations",
        "predictor_input": "prior trajectory only; no current conversation",
        "n_users_evaluated": n_users,
        "n_samples": n_samples,
        "elapsed_sec": round(elapsed_sec, 2),
        "notes": notes,
    }
    if metrics:
        result.update(metrics)
    return result


def bm25_trajectory_predict(
    trajectory_bow: Counter, past: list[Conversation], k: int
) -> list[str]:
    if not past:
        return []
    n_docs = len(past)
    df: Counter = Counter()
    for conv in past:
        for term in set(conv.bow.keys()):
            df[term] += 1
    avg_doc_len = sum(sum(conv.bow.values()) for conv in past) / max(n_docs, 1)
    scored = [
        (conv.session_id, bm25_score(trajectory_bow, conv.bow, avg_doc_len, n_docs, df))
        for conv in past
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return [sid for sid, _score in scored[:k]]


def bm25_trajectory_predict_cached() -> Callable:
    stats_cache: dict[tuple[str, int], tuple[int, Counter, float]] = {}

    def _stats(past: list[Conversation]) -> tuple[int, Counter, float]:
        if not past:
            return 0, Counter(), 0.0
        key = (past[-1].session_id, len(past))
        cached = stats_cache.get(key)
        if cached:
            return cached
        n_docs = len(past)
        df: Counter = Counter()
        for conv in past:
            for term in set(conv.bow.keys()):
                df[term] += 1
        avg_doc_len = sum(sum(conv.bow.values()) for conv in past) / max(n_docs, 1)
        stats_cache[key] = (n_docs, df, avg_doc_len)
        return stats_cache[key]

    def _predict(trajectory_bow: Counter, past: list[Conversation], k: int) -> list[str]:
        if not past:
            return []
        n_docs, df, avg_doc_len = _stats(past)
        scored = [
            (conv.session_id, bm25_score(trajectory_bow, conv.bow, avg_doc_len, n_docs, df))
            for conv in past
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [sid for sid, _score in scored[:k]]

    return _predict


def tfidf_trajectory_predict(idf: Counter) -> Callable:
    conv_cache: dict[str, Counter] = {}

    def _conv_tfidf(conv: Conversation) -> Counter:
        cached = conv_cache.get(conv.session_id)
        if cached is None:
            cached = tfidf_bow(conv.bow, idf)
            conv_cache[conv.session_id] = cached
        return cached

    def _predict(trajectory_bow: Counter, past: list[Conversation], k: int) -> list[str]:
        if not past:
            return []
        trajectory_tfidf = tfidf_bow(trajectory_bow, idf)
        scored = [
            (conv.session_id, cosine(_conv_tfidf(conv), trajectory_tfidf))
            for conv in past
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [sid for sid, _score in scored[:k]]

    return _predict


def cosine(left: Counter, right: Counter) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(term, 0.0) for term, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def lexical_vectorize(bow: Counter, dims: int = 512) -> Counter:
    """
    Dependency-free vector-style lexical proxy.

    Tokens are projected into a fixed hashed vector with signed log-TF weights.
    This is not semantic embedding; it is a local lexical approximation for the
    Pinecone/vector-DB shape without external models or API calls.
    """
    vector: Counter = Counter()
    for term, count in bow.items():
        digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dims
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[str(bucket)] += sign * (1.0 + math.log(count))
    return vector


def lexical_vector_proxy_predict(
    trajectory_bow: Counter, past: list[Conversation], k: int
) -> list[str]:
    if not past:
        return []
    trajectory_vec = lexical_vectorize(trajectory_bow)
    scored = [
        (conv.session_id, cosine(lexical_vectorize(conv.bow), trajectory_vec))
        for conv in past
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return [sid for sid, _score in scored[:k]]


def lexical_vector_proxy_predict_cached() -> Callable:
    conv_cache: dict[str, Counter] = {}

    def _conv_vector(conv: Conversation) -> Counter:
        cached = conv_cache.get(conv.session_id)
        if cached is None:
            cached = lexical_vectorize(conv.bow)
            conv_cache[conv.session_id] = cached
        return cached

    def _predict(trajectory_bow: Counter, past: list[Conversation], k: int) -> list[str]:
        if not past:
            return []
        trajectory_vec = lexical_vectorize(trajectory_bow)
        scored = [
            (conv.session_id, cosine(_conv_vector(conv), trajectory_vec))
            for conv in past
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [sid for sid, _score in scored[:k]]

    return _predict


def run_local_baselines(
    users: dict[str, list[Conversation]], all_samples: list, trajectory_window: int
) -> dict[str, dict]:
    started = time.time()
    all_convs = [conv for convs in users.values() for conv in convs]
    idf = compute_idf(all_convs)
    predictors: list[tuple[str, str, str, Callable]] = [
        ("random", "stdlib-random-seed-42", "Uniform random past conversations", baseline_random),
        ("recency", "local", "Most recent past conversations", baseline_recency),
        (
            "local_bm25_trajectory",
            "local",
            "BM25 over past conversations using trajectory as query",
            bm25_trajectory_predict_cached(),
        ),
        (
            "local_tfidf",
            "local",
            "TF-IDF cosine over past conversations using trajectory as query",
            tfidf_trajectory_predict(idf),
        ),
        (
            "local_vector_lexical_proxy",
            "local-hash-512",
            "Hashed lexical vector cosine; no embeddings or external calls",
            lexical_vector_proxy_predict_cached(),
        ),
    ]

    results = {}
    for system, version, notes, predictor in predictors:
        metrics = score(predictor, all_samples)
        results[system] = metric_result(
            system,
            version,
            "verified",
            metrics,
            n_users=len(users),
            n_samples=len(all_samples),
            elapsed_sec=time.time() - started,
            notes=notes,
            trajectory_window=trajectory_window,
        )
    return results


def trajectory_to_query(trajectory_convs: list[Conversation]) -> str:
    return "\n\n---\n\n".join(conv.content for conv in trajectory_convs)


def score_predicted_session_ids(predicted_sids: list[str], truth: set[str]) -> dict[int, int]:
    return {k: int(bool(set(predicted_sids[:k]) & truth)) for k in (1, 3, 5)}


def blocked_result(
    system: str,
    reason: str,
    *,
    n_users: int,
    n_samples: int,
    notes: str,
    trajectory_window: int,
    version: str = "unverified",
) -> dict:
    return metric_result(
        system,
        version,
        reason,
        None,
        n_users=n_users,
        n_samples=n_samples,
        elapsed_sec=0.0,
        notes=notes,
        trajectory_window=trajectory_window,
    )


def run_mem0(
    users: dict[str, list[Conversation]],
    samples_by_user: dict[str, list],
    args: argparse.Namespace,
) -> dict:
    if not os.environ.get("OPENAI_API_KEY"):
        return blocked_result(
            "mem0",
            "blocked-by-key",
            n_users=len(users),
            n_samples=sum(len(samples) for samples in samples_by_user.values()),
            notes="OPENAI_API_KEY missing; Mem0 local adapter requires OpenAI LLM/embedder.",
            trajectory_window=args.trajectory_window,
        )
    if not has_module("mem0"):
        return blocked_result(
            "mem0",
            "blocked-by-tool",
            n_users=len(users),
            n_samples=sum(len(samples) for samples in samples_by_user.values()),
            notes="Python package mem0 is not installed.",
            trajectory_window=args.trajectory_window,
        )
    if not args.allow_paid_api:
        return blocked_result(
            "mem0",
            "blocked-by-paid-approval",
            n_users=len(users),
            n_samples=sum(len(samples) for samples in samples_by_user.values()),
            notes="OPENAI_API_KEY and mem0 are present, but Mem0 ingestion/search would call paid OpenAI APIs. Re-run with --allow-paid-api only after explicit approval.",
            trajectory_window=args.trajectory_window,
            version="installed-unrun",
        )

    from mem0 import Memory

    chroma_root = Path("/tmp/competitor_pqhr_chroma")
    chroma_root.mkdir(parents=True, exist_ok=True)
    config_template = {
        "vector_store": {
            "provider": "chroma",
            "config": {"collection_name": "pqhr", "path": ""},
        },
        "llm": {
            "provider": "openai",
            "config": {"model": "gpt-4o-mini", "temperature": 0.0},
        },
        "embedder": {
            "provider": "openai",
            "config": {"model": "text-embedding-3-small"},
        },
    }

    hits = {1: 0, 3: 0, 5: 0}
    n_samples = 0
    ingestion_calls = 0
    retrieval_calls = 0
    started = time.time()

    for user_idx, (qid, convs) in enumerate(users.items(), start=1):
        samples = samples_by_user.get(qid, [])
        if not samples:
            continue

        cfg = json.loads(json.dumps(config_template))
        cfg["vector_store"]["config"]["path"] = str(chroma_root / f"user_{user_idx}")
        cfg["vector_store"]["config"]["collection_name"] = f"pqhr_{user_idx}"
        mem = Memory.from_config(cfg)

        for conv in convs:
            messages = [{"role": "user", "content": turn[:2000]} for turn in conv.turns or []]
            if not messages:
                continue
            mem.add(messages=messages, user_id=qid, metadata={"session_id": conv.session_id})
            ingestion_calls += 1

        for _current, _trajectory_bow, past, truth in samples:
            trajectory_convs = past[-args.trajectory_window :]
            query = trajectory_to_query(trajectory_convs)[:4000]
            results = mem.search(query=query, user_id=qid, limit=TOP_K)
            retrieval_calls += 1
            mem_list = results.get("results", []) if isinstance(results, dict) else results
            predicted_sids = []
            for memory in mem_list:
                metadata = memory.get("metadata") or {}
                sid = metadata.get("session_id")
                if sid and sid not in predicted_sids:
                    predicted_sids.append(sid)
            sample_hits = score_predicted_session_ids(predicted_sids, truth)
            for k, hit in sample_hits.items():
                hits[k] += hit
            n_samples += 1

        if args.progress_every and user_idx % args.progress_every == 0:
            print(
                f"  mem0 user {user_idx}/{len(users)} | samples={n_samples} | "
                f"@5={hits[5] / n_samples if n_samples else 0:.3f}",
                flush=True,
            )

    metrics = {f"pqhr@{k}": hits[k] / n_samples if n_samples else 0.0 for k in (1, 3, 5)}
    result = metric_result(
        "mem0",
        "mem0-local-chroma-openai",
        "verified",
        metrics,
        n_users=len(users),
        n_samples=n_samples,
        elapsed_sec=time.time() - started,
        notes="Reactive Mem0 adapter; trajectory adapted as search query.",
        trajectory_window=args.trajectory_window,
    )
    result["ingestion_calls"] = ingestion_calls
    result["retrieval_calls"] = retrieval_calls
    return result


def external_statuses(n_users: int, n_samples: int, trajectory_window: int) -> dict[str, dict]:
    statuses = {}
    zep_key = os.environ.get("ZEP_API_KEY") or os.environ.get("ZEP_PROJECT_API_KEY")
    if not zep_key:
        statuses["zep"] = blocked_result(
            "zep",
            "blocked-by-key",
            n_users=n_users,
            n_samples=n_samples,
            notes="No ZEP_API_KEY/ZEP_PROJECT_API_KEY found; no Zep calls run.",
            trajectory_window=trajectory_window,
        )
    elif not has_module("zep_cloud"):
        statuses["zep"] = blocked_result(
            "zep",
            "blocked-by-tool",
            n_users=n_users,
            n_samples=n_samples,
            notes="Zep key present but zep_cloud package is not installed; no calls run.",
            trajectory_window=trajectory_window,
        )
    else:
        statuses["zep"] = blocked_result(
            "zep",
            "blocked-by-paid-approval",
            n_users=n_users,
            n_samples=n_samples,
            notes="Zep key/tool present but external API calls require explicit approval.",
            trajectory_window=trajectory_window,
            version="installed-unrun",
        )

    letta_key = os.environ.get("LETTA_API_KEY")
    if not letta_key:
        statuses["letta"] = blocked_result(
            "letta",
            "blocked-by-key",
            n_users=n_users,
            n_samples=n_samples,
            notes="No LETTA_API_KEY found; no Letta calls run.",
            trajectory_window=trajectory_window,
        )
    elif not has_module("letta_client"):
        statuses["letta"] = blocked_result(
            "letta",
            "blocked-by-tool",
            n_users=n_users,
            n_samples=n_samples,
            notes="Letta key present but letta_client package is not installed; no calls run.",
            trajectory_window=trajectory_window,
        )
    else:
        statuses["letta"] = blocked_result(
            "letta",
            "blocked-by-paid-approval",
            n_users=n_users,
            n_samples=n_samples,
            notes="Letta key/tool present but external API calls require explicit approval.",
            trajectory_window=trajectory_window,
            version="installed-unrun",
        )

    pinecone_key = os.environ.get("PINECONE_API_KEY")
    if not pinecone_key:
        statuses["pinecone"] = blocked_result(
            "pinecone",
            "blocked-by-key",
            n_users=n_users,
            n_samples=n_samples,
            notes="No PINECONE_API_KEY found; no Pinecone calls run. Local vector-style lexical proxy is reported separately.",
            trajectory_window=trajectory_window,
        )
    elif not has_module("pinecone"):
        statuses["pinecone"] = blocked_result(
            "pinecone",
            "blocked-by-tool",
            n_users=n_users,
            n_samples=n_samples,
            notes="Pinecone key present but pinecone package is not installed; no calls run.",
            trajectory_window=trajectory_window,
        )
    else:
        statuses["pinecone"] = blocked_result(
            "pinecone",
            "blocked-by-paid-approval",
            n_users=n_users,
            n_samples=n_samples,
            notes="Pinecone key/tool present but external API calls require explicit approval.",
            trajectory_window=trajectory_window,
            version="installed-unrun",
        )
    return statuses


def write_results(new_results: dict[str, dict]) -> None:
    payload = {}
    if RESULTS_PATH.exists():
        try:
            payload = json.loads(RESULTS_PATH.read_text())
        except json.JSONDecodeError:
            payload = {}
    payload.update(new_results)
    payload["_metadata"] = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "no_paid_api_calls": "--allow-paid-api" not in sys.argv,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def print_summary(results: dict[str, dict]) -> None:
    print()
    print(f"{'system':<28} {'status':<24} {'pqhr@1':>8} {'pqhr@3':>8} {'pqhr@5':>8}")
    for name, result in results.items():
        print(
            f"{name:<28} {result['status']:<24} "
            f"{result.get('pqhr@1', 0.0):>8.4f} "
            f"{result.get('pqhr@3', 0.0):>8.4f} "
            f"{result.get('pqhr@5', 0.0):>8.4f}"
        )


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--system",
        choices=[
            "all-local",
            "random",
            "recency",
            "local-bm25",
            "local-tfidf",
            "local-vector-proxy",
            "mem0",
            "all",
        ],
        default="all-local",
    )
    parser.add_argument("--max-users", type=int, default=0, help="0 = all users")
    parser.add_argument("--trajectory-window", type=int, default=TRAJECTORY_WINDOW_DEFAULT)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--allow-paid-api",
        action="store_true",
        help="Required before running Mem0/OpenAI or external paid API adapters.",
    )
    args = parser.parse_args()

    random.seed(42)
    print("=" * 72)
    print(" Competitor + DIY PQHR baselines")
    print("=" * 72)
    print(f"trajectory window: {args.trajectory_window} | ground truth: BM25 top-5")
    print("dataset: LongMemEval-S public synthetic users")
    print(f"paid API calls allowed: {args.allow_paid_api}")
    print()

    users = limited_users(load_public_conversations(), args.max_users)
    samples_by_user, all_samples = build_samples_by_user(users, args.trajectory_window)
    print(f"Loaded {len(users)} users; PQHR samples: {len(all_samples)}")
    if not all_samples:
        raise SystemExit("No PQHR samples found.")

    results: dict[str, dict] = {}
    local_results = run_local_baselines(users, all_samples, args.trajectory_window)
    selector = {
        "random": "random",
        "recency": "recency",
        "local-bm25": "local_bm25_trajectory",
        "local-tfidf": "local_tfidf",
        "local-vector-proxy": "local_vector_lexical_proxy",
    }

    if args.system == "all-local":
        results.update(local_results)
        results["mem0"] = run_mem0(users, samples_by_user, args)
        results.update(external_statuses(len(users), len(all_samples), args.trajectory_window))
    elif args.system in selector:
        results[selector[args.system]] = local_results[selector[args.system]]
    elif args.system == "mem0":
        results["mem0"] = run_mem0(users, samples_by_user, args)
    elif args.system == "all":
        results.update(local_results)
        results["mem0"] = run_mem0(users, samples_by_user, args)
        results.update(external_statuses(len(users), len(all_samples), args.trajectory_window))

    write_results(results)
    print_summary(results)
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
