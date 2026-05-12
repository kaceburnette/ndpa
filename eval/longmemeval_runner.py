#!/usr/bin/env python3
"""
Run NDPA retrieval on LongMemEval.

Usage:
    python3 -m eval.longmemeval_runner
    python3 -m eval.longmemeval_runner --split oracle --max-items 25

The runner measures whether NDPA's Predictions API surfaces the labeled
evidence session IDs in top-K. It does not ask an LLM to answer questions.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from eval.external_benchmark_utils import (
    DATA_DIR,
    RateLimiter,
    aggregate_hits,
    download_json,
    load_ndpa_client,
    prediction_session_ids,
    score_predictions,
    session_to_events,
    write_json,
)

RESULTS_PATH = Path(__file__).with_name("longmemeval_results.json")

DATASETS = {
    "s": (
        "xiaowu0162/longmemeval-cleaned",
        "longmemeval_s_cleaned.json",
        "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json",
    ),
    "m": (
        "xiaowu0162/longmemeval-cleaned",
        "longmemeval_m_cleaned.json",
        "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_m_cleaned.json",
    ),
    "oracle": (
        "xiaowu0162/longmemeval-cleaned",
        "longmemeval_oracle.json",
        "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json",
    ),
}

PUBLISHED_RETRIEVAL_NOTES = {
    "source": "LongMemEval paper reports retrieval baselines such as BM25, Contriever, GTE, and Stella, but not Mem0/Zep/MemGPT in the released benchmark paper.",
    "comparison_status": "No Mem0/Zep/MemGPT LongMemEval category numbers found in the paper; runner records NDPA hit@K only.",
}


def category_for(item: dict) -> str:
    if str(item.get("question_id", "")).endswith("_abs"):
        return "abstention"
    return str(item.get("question_type") or "unknown")


def run(args: argparse.Namespace) -> dict:
    dataset_id, filename, url = DATASETS[args.split]
    path = DATA_DIR / "longmemeval" / filename
    items = download_json(url, path, force=args.force_download)
    if args.max_items:
        items = items[: args.max_items]

    client = load_ndpa_client("longmemeval", timeout=args.timeout)
    limiter = RateLimiter(args.requests_per_minute)

    rows = []
    started = time.time()
    for index, item in enumerate(items, start=1):
        question_id = str(item["question_id"])
        end_user_id = f"longmemeval_{question_id}"
        haystack_ids = [str(sid) for sid in item.get("haystack_session_ids") or []]
        haystack_dates = item.get("haystack_dates") or []
        haystack_sessions = item.get("haystack_sessions") or []

        for session_index, session in enumerate(haystack_sessions):
            session_id = haystack_ids[session_index] if session_index < len(haystack_ids) else f"{question_id}_{session_index}"
            fallback_ts = 1_700_000_000.0 + session_index
            events = session_to_events(session, fallback_ts=fallback_ts)
            if not events:
                continue
            if session_index < len(haystack_dates):
                events[0]["source_path"] = f"longmemeval:{haystack_dates[session_index]}"
            limiter.wait()
            client.log_events(session_id, events, end_user_id=end_user_id)

        limiter.wait()
        result = client.get_predictions(
            session_id=f"longmemeval_query_{question_id}",
            query=str(item.get("question") or ""),
            k=5,
            end_user_id=end_user_id,
        )
        predicted = prediction_session_ids(result)
        truth = {str(sid) for sid in item.get("answer_session_ids") or []}
        scored = bool(truth)
        rows.append({
            "question_id": question_id,
            "category": category_for(item),
            "question_type": item.get("question_type"),
            "answer_session_ids": sorted(truth),
            "predicted_session_ids": predicted,
            "hits": score_predictions(predicted, truth) if scored else {},
            "scored": scored,
            "raw_prediction_count": len(result.get("predictions") or []),
        })
        if index % args.progress_every == 0:
            print(f"  {index}/{len(items)} questions processed", flush=True)

    payload = {
        "benchmark": "LongMemEval",
        "dataset": dataset_id,
        "split": args.split,
        "dataset_file": str(path),
        "n_items": len(items),
        "n_scored": sum(1 for row in rows if row["scored"]),
        "elapsed_sec": round(time.time() - started, 2),
        "metrics": aggregate_hits(rows),
        "published_comparison": PUBLISHED_RETRIEVAL_NOTES,
        "rows": rows,
    }
    write_json(RESULTS_PATH, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=sorted(DATASETS), default="s")
    parser.add_argument("--max-items", type=int, default=0, help="Optional smoke-test limit")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--requests-per-minute", type=int, default=540, help="Stay below the 600 req/min API limit")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    results = run(args)
    print(f"Wrote {RESULTS_PATH}")
    print(results["metrics"]["overall"])


if __name__ == "__main__":
    main()

