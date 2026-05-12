#!/usr/bin/env python3
"""
Run NDPA retrieval on LoCoMo QA evidence sessions.

Usage:
    python3 -m eval.locomo_runner
    python3 -m eval.locomo_runner --max-questions 25

The public Snap/USC repository currently ships locomo10.json. The first arXiv
release mentioned 50 conversations, but the maintained public benchmark file
contains 10 conversations with QA annotations.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from eval.external_benchmark_utils import (
    DATA_DIR,
    RateLimiter,
    aggregate_hits,
    append_note,
    download_json,
    load_ndpa_client,
    prediction_session_ids,
    score_predictions,
    session_to_events,
    write_json,
)

RESULTS_PATH = Path(__file__).with_name("locomo_results.json")
LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
EVIDENCE_RE = re.compile(r"^D(\d+):")

CATEGORY_LABELS = {
    1: "single-hop",
    2: "temporal",
    3: "multi-hop",
    4: "open-domain",
    5: "adversarial",
}


def iter_sessions(sample: dict) -> list[tuple[int, list[dict], str | None]]:
    conv = sample.get("conversation") or {}
    sessions = []
    for key, value in conv.items():
        if not key.startswith("session_") or key.endswith("_date_time") or not isinstance(value, list):
            continue
        try:
            number = int(key.split("_", 1)[1])
        except ValueError:
            continue
        sessions.append((number, value, conv.get(f"session_{number}_date_time")))
    sessions.sort(key=lambda item: item[0])
    return sessions


def evidence_session_ids(sample_id: str, qa: dict) -> set[str]:
    ids = set()
    for evidence in qa.get("evidence") or []:
        match = EVIDENCE_RE.match(str(evidence))
        if match:
            ids.add(f"{sample_id}_session_{int(match.group(1))}")
    return ids


def category_for(qa: dict) -> str:
    raw = qa.get("category")
    try:
        return CATEGORY_LABELS.get(int(raw), f"category_{raw}")
    except Exception:
        return f"category_{raw or 'unknown'}"


def run(args: argparse.Namespace) -> dict:
    path = DATA_DIR / "locomo" / "locomo10.json"
    samples = download_json(LOCOMO_URL, path, force=args.force_download)
    if len(samples) != 50:
        append_note(
            "LoCoMo public dataset size",
            (
                f"Requested LoCoMo runner expected 50 conversations, but the maintained "
                f"Snap/USC public file at {LOCOMO_URL} contains {len(samples)} conversations. "
                "The runner scores the official public locomo10.json file and records the "
                "actual dataset size in eval/locomo_results.json."
            ),
        )

    client = load_ndpa_client("locomo", timeout=args.timeout)
    limiter = RateLimiter(args.requests_per_minute)

    rows = []
    started = time.time()
    total_questions = sum(len(sample.get("qa") or []) for sample in samples)
    question_count = 0

    for sample in samples:
        sample_id = str(sample.get("sample_id"))
        sessions = iter_sessions(sample)
        for qa_index, qa in enumerate(sample.get("qa") or []):
            if args.max_questions and question_count >= args.max_questions:
                break
            question_count += 1
            end_user_id = f"locomo_{sample_id}_{qa_index}"

            for session_number, session, date_text in sessions:
                session_id = f"{sample_id}_session_{session_number}"
                fallback_ts = 1_680_000_000.0 + session_number
                events = session_to_events(session, fallback_ts=fallback_ts)
                if not events:
                    continue
                if date_text:
                    events[0]["source_path"] = f"locomo:{date_text}"
                limiter.wait()
                client.log_events(session_id, events, end_user_id=end_user_id)

            limiter.wait()
            result = client.get_predictions(
                session_id=f"locomo_query_{sample_id}_{qa_index}",
                query=str(qa.get("question") or ""),
                k=5,
                end_user_id=end_user_id,
            )
            predicted = prediction_session_ids(result)
            truth = evidence_session_ids(sample_id, qa)
            scored = bool(truth)
            rows.append({
                "sample_id": sample_id,
                "question_index": qa_index,
                "category": category_for(qa),
                "raw_category": qa.get("category"),
                "question": qa.get("question"),
                "answer": qa.get("answer"),
                "evidence": qa.get("evidence") or [],
                "answer_session_ids": sorted(truth),
                "predicted_session_ids": predicted,
                "hits": score_predictions(predicted, truth) if scored else {},
                "scored": scored,
                "raw_prediction_count": len(result.get("predictions") or []),
            })

            if question_count % args.progress_every == 0:
                limit = args.max_questions or total_questions
                print(f"  {question_count}/{limit} questions processed", flush=True)
        if args.max_questions and question_count >= args.max_questions:
            break

    payload = {
        "benchmark": "LoCoMo",
        "dataset": "snap-research/locomo data/locomo10.json",
        "dataset_file": str(path),
        "dataset_conversations": len(samples),
        "requested_conversations_note": "Original arXiv release mentioned 50 conversations; maintained public repo currently ships 10.",
        "n_questions": question_count,
        "n_scored": sum(1 for row in rows if row["scored"]),
        "elapsed_sec": round(time.time() - started, 2),
        "metrics": aggregate_hits(rows),
        "rows": rows,
    }
    write_json(RESULTS_PATH, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-questions", type=int, default=0, help="Optional smoke-test limit")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--requests-per-minute", type=int, default=540)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    results = run(args)
    print(f"Wrote {RESULTS_PATH}")
    print(results["metrics"]["overall"])


if __name__ == "__main__":
    main()

