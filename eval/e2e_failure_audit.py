#!/usr/bin/env python3
"""
Audit e2e misses into Codex's seven failure buckets.

Buckets:
  - retrieval_miss        — answer_session_ids ∩ predicted_session_ids = ∅
  - retrieved_right_reader_wrong — retrieval hit but reader produced wrong answer
  - grader_false_negative — string-overlap=False but llm-judge=True
  - multi_hop_miss        — question references >=2 sessions but predicted <2 truth sessions
  - temporal_miss         — category=temporal-reasoning AND judge=False
  - abstention_miss       — category=abstention AND judge=False
  - duplicate_context     — top-K predicted contains >=2 chunks from same session_id

A row can fall into multiple buckets. We tag each row with all matches and
report counts of each bucket.

Usage:
    python3 -m eval.e2e_failure_audit
    python3 -m eval.e2e_failure_audit --target eval/longmemeval_reasoning_results.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


DEFAULT_RESULTS = Path("/Users/kaceburnette/Desktop/ndp/eval/longmemeval_e2e_results.json")
DEFAULT_SOURCE  = Path("/Users/kaceburnette/Desktop/ndp/eval/data/longmemeval/longmemeval_s_cleaned.json")
DEFAULT_AUDIT_OUT = Path("/Users/kaceburnette/Desktop/ndp/eval/e2e_failure_audit.json")


def load_truth(source_path: Path) -> dict[str, list[str]]:
    items = json.loads(source_path.read_text())
    out: dict[str, list[str]] = {}
    for it in items:
        qid = str(it.get("question_id") or it.get("id") or "")
        ids = it.get("answer_session_ids") or it.get("evidence_session_ids") or []
        out[qid] = [str(x) for x in ids]
    return out


def bucket_row(row: dict, truth_sids: list[str]) -> list[str]:
    buckets = []
    judge = bool(row.get("correct_llm_judge"))
    string_overlap = bool(row.get("correct"))
    predicted = list(row.get("predicted_session_ids") or [])
    category = row.get("category", "")

    truth_set = set(truth_sids)
    pred_set = set(predicted)
    retrieval_hit = bool(truth_set & pred_set)

    # 1. grader false negative — string=False, judge=True
    if not string_overlap and judge:
        buckets.append("grader_false_negative")
        # Not a "miss" by our true measure — still log other categories below

    # If LLM judge says correct, this isn't really a miss to audit further.
    if judge:
        return buckets or ["correct"]

    # From here: judge=False. It IS a miss.

    # 2. retrieval miss
    if not retrieval_hit and truth_set:
        buckets.append("retrieval_miss")

    # 3. retrieved-right / reader-wrong
    if retrieval_hit:
        buckets.append("retrieved_right_reader_wrong")

    # 4. multi-hop miss — truth has >=2 sessions, predicted hit <2 of them
    if len(truth_set) >= 2 and len(truth_set & pred_set) < 2:
        buckets.append("multi_hop_miss")

    # 5. temporal miss
    if category == "temporal-reasoning":
        buckets.append("temporal_miss")

    # 6. abstention miss
    if category == "abstention":
        buckets.append("abstention_miss")

    # 7. duplicate context — predicted list has duplicates by session_id
    if predicted:
        counts = Counter(predicted)
        if any(c >= 2 for c in counts.values()):
            buckets.append("duplicate_context")

    # Catch-all if judge=False with no other bucket (shouldn't happen but log it)
    if not buckets:
        buckets.append("uncategorized_miss")

    return buckets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=str(DEFAULT_RESULTS))
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--out", default=str(DEFAULT_AUDIT_OUT))
    args = parser.parse_args()

    results = json.loads(Path(args.target).read_text())
    rows = results.get("rows", [])
    truth_map = load_truth(Path(args.source))

    bucket_counts: Counter = Counter()
    bucket_by_category: dict[str, Counter] = {}
    audited = []
    judge_correct = 0
    judge_wrong = 0

    for r in rows:
        qid = str(r.get("question_id", ""))
        cat = r.get("category", "unknown")
        truth_sids = truth_map.get(qid, [])
        buckets = bucket_row(r, truth_sids)

        # judge_correct counts rows where the LLM judge said YES
        # (independent of whether the string-overlap grader agreed)
        is_judge_correct = bool(r.get("correct_llm_judge"))
        if is_judge_correct:
            judge_correct += 1
        else:
            judge_wrong += 1

        for b in buckets:
            if b == "correct":
                continue
            bucket_counts[b] += 1
            bucket_by_category.setdefault(cat, Counter())[b] += 1

        audited.append({
            "question_id": qid,
            "category": cat,
            "judge_correct": bool(r.get("correct_llm_judge")),
            "string_correct": bool(r.get("correct")),
            "truth_session_ids": truth_sids,
            "predicted_session_ids": list(r.get("predicted_session_ids") or []),
            "buckets": buckets,
        })

    out = {
        "source_file": str(args.target),
        "n_rows": len(rows),
        "judge_correct": judge_correct,
        "judge_wrong": judge_wrong,
        "judge_accuracy": round(judge_correct / len(rows), 4) if rows else 0.0,
        "bucket_counts": dict(bucket_counts),
        "bucket_by_category": {k: dict(v) for k, v in bucket_by_category.items()},
        "rows": audited,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))

    # Print summary
    print(f"Audited: {args.target}")
    print(f"n_rows: {len(rows)} (judge_correct={judge_correct}, judge_wrong={judge_wrong})\n")
    print("Bucket counts (a row can appear in multiple buckets):")
    width = max(len(b) for b in bucket_counts.keys()) if bucket_counts else 30
    for bucket, count in sorted(bucket_counts.items(), key=lambda x: -x[1]):
        if bucket == "grader_false_negative":
            # These are judge-correct rows where string-overlap was wrong.
            # Show as % of judge-correct, not % of misses.
            share = count / max(judge_correct, 1) * 100
            print(f"  {bucket:<{width}}  {count:>4}  ({share:.1f}% of judge-correct rows)")
        else:
            share = count / max(judge_wrong, 1) * 100
            print(f"  {bucket:<{width}}  {count:>4}  ({share:.1f}% of misses)")

    print("\nBucket counts by category:")
    for cat in sorted(bucket_by_category):
        print(f"  {cat}:")
        for b, c in sorted(bucket_by_category[cat].items(), key=lambda x: -x[1]):
            print(f"    {b:<{width}}  {c}")

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
