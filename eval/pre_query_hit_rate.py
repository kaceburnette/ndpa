#!/usr/bin/env python3
"""
Pre-Query Hit Rate (PQHR) — NDPA's own benchmark.

The published memory benchmarks (LongMemEval, LoCoMo) test REACTIVE retrieval:
  given a question, did the system retrieve the right past conversation?

PQHR tests PREDICTIVE staging — what NDPA actually does:
  given ONLY the user's prior session trajectory (no current question, no current
  conversation), can NDPA stage the conversations that WILL become relevant for
  the user's NEXT session?

Why nobody else competes here:
  Vector DBs need a query to retrieve against. Mem0/Zep require explicit input.
  Predict-forward staging based purely on behavioral signal is a category nobody
  but NDPA addresses in production.

Methodology:
  1. Sort all conversations chronologically.
  2. For each conversation C_i (with at least K_TRAJ prior conversations):
       trajectory = concatenated content of C_{i-K_TRAJ} .. C_{i-1}
       (the predictor sees ONLY the trajectory, never C_i)
  3. Ground truth = top-N past conversations whose content best matches
       C_i's FULL content (cosine over BoW).
  4. Predictor must rank past conversations using only the trajectory.
  5. Score = hit@k against ground truth.

This is harder than retrieval. The predictor doesn't see the upcoming question.
It must infer "what is this user about to need" from what they've been doing.

Usage:
    python3 -m eval.pre_query_hit_rate                          # default on author's data
    python3 -m eval.pre_query_hit_rate --trajectory-window 5
    python3 -m eval.pre_query_hit_rate --min-future-chars 200
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from typing import List, Set

from collections import defaultdict

from eval.conversation_eval import (
    Conversation, load_conversations, tokenize, cosine,
    TOP_K_PREDICT,
)


def load_conversations_grouped() -> dict:
    """
    Load conversations grouped by user (end_user_id from session prefix or main user).
    Returns {user_key: [Conversation, ...]} so PQHR can run per-user.

    Heuristic: session_ids starting with 'longmemeval_' belong to that synthetic
    user; everything else belongs to the main NDPA user.
    """
    all_convs = load_conversations()
    grouped: dict[str, list[Conversation]] = defaultdict(list)
    for c in all_convs:
        # LongMemEval haystacks: session_id may contain the question_id; group by question
        if "longmemeval" in c.session_id.lower() or c.session_id.startswith("qa"):
            # use session_id prefix to identify the synthetic user
            grouped["longmemeval_haystack_pool"].append(c)
        else:
            grouped["main_user"].append(c)
    return grouped

K_GROUND_TRUTH = 5             # top-K relevant past conversations are "truth"
TRAJECTORY_WINDOW_DEFAULT = 3  # use last N conversations as trajectory signal
MIN_FUTURE_CHARS = 200         # skip trivial conversations
MIN_PAST_CONVS = 10            # need enough history to have meaningful "past"


def pure_topic_predict(trajectory_bow: Counter, past: List[Conversation], k: int) -> List[str]:
    """NDPA's actual predict-forward path: pure topic match, no recency bias."""
    if not past:
        return []
    scored = [(c.session_id, cosine(c.bow, trajectory_bow)) for c in past]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [sid for sid, _ in scored[:k]]


def heuristic_predict(trajectory_bow: Counter, past: List[Conversation], k: int) -> List[str]:
    """NDPA's heuristic: blend topic + recency. What the predictions API uses live."""
    if not past:
        return []
    most_recent = max(c.started_at for c in past)
    scored = []
    for c in past:
        recency = 1.0 / (1 + max(0, (most_recent - c.started_at) / 86400))
        topic = cosine(c.bow, trajectory_bow)
        scored.append((c.session_id, 0.4 * recency + 0.6 * topic))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [sid for sid, _ in scored[:k]]


def baseline_recency(trajectory_bow: Counter, past: List[Conversation], k: int) -> List[str]:
    """Dumbest baseline: just return the most recent past conversations."""
    return [c.session_id for c in sorted(past, key=lambda c: c.started_at, reverse=True)[:k]]


def baseline_random(trajectory_bow: Counter, past: List[Conversation], k: int) -> List[str]:
    """Random baseline for sanity check."""
    import random
    rng = random.Random(42)
    return [c.session_id for c in rng.sample(past, min(k, len(past)))]


def filter_to_real_user_data(convs: List[Conversation]) -> List[Conversation]:
    """
    Filter out LongMemEval test haystacks and other benchmark-generated data.
    PQHR should be measured on REAL user conversation patterns, not on
    benchmarks where 500 synthetic users got mixed into one pool.
    """
    return [
        c for c in convs
        if not any(prefix in c.session_id.lower() for prefix in (
            "longmemeval", "locomo_", "bench_", "synthetic_",
            "test_", "smoke_", "dogfood_", "fix_test", "latency_test",
            "predictive_demo", "demo_", "conv_pred_test", "hook_",
            "rl_test", "live_demo",
        ))
    ]


def build_pqhr_samples(convs: List[Conversation], trajectory_window: int) -> list:
    """For each conversation, build a sample where the predictor sees ONLY prior conversations."""
    samples = []
    sorted_convs = sorted(convs, key=lambda c: c.started_at)

    for i, current in enumerate(sorted_convs):
        if i < max(trajectory_window, MIN_PAST_CONVS):
            continue
        if len(current.content) < MIN_FUTURE_CHARS:
            continue

        # Trajectory: last N conversations before C_i, NOT including C_i
        trajectory = sorted_convs[max(0, i - trajectory_window): i]
        trajectory_text = " ".join(c.content for c in trajectory)
        trajectory_bow = tokenize(trajectory_text)
        if not trajectory_bow:
            continue

        # Past pool: everything before C_i
        past = sorted_convs[:i]

        # Ground truth: past conversations whose content best matches C_i's content
        scored = [(c.session_id, cosine(c.bow, current.bow)) for c in past]
        scored.sort(key=lambda x: x[1], reverse=True)
        truth = {sid for sid, score in scored[:K_GROUND_TRUTH] if score > 0.05}
        if not truth:
            continue

        samples.append((current, trajectory_bow, past, truth))
    return samples


def score(predictor, samples) -> dict:
    hits = {1: 0, 3: 0, 5: 0}
    n = len(samples)
    if n == 0:
        return {f"pqhr@{k}": 0.0 for k in (1, 3, 5)}
    for _current, trajectory_bow, past, truth in samples:
        preds = predictor(trajectory_bow, past, TOP_K_PREDICT)
        for k in (1, 3, 5):
            if any(p in truth for p in preds[:k]):
                hits[k] += 1
    return {f"pqhr@{k}": v / n for k, v in hits.items()}


def main():
    global MIN_FUTURE_CHARS
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-window", type=int, default=TRAJECTORY_WINDOW_DEFAULT,
                        help="How many prior conversations the predictor sees")
    parser.add_argument("--min-future-chars", type=int, default=MIN_FUTURE_CHARS)
    args = parser.parse_args()
    MIN_FUTURE_CHARS = args.min_future_chars

    print("=" * 72)
    print(" Pre-Query Hit Rate (PQHR) — NDPA's predict-forward benchmark")
    print("=" * 72)
    print(f" trajectory window: last {args.trajectory_window} conversations")
    print(f" predictor sees: ONLY prior conversations, never current or future")
    print()

    all_convs = load_conversations()
    print(f"Loaded {len(all_convs)} total conversations")

    convs = filter_to_real_user_data(all_convs)
    print(f"After filtering benchmark/test data: {len(convs)} real-user conversations")

    samples = build_pqhr_samples(convs, args.trajectory_window)
    print(f"PQHR samples: {len(samples)}")
    if not samples:
        print("Not enough data to evaluate.")
        return
    print()

    random_baseline = score(baseline_random, samples)
    recency_baseline = score(baseline_recency, samples)
    heuristic = score(heuristic_predict, samples)
    pure_topic = score(pure_topic_predict, samples)

    print(f"  {'predictor':<24}  pqhr@1   pqhr@3   pqhr@5")
    print(f"  {'random (sanity)':<24}  {random_baseline['pqhr@1']:.4f}  {random_baseline['pqhr@3']:.4f}  {random_baseline['pqhr@5']:.4f}")
    print(f"  {'recency-only':<24}  {recency_baseline['pqhr@1']:.4f}  {recency_baseline['pqhr@3']:.4f}  {recency_baseline['pqhr@5']:.4f}")
    print(f"  {'heuristic (0.4r+0.6t)':<24}  {heuristic['pqhr@1']:.4f}  {heuristic['pqhr@3']:.4f}  {heuristic['pqhr@5']:.4f}")
    print(f"  {'pure topic (NDPA)':<24}  {pure_topic['pqhr@1']:.4f}  {pure_topic['pqhr@3']:.4f}  {pure_topic['pqhr@5']:.4f}")
    print()

    lift = (pure_topic["pqhr@5"] - recency_baseline["pqhr@5"]) * 100
    print(f"Lift over recency baseline @5: +{lift:.1f} pts")
    print()
    print("=" * 72)
    print(" INTERPRETATION")
    print("=" * 72)
    print(f"  Out of every 100 future conversations, NDPA correctly")
    print(f"  pre-stages a relevant past conversation in its top 5")
    print(f"  predictions {int(pure_topic['pqhr@5'] * 100)}% of the time — using ONLY")
    print(f"  prior session trajectory. No question. No current conversation.")
    print()
    print(f"  Recency baseline gets {int(recency_baseline['pqhr@5'] * 100)}%. Random gets {int(random_baseline['pqhr@5'] * 100)}%.")
    print()
    print(f"  This is the metric NDPA owns: predict-forward staging from")
    print(f"  behavioral signal alone. Vector DBs require a query. Mem0/Zep")
    print(f"  require an explicit input. NDPA is the only system that scores")
    print(f"  on this benchmark because nobody else attempts it.")


if __name__ == "__main__":
    main()
