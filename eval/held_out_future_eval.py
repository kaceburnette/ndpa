"""
Held-out future evaluation — the gold-standard test for predict-forward memory.

The previous conversation eval (eval/conversation_eval.py) and the novel-topic
eval (eval/novel_topic_eval.py) both score predictions WITHIN a conversation
(split at midpoint, predict the back half from the front half).

This eval is stricter and more honest about what NDPA actually claims to do:

  1. Take all of a user's conversations, sort by timestamp.
  2. Split at date D (e.g. 30 days before the most recent).
  3. "Past" = everything before D. "Future" = the held-out 30 days.
  4. For each future conversation C:
       - The predictor only sees C's FIRST few turns (the user's current trajectory).
       - The predictor only knows about conversations from the "past" set.
       - It must surface the past conversations that would actually have been
         helpful for the rest of C.
  5. Ground truth = past conversations whose top_terms overlap most with C's
       complete content (front + back).

This simulates what NDPA does in production:
  - User starts a new session
  - NDPA has never seen this session's future
  - NDPA must promote/stage the right past conversations from cold tier
  - We measure whether it picked right

Usage:
    python3 -m eval.held_out_future_eval                  # default: 30-day holdout
    python3 -m eval.held_out_future_eval --holdout-days 60
    python3 -m eval.held_out_future_eval --first-turns 2  # only see first 2 turns
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

from eval.conversation_eval import (
    Conversation, load_conversations, tokenize, cosine,
    TOP_K_PREDICT,
)

# How many past conversations are "relevant" (ground truth K)
GROUND_TRUTH_K = 5
# Filter: only score future conversations that have at least this many turns
MIN_FUTURE_TURNS = 3
# Filter: must have at least this many past conversations to even consider
MIN_PAST_CONVS = 10


def baseline_recency(history_bow: Counter, past: list[Conversation], k: int) -> list[str]:
    """Baseline: just return the most recent past conversations."""
    return [c.session_id for c in sorted(past, key=lambda c: c.started_at, reverse=True)[:k]]


def heuristic_predict(history_bow: Counter, past: list[Conversation], k: int) -> list[str]:
    """NDPA heuristic: 0.4 recency + 0.6 topic. Same as production kernel."""
    if not past:
        return []
    most_recent = max(c.started_at for c in past)
    scored = []
    for c in past:
        recency = 1.0 / (1 + max(0, (most_recent - c.started_at) / 86400))
        topic = cosine(c.bow, history_bow)
        scored.append((c.session_id, 0.4 * recency + 0.6 * topic))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [sid for sid, _ in scored[:k]]


def pure_topic_predict(history_bow: Counter, past: list[Conversation], k: int) -> list[str]:
    """Topic-only — what predictive_promote uses (no recency bias)."""
    if not past:
        return []
    scored = [(c.session_id, cosine(c.bow, history_bow)) for c in past]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [sid for sid, _ in scored[:k]]


def build_held_out_samples(
    convs: list[Conversation],
    holdout_days: int,
    first_turns: int,
) -> tuple[list, float]:
    """
    Split conversations into past + held-out future. Build eval samples from
    the future, where each sample only sees first_turns of its conversation.
    """
    if not convs:
        return [], 0.0

    convs_sorted = sorted(convs, key=lambda c: c.started_at)
    most_recent_ts = convs_sorted[-1].started_at
    cutoff_ts = most_recent_ts - holdout_days * 86400

    past = [c for c in convs_sorted if c.started_at < cutoff_ts]
    future = [c for c in convs_sorted if c.started_at >= cutoff_ts]

    if len(past) < MIN_PAST_CONVS:
        return [], cutoff_ts

    samples = []
    for current in future:
        if len(current.turns) < MIN_FUTURE_TURNS:
            continue

        # The predictor only sees the first N turns of this future conversation
        n_visible = min(first_turns, max(1, len(current.turns) // 4))
        visible_text = " ".join(current.turns[:n_visible])
        visible_bow = tokenize(visible_text)
        if not visible_bow:
            continue

        # Ground truth: past conversations whose content best matches the
        # FULL content of this future conversation (we know what would have
        # been relevant; predictor doesn't, since it only sees first turns)
        full_bow = current.bow
        scored = [(c.session_id, cosine(c.bow, full_bow)) for c in past]
        scored.sort(key=lambda x: x[1], reverse=True)
        truth = {sid for sid, score in scored[:GROUND_TRUTH_K] if score > 0.05}
        if not truth:
            continue

        samples.append((current, visible_bow, past, truth))

    return samples, cutoff_ts


def score(predictor, samples) -> dict:
    hits = {1: 0, 3: 0, 5: 0}
    n = len(samples)
    if n == 0:
        return {f"hit@{k}": 0.0 for k in (1, 3, 5)}
    for _current, visible_bow, past, truth in samples:
        preds = predictor(visible_bow, past, TOP_K_PREDICT)
        for k in (1, 3, 5):
            if any(p in truth for p in preds[:k]):
                hits[k] += 1
    return {f"hit@{k}": v / n for k, v in hits.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout-days", type=int, default=30,
                        help="Hide conversations from the last N days (default 30)")
    parser.add_argument("--first-turns", type=int, default=2,
                        help="How many opening turns the predictor sees (default 2)")
    args = parser.parse_args()

    print("=" * 72)
    print(" Held-Out Future Eval — the strict test")
    print("=" * 72)
    print(f" holdout window: last {args.holdout_days} days")
    print(f" predictor sees: first {args.first_turns} turns of each held-out conversation")
    print()

    print("Loading conversations...")
    convs = load_conversations()
    print(f"  loaded {len(convs)} conversations")

    samples, cutoff_ts = build_held_out_samples(convs, args.holdout_days, args.first_turns)
    if not samples:
        print(f"\nNot enough data: need {MIN_PAST_CONVS}+ past conversations and held-out future convs.")
        return

    cutoff_iso = time.strftime("%Y-%m-%d", time.localtime(cutoff_ts))
    print(f"  cutoff date: {cutoff_iso}")
    print(f"  past conversations: {sum(1 for c in convs if c.started_at < cutoff_ts)}")
    print(f"  held-out future: {sum(1 for c in convs if c.started_at >= cutoff_ts)}")
    print(f"  eval samples: {len(samples)}")
    print()

    baseline = score(baseline_recency, samples)
    heuristic = score(heuristic_predict, samples)
    topic_only = score(pure_topic_predict, samples)

    print(f"  {'':12s}  hit@1   hit@3   hit@5")
    print(f"  {'baseline':12s}  {baseline['hit@1']:.4f}  {baseline['hit@3']:.4f}  {baseline['hit@5']:.4f}")
    print(f"  {'topic-only':12s}  {topic_only['hit@1']:.4f}  {topic_only['hit@3']:.4f}  {topic_only['hit@5']:.4f}")
    print(f"  {'heuristic':12s}  {heuristic['hit@1']:.4f}  {heuristic['hit@3']:.4f}  {heuristic['hit@5']:.4f}")
    print()
    for k in (1, 3, 5):
        lift = (heuristic[f"hit@{k}"] - baseline[f"hit@{k}"]) * 100
        print(f"  lift@{k} (heuristic vs baseline): {lift:+.1f} pts")
    print()
    print("Interpretation:")
    print(f"  Out of every 100 held-out future conversations, NDPA correctly")
    print(f"  identifies a relevant past conversation in its top 5 predictions")
    print(f"  {int(heuristic['hit@5'] * 100)} times. Baseline (recency-only) gets it")
    print(f"  {int(baseline['hit@5'] * 100)} times.")
    print()
    print("Methodology:")
    print("  - Past = conversations BEFORE cutoff date (predictor's only knowledge)")
    print("  - Future = conversations AFTER cutoff (held-out)")
    print(f"  - Predictor sees first {args.first_turns} turns of each future conv only")
    print("  - Ground truth: past conversations whose content best matches the")
    print("    FULL future conversation (which the predictor doesn't see)")
    print()
    print("This is the strictest of the three evals. Closer to production:")
    print("  user starts a new session → NDPA must promote the right past")
    print("  context from cold tier knowing only the first few turns.")


if __name__ == "__main__":
    main()
