"""
Honest "novel topic prediction" eval.

The original conversation eval (eval/conversation_eval.py) has a subtle
issue: ground truth uses BoW(future), predictor uses BoW(history), and
within one conversation those BoWs are highly correlated (same topic
discussed throughout). The 85% hit@5 measures topic CONTINUITY, not
genuinely novel prediction.

This eval is stricter. For each conversation:
  1. Take the FIRST half (history)
  2. Extract NOVEL terms in the SECOND half — words that appear in future
     turns but NOT in history. These are the "new things" introduced.
  3. Ground truth = past conversations whose content contains those NOVEL
     terms (you couldn't guess them from history alone).
  4. Predictor must surface past conversations that introduced those novel
     terms — proving it can predict topics the user is ABOUT to bring up.

This is closer to what platforms care about: predicting context the user
will need but hasn't asked for yet, based on behavioral signal alone.

Usage:
    python3 -m eval.novel_topic_eval
"""

import re
import time
from collections import Counter
from typing import List, Set, Tuple

from eval.conversation_eval import (
    Conversation, load_conversations, tokenize, cosine,
    TOP_K_PREDICT, MIN_PAST_CONVS,
)

# Need enough turns to split into "history" and "future" meaningfully
MIN_TURNS_FOR_NOVEL = 4
# How many of the most-novel future terms count as "truly novel"
NOVEL_TERM_TOP_K = 10
# A past conversation is "relevant" if it contains at least this many novel terms
MIN_NOVEL_OVERLAP = 2


def novel_terms(history_bow: Counter, future_bow: Counter, k: int = NOVEL_TERM_TOP_K) -> Set[str]:
    """
    Terms that appear in the FUTURE but NOT in the history (or appear much
    more frequently in future). These are the new topics introduced.
    """
    novel = []
    for word, future_count in future_bow.items():
        hist_count = history_bow.get(word, 0)
        # Truly novel: word absent or rare in history but prominent in future
        if hist_count == 0 and future_count >= 1:
            novel.append((word, future_count))
        elif future_count > hist_count * 2 and future_count >= 2:
            novel.append((word, future_count - hist_count))
    novel.sort(key=lambda x: x[1], reverse=True)
    return {w for w, _ in novel[:k]}


def relevant_past_with_novel(novel: Set[str], past: List[Conversation], k: int) -> Set[str]:
    """A past conv is relevant if it contains at least MIN_NOVEL_OVERLAP of the novel terms."""
    if not novel:
        return set()
    scored = []
    for c in past:
        overlap = sum(1 for w in novel if w in c.bow and c.bow[w] >= 2)
        if overlap >= MIN_NOVEL_OVERLAP:
            scored.append((c.session_id, overlap))
    scored.sort(key=lambda x: x[1], reverse=True)
    return {sid for sid, _ in scored[:k]}


def baseline_recency(history_bow: Counter, past: List[Conversation], k: int) -> List[str]:
    """Recency-only — the dumb baseline."""
    return [c.session_id for c in sorted(past, key=lambda c: c.started_at, reverse=True)[:k]]


def heuristic_predict(history_bow: Counter, past: List[Conversation], k: int) -> List[str]:
    """NDPA's actual predictor: topic + recency."""
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


def build_novel_samples(convs: List[Conversation]) -> list:
    """For each conversation, build (history_bow, past, ground_truth_novel_relevant)."""
    samples = []
    sorted_convs = sorted(convs, key=lambda c: c.started_at)

    for i, current in enumerate(sorted_convs):
        if len(current.turns) < MIN_TURNS_FOR_NOVEL:
            continue
        past = sorted_convs[:i]
        if len(past) < MIN_PAST_CONVS:
            continue

        split = len(current.turns) // 2
        history_text = " ".join(current.turns[:split])
        future_text = " ".join(current.turns[split:])
        history_bow = tokenize(history_text)
        future_bow = tokenize(future_text)
        if not history_bow or not future_bow:
            continue

        novel = novel_terms(history_bow, future_bow)
        if not novel:
            continue

        truth = relevant_past_with_novel(novel, past, k=5)
        if truth:
            samples.append((current, history_bow, past, truth, novel))

    return samples


def score(predictor, samples) -> dict:
    hits = {1: 0, 3: 0, 5: 0}
    n = len(samples)
    if n == 0:
        return {f"hit@{k}": 0.0 for k in (1, 3, 5)}
    for _current, history_bow, past, truth, _novel in samples:
        preds = predictor(history_bow, past, TOP_K_PREDICT)
        for k in (1, 3, 5):
            if any(p in truth for p in preds[:k]):
                hits[k] += 1
    return {f"hit@{k}": v / n for k, v in hits.items()}


def main():
    print("Loading conversations...")
    convs = load_conversations()
    print(f"Loaded {len(convs)} conversations")

    samples = build_novel_samples(convs)
    print(f"\nNovel-topic samples: {len(samples)}")
    if not samples:
        print("Need more conversations to evaluate novel prediction.")
        return

    avg_novel = sum(len(s[4]) for s in samples) / len(samples)
    avg_truth = sum(len(s[3]) for s in samples) / len(samples)
    print(f"Avg novel terms per sample: {avg_novel:.1f}")
    print(f"Avg relevant past convs per sample: {avg_truth:.1f}")

    baseline = score(baseline_recency, samples)
    heuristic = score(heuristic_predict, samples)

    print()
    print(f"{'':12s}  hit@1   hit@3   hit@5")
    print(f"{'baseline':12s}  {baseline['hit@1']:.4f}  {baseline['hit@3']:.4f}  {baseline['hit@5']:.4f}")
    print(f"{'heuristic':12s}  {heuristic['hit@1']:.4f}  {heuristic['hit@3']:.4f}  {heuristic['hit@5']:.4f}")

    print()
    for k in (1, 3, 5):
        lift = (heuristic[f"hit@{k}"] - baseline[f"hit@{k}"]) * 100
        print(f"lift@{k}: {lift:+.1f} pts")

    print()
    print("Methodology note:")
    print("  This eval measures prediction of NOVEL topics — terms introduced in")
    print("  the future of a conversation that were NOT in its history. Stricter")
    print("  than the original conversation eval, which measured topic continuity.")


if __name__ == "__main__":
    main()
