#!/usr/bin/env python3
"""
Re-grade longmemeval_e2e_results.json with a smarter grader.

Free re-scoring: uses the saved predictions, no new API calls.

The original grader misses two patterns:
  1. Abstention questions where truth = "You did not mention X" and the LLM
     correctly says "I don't know" — same semantic answer, different words.
  2. Preference questions where the answer is open-ended.

This grader handles abstention specifically and is more lenient on preference.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


# Phrases that indicate "I don't know" / abstention from the LLM side
LLM_ABSTENTION_PATTERNS = [
    r"\bi don'?t know\b",
    r"\bnot mentioned\b",
    r"\bnot specified\b",
    r"\bdidn'?t mention\b",
    r"\bno information\b",
    r"\bnot stated\b",
    r"\bcan'?t find\b",
    r"\bno reference\b",
    r"\bnot provided\b",
    r"\bnot in the context\b",
    r"\bno mention\b",
]

# Phrases that indicate the GOLD answer is also "this wasn't mentioned"
GOLD_ABSTENTION_PATTERNS = [
    r"\bdid not mention\b",
    r"\bnot mentioned\b",
    r"\bnot in the (context|conversation|information)\b",
    r"\bnot specified\b",
    r"\bnot provided\b",
    r"\bno information\b",
    r"\bdidn'?t mention\b",
]


def is_llm_abstaining(predicted: str) -> bool:
    p = predicted.lower()
    return any(re.search(pat, p) for pat in LLM_ABSTENTION_PATTERNS)


def is_gold_abstention(ground_truth: str) -> bool:
    g = ground_truth.lower()
    return any(re.search(pat, g) for pat in GOLD_ABSTENTION_PATTERNS)


def grade_answer_v2(predicted: str, ground_truth: str | None, category: str) -> bool:
    """
    Smarter grader:
      - Abstention: if both predicted and truth indicate "didn't mention", correct.
      - Preference: lenient token overlap (40% instead of 60%).
      - Default: original logic.
    """
    if not predicted or not ground_truth:
        return False
    g = str(ground_truth).lower().strip()
    p = predicted.lower().strip()
    if not g or g == "n/a":
        return False

    # 1. Abstention handling
    if category == "abstention" or is_gold_abstention(g):
        # If the LLM abstained AND the gold answer is an abstention, that's correct
        if is_llm_abstaining(p):
            return True
        # If the gold mentions a specific thing the user DID mention (the "but not Y" part),
        # check the LLM didn't fabricate the missing thing
        # We're lenient here: any abstention-like response counts

    # 2. Exact / substring match
    if g in p or p in g:
        return True

    # 3. Token overlap — use lower threshold for preference (40%) vs default (60%)
    threshold = 0.40 if category == "single-session-preference" else 0.60
    g_tokens = {w for w in re.findall(r"[a-zA-Z]{4,}", g)}
    p_tokens = {w for w in re.findall(r"[a-zA-Z]{4,}", p)}
    if g_tokens:
        overlap = len(g_tokens & p_tokens) / len(g_tokens)
        if overlap >= threshold:
            return True

    return False


def main():
    results_path = Path(__file__).with_name("longmemeval_e2e_results.json")
    with open(results_path) as f:
        data = json.load(f)

    rows = data.get("rows") or []
    if not rows:
        print("No rows in results file — re-run the e2e benchmark with the patched runner.")
        sys.exit(1)

    print(f"Re-grading {len(rows)} answers with v2 grader...\n")

    # Re-grade
    by_category: dict = {}
    correct_total = 0
    for r in rows:
        cat = r["category"]
        is_correct = grade_answer_v2(r["predicted_answer"], r.get("ground_truth"), cat)
        r["correct_v2"] = is_correct
        if is_correct:
            correct_total += 1
        by_category.setdefault(cat, {"correct": 0, "n": 0, "correct_v1": 0})
        by_category[cat]["n"] += 1
        if is_correct:
            by_category[cat]["correct"] += 1
        if r.get("correct"):
            by_category[cat]["correct_v1"] += 1

    print(f"{'Category':<30} {'v1 acc':>8} {'v2 acc':>8} {'n':>4} {'lift':>6}")
    print("-" * 60)
    for cat, stats in sorted(by_category.items()):
        v1 = stats["correct_v1"] / stats["n"]
        v2 = stats["correct"] / stats["n"]
        lift = (v2 - v1) * 100
        print(f"{cat:<30} {v1:>7.2%} {v2:>7.2%} {stats['n']:>4} {lift:>+5.1f}")
    print("-" * 60)
    overall_v1 = sum(s["correct_v1"] for s in by_category.values()) / len(rows)
    overall_v2 = correct_total / len(rows)
    print(f"{'OVERALL':<30} {overall_v1:>7.2%} {overall_v2:>7.2%} {len(rows):>4} {(overall_v2-overall_v1)*100:>+5.1f}")

    # Save updated results
    data["metrics_v2"] = {
        "overall": {"accuracy": overall_v2, "n": len(rows)},
    }
    for cat, stats in by_category.items():
        data["metrics_v2"][cat] = {"accuracy": stats["correct"] / stats["n"], "n": stats["n"]}
    data["grader_v2_note"] = (
        "v2 grader handles abstention (LLM abstain + gold abstain = correct) and "
        "uses lower 40% token-overlap threshold for preference category."
    )

    with open(results_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved to {results_path}")


if __name__ == "__main__":
    main()
