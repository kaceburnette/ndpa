#!/usr/bin/env python3
"""
LLM-as-judge re-grader.

The published Mem0/Zep/MemPalace numbers use LLM-as-judge grading — they
ask an LLM "did this answer match the ground truth?" instead of doing
string matching. This is the field standard for memory benchmarks.

Our string-overlap grader is conservative. It marks correct paraphrases
as wrong. This script re-grades saved answers using an LLM judge
as judge.

Re-grading is free for retrieval (no new API calls for that) — only the
Judge calls cost money.

Usage:
    export OPENAI_API_KEY="sk-..."
    python3 -m eval.llm_judge_regrade
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

DEFAULT_RESULTS_PATH = Path(__file__).with_name("longmemeval_e2e_results.json")
MODEL_PRICES_PER_MILLION = {
    "gpt-5.5": {"input": 5.00, "output": 30.00},
    "gpt-5.4": {"input": 2.50, "output": 15.00},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "gpt-5.4-nano": {"input": 0.20, "output": 1.25},
    "gpt-5.1": {"input": 1.25, "output": 10.00},
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = MODEL_PRICES_PER_MILLION.get(model, MODEL_PRICES_PER_MILLION["gpt-4.1-mini"])
    return (
        input_tokens / 1_000_000 * prices["input"]
        + output_tokens / 1_000_000 * prices["output"]
    )

# Load .env if present
ENV_FILE = Path("/Users/kaceburnette/Desktop/ndp/.env")
if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, _, v = line.partition("=")
            os.environ[k.strip()] = v.strip().strip("\"'")

JUDGE_PROMPT = """You are grading a memory-system answer.

The system was asked a question about a user. You'll see:
- The question
- The ground-truth correct answer
- The system's answer

Decide if the system's answer is semantically equivalent to the ground truth.

Rules:
- "Yes, semantically equivalent" if the answer captures the same fact/information
- "Yes" even if wording differs significantly, as long as the meaning matches
- "Yes" for abstention: if ground truth says "user did not mention X" and the answer
  says "I don't know" or "you didn't mention X" — those are equivalent.
- "Yes" for preferences: if ground truth describes preferences and the answer
  captures the same preferences (even via different phrasing).
- "No" only if the system's answer is wrong, incomplete in a meaningful way, or
  fabricates information not in the ground truth.

Respond with ONLY one word: YES or NO."""


def judge(client, model: str, question: str, truth: str, predicted: str) -> tuple[bool, int, int]:
    user_msg = (
        f"Question: {question}\n\n"
        f"Ground truth: {truth}\n\n"
        f"System answer: {predicted}\n\n"
        f"Is the system answer semantically equivalent? Answer YES or NO."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=5,
    )
    answer = (resp.choices[0].message.content or "").strip().upper()
    return ("YES" in answer), getattr(resp.usage, "prompt_tokens", 0), getattr(resp.usage, "completion_tokens", 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=str(DEFAULT_RESULTS_PATH),
                        help="results JSON to regrade (e2e or reasoning)")
    parser.add_argument("--model", default="gpt-4.1-mini")
    args = parser.parse_args()

    results_path = Path(args.target)
    if not results_path.exists():
        print(f"ERROR: results file not found: {results_path}", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: pip install openai", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    model = args.model

    print(f"Regrading: {results_path.name} with judge model: {model}")
    with open(results_path) as f:
        data = json.load(f)
    rows = data.get("rows") or []
    if not rows:
        print("No rows to re-grade", file=sys.stderr)
        sys.exit(1)

    print(f"Re-grading {len(rows)} answers with LLM-as-judge ({model})...")
    print(f"Estimated cost: ~${estimate_cost_usd(model, len(rows) * 300, len(rows) * 5):.2f}\n")

    total_in = 0
    total_out = 0
    correct = 0
    by_category: dict = {}

    for i, r in enumerate(rows, start=1):
        truth = r.get("ground_truth")
        pred = r.get("predicted_answer", "")
        cat = r.get("category", "unknown")
        question = r.get("question", "")

        if not pred or not truth:
            r["correct_llm_judge"] = False
            by_category.setdefault(cat, {"correct": 0, "n": 0})
            by_category[cat]["n"] += 1
            continue

        try:
            is_correct, in_tok, out_tok = judge(client, model, question, str(truth), str(pred))
        except Exception as e:
            print(f"  [warn] judge failed at q={r.get('question_id')}: {e}", flush=True)
            is_correct = False
            in_tok = 0
            out_tok = 0

        r["correct_llm_judge"] = is_correct
        if is_correct:
            correct += 1
        total_in += in_tok
        total_out += out_tok

        by_category.setdefault(cat, {"correct": 0, "n": 0})
        by_category[cat]["n"] += 1
        if is_correct:
            by_category[cat]["correct"] += 1

        if i % 25 == 0:
            acc = correct / i
            cost = estimate_cost_usd(model, total_in, total_out)
            print(f"  {i}/{len(rows)} | acc={acc:.3f} | ${cost:.3f}", flush=True)

    print()
    print(f"{'Category':<30} {'v1':>6} {'v2':>6} {'judge':>6}  {'n':>4}")
    print("-" * 60)
    for cat, stats in sorted(by_category.items()):
        v1 = sum(1 for r in rows if r["category"] == cat and r.get("correct")) / stats["n"]
        v2 = sum(1 for r in rows if r["category"] == cat and r.get("correct_v2")) / stats["n"]
        judge_acc = stats["correct"] / stats["n"]
        print(f"{cat:<30} {v1:>5.1%} {v2:>5.1%} {judge_acc:>5.1%}  {stats['n']:>4}")
    print("-" * 60)
    overall_v1 = sum(1 for r in rows if r.get("correct")) / len(rows)
    overall_v2 = sum(1 for r in rows if r.get("correct_v2")) / len(rows)
    overall_judge = correct / len(rows)
    print(f"{'OVERALL':<30} {overall_v1:>5.1%} {overall_v2:>5.1%} {overall_judge:>5.1%}  {len(rows):>4}")

    total_cost = estimate_cost_usd(model, total_in, total_out)
    print(f"\nTotal cost: ${total_cost:.4f}")

    data["metrics_llm_judge"] = {
        "overall": {"accuracy": overall_judge, "n": len(rows)},
        "judge_model": model,
        "judge_cost_usd": round(total_cost, 4),
    }
    for cat, stats in by_category.items():
        data["metrics_llm_judge"][cat] = {"accuracy": stats["correct"] / stats["n"], "n": stats["n"]}

    with open(results_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved to {results_path}")


if __name__ == "__main__":
    main()
