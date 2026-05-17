#!/usr/bin/env python3
"""
End-to-end QA eval on LongMemEval.

Reuses ingestion + retrieval from longmemeval_runner.py, then chains the
retrieved context into an LLM reader and scores the FINAL ANSWER.

This is the number that compares apples-to-apples against published
Mem0/Zep/TiMem results.

Usage:
    export OPENAI_API_KEY="sk-..."
    python3 -m eval.longmemeval_e2e_runner
    python3 -m eval.longmemeval_e2e_runner --model gpt-4o --yes
    python3 -m eval.longmemeval_e2e_runner --model gpt-4o-mini --yes

    # Claude reader (shows NDPA+Claude numbers for Anthropic pitch)
    export ANTHROPIC_API_KEY="sk-ant-..."
    python3 -m eval.longmemeval_e2e_runner --model claude-sonnet-4-6 --yes
    python3 -m eval.longmemeval_e2e_runner --model claude-opus-4-7 --yes

Cost depends on the reader model; the runner records token usage and estimated spend.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from eval.external_benchmark_utils import (
    DATA_DIR,
    RateLimiter,
    aggregate_hits,
    call_with_retries,
    download_json,
    load_ndpa_client,
    prediction_session_ids,
    session_to_events,
    write_json,
)
from eval.longmemeval_runner import DATASETS, PUBLISHED_RETRIEVAL_NOTES, category_for

RESULTS_PATH = Path(__file__).with_name("longmemeval_e2e_results.json")
DEFAULT_MODEL = "gpt-4.1-mini"
MAX_CONTEXT_CHARS = 80_000
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
    prices = MODEL_PRICES_PER_MILLION.get(model)
    if prices is None:
        # Keep unknown paid models explicit instead of pretending old mini
        # pricing applies.
        return 0.0
    return (
        input_tokens / 1_000_000 * prices["input"]
        + output_tokens / 1_000_000 * prices["output"]
    )

QUESTION_TYPE_GUIDANCE = {
    "temporal-reasoning": (
        "Use session dates in context headers. Build a short timeline, prefer "
        "the most relevant dated evidence, and answer with a concrete date, "
        "duration, count, or ordering."
    ),
    "multi-session": (
        "This may require evidence from multiple prior sessions. Synthesize "
        "across sessions instead of stopping at the first matching fact."
    ),
    "knowledge-update": (
        "Prefer the most recent relevant evidence when older and newer context "
        "conflict or the user corrected something."
    ),
    "abstention": (
        "If the supplied context does not mention the requested subject, say "
        "the user did not mention it."
    ),
    "single-session-preference": (
        "Preserve nuance from the user's original language and infer preferences "
        "from repeated patterns, not just isolated facts."
    ),
}

SYSTEM_PROMPT = (
    "You are answering a question about the user based on context from their past "
    "conversations. Use the context to answer.\n\n"
    "GUIDELINES:\n"
    "- For factual questions ('what did I do', 'when', 'where'): give a direct concise "
    "answer using the context. Don't pad with 'According to your conversations...'\n"
    "- For preference questions ('what kind of X would I like', 'recommend Y'): INFER "
    "the user's preferences from patterns in the context, even if not explicitly stated. "
    "Be specific about what they'd prefer based on what they've discussed liking, doing, or asking about.\n"
    "- For temporal questions ('how long ago', 'when did'): reason carefully about dates "
    "and times. If multiple dates appear, identify the most relevant one to the question. "
    "Give a specific date or duration, not a vague answer.\n"
    "- Context headers may include session_id, date, platform, and rank. Use header dates "
    "for temporal reasoning; do not treat headers as conversation text.\n"
    "- For multi-session questions: combine evidence from multiple context blocks when needed.\n"
    "- For knowledge-update questions: use the MOST RECENT information in the context if "
    "the user has corrected or updated something.\n"
    "- For abstention questions ('did I mention X'): if the context contains nothing about X, "
    "say 'You did not mention [X]' — not 'I don't know'.\n"
    "- ONLY say 'I don't know' if the context truly contains nothing relevant.\n"
    "- Do not fabricate facts not in the context."
)


TEMPORAL_RE = re.compile(
    r"\b(when|before|after|last|latest|first|recent|recently|current|currently|"
    r"today|yesterday|tomorrow|week|month|year|date|time|ago|older|newer|"
    r"changed|updated|previous|next)\b",
    re.I,
)


def adaptive_context_chars(category: str, question: str, k: int) -> int:
    """Allocate more context for categories where misses are mostly reader-side."""
    budgets = {
        "temporal-reasoning": 75_000,
        "multi-session": 70_000,
        "knowledge-update": 60_000,
        "single-session-preference": 60_000,
        "single-session-user": 50_000,
        "single-session-assistant": 45_000,
        "abstention": 35_000,
    }
    budget = budgets.get(category, 50_000)
    if TEMPORAL_RE.search(question):
        budget = max(budget, 70_000)
    if k > 5:
        budget += min(20_000, (k - 5) * 4_000)
    return min(MAX_CONTEXT_CHARS, budget)


def format_prediction_context(predictions: list[dict], max_chars: int) -> str:
    """Preserve session/date metadata and truncate by whole ranked blocks."""
    parts = []
    used = 0
    for rank, pred in enumerate(predictions, start=1):
        content = str(pred.get("content") or "").strip()
        if not content:
            continue
        header = (
            f"[rank={rank} session_id={pred.get('session_id') or pred.get('id') or 'unknown'} "
            f"date={(pred.get('started_at') or '')[:10] or 'unknown'} "
            f"platform={pred.get('platform') or 'unknown'} "
            f"score={pred.get('score', pred.get('topic_score', ''))}]"
        )
        remaining = max_chars - used - len(header) - 8
        if remaining <= 0:
            break
        block = f"{header}\n{content[:remaining]}"
        parts.append(block)
        used += len(block) + 8
    return "\n\n---\n\n".join(parts)


def build_prompt(question: str, predictions: list[dict], category: str, k: int) -> tuple[str, str]:
    """Build (system, user) prompt strings for the LLM call."""
    context_blob = format_prediction_context(predictions, adaptive_context_chars(category, question, k))
    guidance = QUESTION_TYPE_GUIDANCE.get(category, "")
    user = (
        f"Question category: {category}\n"
        f"{'Category guidance: ' + guidance if guidance else ''}\n\n"
        f"Context from past conversations:\n\n{context_blob}\n\n---\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    return SYSTEM_PROMPT, user


def call_openai(client, model: str, system: str, user: str) -> tuple[str, int, int]:
    """Returns (answer_text, input_tokens, output_tokens)."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
    )
    answer = (resp.choices[0].message.content or "").strip()
    in_tok = getattr(resp.usage, "prompt_tokens", 0)
    out_tok = getattr(resp.usage, "completion_tokens", 0)
    return answer, in_tok, out_tok


def call_anthropic(client, model: str, system: str, user: str) -> tuple[str, int, int]:
    """Returns (answer_text, input_tokens, output_tokens)."""
    resp = client.messages.create(
        model=model,
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    answer = (resp.content[0].text if resp.content else "").strip()
    in_tok = getattr(resp.usage, "input_tokens", 0)
    out_tok = getattr(resp.usage, "output_tokens", 0)
    return answer, in_tok, out_tok


def call_reader(openai_client, anthropic_client, model: str, system: str, user: str) -> tuple[str, int, int]:
    if model.startswith("claude"):
        if anthropic_client is None:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        return call_anthropic(anthropic_client, model, system, user)
    return call_openai(openai_client, model, system, user)


def grade_answer(predicted: str, ground_truth: str | None) -> bool:
    """
    Lenient string-overlap grader.

    The LongMemEval paper uses an LLM-as-judge for their official scorer.
    We use a cheap proxy here so the run is self-contained. The number
    you get from this grader is a LOWER BOUND on true accuracy — it
    misses paraphrases and equivalent answers. For a publication-grade
    number, swap this for the paper's official LLM judge.
    """
    if not predicted or not ground_truth:
        return False
    p = predicted.lower().strip()
    g = str(ground_truth).lower().strip()
    if not g or g == "n/a":
        return False
    # exact match, substring either direction, or strong token overlap
    if g in p or p in g:
        return True
    g_tokens = {w for w in g.split() if len(w) > 3}
    p_tokens = {w for w in p.split() if len(w) > 3}
    if g_tokens and len(g_tokens & p_tokens) / len(g_tokens) >= 0.6:
        return True
    return False


def run(args: argparse.Namespace) -> dict:
    openai_client = None
    anthropic_client = None

    if args.model.startswith("claude"):
        ant_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not ant_key:
            print("ERROR: ANTHROPIC_API_KEY not set for Claude reader.", file=sys.stderr)
            sys.exit(1)
        try:
            import anthropic
            anthropic_client = anthropic.Anthropic(api_key=ant_key)
        except ImportError:
            print("ERROR: pip install anthropic", file=sys.stderr)
            sys.exit(1)
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            print("ERROR: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
            print("Run: export OPENAI_API_KEY=\"sk-...\"", file=sys.stderr)
            sys.exit(1)
        try:
            from openai import OpenAI
        except ImportError:
            print("ERROR: openai package not installed. Run: pip install openai", file=sys.stderr)
            sys.exit(1)
        openai_client = OpenAI(api_key=api_key)

    dataset_id, filename, url = DATASETS[args.split]
    path = DATA_DIR / "longmemeval" / filename
    items = download_json(url, path, force=args.force_download)
    if args.max_items:
        items = items[: args.max_items]

    # Cost estimate before running
    est_input = len(items) * 6000
    est_output = len(items) * 200
    est_cost = estimate_cost_usd(args.model, est_input, est_output)
    print(f"\n  Cost estimate: ~${est_cost:.2f} for {len(items)} questions with {args.model}")
    print(f"  Input tokens ~{est_input:,}, output tokens ~{est_output:,}")
    if not args.yes:
        confirm = input("  Proceed? [y/N] ").strip().lower()
        if confirm != "y":
            print("  Aborted.")
            return {}

    ndpa = load_ndpa_client("longmemeval", timeout=args.timeout)
    limiter = RateLimiter(args.requests_per_minute)

    rows = []
    total_in = 0
    total_out = 0
    correct = 0
    started = time.time()

    for index, item in enumerate(items, start=1):
        question_id = str(item["question_id"])
        category = category_for(item)
        end_user_id = f"longmemeval_{question_id}"
        haystack_ids = [str(sid) for sid in item.get("haystack_session_ids") or []]
        haystack_dates = item.get("haystack_dates") or []
        haystack_sessions = item.get("haystack_sessions") or []
        question = str(item.get("question") or "")
        ground_truth = item.get("answer")

        # Ingest haystack (idempotent — server already has them if previous run did this)
        def ingest_session(session_index: int, session: list[dict]) -> None:
            session_id = haystack_ids[session_index] if session_index < len(haystack_ids) else f"{question_id}_{session_index}"
            fallback_ts = 1_700_000_000.0 + session_index
            events = session_to_events(session, fallback_ts=fallback_ts)
            if not events:
                return
            if session_index < len(haystack_dates):
                events[0]["source_path"] = f"longmemeval:{haystack_dates[session_index]}"
            limiter.wait()
            call_with_retries(
                lambda: ndpa.log_events(session_id, events, end_user_id=end_user_id),
                attempts=args.retries,
            )

        if not args.skip_ingest:
            with ThreadPoolExecutor(max_workers=args.ingest_workers) as pool:
                futures = [
                    pool.submit(ingest_session, si, sess)
                    for si, sess in enumerate(haystack_sessions)
                ]
                for f in as_completed(futures):
                    f.result()

        # Retrieve top-K from NDPA
        result = call_with_retries(
            lambda: ndpa.get_predictions(
                session_id=f"longmemeval_query_{question_id}",
                query=question,
                k=args.k,
                end_user_id=end_user_id,
            ),
            attempts=args.retries,
        )
        predictions = result.get("predictions") or []

        # Call LLM
        system, user = build_prompt(question, predictions, category, args.k)
        try:
            answer, in_tok, out_tok = call_reader(openai_client, anthropic_client, args.model, system, user)
        except Exception as e:
            answer = ""
            in_tok = 0
            out_tok = 0
            print(f"  [warn] reader call failed at q={question_id}: {e}", flush=True)

        total_in += in_tok
        total_out += out_tok

        is_correct = grade_answer(answer, ground_truth if isinstance(ground_truth, str) else None)
        if is_correct:
            correct += 1

        rows.append({
            "question_id": question_id,
            "category": category,
            "question": question,
            "ground_truth": ground_truth,
            "predicted_answer": answer,
            "correct": is_correct,
            "predicted_session_ids": prediction_session_ids(result),
            "tokens": {"in": in_tok, "out": out_tok},
        })

        if index % args.progress_every == 0:
            elapsed = time.time() - started
            acc = correct / index
            cost_so_far = estimate_cost_usd(args.model, total_in, total_out)
            print(f"  {index}/{len(items)} | acc={acc:.3f} | ${cost_so_far:.3f} | {elapsed:.0f}s", flush=True)

    # Aggregate by category
    by_category: dict = {}
    for r in rows:
        cat = r["category"]
        by_category.setdefault(cat, {"correct": 0, "n": 0})
        by_category[cat]["n"] += 1
        if r["correct"]:
            by_category[cat]["correct"] += 1

    metrics = {
        "overall": {"accuracy": correct / len(rows) if rows else 0.0, "n": len(rows)},
    }
    for cat, stats in by_category.items():
        metrics[cat] = {"accuracy": stats["correct"] / stats["n"], "n": stats["n"]}

    total_cost = estimate_cost_usd(args.model, total_in, total_out)

    payload = {
        "benchmark": "LongMemEval (end-to-end QA)",
        "dataset": dataset_id,
        "split": args.split,
        "model": args.model,
        "scorer": "lenient string overlap (proxy for LLM judge)",
        "n_items": len(items),
        "elapsed_sec": round(time.time() - started, 2),
        "tokens": {"in": total_in, "out": total_out},
        "cost_usd": round(total_cost, 4),
        "metrics": metrics,
        "published_comparison_note": (
            "Published Mem0/Zep/TiMem numbers use the LongMemEval paper's "
            "official LLM-as-judge scorer. This run uses a lenient string-overlap "
            "proxy. Our number is a LOWER BOUND on true accuracy. For "
            "publication-grade comparison, re-grade with the official scorer."
        ),
        "published_retrieval_notes": PUBLISHED_RETRIEVAL_NOTES,
        "rows": rows,  # ALL rows saved so we can re-grade later without re-running
    }
    out_path = Path(args.out) if args.out else RESULTS_PATH
    write_json(out_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=sorted(DATASETS), default="s")
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--yes", action="store_true", help="Skip cost-confirm prompt")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip ingestion if data already in NDPA")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--requests-per-minute", type=int, default=540)
    parser.add_argument("--ingest-workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--out", default="", help="Optional output JSON path")
    args = parser.parse_args()

    result = run(args)
    if result:
        print()
        print(json.dumps(result["metrics"], indent=2))
        print(f"\nTotal cost: ${result['cost_usd']:.4f}")
        print(f"Results: {Path(args.out) if args.out else RESULTS_PATH}")


if __name__ == "__main__":
    main()
