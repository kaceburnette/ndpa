#!/usr/bin/env python3
"""Apply the Reasoning repair pass to an existing LongMemEval result file.

This is production-shaped: it repairs every hard-category row, not only rows
known to be wrong. It rehydrates benchmark evidence locally, uses the saved
draft answer and extracted facts, and writes a new result JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from eval.external_benchmark_utils import DATA_DIR, download_json, write_json
from eval.longmemeval_e2e_runner import adaptive_context_chars, grade_answer
from eval.longmemeval_runner import DATASETS, category_for
from eval.longmemeval_reasoning_runner import (
    REPAIR_CATEGORIES,
    estimate_cost_usd,
    focused_hydration_context,
    local_memory_predictions,
    repair_answer,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", choices=sorted(DATASETS), default="s")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: pip install openai", file=sys.stderr)
        sys.exit(1)

    import os

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    client = OpenAI(api_key=api_key)

    source_path = Path(args.target)
    data = json.loads(source_path.read_text())
    rows = data.get("rows") or []

    dataset_id, filename, url = DATASETS[args.split]
    items = download_json(url, DATA_DIR / "longmemeval" / filename)
    items_by_id = {str(item["question_id"]): item for item in items}

    total_in = 0
    total_out = 0
    repaired_count = 0
    started = time.time()

    for index, row in enumerate(rows, start=1):
        category = row.get("category", "")
        if category not in REPAIR_CATEGORIES:
            continue
        question_id = str(row.get("question_id"))
        item = items_by_id.get(question_id)
        if not item:
            continue

        question = str(row.get("question") or item.get("question") or "")
        reference_date = str(row.get("question_date") or item.get("question_date") or "")
        predictions, _predicted_ids = local_memory_predictions(item, question, args.k)
        context = focused_hydration_context(
            question,
            predictions,
            category,
            adaptive_context_chars(category, question, args.k),
        )
        evidence = (
            f"Question reference date: {reference_date}\n\n"
            "Focused hydrated memory evidence:\n"
            f"{context}"
        )
        draft = str(row.get("predicted_answer") or "")
        facts = str(row.get("extracted_facts") or "")
        if not draft:
            continue

        repaired, in_tok, out_tok = repair_answer(
            client,
            args.model,
            question,
            category,
            evidence,
            facts,
            draft,
            reference_date,
        )
        total_in += in_tok
        total_out += out_tok
        if repaired:
            row["pre_repair_answer"] = draft
            row["predicted_answer"] = repaired
            row["routed_mode"] = f"{row.get('routed_mode', 'unknown')}+repair_only"
            row["correct"] = grade_answer(
                repaired,
                row.get("ground_truth") if isinstance(row.get("ground_truth"), str) else None,
            )
            tokens = row.setdefault("tokens", {})
            tokens["repair_in"] = tokens.get("repair_in", 0) + in_tok
            tokens["repair_out"] = tokens.get("repair_out", 0) + out_tok
            repaired_count += 1

        if args.progress_every and repaired_count % args.progress_every == 0:
            cost = estimate_cost_usd(args.model, total_in, total_out)
            print(f"{repaired_count} repaired | ${cost:.3f} | {time.time() - started:.0f}s", flush=True)

    data["repair_pass"] = {
        "source": str(source_path),
        "model": args.model,
        "k": args.k,
        "categories": sorted(REPAIR_CATEGORIES),
        "rows_repaired": repaired_count,
        "tokens": {"in": total_in, "out": total_out},
        "cost_usd": round(estimate_cost_usd(args.model, total_in, total_out), 4),
    }

    # Recompute string metrics after repair.
    by_category: dict[str, dict[str, int]] = {}
    correct = 0
    for row in rows:
        category = row.get("category", "unknown")
        by_category.setdefault(category, {"correct": 0, "n": 0})
        by_category[category]["n"] += 1
        if row.get("correct"):
            correct += 1
            by_category[category]["correct"] += 1
    data["metrics"] = {"overall": {"accuracy": correct / len(rows), "n": len(rows)}}
    for category, stats in by_category.items():
        data["metrics"][category] = {
            "accuracy": stats["correct"] / stats["n"],
            "n": stats["n"],
        }

    write_json(Path(args.out), data)
    print(f"Wrote {args.out}")
    print(json.dumps(data["repair_pass"], indent=2))
    print(json.dumps(data["metrics"]["overall"], indent=2))


if __name__ == "__main__":
    main()
