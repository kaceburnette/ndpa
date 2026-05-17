#!/usr/bin/env python3
"""
Cross-encoder reranker for NDPA Memory tier.

Stage 1: BM25 retrieval returns top-50 candidate conversations (cheap, fast).
Stage 2: Cross-encoder reranks top-50 → top-5 (accurate, run on candidates only).

This is what separates NDPA Memory from NDPA core:
- NDPA core: BM25 only, 110ms, voice AI
- NDPA Memory: BM25 + rerank, ~400ms, QA/chat accuracy

Usage (eval):
    python3 -m eval.reranker --eval longmemeval --skip-ingest

Production integration:
    from eval.reranker import rerank
    candidates = ndpa.get_predictions(session_id, k=50)
    top5 = rerank(query, candidates, k=5)
"""

from __future__ import annotations

import sys
from typing import Any


CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CANDIDATE_K = 50   # BM25 retrieves this many
RERANK_K = 5       # cross-encoder selects this many


def load_cross_encoder():
    """Load the cross-encoder model. ~80MB download on first run."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        print("ERROR: pip install sentence-transformers", file=sys.stderr)
        sys.exit(1)
    return CrossEncoder(CROSS_ENCODER_MODEL)


def rerank(query: str, candidates: list[dict], k: int = RERANK_K, model=None) -> list[dict]:
    """
    Rerank candidates using a cross-encoder.

    Args:
        query: The search query / trajectory text.
        candidates: List of prediction dicts with 'content' and 'session_id'.
        k: Number of results to return.
        model: CrossEncoder instance (loaded once, reused).

    Returns:
        Top-k candidates, reranked, with 'rerank_score' added.
    """
    if not candidates:
        return []
    if model is None:
        model = load_cross_encoder()

    pairs = [(query, (c.get("content") or "")[:2000]) for c in candidates]
    scores = model.predict(pairs)

    ranked = sorted(
        zip(scores, candidates),
        key=lambda x: x[0],
        reverse=True,
    )
    result = []
    for score, candidate in ranked[:k]:
        candidate = dict(candidate)
        candidate["rerank_score"] = float(score)
        result.append(candidate)
    return result


def eval_longmemeval_with_reranker(args) -> None:
    """
    Run LongMemEval retrieval with BM25 + cross-encoder reranker.
    Compares hit@k for BM25-only vs BM25+rerank.
    """
    import json
    from pathlib import Path
    from eval.external_benchmark_utils import (
        DATA_DIR, load_ndpa_client, call_with_retries,
    )
    from eval.longmemeval_runner import DATASETS, category_for

    print(f"Loading cross-encoder: {CROSS_ENCODER_MODEL}")
    model = load_cross_encoder()
    print("Model loaded.\n")

    dataset_id, filename, url = DATASETS[args.split]
    path = DATA_DIR / "longmemeval" / filename
    if not path.exists():
        from eval.external_benchmark_utils import download_json
        items = download_json(url, path)
    else:
        items = json.loads(path.read_text())

    if args.max_items:
        items = items[: args.max_items]

    ndpa = load_ndpa_client("longmemeval", timeout=args.timeout)

    hits_bm25 = {1: 0, 3: 0, 5: 0}
    hits_rerank = {1: 0, 3: 0, 5: 0}
    by_category_bm25: dict = {}
    by_category_rerank: dict = {}
    n = 0

    for item in items:
        question_id = item.get("question_id") or item.get("id") or str(n)
        question = item.get("question") or ""
        evidence_ids = set(item.get("answer_session_ids") or item.get("evidence_session_ids") or [])
        cat = category_for(item)

        if not question or not evidence_ids:
            continue

        result = call_with_retries(
            lambda: ndpa.get_predictions(
                session_id=f"longmemeval_query_{question_id}",
                query=question,
                k=CANDIDATE_K,
                end_user_id=f"longmemeval_{question_id}",
            ),
            attempts=3,
        )
        candidates = result.get("predictions") or []
        pred_ids_bm25 = [c.get("session_id", "") for c in candidates]

        # Rerank
        reranked = rerank(question, candidates, k=RERANK_K, model=model)
        pred_ids_rerank = [c.get("session_id", "") for c in reranked]

        for k in (1, 3, 5):
            if evidence_ids & set(pred_ids_bm25[:k]):
                hits_bm25[k] += 1
            if evidence_ids & set(pred_ids_rerank[:min(k, RERANK_K)]):
                hits_rerank[k] += 1

        by_category_bm25.setdefault(cat, {1: 0, 3: 0, 5: 0, "n": 0})
        by_category_rerank.setdefault(cat, {1: 0, 3: 0, 5: 0, "n": 0})
        by_category_bm25[cat]["n"] += 1
        by_category_rerank[cat]["n"] += 1
        for k in (1, 3, 5):
            if evidence_ids & set(pred_ids_bm25[:k]):
                by_category_bm25[cat][k] += 1
            if evidence_ids & set(pred_ids_rerank[:min(k, RERANK_K)]):
                by_category_rerank[cat][k] += 1

        n += 1
        if n % 25 == 0:
            print(f"  {n}/{len(items)}  bm25@5={hits_bm25[5]/n:.3f}  rerank@5={hits_rerank[5]/n:.3f}", flush=True)

    print()
    print(f"{'Category':<32} {'BM25@5':>7} {'Rerank@5':>9} {'lift':>6}  n")
    print("-" * 65)
    for cat in sorted(by_category_bm25):
        cn = by_category_bm25[cat]["n"]
        b5 = by_category_bm25[cat][5] / cn
        r5 = by_category_rerank[cat][5] / cn
        print(f"{cat:<32} {b5:>6.1%} {r5:>8.1%} {(r5-b5)*100:>+5.1f}  {cn}")
    print("-" * 65)
    if n == 0:
        print("No items scored — check that LongMemEval data is ingested (run without --skip-ingest first)")
        return
    b5 = hits_bm25[5] / n
    r5 = hits_rerank[5] / n
    print(f"{'OVERALL':<32} {b5:>6.1%} {r5:>8.1%} {(r5-b5)*100:>+5.1f}  {n}")
    print()
    print(f"BM25  hit@1={hits_bm25[1]/n:.3f}  hit@3={hits_bm25[3]/n:.3f}  hit@5={b5:.3f}")
    print(f"Rerank hit@1={hits_rerank[1]/n:.3f}  hit@3={hits_rerank[3]/n:.3f}  hit@5={r5:.3f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", choices=["longmemeval"], default="longmemeval")
    parser.add_argument("--split", choices=["s", "m", "o"], default="s")
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    if args.eval == "longmemeval":
        eval_longmemeval_with_reranker(args)


if __name__ == "__main__":
    main()
