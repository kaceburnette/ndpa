#!/usr/bin/env python3
"""
Metadata-only retrieval quality eval for NDPA Memory.

This runner is deliberately local-only:
- no Supabase writes
- no benchmark import
- no LLM judge
- no server/SDK dependency

It scores explicit-query retrieval using only compact per-session metadata:
top_terms, content_preview, timestamps, storage_key, and session_id.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from eval.external_benchmark_utils import DATA_DIR, aggregate_hits, download_json, score_predictions, write_json
from eval.locomo_runner import LOCOMO_URL, category_for as locomo_category_for, evidence_session_ids, iter_sessions
from eval.longmemeval_runner import DATASETS, category_for as longmemeval_category_for

RESULTS_PATH = Path(__file__).with_name("metadata_retrieval_results.json")
TOP_KS = (1, 3, 5)
WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_'-]{2,}")
DATE_RE = re.compile(r"\b(?:19|20)\d{2}(?:[/.-]\d{1,2})?(?:[/.-]\d{1,2})?\b")
STOP = set(
    "the a an and or but if then else for of to in on with at by from as is was are "
    "be been being have has had do does did this that these those it its they them "
    "their there here what when where who whom why how which would could should can "
    "may might about into over under between after before during while again very"
    .split()
)


@dataclass
class MetadataDoc:
    session_id: str
    storage_key: str
    started_at: float
    content_preview: str
    top_terms: Counter
    preview_terms: Counter
    date_terms: set[str]

    @property
    def score_terms(self) -> Counter:
        terms = Counter()
        terms.update({term: count * 3 for term, count in self.top_terms.items()})
        terms.update(self.preview_terms)
        for term in self.date_terms:
            terms[term] += 2
        return terms


def tokenize(text: str) -> Counter:
    words = (w.lower().strip("'") for w in WORD_RE.findall(text or ""))
    return Counter(w for w in words if w and w not in STOP)


def text_from_turns(turns: Iterable[dict]) -> str:
    parts = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role") or turn.get("speaker") or "user")
        content = str(turn.get("content") or turn.get("text") or "").strip()
        if content:
            parts.append(f"[{role}] {content}")
    return "\n".join(parts)


def date_terms(text: str) -> set[str]:
    return {m.group(0).replace("/", "-").replace(".", "-") for m in DATE_RE.finditer(text or "")}


def make_doc(
    session_id: str,
    content: str,
    *,
    started_at: float,
    date_text: str = "",
    top_terms: int,
    preview_chars: int,
) -> MetadataDoc:
    preview = content[:preview_chars]
    full_terms = tokenize(content)
    return MetadataDoc(
        session_id=session_id,
        storage_key=session_id,
        started_at=started_at,
        content_preview=preview,
        top_terms=Counter(dict(full_terms.most_common(top_terms))),
        preview_terms=tokenize(preview),
        date_terms=date_terms(date_text),
    )


def bm25_metadata_scores(query_terms: Counter, docs: list[MetadataDoc]) -> list[tuple[str, float]]:
    if not docs or not query_terms:
        return []

    doc_terms = [(doc, doc.score_terms) for doc in docs]
    lengths = {doc.session_id: sum(terms.values()) for doc, terms in doc_terms}
    avg_len = sum(lengths.values()) / max(1, len(lengths))
    df: Counter = Counter()
    for _doc, terms in doc_terms:
        for term in set(terms):
            df[term] += 1

    n_docs = len(docs)
    scored = []
    for doc, terms in doc_terms:
        score = 0.0
        doc_len = max(1, lengths[doc.session_id])
        for term, qf in query_terms.items():
            tf = terms.get(term, 0)
            if not tf:
                continue
            idf = math.log((n_docs - df[term] + 0.5) / (df[term] + 0.5) + 1)
            tf_norm = tf * 2.5 / (tf + 1.5 * (1 - 0.75 + 0.75 * doc_len / max(avg_len, 1)))
            score += idf * tf_norm * (1 + math.log1p(qf))
        scored.append((doc.session_id, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


def retrieve(query: str, docs: list[MetadataDoc], k: int) -> list[str]:
    query_terms = tokenize(query)
    for term in date_terms(query):
        query_terms[term] += 2
    scored = bm25_metadata_scores(query_terms, docs)
    return [sid for sid, score in scored[:k] if score > 0.0]


def run_longmemeval(args: argparse.Namespace) -> dict:
    dataset_id, filename, url = DATASETS[args.longmemeval_split]
    path = DATA_DIR / "longmemeval" / filename
    items = download_json(url, path, force=args.force_download)
    if args.max_longmemeval_items:
        items = items[: args.max_longmemeval_items]

    rows = []
    for item in items:
        question_id = str(item["question_id"])
        haystack_ids = [str(sid) for sid in item.get("haystack_session_ids") or []]
        haystack_dates = item.get("haystack_dates") or []
        docs = []
        for index, session in enumerate(item.get("haystack_sessions") or []):
            session_id = haystack_ids[index] if index < len(haystack_ids) else f"{question_id}_{index}"
            date_text = haystack_dates[index] if index < len(haystack_dates) else ""
            docs.append(
                make_doc(
                    session_id,
                    text_from_turns(session),
                    started_at=float(index),
                    date_text=date_text,
                    top_terms=args.top_terms,
                    preview_chars=args.preview_chars,
                )
            )

        predicted = retrieve(str(item.get("question") or ""), docs, 5)
        truth = {str(sid) for sid in item.get("answer_session_ids") or []}
        scored = bool(truth)
        rows.append({
            "question_id": question_id,
            "category": longmemeval_category_for(item),
            "question_type": item.get("question_type"),
            "answer_session_ids": sorted(truth),
            "predicted_session_ids": predicted,
            "hits": score_predictions(predicted, truth) if scored else {},
            "scored": scored,
        })

    return {
        "benchmark": "LongMemEval",
        "dataset": dataset_id,
        "split": args.longmemeval_split,
        "dataset_file": str(path),
        "n_items": len(items),
        "n_scored": sum(1 for row in rows if row["scored"]),
        "metrics": aggregate_hits(rows),
        "rows": rows,
    }


def run_locomo(args: argparse.Namespace) -> dict:
    path = DATA_DIR / "locomo" / "locomo10.json"
    samples = download_json(LOCOMO_URL, path, force=args.force_download)

    rows = []
    question_count = 0
    for sample in samples:
        sample_id = str(sample.get("sample_id"))
        docs = []
        for session_number, session, date_text in iter_sessions(sample):
            session_id = f"{sample_id}_session_{session_number}"
            docs.append(
                make_doc(
                    session_id,
                    text_from_turns(session),
                    started_at=float(session_number),
                    date_text=date_text or "",
                    top_terms=args.top_terms,
                    preview_chars=args.preview_chars,
                )
            )

        for qa_index, qa in enumerate(sample.get("qa") or []):
            if args.max_locomo_questions and question_count >= args.max_locomo_questions:
                break
            question_count += 1
            predicted = retrieve(str(qa.get("question") or ""), docs, 5)
            truth = evidence_session_ids(sample_id, qa)
            scored = bool(truth)
            rows.append({
                "sample_id": sample_id,
                "question_index": qa_index,
                "category": locomo_category_for(qa),
                "raw_category": qa.get("category"),
                "answer_session_ids": sorted(truth),
                "predicted_session_ids": predicted,
                "hits": score_predictions(predicted, truth) if scored else {},
                "scored": scored,
            })
        if args.max_locomo_questions and question_count >= args.max_locomo_questions:
            break

    return {
        "benchmark": "LoCoMo",
        "dataset": "snap-research/locomo data/locomo10.json",
        "dataset_file": str(path),
        "dataset_conversations": len(samples),
        "n_questions": question_count,
        "n_scored": sum(1 for row in rows if row["scored"]),
        "metrics": aggregate_hits(rows),
        "rows": rows,
    }


def load_old_metrics(path: str) -> dict:
    try:
        data = json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {}
    return data.get("metrics", {}).get("overall", {})


def delta(new_metrics: dict, old_metrics: dict) -> dict:
    return {
        key: round(new_metrics.get(key, 0.0) - old_metrics.get(key, 0.0), 4)
        for key in ("hit@1", "hit@3", "hit@5")
        if key in new_metrics and key in old_metrics
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("longmemeval", "locomo", "both"), default="both")
    parser.add_argument("--longmemeval-split", choices=sorted(DATASETS), default="s")
    parser.add_argument("--max-longmemeval-items", type=int, default=0)
    parser.add_argument("--max-locomo-questions", type=int, default=0)
    parser.add_argument("--top-terms", type=int, default=300)
    parser.add_argument("--preview-chars", type=int, default=600)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()

    started = time.time()
    results = {
        "architecture": "metadata-only",
        "methodology": {
            "uses_supabase": False,
            "uses_llm_judge": False,
            "imports_benchmarks": False,
            "allowed_fields": ["top_terms", "content_preview", "timestamps", "storage_key", "session_id"],
            "top_terms": args.top_terms,
            "preview_chars": args.preview_chars,
        },
        "benchmarks": {},
        "old_reference": {
            "longmemeval_results_json": load_old_metrics("eval/longmemeval_results.json"),
            "locomo_results_json": load_old_metrics("eval/locomo_results.json"),
            "published_claims": {
                "longmemeval_hit@5": 0.842,
                "locomo_hit@5": 0.825,
            },
            "assessment": "Existing runners use live NDPA API ingestion of full benchmark session content, so they are not clean metadata-only measurements.",
        },
    }

    if args.benchmark in {"longmemeval", "both"}:
        results["benchmarks"]["longmemeval"] = run_longmemeval(args)
        results["benchmarks"]["longmemeval"]["delta_vs_old_results_json"] = delta(
            results["benchmarks"]["longmemeval"]["metrics"]["overall"],
            results["old_reference"]["longmemeval_results_json"],
        )
    if args.benchmark in {"locomo", "both"}:
        results["benchmarks"]["locomo"] = run_locomo(args)
        results["benchmarks"]["locomo"]["delta_vs_old_results_json"] = delta(
            results["benchmarks"]["locomo"]["metrics"]["overall"],
            results["old_reference"]["locomo_results_json"],
        )

    results["elapsed_sec"] = round(time.time() - started, 2)
    write_json(args.output, results)
    print(f"Wrote {args.output}")
    for name, payload in results["benchmarks"].items():
        print(name, payload["metrics"]["overall"])


if __name__ == "__main__":
    main()
