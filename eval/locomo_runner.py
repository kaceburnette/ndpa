#!/usr/bin/env python3
"""
Run NDPA retrieval on LoCoMo QA evidence sessions.

Usage:
    python3 -m eval.locomo_runner
    python3 -m eval.locomo_runner --max-questions 25

The public Snap/USC repository currently ships locomo10.json. The first arXiv
release mentioned 50 conversations, but the maintained public benchmark file
contains 10 conversations with QA annotations.
"""

from __future__ import annotations

import argparse
import math
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from eval.external_benchmark_utils import (
    DATA_DIR,
    RateLimiter,
    aggregate_hits,
    call_with_retries,
    download_json,
    load_ndpa_client,
    prediction_session_ids,
    score_predictions,
    session_to_events,
    write_json,
)

RESULTS_PATH = Path(__file__).with_name("locomo_results.json")
LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
EVIDENCE_RE = re.compile(r"^D(\d+):")
WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+-]{1,}")
ENTITY_RE = re.compile(r"\b(?:[A-Z][a-zA-Z0-9_+-]{2,}|[A-Z]{2,})\b")
DATE_RE = re.compile(
    r"\b(?:\d{4}|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
    r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?|monday|tuesday|wednesday|thursday|"
    r"friday|saturday|sunday|yesterday|today|tomorrow|last year|"
    r"last month|last week|recently|before|after)\b",
    re.I,
)
TEMPORAL_RE = re.compile(
    r"\b(?:when|date|time|before|after|last|latest|first|recent|recently|"
    r"currently|current|year|month|week|day|ago|older|newer|previous|next)\b",
    re.I,
)
MULTI_HOP_RE = re.compile(
    r"\b(?:both|together|relationship|likely|what fields|common|same|shared|"
    r"compare|between|and .* and)\b",
    re.I,
)
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "for", "of", "to",
    "in", "on", "with", "at", "by", "from", "as", "is", "was", "are", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "this",
    "that", "these", "those", "it", "its", "they", "them", "their", "what",
    "which", "who", "whom", "whose", "can", "could", "would", "should",
    "about", "tell", "me", "my", "you", "your", "i", "we", "our", "when",
    "where", "why", "how", "there", "here", "caroline", "melanie", "sam",
    "charlie", "sarah", "james", "maya", "john", "emma",
}

CATEGORY_LABELS = {
    1: "single-hop",
    2: "temporal",
    3: "multi-hop",
    4: "open-domain",
    5: "adversarial",
}


def iter_sessions(sample: dict) -> list[tuple[int, list[dict], str | None]]:
    conv = sample.get("conversation") or {}
    sessions = []
    for key, value in conv.items():
        if not key.startswith("session_") or key.endswith("_date_time") or not isinstance(value, list):
            continue
        try:
            number = int(key.split("_", 1)[1])
        except ValueError:
            continue
        sessions.append((number, value, conv.get(f"session_{number}_date_time")))
    sessions.sort(key=lambda item: item[0])
    return sessions


def evidence_session_ids(sample_id: str, qa: dict) -> set[str]:
    ids = set()
    for evidence in qa.get("evidence") or []:
        match = EVIDENCE_RE.match(str(evidence))
        if match:
            ids.add(f"{sample_id}_session_{int(match.group(1))}")
    return ids


def category_for(qa: dict) -> str:
    raw = qa.get("category")
    try:
        return CATEGORY_LABELS.get(int(raw), f"category_{raw}")
    except Exception:
        return f"category_{raw or 'unknown'}"


def normalize_word(word: str) -> str:
    word = word.lower()
    if word.endswith("'s"):
        word = word[:-2]
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 3 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s"):
        return word[:-1]
    return word


def content_words(text: str) -> list[str]:
    words = []
    for raw in WORD_RE.findall(text):
        word = normalize_word(raw)
        if word not in STOPWORDS:
            words.append(word)
    return words


def query_features(question: str) -> dict:
    terms = set(content_words(question))
    entities = {
        m.group(0).lower()
        for m in ENTITY_RE.finditer(question)
        if m.group(0).lower() not in STOPWORDS
    }
    dates = {m.group(0).lower() for m in DATE_RE.finditer(question)}
    return {
        "terms": terms,
        "entities": entities,
        "dates": dates,
        "is_temporal": bool(TEMPORAL_RE.search(question)),
        "is_multi_hop": bool(MULTI_HOP_RE.search(question)) or len(entities) >= 2,
    }


def flatten_event_summary(value) -> str:
    if not isinstance(value, dict):
        return ""
    parts = []
    for key, item in value.items():
        if isinstance(item, list):
            parts.extend(str(x) for x in item)
        elif item:
            parts.append(f"{key}: {item}")
    return " ".join(parts)


def session_text(session: list[dict]) -> str:
    parts = []
    for turn in session:
        speaker = str(turn.get("speaker") or turn.get("role") or "")
        text = str(turn.get("text") or turn.get("content") or "")
        caption = str(turn.get("blip_caption") or "")
        query = str(turn.get("query") or "")
        parts.append(" ".join(x for x in (speaker, text, caption, query) if x))
    return "\n".join(parts)


def parse_locomo_date(date_text: str | None, fallback: int) -> float:
    if not date_text:
        return float(fallback)
    match = re.search(r"(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})", date_text)
    if not match:
        return float(fallback)
    try:
        dt = datetime.strptime(" ".join(match.groups()), "%d %B %Y").replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return float(fallback)


def build_local_metadata(sample: dict) -> list[dict]:
    sample_id = str(sample.get("sample_id"))
    summaries = sample.get("session_summary") or {}
    events = sample.get("event_summary") or {}
    rows = []
    for session_number, session, date_text in iter_sessions(sample):
        raw_text = session_text(session)
        summary = str(summaries.get(f"session_{session_number}_summary") or "")
        event_text = flatten_event_summary(events.get(f"events_session_{session_number}"))
        metadata_text = "\n".join(x for x in (date_text or "", summary, event_text, raw_text[:1200]) if x)
        term_counts = Counter(content_words(metadata_text))
        top_terms = [term for term, _ in term_counts.most_common(80)]
        rows.append({
            "session_id": f"{sample_id}_session_{session_number}",
            "session_number": session_number,
            "content_preview": raw_text[:1600],
            "summary": summary,
            "event_summary": event_text,
            "date_text": date_text or "",
            "updated_epoch": parse_locomo_date(date_text, session_number),
            "top_terms": top_terms,
            "term_counts": term_counts,
            "metadata_text": metadata_text,
        })
    return rows


def build_idf(candidates: list[dict]) -> Counter:
    n_docs = len(candidates)
    df = Counter()
    for item in candidates:
        for term in set(item["term_counts"]):
            df[term] += 1
    return Counter({term: math.log((n_docs + 1) / (count + 0.5)) for term, count in df.items()})


def overlap_score(query_terms: set[str], candidate_terms: set[str]) -> float:
    if not query_terms or not candidate_terms:
        return 0.0
    return len(query_terms & candidate_terms) / math.sqrt(len(query_terms) * len(candidate_terms))


def bm25_like(features: dict, candidate: dict, idf: Counter, avg_len: float) -> float:
    score = 0.0
    doc_len = sum(candidate["term_counts"].values())
    for term in features["terms"]:
        tf = candidate["term_counts"].get(term, 0)
        if not tf:
            continue
        tf_norm = tf * 2.5 / (tf + 1.5 * (1 - 0.75 + 0.75 * doc_len / max(avg_len, 1)))
        score += idf.get(term, 0.0) * tf_norm
    return score


def entity_score(features: dict, candidate: dict) -> float:
    entities = features["entities"]
    if not entities:
        return 0.0
    haystack = candidate["metadata_text"].lower()
    return sum(1 for entity in entities if entity in haystack) / len(entities)


def date_score(features: dict, candidate: dict) -> float:
    if not features["is_temporal"]:
        return 0.0
    haystack = " ".join((candidate["date_text"], candidate["summary"], candidate["event_summary"])).lower()
    explicit_hits = sum(1 for date in features["dates"] if date in haystack)
    if features["dates"]:
        return min(1.0, explicit_hits / len(features["dates"]) + (0.15 if DATE_RE.search(haystack) else 0.0))
    return 0.35 if candidate["date_text"] else 0.0


def recency_score(candidate: dict, newest_epoch: float) -> float:
    if newest_epoch <= 0:
        return 0.0
    # LoCoMo sessions are chronological. Use a gentle decay so recency can
    # break ties without drowning evidence matches.
    age_days = max(0.0, (newest_epoch - candidate["updated_epoch"]) / 86400.0)
    return 0.5 ** (age_days / 120.0)


def lexical_rerank_score(question: str, candidate: dict) -> float:
    q_terms = content_words(question)
    text = " ".join((candidate["summary"], candidate["event_summary"], candidate["content_preview"])).lower()
    if not q_terms or not text:
        return 0.0
    adjacent = 0
    for left, right in zip(q_terms, q_terms[1:]):
        if f"{left} {right}" in text:
            adjacent += 1
    phrase_score = adjacent / max(1, len(q_terms) - 1)
    term_score = sum(1 for term in set(q_terms) if term in text) / len(set(q_terms))
    return 0.55 * term_score + 0.45 * phrase_score


def local_rank(
    question: str,
    candidates: list[dict],
    *,
    boosted: bool,
    hybrid_rerank: bool,
    k: int = 5,
) -> list[dict]:
    features = query_features(question)
    idf = build_idf(candidates)
    avg_len = sum(sum(c["term_counts"].values()) for c in candidates) / max(1, len(candidates))
    newest_epoch = max((c["updated_epoch"] for c in candidates), default=0.0)
    raw_scores = []
    for candidate in candidates:
        candidate_terms = set(candidate["top_terms"])
        bm25 = bm25_like(features, candidate, idf, avg_len)
        overlap = overlap_score(features["terms"], candidate_terms)
        entity = entity_score(features, candidate)
        date = date_score(features, candidate)
        recency = recency_score(candidate, newest_epoch)
        score = 0.82 * bm25 + 0.18 * overlap
        item = dict(candidate)
        item["_score"] = score
        item["_boost_score"] = boost_score(score, entity, date, recency, overlap, features)
        item["_terms"] = candidate_terms
        raw_scores.append(item)

    raw_scores.sort(key=lambda item: item["_score"], reverse=True)
    selected = mmr_select(raw_scores, k)
    if boosted:
        for item in selected:
            item["_lexical_score"] = item["_score"]
            item["_score"] = item["_boost_score"]
            if hybrid_rerank:
                item["_score"] = 0.9 * item["_score"] + 0.1 * lexical_rerank_score(question, item)
        cautious_boost_reorder(selected)
    return selected


def cautious_boost_reorder(selected: list[dict], threshold: float = 0.01) -> None:
    """
    Preserve the MMR top-k set and order unless a metadata boost is meaningful.

    LoCoMo has many near-duplicate sessions; blind score sorting hurts rank@3.
    This bounded adjacent swap lets entity/date boosts break close ties while
    keeping lexical/MMR order as the default.
    """
    changed = True
    while changed:
        changed = False
        for idx in range(len(selected) - 1):
            if float(selected[idx + 1]["_score"]) - float(selected[idx]["_score"]) > threshold:
                selected[idx], selected[idx + 1] = selected[idx + 1], selected[idx]
                changed = True


def boost_score(
    lexical: float,
    entity: float,
    date: float,
    recency: float,
    overlap: float,
    features: dict,
) -> float:
    if features["is_temporal"]:
        return lexical + 0.08 * entity + 0.06 * date + 0.01 * recency
    if features["is_multi_hop"]:
        return lexical + 0.10 * entity + 0.02 * overlap
    return lexical + 0.04 * entity


def mmr_select(ranked: list[dict], k: int) -> list[dict]:
    selected = []
    remaining = list(ranked)
    while remaining and len(selected) < k:
        best_idx = 0
        best_score = float("-inf")
        for idx, item in enumerate(remaining):
            redundancy = 0.0
            if selected:
                redundancy = max(jaccard(item["_terms"], prev["_terms"]) for prev in selected)
            score = 0.9 * float(item["_score"]) - 0.1 * redundancy
            if score > best_score:
                best_idx = idx
                best_score = score
        selected.append(remaining.pop(best_idx))
    return selected


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def run_local(args: argparse.Namespace, samples: list[dict], path: Path) -> dict:
    rows = []
    baseline_rows = []
    started = time.time()
    question_count = 0

    for sample in samples:
        sample_id = str(sample.get("sample_id"))
        candidates = build_local_metadata(sample)
        for qa_index, qa in enumerate(sample.get("qa") or []):
            if args.max_questions and question_count >= args.max_questions:
                break
            question_count += 1
            question = str(qa.get("question") or "")
            truth = evidence_session_ids(sample_id, qa)
            scored = bool(truth)

            baseline = local_rank(question, candidates, boosted=False, hybrid_rerank=False, k=5)
            predicted_baseline = [item["session_id"] for item in baseline]
            baseline_rows.append({
                "category": category_for(qa),
                "hits": score_predictions(predicted_baseline, truth) if scored else {},
                "scored": scored,
            })

            ranked = local_rank(
                question,
                candidates,
                boosted=True,
                hybrid_rerank=args.hybrid_rerank,
                k=5,
            )
            predicted = [item["session_id"] for item in ranked]
            rows.append({
                "sample_id": sample_id,
                "question_index": qa_index,
                "category": category_for(qa),
                "raw_category": qa.get("category"),
                "question": qa.get("question"),
                "answer": qa.get("answer"),
                "evidence": qa.get("evidence") or [],
                "answer_session_ids": sorted(truth),
                "predicted_session_ids": predicted,
                "baseline_predicted_session_ids": predicted_baseline,
                "hits": score_predictions(predicted, truth) if scored else {},
                "baseline_hits": score_predictions(predicted_baseline, truth) if scored else {},
                "scored": scored,
                "raw_prediction_count": len(predicted),
            })

            if question_count % args.progress_every == 0:
                limit = args.max_questions or sum(len(sample.get("qa") or []) for sample in samples)
                print(f"  {question_count}/{limit} questions processed", flush=True)
        if args.max_questions and question_count >= args.max_questions:
            break

    metrics = aggregate_hits(rows)
    baseline_metrics = aggregate_hits(baseline_rows)
    payload = {
        "benchmark": "LoCoMo",
        "dataset": "snap-research/locomo data/locomo10.json",
        "dataset_file": str(path),
        "dataset_conversations": len(samples),
        "requested_conversations_note": "Original arXiv release mentioned 50 conversations; maintained public repo currently ships 10.",
        "retrieval_mode": "local_metadata_only",
        "metadata_fields": ["session_date", "session_summary", "event_summary", "content_preview", "top_terms"],
        "improvements": ["temporal boost", "multi-hop/entity boost", "date boost", "MMR diversity", "optional lexical hybrid rerank"],
        "hybrid_rerank": args.hybrid_rerank,
        "n_questions": question_count,
        "n_scored": sum(1 for row in rows if row["scored"]),
        "elapsed_sec": round(time.time() - started, 2),
        "baseline_metrics": baseline_metrics,
        "metrics": metrics,
        "rows": rows,
    }
    write_json(RESULTS_PATH, payload)
    return payload


def run_api(args: argparse.Namespace, samples: list[dict], path: Path) -> dict:
    path = DATA_DIR / "locomo" / "locomo10.json"
    client = load_ndpa_client("locomo", timeout=args.timeout)
    limiter = RateLimiter(args.requests_per_minute)

    rows = []
    started = time.time()
    total_questions = sum(len(sample.get("qa") or []) for sample in samples)
    question_count = 0

    for sample in samples:
        sample_id = str(sample.get("sample_id"))
        sessions = iter_sessions(sample)
        end_user_id = f"locomo_{sample_id}"

        def ingest_session(session_info: tuple[int, list[dict], str | None]) -> None:
            session_number, session, date_text = session_info
            session_id = f"{sample_id}_session_{session_number}"
            fallback_ts = 1_680_000_000.0 + session_number
            events = session_to_events(session, fallback_ts=fallback_ts)
            if not events:
                return
            if date_text:
                events[0]["source_path"] = f"locomo:{date_text}"
            limiter.wait()
            call_with_retries(
                lambda: client.log_events(session_id, events, end_user_id=end_user_id),
                attempts=args.retries,
            )

        with ThreadPoolExecutor(max_workers=args.ingest_workers) as pool:
            futures = [pool.submit(ingest_session, session_info) for session_info in sessions]
            for future in as_completed(futures):
                future.result()

        for qa_index, qa in enumerate(sample.get("qa") or []):
            if args.max_questions and question_count >= args.max_questions:
                break
            question_count += 1

            result = call_with_retries(
                lambda: client.get_predictions(
                    session_id=f"locomo_query_{sample_id}_{qa_index}",
                    query=str(qa.get("question") or ""),
                    k=5,
                    end_user_id=end_user_id,
                ),
                attempts=args.retries,
            )
            predicted = prediction_session_ids(result)
            truth = evidence_session_ids(sample_id, qa)
            scored = bool(truth)
            rows.append({
                "sample_id": sample_id,
                "question_index": qa_index,
                "category": category_for(qa),
                "raw_category": qa.get("category"),
                "question": qa.get("question"),
                "answer": qa.get("answer"),
                "evidence": qa.get("evidence") or [],
                "answer_session_ids": sorted(truth),
                "predicted_session_ids": predicted,
                "hits": score_predictions(predicted, truth) if scored else {},
                "scored": scored,
                "raw_prediction_count": len(result.get("predictions") or []),
            })

            if question_count % args.progress_every == 0:
                limit = args.max_questions or total_questions
                print(f"  {question_count}/{limit} questions processed", flush=True)
        if args.max_questions and question_count >= args.max_questions:
            break

    payload = {
        "benchmark": "LoCoMo",
        "dataset": "snap-research/locomo data/locomo10.json",
        "dataset_file": str(path),
        "dataset_conversations": len(samples),
        "requested_conversations_note": "Original arXiv release mentioned 50 conversations; maintained public repo currently ships 10.",
        "n_questions": question_count,
        "n_scored": sum(1 for row in rows if row["scored"]),
        "elapsed_sec": round(time.time() - started, 2),
        "metrics": aggregate_hits(rows),
        "rows": rows,
    }
    write_json(RESULTS_PATH, payload)
    return payload


def run(args: argparse.Namespace) -> dict:
    path = DATA_DIR / "locomo" / "locomo10.json"
    samples = download_json(LOCOMO_URL, path, force=args.force_download)

    if args.mode == "api":
        return run_api(args, samples, path)
    return run_local(args, samples, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["local", "api"], default="local")
    parser.add_argument("--max-questions", type=int, default=0, help="Optional smoke-test limit")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--hybrid-rerank", action="store_true")
    parser.add_argument("--requests-per-minute", type=int, default=540)
    parser.add_argument("--ingest-workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    results = run(args)
    print(f"Wrote {RESULTS_PATH}")
    print(results["metrics"]["overall"])


if __name__ == "__main__":
    main()
