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
import importlib.util
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Callable, List

from collections import defaultdict

from eval.conversation_eval import (
    Conversation, load_conversations, tokenize, STOP, WORD_RE,
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
TRAJECTORY_WINDOW_DEFAULT = 7  # use last N conversations as trajectory signal
ADAPTIVE_TRAJECTORY_WINDOWS = (3, 5, 7, 10, 15)
MIN_FUTURE_CHARS = 200         # skip trivial conversations
MIN_PAST_CONVS = 10            # need enough history to have meaningful "past"
PREDICTOR_TOPIC_TERMS = 250    # predictor-side cap; BM25 labels still use full text
RESULTS_PATH = Path(__file__).with_name("pre_query_hit_rate_results.json")
ACCURACY_MODE_DEFAULT = "no-embedding"
SEMANTIC_MODEL_DEFAULT = "sentence-transformers/all-MiniLM-L6-v2"

REF_RE = re.compile(
    r"(?:(?:[\w.-]+/)+[\w.-]+\.\w+|[\w.-]+\.(?:py|ts|tsx|js|jsx|md|json|sql|yaml|yml|toml|go|rs|java|cpp|h|css|html)|#[A-Za-z0-9_-]+|`[^`]{2,80}`)"
)
_REF_CACHE: dict[str, Counter] = {}
_KEYPHRASE_CACHE: dict[str, Counter] = {}
_BOW_NORM_CACHE: dict[str, float] = {}


def compute_idf(convs: List[Conversation]) -> Counter:
    """Corpus IDF from all conversations. Applied to both trajectory and past.bow."""
    N = len(convs)
    df: Counter = Counter()
    for c in convs:
        for term in set(c.bow.keys()):
            df[term] += 1
    return Counter({t: math.log((N + 1) / (cnt + 1)) for t, cnt in df.items()})


def tfidf_bow(bow: Counter, idf: Counter) -> Counter:
    """Weight raw term counts by IDF."""
    return Counter({t: cnt * idf.get(t, 1.0) for t, cnt in bow.items()})


def top_terms(bow: Counter, limit: int = PREDICTOR_TOPIC_TERMS) -> Counter:
    return Counter(dict(bow.most_common(limit)))


def vector_norm(vec: Counter) -> float:
    return sum(v * v for v in vec.values()) ** 0.5


def cosine_with_norm(a: Counter, b: Counter, a_norm: float, b_norm: float) -> float:
    if not a or not b or not a_norm or not b_norm:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(value * b.get(term, 0) for term, value in a.items())
    return dot / (a_norm * b_norm) if dot else 0.0


def conversation_norm(c: Conversation) -> float:
    if c.session_id not in _BOW_NORM_CACHE:
        _BOW_NORM_CACHE[c.session_id] = vector_norm(c.bow)
    return _BOW_NORM_CACHE[c.session_id]


def pure_topic_predict(trajectory_bow: Counter, past: List[Conversation], k: int) -> List[str]:
    """NDPA's actual predict-forward path: pure topic match, no recency bias."""
    if not past:
        return []
    trajectory_bow = top_terms(trajectory_bow)
    trajectory_norm = vector_norm(trajectory_bow)
    scored = [
        (c.session_id, cosine_with_norm(c.bow, trajectory_bow, conversation_norm(c), trajectory_norm))
        for c in past
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [sid for sid, _ in scored[:k]]


def tfidf_predict(idf: Counter):
    """TF-IDF weighted topic predictor. Rare terms matter more."""
    doc_cache: dict[str, Counter] = {}
    norm_cache: dict[str, float] = {}

    def _predict(trajectory_bow: Counter, past: List[Conversation], k: int) -> List[str]:
        if not past:
            return []
        traj_tfidf = tfidf_bow(trajectory_bow, idf)
        traj_norm = vector_norm(traj_tfidf)
        scored = []
        for c in past:
            if c.session_id not in doc_cache:
                doc_cache[c.session_id] = tfidf_bow(c.bow, idf)
                norm_cache[c.session_id] = vector_norm(doc_cache[c.session_id])
            doc = doc_cache[c.session_id]
            scored.append((c.session_id, cosine_with_norm(doc, traj_tfidf, norm_cache[c.session_id], traj_norm)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [sid for sid, _ in scored[:k]]
    return _predict


def heuristic_predict(trajectory_bow: Counter, past: List[Conversation], k: int) -> List[str]:
    """NDPA's heuristic: blend topic + recency. What the predictions API uses live."""
    if not past:
        return []
    most_recent = max(c.started_at for c in past)
    trajectory_bow = top_terms(trajectory_bow)
    trajectory_norm = vector_norm(trajectory_bow)
    scored = []
    for c in past:
        recency = 1.0 / (1 + max(0, (most_recent - c.started_at) / 86400))
        topic = cosine_with_norm(c.bow, trajectory_bow, conversation_norm(c), trajectory_norm)
        scored.append((c.session_id, 0.4 * recency + 0.6 * topic))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [sid for sid, _ in scored[:k]]


def extract_refs(text: str) -> Counter:
    """Explicit file/code/path refs in the trajectory; used only as predictor signal."""
    refs = Counter()
    for match in REF_RE.findall(text):
        ref = match.strip("`").lower()
        if ref:
            refs[ref] += 1
    return refs


def keyphrases(text: str, limit: int = 20) -> Counter:
    """Lightweight lexical keyphrases without embeddings or external models."""
    words = [w.lower() for w in WORD_RE.findall(text)]
    words = [w for w in words if w not in STOP]
    counts: Counter = Counter()
    counts.update(words)
    counts.update(f"{a} {b}" for a, b in zip(words, words[1:]) if a != b)
    return Counter(dict(counts.most_common(limit)))


def overlap_ratio(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    if not common:
        return 0.0
    return sum(min(a[t], b[t]) for t in common) / max(1, sum(a.values()))


def temporal_coherence(candidate: Conversation, trajectory: List[Conversation]) -> float:
    """
    Trajectory-only temporal signal.

    Scores whether a past conversation sits near the user's recent cadence. This
    deliberately does not look at the target conversation.
    """
    if not trajectory:
        return 0.0
    last_seen = max(c.started_at for c in trajectory)
    gaps = [
        b.started_at - a.started_at
        for a, b in zip(trajectory, trajectory[1:])
        if b.started_at > a.started_at
    ]
    cadence = sorted(gaps)[len(gaps) // 2] if gaps else 86400.0
    cadence = max(cadence, 3600.0)
    age = max(0.0, last_seen - candidate.started_at)
    return 1.0 / (1.0 + age / cadence)


def multi_signal_predict(trajectory_window: int) -> Callable[[Counter, List[Conversation], int], List[str]]:
    """
    Hybrid trajectory-only predictor.

    Signals:
    - topic similarity over full trajectory
    - topic frequency from repeated terms in the trajectory
    - explicit refs/file paths/code refs
    - lightweight entity/keyphrase overlap
    - temporal coherence against the recent trajectory cadence
    """
    trajectory_cache: dict[tuple[str, ...], tuple[Counter, Counter]] = {}

    def cached_refs(c: Conversation) -> Counter:
        if c.session_id not in _REF_CACHE:
            _REF_CACHE[c.session_id] = extract_refs(c.content)
        return _REF_CACHE[c.session_id]

    def cached_keyphrases(c: Conversation) -> Counter:
        if c.session_id not in _KEYPHRASE_CACHE:
            _KEYPHRASE_CACHE[c.session_id] = keyphrases(c.content)
        return _KEYPHRASE_CACHE[c.session_id]

    def _predict(trajectory_bow: Counter, past: List[Conversation], k: int) -> List[str]:
        if not past:
            return []
        trajectory_bow = top_terms(trajectory_bow)
        trajectory = past[-trajectory_window:]
        trajectory_key = tuple(c.session_id for c in trajectory)
        if trajectory_key not in trajectory_cache:
            refs = Counter()
            phrases = Counter()
            for c in trajectory:
                refs.update(cached_refs(c))
                phrases.update(cached_keyphrases(c))
            trajectory_cache[trajectory_key] = (refs, Counter(dict(phrases.most_common(30))))
        trajectory_refs, trajectory_keyphrases = trajectory_cache[trajectory_key]
        trajectory_freq = Counter(dict(trajectory_bow.most_common(40)))

        scored = []
        trajectory_norm = vector_norm(trajectory_bow)
        for c in past:
            topic = cosine_with_norm(c.bow, trajectory_bow, conversation_norm(c), trajectory_norm)
            frequency = overlap_ratio(trajectory_freq, c.bow)
            refs = overlap_ratio(trajectory_refs, cached_refs(c))
            entities = overlap_ratio(trajectory_keyphrases, cached_keyphrases(c))
            temporal = temporal_coherence(c, trajectory)
            score_value = (
                0.50 * topic
                + 0.18 * frequency
                + 0.14 * refs
                + 0.12 * entities
                + 0.06 * temporal
            )
            scored.append((c.session_id, score_value))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [sid for sid, _ in scored[:k]]

    return _predict


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


def bm25_score(query_bow: Counter, doc_bow: Counter, avg_doc_len: float, N: int, df: Counter) -> float:
    """BM25 score — independent from cosine BoW. Used for ground truth only."""
    K1, B = 1.5, 0.75
    doc_len = sum(doc_bow.values())
    score = 0.0
    for term in doc_bow:
        if term not in query_bow:
            continue
        tf = doc_bow[term]
        idf = math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1)
        tf_norm = tf * (K1 + 1) / (tf + K1 * (1 - B + B * doc_len / max(avg_doc_len, 1)))
        score += idf * tf_norm
    return score


def build_pqhr_samples(convs: List[Conversation], trajectory_window: int) -> list:
    """
    For each conversation, build a sample where the predictor sees ONLY prior conversations.

    Ground truth uses BM25 (independent from BoW cosine predictor).
    Predictor uses BoW cosine on trajectory.
    These are genuinely different methods on different inputs — no circularity.
    """
    import math
    samples = []
    sorted_convs = sorted(convs, key=lambda c: c.started_at)

    # Precompute corpus stats for BM25 ground truth
    N = len(sorted_convs)
    df: Counter = Counter()
    for c in sorted_convs:
        for term in set(c.bow.keys()):
            df[term] += 1
    avg_doc_len = sum(sum(c.bow.values()) for c in sorted_convs) / max(N, 1)

    postings: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    past: list[Conversation] = []

    for i, current in enumerate(sorted_convs):
        if i >= max(trajectory_window, MIN_PAST_CONVS) and len(current.content) >= MIN_FUTURE_CHARS:
            # Trajectory: last N conversations before C_i, NOT including C_i
            trajectory = sorted_convs[max(0, i - trajectory_window): i]
            trajectory_bow: Counter = Counter()
            for c in trajectory:
                trajectory_bow.update(c.bow)

            if trajectory_bow:
                # Ground truth: BM25 score of past convs against C_i's content.
                # Incremental postings contain only conversations before C_i.
                scores: Counter = Counter()
                for term in current.bow:
                    term_df = df.get(term, 0)
                    if term_df == 0:
                        continue
                    idf = math.log((N - term_df + 0.5) / (term_df + 0.5) + 1)
                    for sid, tf, doc_len in postings.get(term, []):
                        tf_norm = tf * 2.5 / (tf + 1.5 * (1 - 0.75 + 0.75 * doc_len / max(avg_doc_len, 1)))
                        scores[sid] += idf * tf_norm

                truth = {sid for sid, score_value in scores.most_common(K_GROUND_TRUTH) if score_value > 0.0}
                if truth:
                    samples.append((current, trajectory_bow, list(past), truth))

        doc_len = sum(current.bow.values())
        for term, tf in current.bow.items():
            postings[term].append((current.session_id, tf, doc_len))
        past.append(current)
    return samples


def ndcg(preds: List[str], truth: set[str], k: int) -> float:
    dcg = 0.0
    for rank, sid in enumerate(preds[:k], start=1):
        if sid in truth:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(truth), k)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal if ideal else 0.0


def score(predictor, samples) -> dict:
    hits = {1: 0, 3: 0, 5: 0}
    reciprocal_rank = 0.0
    ndcg_5 = 0.0
    n = len(samples)
    if n == 0:
        empty = {f"pqhr@{k}": 0.0 for k in (1, 3, 5)}
        empty.update({"mrr@5": 0.0, "ndcg@5": 0.0})
        return empty
    for _current, trajectory_bow, past, truth in samples:
        preds = predictor(trajectory_bow, past, TOP_K_PREDICT)
        for k in (1, 3, 5):
            if any(p in truth for p in preds[:k]):
                hits[k] += 1
        rr = 0.0
        for rank, sid in enumerate(preds[:5], start=1):
            if sid in truth:
                rr = 1.0 / rank
                break
        reciprocal_rank += rr
        ndcg_5 += ndcg(preds, truth, 5)
    result = {f"pqhr@{k}": v / n for k, v in hits.items()}
    result["mrr@5"] = reciprocal_rank / n
    result["ndcg@5"] = ndcg_5 / n
    return result


def evaluate_samples(samples: list, convs: list[Conversation], trajectory_window: int) -> dict:
    return {
        "random": score(baseline_random, samples),
        "recency": score(baseline_recency, samples),
        "heuristic": score(heuristic_predict, samples),
        "pure_topic": score(pure_topic_predict, samples),
        "multi_signal": score(multi_signal_predict(trajectory_window), samples),
    }


def semantic_embedding_predict(trajectory_window: int, model_name: str) -> Callable[[Counter, List[Conversation], int], List[str]]:
    """
    Optional semantic accuracy ablation.

    This is deliberately not part of the default no-embedding PQHR path. It
    requires sentence-transformers to be installed locally and must be selected
    with --accuracy-mode semantic-ablation.
    """
    if importlib.util.find_spec("sentence_transformers") is None:
        raise RuntimeError(
            "semantic-ablation requires optional package sentence-transformers; "
            "default PQHR remains no-embedding"
        )

    from sentence_transformers import SentenceTransformer  # type: ignore

    model = SentenceTransformer(model_name)
    doc_cache: dict[str, object] = {}
    trajectory_cache: dict[tuple[str, ...], object] = {}

    def encode(text: str):
        return model.encode(text, normalize_embeddings=True, show_progress_bar=False)

    def doc_embedding(c: Conversation):
        if c.session_id not in doc_cache:
            doc_cache[c.session_id] = encode(c.content)
        return doc_cache[c.session_id]

    def _predict(_trajectory_bow: Counter, past: List[Conversation], k: int) -> List[str]:
        if not past:
            return []
        trajectory = past[-trajectory_window:]
        trajectory_key = tuple(c.session_id for c in trajectory)
        if trajectory_key not in trajectory_cache:
            trajectory_cache[trajectory_key] = encode("\n\n".join(c.content for c in trajectory))
        trajectory_embedding = trajectory_cache[trajectory_key]

        scored = []
        for c in past:
            doc_vec = doc_embedding(c)
            scored.append((c.session_id, float(doc_vec @ trajectory_embedding)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [sid for sid, _ in scored[:k]]

    return _predict


def maybe_add_accuracy_ablation(
    metrics: dict,
    samples: list,
    trajectory_window: int,
    accuracy_mode: str,
    semantic_model: str,
) -> dict | None:
    if accuracy_mode == ACCURACY_MODE_DEFAULT:
        return None
    if accuracy_mode != "semantic-ablation":
        raise ValueError(f"Unknown accuracy mode: {accuracy_mode}")

    print()
    print(f"Optional accuracy mode: semantic-ablation ({semantic_model})")
    try:
        result = score(semantic_embedding_predict(trajectory_window, semantic_model), samples)
    except RuntimeError as exc:
        skipped = {
            "status": "skipped",
            "reason": str(exc),
            "default_unchanged": True,
        }
        metrics["semantic_embedding_ablation"] = skipped
        print(f"  skipped: {exc}")
        return skipped

    metrics["semantic_embedding_ablation"] = result
    print(
        f"  {'semantic embedding':<24}  {result['pqhr@1']:.4f}  {result['pqhr@3']:.4f}  "
        f"{result['pqhr@5']:.4f}   {result['mrr@5']:.4f}   {result['ndcg@5']:.4f}"
    )
    return result


def print_scores(results: dict) -> None:
    rows = [
        ("random (sanity)", results["random"]),
        ("recency-only", results["recency"]),
        ("heuristic (0.4r+0.6t)", results["heuristic"]),
        ("pure topic (BoW)", results["pure_topic"]),
        ("multi-signal (NDPA)", results["multi_signal"]),
    ]
    print(f"  {'predictor':<24}  pqhr@1   pqhr@3   pqhr@5    mrr@5   ndcg@5")
    for name, metrics in rows:
        print(
            f"  {name:<24}  {metrics['pqhr@1']:.4f}  {metrics['pqhr@3']:.4f}  "
            f"{metrics['pqhr@5']:.4f}   {metrics['mrr@5']:.4f}   {metrics['ndcg@5']:.4f}"
        )


def run_adaptive_experiment(users: dict[str, list[Conversation]], public: bool) -> dict:
    windows = {}
    for window in ADAPTIVE_TRAJECTORY_WINDOWS:
        print(f"  scoring adaptive W={window} ...")
        all_samples = []
        for convs in users.values():
            all_samples.extend(build_pqhr_samples(convs, window))
        metrics = {}
        if all_samples:
            metrics = {
                "random": score(baseline_random, all_samples),
                "recency": score(baseline_recency, all_samples),
                "pure_topic": score(pure_topic_predict, all_samples),
                "multi_signal": score(multi_signal_predict(window), all_samples),
            }
        windows[str(window)] = {
            "trajectory_window": window,
            "sample_count": len(all_samples),
            "metrics": metrics,
        }
    best_window = max(
        windows.values(),
        key=lambda row: row["metrics"].get("multi_signal", {}).get("pqhr@5", 0.0),
    )["trajectory_window"]
    return {
        "dataset": "longmemeval_public_s" if public else "real_user_internal",
        "bm25_labeler": True,
        "trajectory_only_predictors": True,
        "windows": windows,
        "best_multi_signal_window_by_pqhr@5": best_window,
    }


def write_results(payload: dict, output: Path) -> None:
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote results JSON: {output}")


def load_public_conversations() -> dict[str, list["Conversation"]]:
    """
    Load LongMemEval synthetic users as a public PQHR corpus.

    Each LongMemEval question has 40 haystack sessions for a single synthetic
    user. We group sessions by question_id (one synthetic user per question)
    and run PQHR per-user.

    Data: xiaowu0162/longmemeval-cleaned (already downloaded by longmemeval_runner)
    No Supabase credentials needed.
    """
    import json
    from eval.external_benchmark_utils import DATA_DIR
    from eval.longmemeval_runner import DATASETS

    dataset_id, filename, url = DATASETS["s"]
    path = DATA_DIR / "longmemeval" / filename
    if not path.exists():
        from eval.external_benchmark_utils import download_json
        download_json(url, path)

    items = json.loads(path.read_text())
    users: dict[str, list[Conversation]] = {}

    for item in items:
        qid = item.get("question_id") or item.get("id") or "unknown"
        session_ids = item.get("haystack_session_ids") or []
        sessions = item.get("haystack_sessions") or []
        convs = []
        for idx, turns in enumerate(sessions):
            if not isinstance(turns, list):
                continue
            sid = session_ids[idx] if idx < len(session_ids) else f"{qid}_{idx}"
            content = "\n".join(
                f"[{t.get('role','user')}] {t.get('content','')}"
                for t in turns if isinstance(t, dict)
            )
            if not content.strip():
                continue
            bow = tokenize(content)
            if not bow:
                continue
            convs.append(Conversation(
                session_id=sid,
                content=content,
                started_at=float(idx),
                turns=[t.get("content", "") for t in turns if isinstance(t, dict) and t.get("content")],
                bow=bow,
            ))
        if len(convs) >= MIN_PAST_CONVS + 1:
            users[qid] = convs

    return users


def run_public_pqhr(
    trajectory_window: int,
    adaptive_windows: bool,
    output: Path,
    accuracy_mode: str,
    semantic_model: str,
) -> None:
    """Run PQHR on LongMemEval synthetic users. No personal data needed."""
    print("=" * 72)
    print(" PQHR (Public) — LongMemEval synthetic users")
    print("=" * 72)
    print(f" trajectory window: {trajectory_window}  |  ground truth: BM25")
    print(f" data: xiaowu0162/longmemeval-cleaned (public)")
    print()

    users = load_public_conversations()
    print(f"Loaded {len(users)} synthetic users from LongMemEval")

    all_samples = []
    for qid, convs in users.items():
        samples = build_pqhr_samples(convs, trajectory_window)
        all_samples.extend(samples)

    print(f"Total PQHR samples: {len(all_samples)}")
    if not all_samples:
        print("Not enough data.")
        return
    print()

    all_convs = [c for convs in users.values() for c in convs]
    metrics = evaluate_samples(all_samples, all_convs, trajectory_window)

    print_scores(metrics)
    maybe_add_accuracy_ablation(metrics, all_samples, trajectory_window, accuracy_mode, semantic_model)
    lift = (metrics["multi_signal"]["pqhr@5"] - metrics["recency"]["pqhr@5"]) * 100
    print()
    print(f"Lift over recency baseline @5: +{lift:.1f} pts")
    print()
    print(f"  {len(users)} synthetic users · {len(all_samples)} samples")
    print(f"  Ground truth: BM25 (independent from BoW cosine predictor)")
    print(f"  Reproducible: python3 -m eval.pre_query_hit_rate --public")

    payload = {
        "dataset": "longmemeval_public_s",
        "trajectory_window": trajectory_window,
        "sample_count": len(all_samples),
        "user_count": len(users),
        "ground_truth": "BM25 top-5",
        "predictor_discipline": "trajectory-only; predictors never see target/current conversation",
        "accuracy_mode": accuracy_mode,
        "default_no_embedding": accuracy_mode == ACCURACY_MODE_DEFAULT,
        "metrics": metrics,
    }
    if adaptive_windows:
        print()
        print("Adaptive trajectory window experiment")
        experiment = run_adaptive_experiment(users, public=True)
        payload["adaptive_trajectory_windows"] = experiment
        for window, row in experiment["windows"].items():
            result = row["metrics"]["multi_signal"]
            print(
                f"  W={window:<2} samples={row['sample_count']:<5} "
                f"multi_signal pqhr@5={result['pqhr@5']:.4f} mrr@5={result['mrr@5']:.4f} ndcg@5={result['ndcg@5']:.4f}"
            )
    write_results(payload, output)


def main():
    global MIN_FUTURE_CHARS
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-window", type=int, default=TRAJECTORY_WINDOW_DEFAULT,
                        help="How many prior conversations the predictor sees")
    parser.add_argument("--min-future-chars", type=int, default=MIN_FUTURE_CHARS)
    parser.add_argument("--public", action="store_true",
                        help="Run on LongMemEval public data (no personal data needed)")
    parser.add_argument("--adaptive-windows", action="store_true",
                        help="Run trajectory window sweep over 3,5,7,10,15")
    parser.add_argument("--output", type=Path, default=RESULTS_PATH,
                        help="Where to write results JSON")
    parser.add_argument("--accuracy-mode", choices=(ACCURACY_MODE_DEFAULT, "semantic-ablation"),
                        default=ACCURACY_MODE_DEFAULT,
                        help="Optional accuracy-only ablation. Default PQHR remains no-embedding.")
    parser.add_argument("--semantic-model", default=SEMANTIC_MODEL_DEFAULT,
                        help="sentence-transformers model for --accuracy-mode semantic-ablation")
    args = parser.parse_args()
    MIN_FUTURE_CHARS = args.min_future_chars

    if args.public:
        run_public_pqhr(
            args.trajectory_window,
            args.adaptive_windows,
            args.output,
            args.accuracy_mode,
            args.semantic_model,
        )
        return

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

    metrics = evaluate_samples(samples, convs, args.trajectory_window)
    print_scores(metrics)
    maybe_add_accuracy_ablation(metrics, samples, args.trajectory_window, args.accuracy_mode, args.semantic_model)
    print()

    lift = (metrics["multi_signal"]["pqhr@5"] - metrics["recency"]["pqhr@5"]) * 100
    print(f"Lift over recency baseline @5: +{lift:.1f} pts  [multi-signal (NDPA)]")
    print()
    print("=" * 72)
    print(" INTERPRETATION")
    print("=" * 72)
    print(f"  Out of every 100 future conversations, NDPA correctly")
    print(f"  pre-stages a relevant past conversation in its top 5")
    print(f"  predictions {int(metrics['multi_signal']['pqhr@5'] * 100)}% of the time — using ONLY")
    print(f"  prior session trajectory. No question. No current conversation.")
    print()
    print(f"  Recency baseline gets {int(metrics['recency']['pqhr@5'] * 100)}%. Random gets {int(metrics['random']['pqhr@5'] * 100)}%.")
    print()
    print(f"  This is the metric NDPA owns: predict-forward staging from")
    print(f"  behavioral signal alone. Vector DBs require a query. Mem0/Zep")
    print(f"  require an explicit input. NDPA is the only system that scores")
    print(f"  on this benchmark because nobody else attempts it.")

    payload = {
        "dataset": "real_user_internal",
        "trajectory_window": args.trajectory_window,
        "sample_count": len(samples),
        "conversation_count": len(convs),
        "ground_truth": "BM25 top-5",
        "predictor_discipline": "trajectory-only; predictors never see target/current conversation",
        "accuracy_mode": args.accuracy_mode,
        "default_no_embedding": args.accuracy_mode == ACCURACY_MODE_DEFAULT,
        "metrics": metrics,
    }
    if args.adaptive_windows:
        payload["adaptive_trajectory_windows"] = run_adaptive_experiment({"main_user": convs}, public=False)
    write_results(payload, args.output)


if __name__ == "__main__":
    main()
