"""
Predictions kernel — metadata-only lexical retrieval.

Strategy:
- Rank candidates by top_terms array overlap against query terms (GIN index).
- Final ranking blends topic overlap, entity/date features, recency, length,
  and MMR diversity. No tsv_content, no full-content scan, no LLM, no GPU.

The kernel operates entirely over ndp_conversations metadata
(top_terms, content_preview, storage_key, content_bytes, updated_at).
Raw content is fetched only by /hydrate after a handle is selected.
"""

from __future__ import annotations

import math
import re
import time
from typing import Optional

import asyncpg

# Recency half-life — older conversations decay faster
RECENCY_HALF_LIFE_DAYS = 180.0
TOPIC_WEIGHT = 0.7
RECENCY_WEIGHT = 0.3
MAX_CANDIDATES = 200
FALLBACK_SCAN_LIMIT = 1200

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+-]{1,}")
_ENTITY_RE = re.compile(r"\b(?:[A-Z][a-zA-Z0-9_+-]{2,}|[A-Z]{2,}|[a-zA-Z0-9_+-]+(?:\.[a-zA-Z0-9_+-]+)+)\b")
_DATE_RE = re.compile(
    r"\b(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4}|"
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"yesterday|today|tomorrow|last|next|before|after|recent|latest|first|"
    r"second|third|fourth|fifth|\d{4})\b",
    re.I,
)
_TEMPORAL_RE = re.compile(
    r"\b(when|before|after|last|latest|first|recent|recently|current|currently|"
    r"today|yesterday|tomorrow|week|month|year|date|time|ago|older|newer|"
    r"changed|updated|previous|next)\b",
    re.I,
)
_STOP = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "for", "of",
    "to", "in", "on", "with", "at", "by", "from", "as", "is", "was", "are",
    "be", "been", "being", "have", "has", "had", "do", "does", "did", "this",
    "that", "these", "those", "it", "its", "they", "them", "their", "what",
    "which", "who", "whom", "whose", "can", "could", "would", "should", "about",
    "tell", "me", "my", "you", "your", "i", "when", "where", "why", "how",
    "did", "does", "done", "there", "here", "before", "after", "last", "next",
    "latest", "first", "recent", "recently", "current", "currently", "previous",
}
_SYNONYMS = {
    "buy": ("bought", "purchase", "purchased"),
    "bought": ("buy", "purchase", "purchased"),
    "purchase": ("buy", "bought", "purchased"),
    "recommend": ("suggest", "suggested", "recommendation"),
    "suggest": ("recommend", "recommended", "suggestion"),
    "restaurant": ("cafe", "dinner", "lunch", "food"),
    "movie": ("film", "show"),
    "job": ("role", "work", "career"),
    "trip": ("travel", "vacation", "visited"),
    "doctor": ("physician", "appointment"),
}


async def predict_topk(
    pool: asyncpg.Pool,
    *,
    user_id: str,
    end_user_id: Optional[str],
    session_id: str,
    query: Optional[str],
    k: int = 5,
) -> tuple[list[dict], str, dict[str, int]]:
    """
    Return (predictions, mode). mode is "query" or "live".

    - Query mode: pure topic (BM25/tsvector). session_id is empty.
    - Live mode: blend topic + recency from the active session's trajectory.

    Both modes scope by user_id (your platform key) and end_user_id (which
    of your users).
    """
    # Determine query text
    if query and query.strip():
        q_text = query.strip()
        mode = "query"
    elif session_id:
        # Live mode: pull trajectory from the active session
        q_text = await _build_trajectory_query(pool, user_id, end_user_id, session_id)
        mode = "live"
        if not q_text:
            return [], mode, {"db_candidate_ms": 0, "rank_ms": 0}
    else:
        return [], "empty", {"db_candidate_ms": 0, "rank_ms": 0}

    terms = _expand_query(q_text)
    if not terms:
        return [], mode, {"db_candidate_ms": 0, "rank_ms": 0}

    db_t0 = time.monotonic()
    async with pool.acquire() as conn:
        rows = await _fetch_metadata_candidates(conn, q_text, terms, user_id, end_user_id, session_id)
    db_candidate_ms = int((time.monotonic() - db_t0) * 1000)

    if not rows:
        return [], mode, {"db_candidate_ms": db_candidate_ms, "rank_ms": 0}

    rank_t0 = time.monotonic()
    query_features = _query_features(q_text)
    ranked = _rank_candidates(rows, q_text, query_features, mode)
    selected = _mmr_select(ranked, k)
    payload = [_prediction_payload(item) for item in selected]
    rank_ms = int((time.monotonic() - rank_t0) * 1000)
    return payload, mode, {"db_candidate_ms": db_candidate_ms, "rank_ms": rank_ms}


async def _fetch_metadata_candidates(
    conn: asyncpg.Connection,
    query: str,
    terms: list[str],
    user_id: str,
    end_user_id: Optional[str],
    session_id: str,
) -> list[dict]:
    rows = await conn.fetch(
        """
        SELECT
          session_id,
          COALESCE(content_preview, '') AS content,
          storage_key,
          content_bytes,
          platform,
          EXTRACT(EPOCH FROM updated_at) AS updated_epoch,
          updated_at,
          1.0::float AS topic_score
        FROM ndp_conversations
        WHERE user_id = $1
          AND ($2::text IS NULL OR end_user_id = $2)
          AND ($3::text = '' OR session_id <> $3)
          AND (
            cardinality($4::text[]) = 0
            OR top_terms && $4::text[]
          )
        ORDER BY updated_at DESC
        LIMIT $5
        """,
        user_id,
        end_user_id,
        session_id,
        terms,
        FALLBACK_SCAN_LIMIT,
    )
    candidates = [dict(r) for r in rows]
    features = _query_features(query)
    query_terms = features["terms"]
    for item in candidates:
        content_terms = set(_content_words((item.get("content") or "")[:20_000]))
        item["topic_score"] = _overlap_score(query_terms, content_terms)
    candidates.sort(key=lambda r: (float(r["topic_score"] or 0.0), float(r["updated_epoch"] or 0)), reverse=True)
    return candidates[:MAX_CANDIDATES]



def _content_words(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text) if w.lower() not in _STOP]


def _expand_query(query: str) -> list[str]:
    words = _content_words(query)
    expanded: list[str] = []
    seen = set()
    for word in words:
        variants = (word, *_SYNONYMS.get(word, ()))
        for variant in variants:
            if variant not in seen:
                seen.add(variant)
                expanded.append(variant)
    return expanded[:32]



def _query_features(query: str) -> dict:
    entities = {
        m.group(0).lower()
        for m in _ENTITY_RE.finditer(query)
        if m.group(0).lower() not in _STOP
    }
    terms = set(_content_words(query))
    return {
        "terms": terms,
        "entities": entities,
        "is_temporal": bool(_TEMPORAL_RE.search(query)),
        "dates": {m.group(0).lower() for m in _DATE_RE.finditer(query)},
    }


def _rank_candidates(rows: list[dict], query: str, features: dict, mode: str) -> list[dict]:
    now = time.time()
    half_life_sec = RECENCY_HALF_LIFE_DAYS * 86400
    ranked = []
    max_topic = max(float(r["topic_score"] or 0.0) for r in rows) or 1.0

    for r in rows:
        content = (r.get("content") or "").lower()
        content_terms = set(_content_words(content[:20_000]))
        topic = float(r["topic_score"] or 0.0) / max_topic
        overlap = _overlap_score(features["terms"], content_terms)
        entity = _entity_score(features["entities"], content)
        date = _date_score(features, content)
        age_sec = max(0.0, now - float(r.get("updated_epoch") or 0))
        recency = 0.5 ** (age_sec / half_life_sec)
        length_norm = _length_score(content)

        if mode == "live":
            score = (
                0.46 * topic
                + 0.22 * overlap
                + 0.12 * entity
                + 0.10 * recency
                + 0.07 * date
                + 0.03 * length_norm
            )
        else:
            score = (
                0.58 * topic
                + 0.23 * overlap
                + 0.11 * entity
                + 0.06 * date
                + 0.02 * length_norm
            )

        item = dict(r)
        item["_score"] = score
        item["_topic_norm"] = topic
        item["_recency"] = recency
        item["_terms"] = content_terms
        ranked.append(item)

    ranked.sort(key=lambda r: r["_score"], reverse=True)
    return ranked


def _overlap_score(query_terms: set[str], content_terms: set[str]) -> float:
    if not query_terms or not content_terms:
        return 0.0
    return len(query_terms & content_terms) / math.sqrt(len(query_terms) * len(content_terms))


def _entity_score(entities: set[str], content: str) -> float:
    if not entities:
        return 0.0
    hits = sum(1 for e in entities if e in content)
    return hits / len(entities)


def _date_score(features: dict, content: str) -> float:
    if not features["is_temporal"]:
        return 0.0
    date_hits = sum(1 for d in features["dates"] if d in content)
    if features["dates"]:
        return min(1.0, date_hits / len(features["dates"]) + (0.15 if _DATE_RE.search(content) else 0.0))
    return 0.3 if _DATE_RE.search(content) else 0.0


def _length_score(content: str) -> float:
    # Prefer substantial but not enormous sessions.
    n = len(content)
    if n <= 0:
        return 0.0
    if 1_000 <= n <= 30_000:
        return 1.0
    if n < 1_000:
        return n / 1_000
    return max(0.2, 30_000 / n)


def _mmr_select(ranked: list[dict], k: int) -> list[dict]:
    selected: list[dict] = []
    remaining = list(ranked)
    while remaining and len(selected) < k:
        best_idx = 0
        best_score = float("-inf")
        for idx, item in enumerate(remaining):
            redundancy = 0.0
            if selected:
                redundancy = max(_jaccard(item["_terms"], s["_terms"]) for s in selected)
            score = 0.86 * float(item["_score"]) - 0.14 * redundancy
            if score > best_score:
                best_score = score
                best_idx = idx
        selected.append(remaining.pop(best_idx))
    return selected


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _prediction_payload(r: dict) -> dict:
    preview = (r.get("content") or "")[:2000]
    return {
        "memory_handle": r["session_id"],
        "session_id": r["session_id"],
        "content_preview": preview,
        # Backward-compatible alias. Hosted Memory returns previews; Hydration
        # is responsible for fetching raw full context.
        "content": preview,
        "score": float(r["_score"]),
        "topic_score": float(r.get("_topic_norm", r.get("topic_score") or 0.0)),
        "recency_score": float(r.get("_recency", 0.0)),
        "platform": r["platform"],
        "storage_key": r.get("storage_key"),
        "content_bytes": r.get("content_bytes"),
        "started_at": r.get("updated_at").isoformat() if r.get("updated_at") else None,
    }


async def _build_trajectory_query(
    pool: asyncpg.Pool,
    user_id: str,
    end_user_id: Optional[str],
    session_id: str,
    window: int = 7,
) -> str:
    """Concatenate the user's last `window` conversation texts as the query."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT COALESCE(content_preview, '') AS content
            FROM ndp_conversations
            WHERE user_id = $1
              AND ($2::text IS NULL OR end_user_id = $2)
              AND session_id <> $3
            ORDER BY updated_at DESC
            LIMIT $4
            """,
            user_id, end_user_id, session_id, window,
        )
    return "\n\n".join(r["content"] for r in rows if r["content"])[:8000]
