"""
Conversation-level eval.

Question: given the current conversation up to turn T, can we predict
which PAST conversations should be staged in context?

Ground truth = past conversations whose content most overlaps with the
FUTURE of the current conversation (turns T+1 onward). The predictor's
job: pick those past conversations from history alone, without seeing
the future.

This is the right eval for the actual product. File-prediction was a
warmup. Conversation prediction is what NDPA actually has to do.

Usage:
    python3 -m eval.conversation_eval
"""

import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple

from ndp.config import load_config
from supabase import create_client

K_GROUND_TRUTH = 3     # top-K past conversations are "relevant"
TOP_K_PREDICT = 5      # predictor outputs top-K
MIN_TURNS = 2          # tiny conversations OK at this data scale
MIN_PAST_CONVS = 1     # need at least this many past convs to score

STOP = set("the a an and or but if then else for of to in on with at by from as is was are be been being have has had do does did this that these those it its they them their".split())
WORD_RE = re.compile(r"[a-zA-Z]{3,}")


def tokenize(text: str) -> Counter:
    """Cheap bag-of-words. No embeddings. Production would use better."""
    words = (w.lower() for w in WORD_RE.findall(text))
    return Counter(w for w in words if w not in STOP)


def cosine(a: Counter, b: Counter) -> float:
    """Cosine similarity between two word-count vectors."""
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[w] * b[w] for w in common)
    na = sum(v * v for v in a.values()) ** 0.5
    nb = sum(v * v for v in b.values()) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


@dataclass
class Conversation:
    session_id: str
    content: str
    started_at: float
    turns: List[str] = field(default_factory=list)  # one string per turn
    bow: Counter = field(default_factory=Counter)


SESSION_DIR = Path.home() / ".ndp" / "sessions"


def load_conversations() -> List[Conversation]:
    """
    Build conversation objects from BOTH sources:
      1. ndp_conversations table (full text, post-today sessions)
      2. JSONL prompts (older sessions that pre-date conv capture)

    Prefer Supabase content when present, fall back to JSONL prompts.
    """
    cfg = load_config()
    client = create_client(cfg["supabase_url"], cfg["supabase_key"])

    sb_content: Dict[str, Tuple[str, float]] = {}
    # By default, load ONLY conversations without end_user_id (real user data,
    # not benchmark haystacks). Benchmark data lives under end_user_id values.
    import os
    include_benchmark = os.environ.get("NDPA_INCLUDE_BENCHMARK") == "1"
    try:
        offset = 0
        page_size = 1000
        while True:
            q = (client.table("ndp_conversations")
                 .select("session_id, content, started_at")
                 .eq("user_id", cfg["user_id"]))
            if not include_benchmark:
                q = q.is_("end_user_id", "null")
            result = q.range(offset, offset + page_size - 1).execute()
            if not result.data:
                break
            for row in result.data:
                content = (row.get("content") or "").strip()
                ts = time.mktime(time.strptime(row["started_at"][:19], "%Y-%m-%dT%H:%M:%S"))
                if content:
                    sb_content[row["session_id"]] = (content, ts)
            if len(result.data) < page_size:
                break
            offset += page_size
    except Exception:
        pass

    convs: List[Conversation] = []
    seen: set = set()

    # Pull turn lists from each JSONL — prompts are our turn proxy
    for jsonl in SESSION_DIR.glob("*.jsonl"):
        session_id = jsonl.stem
        seen.add(session_id)
        prompts: List[str] = []
        first_ts = None
        for line in jsonl.read_text(errors="replace").splitlines():
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if first_ts is None:
                first_ts = ev.get("ts", time.time())
            if ev.get("event") == "turn_start":
                p = (ev.get("prompt") or "").strip()
                if p:
                    prompts.append(p)

        if session_id in sb_content:
            content, ts = sb_content[session_id]
            turns = [t.strip() for t in re.split(r"\n\[(?:user|tool)\]\s*", content) if t.strip()]
        else:
            content = "\n".join(prompts)
            ts = first_ts or time.time()
            turns = prompts

        if len(turns) < MIN_TURNS:
            continue
        convs.append(Conversation(
            session_id=session_id,
            content=content,
            started_at=ts,
            turns=turns,
            bow=tokenize(content),
        ))

    # Add Supabase-only sessions (e.g. imported ChatGPT conversations)
    for session_id, (content, ts) in sb_content.items():
        if session_id in seen:
            continue
        turns = [t.strip() for t in re.split(r"\n\[(?:user|tool|assistant)\]\s*", content) if t.strip()]
        if not turns:
            turns = [s.strip() for s in content.split("\n\n") if s.strip()]
        if len(turns) < MIN_TURNS:
            continue
        convs.append(Conversation(
            session_id=session_id,
            content=content,
            started_at=ts,
            turns=turns,
            bow=tokenize(content),
        ))

    convs.sort(key=lambda c: c.started_at)
    return convs


# ── Ground truth + predictors ────────────────────────────────────────────────

def relevant_past_for(current: Conversation, split_at: int, past: List[Conversation], k: int) -> Set[str]:
    """Top-K past conversations whose content most overlaps with current's future."""
    future_text = " ".join(current.turns[split_at:])
    future_bow = tokenize(future_text)
    if not future_bow:
        return set()
    scored = [(c.session_id, cosine(c.bow, future_bow)) for c in past]
    scored.sort(key=lambda x: x[1], reverse=True)
    return {sid for sid, score in scored[:k] if score > 0}


def baseline_recency(history_bow: Counter, past: List[Conversation], k: int) -> List[str]:
    """Most recent past conversations. The bar to beat."""
    return [c.session_id for c in sorted(past, key=lambda c: c.started_at, reverse=True)[:k]]


def heuristic_predict(history_bow: Counter, past: List[Conversation], k: int) -> List[str]:
    """Recency + topic-overlap with current conversation history. No LLM."""
    if not past:
        return []
    scored = []
    most_recent_ts = max(c.started_at for c in past)
    for c in past:
        # recency: 0 (oldest) to 1 (newest)
        recency = 1.0 / (1 + max(0, (most_recent_ts - c.started_at) / 86400))
        # topic overlap with what user has talked about in current chat
        topic = cosine(c.bow, history_bow)
        scored.append((c.session_id, 0.4 * recency + 0.6 * topic))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [sid for sid, _ in scored[:k]]


# ── Eval ──────────────────────────────────────────────────────────────────────

def build_samples(convs: List[Conversation]) -> list:
    """Each sample = (current conv, split point, past convs available, ground truth)."""
    samples = []
    sorted_convs = sorted(convs, key=lambda c: c.started_at)

    for i, current in enumerate(sorted_convs):
        past = sorted_convs[:i]
        if len(past) < MIN_PAST_CONVS:
            continue
        # Split current conversation in half
        split = max(2, len(current.turns) // 2)
        if split >= len(current.turns):
            continue
        history_text = " ".join(current.turns[:split])
        history_bow = tokenize(history_text)
        truth = relevant_past_for(current, split, past, K_GROUND_TRUTH)
        if truth:
            samples.append((current, history_bow, past, truth))
    return samples


def score(predictor, samples) -> dict:
    hits = {1: 0, 3: 0, 5: 0}
    n = len(samples)
    if n == 0:
        return {f"hit@{k}": 0.0 for k in (1, 3, 5)}
    for current, history_bow, past, truth in samples:
        preds = predictor(history_bow, past, TOP_K_PREDICT)
        for k in (1, 3, 5):
            if set(preds[:k]) & truth:
                hits[k] += 1
    return {f"hit@{k}": round(hits[k] / n, 4) for k in (1, 3, 5)}


def main():
    convs = load_conversations()
    print(f"Loaded {len(convs)} conversations")
    for c in convs:
        print(f"  {c.session_id[:8]}  {len(c.turns)} turns  {len(c.content)} chars")
    samples = build_samples(convs)
    print(f"Eval samples: {len(samples)}")
    if not samples:
        print("\nNot enough conversation data yet.")
        print(f"Need conversations with {MIN_TURNS}+ turns AND at least {MIN_PAST_CONVS} prior conversations.")
        print("Keep using Claude — more data unlocks this eval.")
        return

    bres = score(baseline_recency, samples)
    hres = score(heuristic_predict, samples)

    print()
    print(f'{"":12s}  hit@1   hit@3   hit@5')
    print(f'{"baseline":12s}  {bres["hit@1"]:.4f}  {bres["hit@3"]:.4f}  {bres["hit@5"]:.4f}')
    print(f'{"heuristic":12s}  {hres["hit@1"]:.4f}  {hres["hit@3"]:.4f}  {hres["hit@5"]:.4f}')
    print()
    for k in (1, 3, 5):
        lift = (hres[f"hit@{k}"] - bres[f"hit@{k}"]) * 100
        print(f"lift@{k}: {lift:+.1f} pts")


if __name__ == "__main__":
    main()
