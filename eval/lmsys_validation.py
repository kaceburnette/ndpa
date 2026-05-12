#!/usr/bin/env python3
"""
Validate NDPA's held-out future heuristic on LMSYS-Chat-1M samples.

Usage:
    python3 -m eval.lmsys_validation --input path/to/lmsys_sample.jsonl
    python3 -m eval.lmsys_validation --sample-size 5000

The HuggingFace dataset is gated and stored as parquet. This runner avoids
adding hard dependencies: it uses a provided JSON/JSONL sample, or an optional
installed `datasets` package when the user has accepted the dataset terms.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from eval.conversation_eval import Conversation, cosine, tokenize
from eval.external_benchmark_utils import DATA_DIR, append_note, write_json
from eval.held_out_future_eval import (
    baseline_recency,
    build_held_out_samples,
    heuristic_predict,
    pure_topic_predict,
    score,
)

RESULTS_PATH = Path(__file__).with_name("lmsys_results.json")


def load_rows_from_file(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("rows", "data", "train"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError(f"Unsupported LMSYS input shape: {path}")


def load_rows_from_huggingface(sample_size: int) -> list[dict]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:
        raise RuntimeError("Python package `datasets` is not installed") from exc

    ds = load_dataset("lmsys/lmsys-chat-1m", split=f"train[:{sample_size}]")
    return [dict(row) for row in ds]


def normalize_conversation(row: dict) -> list[dict]:
    conv = row.get("conversation")
    if isinstance(conv, str):
        try:
            conv = json.loads(conv)
        except json.JSONDecodeError:
            return []
    if not isinstance(conv, list):
        return []

    turns = []
    for turn in conv:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role") or "").lower()
        content = str(turn.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            turns.append({"role": role, "content": content})
    return turns


def row_text(row: dict) -> str:
    return "\n".join(turn["content"] for turn in normalize_conversation(row))


def select_cluster(rows: list[dict], limit: int) -> tuple[list[dict], dict]:
    """
    LMSYS public schema has no stable user_id. Simulate one user's topical
    continuity by choosing a seed conversation and its nearest neighbors by
    the existing bag-of-words cosine scorer.
    """
    candidates = []
    for idx, row in enumerate(rows):
        turns = normalize_conversation(row)
        if len(turns) < 2:
            continue
        bow = tokenize("\n".join(t["content"] for t in turns))
        if bow:
            candidates.append((idx, row, bow))
    if not candidates:
        return [], {"method": "none", "reason": "No usable conversations with >=2 turns"}

    seed_idx, seed_row, seed_bow = max(candidates, key=lambda item: sum(item[2].values()))
    scored = []
    for idx, row, bow in candidates:
        scored.append((cosine(seed_bow, bow), idx, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [row for _sim, _idx, row in scored[:limit]]
    return selected, {
        "method": "topic_cluster",
        "seed_conversation_id": seed_row.get("conversation_id"),
        "input_candidates": len(candidates),
        "selected": len(selected),
        "caveat": "LMSYS public schema exposes conversation_id but not user_id; this simulates a single user by topic clustering.",
    }


def rows_to_conversations(rows: list[dict]) -> list[Conversation]:
    convs = []
    base_ts = 1_690_000_000.0
    for idx, row in enumerate(rows):
        turns = normalize_conversation(row)
        if len(turns) < 2:
            continue
        content = "\n".join(f"[{t['role']}] {t['content']}" for t in turns)
        session_id = str(row.get("conversation_id") or f"lmsys_{idx:06d}")
        convs.append(Conversation(
            session_id=session_id,
            content=content,
            started_at=base_ts + idx * 3600,
            turns=[t["content"] for t in turns],
            bow=tokenize(content),
        ))
    return convs


def blocked_result(reason: str) -> dict:
    body = (
        "LMSYS-Chat-1M validation could not run automatically. "
        f"Reason: {reason}. The dataset is gated on HuggingFace and distributed "
        "as parquet; this repo intentionally does not add datasets/pyarrow as "
        "required dependencies. Provide --input with an accepted/exported JSON "
        "or JSONL sample, or install datasets after accepting the license."
    )
    append_note("LMSYS validation blocked", body)
    payload = {
        "benchmark": "LMSYS-Chat-1M validation",
        "status": "blocked",
        "reason": reason,
        "metrics": {},
        "rows": [],
    }
    write_json(RESULTS_PATH, payload)
    return payload


def run(args: argparse.Namespace) -> dict:
    try:
        if args.input:
            rows = load_rows_from_file(Path(args.input).expanduser())
            source = str(Path(args.input).expanduser())
        else:
            rows = load_rows_from_huggingface(args.sample_size)
            source = "lmsys/lmsys-chat-1m"
    except Exception as exc:
        return blocked_result(str(exc))

    selected, selection = select_cluster(rows, args.cluster_size)
    convs = rows_to_conversations(selected)
    samples, cutoff_ts = build_held_out_samples(convs, args.holdout_days, args.first_turns)
    if not samples:
        return blocked_result(
            f"Loaded {len(rows)} rows from {source}, selected {len(selected)}, but built 0 held-out samples"
        )

    baseline = score(baseline_recency, samples)
    topic_only = score(pure_topic_predict, samples)
    heuristic = score(heuristic_predict, samples)
    payload = {
        "benchmark": "LMSYS-Chat-1M validation",
        "status": "complete",
        "source": source,
        "selection": selection,
        "holdout_days": args.holdout_days,
        "first_turns": args.first_turns,
        "n_rows_loaded": len(rows),
        "n_conversations_selected": len(convs),
        "n_samples": len(samples),
        "cutoff_ts": cutoff_ts,
        "metrics": {
            "baseline": baseline,
            "topic_only": topic_only,
            "heuristic": heuristic,
        },
        "methodology_caveat": "LMSYS lacks public user IDs and timestamps, so single-user continuity is approximated with topic clustering and input-order timestamps.",
    }
    write_json(RESULTS_PATH, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Local LMSYS JSON/JSONL sample")
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--cluster-size", type=int, default=500)
    parser.add_argument("--holdout-days", type=int, default=7)
    parser.add_argument("--first-turns", type=int, default=1)
    args = parser.parse_args()

    started = time.time()
    result = run(args)
    result["elapsed_sec"] = round(time.time() - started, 2)
    write_json(RESULTS_PATH, result)
    print(f"Wrote {RESULTS_PATH}")
    print(result.get("status"), result.get("metrics") or result.get("reason"))


if __name__ == "__main__":
    main()

