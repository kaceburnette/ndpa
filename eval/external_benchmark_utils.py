"""
Shared helpers for external NDPA benchmark runners.

Keep this module dependency-free. The benchmark datasets are external and
large enough that the runners should degrade to clear blocker notes instead
of silently changing methodology when optional tooling is missing.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sdk" / "python"))

DATA_DIR = ROOT / "eval" / "data"
NOTES_PATH = ROOT / "NOTES.md"

TOP_KS = (1, 3, 5)


def load_ndpa_client(platform: str, timeout: float = 20.0):
    from ndp.config import load_config
    from ndpa import Client

    cfg = load_config()
    api_key = os.environ.get("NDPA_API_KEY") or cfg.get("api_key")
    if not api_key:
        raise RuntimeError("NDPA API key not found. Set NDPA_API_KEY or cfg['api_key'].")
    return Client(api_key=api_key, platform=platform, async_send=False, timeout=timeout)


def download_json(url: str, dest: Path, *, force: bool = False) -> Any:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return json.loads(dest.read_text())

    req = urllib.request.Request(url, headers={"User-Agent": "ndpa-eval/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read()
    dest.write_bytes(raw)
    return json.loads(raw.decode("utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def append_note(title: str, body: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    entry = f"\n## {stamp} - {title}\n\n{body.strip()}\n"
    if NOTES_PATH.exists():
        NOTES_PATH.write_text(NOTES_PATH.read_text() + entry)
    else:
        NOTES_PATH.write_text("# NDPA Notes\n" + entry)


class RateLimiter:
    def __init__(self, requests_per_minute: int):
        self.delay = 60.0 / max(1, requests_per_minute)
        self._next_at = 0.0

    def wait(self) -> None:
        now = time.time()
        if now < self._next_at:
            time.sleep(self._next_at - now)
        self._next_at = time.time() + self.delay


def session_to_events(session: Iterable[dict], *, fallback_ts: float) -> list[dict]:
    events = []
    ts = fallback_ts
    for turn in session:
        role = str(turn.get("role") or turn.get("speaker") or "user").lower()
        if role not in {"user", "assistant", "tool"}:
            role = "user"
        content = str(turn.get("content") or turn.get("text") or "").strip()
        if not content:
            continue
        events.append({"role": role, "content": content[:4000], "ts": ts})
        ts += 0.001
    return events


def prediction_session_ids(result: dict) -> list[str]:
    ids = []
    for pred in result.get("predictions") or []:
        sid = pred.get("session_id") or pred.get("id")
        if sid is not None:
            ids.append(str(sid))
    return ids


def empty_hit_counts() -> dict[str, int]:
    return {f"hit@{k}": 0 for k in TOP_KS}


def score_predictions(predicted: list[str], truth: set[str]) -> dict[str, bool]:
    return {f"hit@{k}": bool(set(predicted[:k]) & truth) for k in TOP_KS}


def aggregate_hits(rows: list[dict], *, category_field: str = "category") -> dict:
    buckets: dict[str, list[dict]] = defaultdict(list)
    buckets["overall"] = []
    for row in rows:
        if row.get("scored"):
            buckets["overall"].append(row)
            buckets[str(row.get(category_field) or "unknown")].append(row)

    out = {}
    for category, scored_rows in sorted(buckets.items()):
        n = len(scored_rows)
        metrics = {f"hit@{k}": 0.0 for k in TOP_KS}
        if n:
            for k in TOP_KS:
                metrics[f"hit@{k}"] = round(
                    sum(1 for row in scored_rows if row["hits"][f"hit@{k}"]) / n,
                    4,
                )
        out[category] = {"n": n, **metrics}
    return out

