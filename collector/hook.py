#!/usr/bin/env python3
"""
NDP event daemon.

Runs on every PostToolUse and UserPromptSubmit event.

Sequence on file tool calls:
  1. Log event to session JSONL
  2. Rebuild workflow state from session log
  3. Score all indexed candidates with kernel
  4. Promote top 3 → hot, next 7 → warm, rest → cold
  5. Write hot-tier content to ~/.ndp/context.md

Never raises — a crash here must never break a Claude Code session.
"""

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

NDP_DIR = Path.home() / ".ndp"
SESSION_DIR = NDP_DIR / "sessions"
CONTEXT_FILE = NDP_DIR / "context.md"

FILE_TOOLS = {"Read", "Edit", "Write", "MultiEdit"}
HOT_K = 3
WARM_K = 7
TOKEN_BUDGET = 3000

# Never stage these in hot tier — they're infra, not workflow context
EXCLUDE_FROM_HOT = {
    "settings.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    ".gitignore", ".env", ".env.local", "CLAUDE.md",
}


def extract_paths(tool_name: str, tool_input: dict) -> list:
    paths = []
    for key in ("file_path", "path", "new_path", "old_path"):
        if key in tool_input:
            paths.append(tool_input[key])
    raw = tool_input.get("paths")
    if isinstance(raw, list):
        paths.extend(raw)
    return [str(p) for p in paths if p]


def log_event(session_id: str, record: dict):
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    with open(SESSION_DIR / f"{session_id}.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")


def load_state(session_id: str) -> dict:
    log = SESSION_DIR / f"{session_id}.jsonl"
    if not log.exists():
        return {"accesses": [], "cwd": "", "turn_idx": 0}

    events = []
    for line in log.read_text(errors="replace").splitlines():
        try:
            events.append(json.loads(line))
        except Exception:
            continue

    turn_idx = sum(1 for e in events if e.get("event") == "turn_start")
    seen_counts: dict = {}
    last_seen: dict = {}
    cwd = ""
    current_turn = 0

    for ev in events:
        if ev.get("event") == "turn_start":
            current_turn += 1
            cwd = ev.get("cwd", cwd)
        elif ev.get("tool") in FILE_TOOLS:
            cwd = ev.get("cwd", cwd)
            for path in ev.get("paths", []):
                seen_counts[path] = seen_counts.get(path, 0) + 1
                last_seen[path] = current_turn

    return {
        "accesses": [
            {"path": p, "count": seen_counts[p], "last_turn": last_seen[p]}
            for p in seen_counts
        ],
        "cwd": cwd,
        "turn_idx": turn_idx,
    }


def score_and_stage(session_id: str):
    from ndp.store import NDPStore
    from ndp.schema import NDPObject
    from kernel.predictor import HeuristicPredictor

    store = NDPStore()
    state = load_state(session_id)
    known_paths = store.all_paths()

    # Extend state with all indexed-but-unaccessed candidates
    seen = {a["path"] for a in state["accesses"]}
    extended_accesses = list(state["accesses"]) + [
        {"path": p, "count": 0, "last_turn": -1}
        for p in known_paths if p not in seen
    ]
    extended_state = dict(state, accesses=extended_accesses)

    kernel = HeuristicPredictor()
    predictions = kernel.predict(extended_state, k=HOT_K + WARM_K)

    hot_set = {p for p, _ in predictions[:HOT_K]}
    warm_set = {p for p, _ in predictions[HOT_K:HOT_K + WARM_K]}

    for path, score in predictions:
        p = Path(path)
        obj = store.get_by_path(path)

        if obj is None:
            if not p.exists():
                continue
            try:
                content = p.read_text(errors="replace")
            except Exception:
                continue
            obj = NDPObject(
                source_path=path,
                token_estimate=max(1, len(content) // 4),
                tier="cold",
            )

        obj.tier = "hot" if path in hot_set else "warm" if path in warm_set else "cold"
        obj.prior_usefulness = min(1.0, obj.prior_usefulness * 0.95 + score * 0.05)

        if obj.tier == "hot" and Path(path).name in EXCLUDE_FROM_HOT:
            obj.tier = "warm"
            obj.content = None

        if obj.tier == "hot" and p.exists():
            try:
                obj.content = p.read_text(errors="replace")
            except Exception:
                obj.content = None
        else:
            obj.content = None

        store.upsert(obj)

    _write_context(store)


def _write_context(store):
    from ndp.store import NDPStore

    hot = store.get_by_tier("hot")
    if not hot:
        CONTEXT_FILE.unlink(missing_ok=True)
        return

    parts = [f"<!-- NDP staged context {time.strftime('%H:%M:%S')} -->"]
    used = 0
    for obj in hot:
        if obj.content is None:
            continue
        if used + obj.token_estimate > TOKEN_BUDGET:
            continue
        parts.append(f"\n### {obj.source_path}\n```\n{obj.content}\n```")
        used += obj.token_estimate

    NDP_DIR.mkdir(parents=True, exist_ok=True)
    CONTEXT_FILE.write_text("\n".join(parts))


def main():
    try:
        raw = sys.stdin.read()
        data = json.loads(raw) if raw.strip() else {}
    except Exception:
        sys.exit(0)

    session_id = data.get("session_id", "unknown")
    ts = time.time()
    tool_name = data.get("tool_name", "")

    # Log the event
    try:
        if tool_name in FILE_TOOLS:
            paths = extract_paths(tool_name, data.get("tool_input", {}))
            if paths:
                log_event(session_id, {
                    "event": "tool_use",
                    "ts": ts,
                    "session_id": session_id,
                    "tool": tool_name,
                    "paths": paths,
                    "cwd": data.get("cwd", ""),
                })
        elif "prompt" in data and not tool_name:
            log_event(session_id, {
                "event": "turn_start",
                "ts": ts,
                "session_id": session_id,
                "prompt": str(data.get("prompt", ""))[:500],
                "cwd": data.get("cwd", ""),
            })
    except Exception:
        pass

    # Run kernel + stage (only on file tool calls)
    if tool_name in FILE_TOOLS:
        try:
            score_and_stage(session_id)
        except Exception:
            pass

    sys.exit(0)


if __name__ == "__main__":
    main()
