#!/usr/bin/env python3
"""
NDP event daemon.

Runs on every PostToolUse and UserPromptSubmit event.

Sequence on file tool calls:
  1. Log event to local JSONL (never lost, works offline)
  2. Write event to Supabase ndp_events (background thread)
  3. Upsert session row in ndp_sessions
  4. Rebuild workflow state from session log
  5. Score candidates with kernel
  6. Promote top 3 → hot, next 7 → warm, rest → cold
  7. Write hot-tier content to ~/.ndp/context.md

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


# ── Local JSONL (always written, offline fallback) ───────────────────────────

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


# ── Conversation capture ──────────────────────────────────────────────────────

def _append_conv(session_id: str, role: str, text: str):
    """Append a turn to the local conversation buffer."""
    conv_file = SESSION_DIR / f"{session_id}.conv"
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    with open(conv_file, "a") as f:
        f.write(f"[{role}] {text}\n\n")


def _sync_conversation(session_id: str, cwd: str, ts: float):
    """No-op now — conversation content rides on the events posted via the API."""
    return


# ── Ingestion API write (dogfood: same path as every other platform) ─────────

def _supabase_write(session_id: str, ts: float, tool_name: str,
                    paths: list, cwd: str, is_prompt: bool, turn_idx: int,
                    prompt_text: str = ""):
    """Post events through the public Ingestion API. Same path SDKs use."""
    try:
        from ndp.config import load_config, is_configured
        if not is_configured():
            return
        cfg = load_config()
        api_key = cfg.get("api_key")
        if not api_key:
            return

        sdk_path = Path(__file__).parent.parent / "sdk" / "python"
        if str(sdk_path) not in sys.path:
            sys.path.insert(0, str(sdk_path))
        from ndpa import Client

        client = Client(api_key=api_key, platform="claude_code", async_send=False, timeout=3.0)

        events = []
        if is_prompt and prompt_text:
            events.append({"role": "user", "content": prompt_text, "ts": ts})
        elif paths:
            for p in paths:
                events.append({
                    "role": "tool",
                    "tool_name": tool_name,
                    "source_path": p,
                    "ts": ts,
                })

        if events:
            client.log_events(session_id, events)
    except Exception:
        pass


def fire_supabase(session_id: str, ts: float, tool_name: str,
                  paths: list, cwd: str, is_prompt: bool, turn_idx: int,
                  prompt_text: str = ""):
    """Call synchronously — threading causes urllib teardown races on sys.exit."""
    _supabase_write(session_id, ts, tool_name, paths, cwd, is_prompt, turn_idx, prompt_text)


# ── Kernel scoring + context staging ─────────────────────────────────────────

def score_and_stage(session_id: str):
    from ndp.store import NDPStore
    from ndp.schema import NDPObject
    from kernel.predictor import HeuristicPredictor

    store = NDPStore()
    state = load_state(session_id)

    # One network round-trip to get everything we know about this user
    known = store.get_all()  # {path: NDPObject}

    seen = {a["path"] for a in state["accesses"]}
    extended_accesses = list(state["accesses"]) + [
        {"path": p, "count": 0, "last_turn": -1}
        for p in known if p not in seen
    ]
    extended_state = dict(state, accesses=extended_accesses)

    kernel = HeuristicPredictor()
    predictions = kernel.predict(extended_state, k=HOT_K + WARM_K)

    hot_set = {p for p, _ in predictions[:HOT_K]}
    warm_set = {p for p, _ in predictions[HOT_K:HOT_K + WARM_K]}

    for path, score in predictions:
        p = Path(path)
        obj = known.get(path)

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


def _build_trajectory_query(session_id: str) -> str:
    """
    Build a conversation prediction query from recent user prompts only.
    File paths drive the file-level kernel. Prompts drive conversation prediction.
    This is what makes it predictive: the trajectory of what you've been ASKING
    determines what past conversations surface before you type the next question.
    """
    log = SESSION_DIR / f"{session_id}.jsonl"
    if not log.exists():
        return ""

    prompts = []
    for line in log.read_text(errors="replace").splitlines():
        try:
            ev = json.loads(line)
            if ev.get("event") == "turn_start":
                p = (ev.get("prompt") or "").strip()
                if p:
                    prompts.append(p[:200])
        except Exception:
            continue

    # Use last 5 prompts — enough context for trajectory, not so much it's noisy
    return " ".join(prompts[-5:])


def _inject_conversation_predictions(session_id: str):
    """
    Predictive: runs AFTER tool use, uses session trajectory to pre-stage
    relevant past conversations BEFORE the next prompt arrives.
    Not reactive search — prediction from behavioral signal.
    """
    try:
        from ndp.config import load_config, is_configured
        if not is_configured():
            return
        cfg = load_config()
        api_key = cfg.get("api_key")
        if not api_key:
            return

        query = _build_trajectory_query(session_id)
        if not query.strip():
            return

        sdk_path = Path(__file__).parent.parent / "sdk" / "python"
        if str(sdk_path) not in sys.path:
            sys.path.insert(0, str(sdk_path))
        from ndpa import Client

        client = Client(api_key=api_key, async_send=False, timeout=4.0)
        # Pass empty session_id: trajectory is already in query, no live-session recency bias needed
        result = client.get_predictions("", query=query, k=10)
        preds = [
            p for p in result.get("predictions", [])
            if p.get("topic_score", 0) > 0.05
            and len(p.get("content", "")) > 500  # skip shallow/test sessions
        ]
        if not preds:
            return

        lines = ["\n<!-- NDPA: predicted context -->"]
        for p in preds:
            platform = p.get("platform") or "unknown"
            date = (p.get("started_at") or "")[:10]
            content = p.get("content", "")
            title = ""
            for chunk in content.split("[user]")[1:]:
                candidate = chunk.strip().split("\n")[0][:120].strip()
                if candidate and not candidate.startswith("[tool]") and len(candidate) > 15:
                    title = candidate
                    break
            if not title:
                # Fallback: first non-empty, non-tool line anywhere in content
                for line in content.splitlines():
                    line = line.strip()
                    if line and not line.startswith("[tool]") and not line.startswith("[assistant]") and not line.startswith("[user]") and len(line) > 20:
                        title = line[:120]
                        break
            if title:
                lines.append(f"[{date} | {platform}] {title}")

        if len(lines) > 1:
            NDP_DIR.mkdir(parents=True, exist_ok=True)
            existing = CONTEXT_FILE.read_text() if CONTEXT_FILE.exists() else ""
            if "<!-- NDPA: predicted context -->" in existing:
                existing = existing[:existing.index("<!-- NDPA: predicted context -->")].rstrip()
            CONTEXT_FILE.write_text(existing + "\n" + "\n".join(lines))
    except Exception:
        pass


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    try:
        raw = sys.stdin.read()
        data = json.loads(raw) if raw.strip() else {}
    except Exception:
        sys.exit(0)

    session_id = data.get("session_id", "unknown")
    ts = time.time()
    tool_name = data.get("tool_name", "")
    cwd = data.get("cwd", "")

    is_file_event = tool_name in FILE_TOOLS
    is_prompt_event = "prompt" in data and not tool_name

    paths = []
    if is_file_event:
        paths = extract_paths(tool_name, data.get("tool_input", {}))

    # Load state first so we have turn_idx for Supabase
    state = load_state(session_id) if is_file_event else {"turn_idx": 0}
    turn_idx = state.get("turn_idx", 0)

    # 1. Local JSONL write + conversation capture
    try:
        if is_file_event and paths:
            log_event(session_id, {
                "event": "tool_use",
                "ts": ts,
                "session_id": session_id,
                "tool": tool_name,
                "paths": paths,
                "cwd": cwd,
            })
            _append_conv(session_id, "tool", f"{tool_name}: {', '.join(paths)}")
        elif is_prompt_event:
            prompt_text = str(data.get("prompt", ""))
            log_event(session_id, {
                "event": "turn_start",
                "ts": ts,
                "session_id": session_id,
                "prompt": prompt_text[:500],
                "cwd": cwd,
            })
            _append_conv(session_id, "user", prompt_text)
    except Exception:
        pass

    # 2. Ingestion API write (sync, before score_and_stage to avoid teardown races)
    if is_file_event or is_prompt_event:
        prompt_text = str(data.get("prompt", "")) if is_prompt_event else ""
        try:
            fire_supabase(
                session_id, ts, tool_name, paths, cwd, is_prompt_event, turn_idx, prompt_text
            )
        except Exception:
            pass

    # 3. After every file tool use: score files + predict relevant past conversations.
    #    Both run from session trajectory — staged BEFORE the next prompt arrives.
    if is_file_event:
        try:
            score_and_stage(session_id)
        except Exception:
            pass
        try:
            _inject_conversation_predictions(session_id)
        except Exception:
            pass

    sys.exit(0)


if __name__ == "__main__":
    main()
