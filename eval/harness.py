"""
Eval harness: replay session logs, score predictors on hit@k.

Usage:
    python3 -m eval.harness                    # baseline only
    python3 -m eval.harness --session-dir /path/to/sessions
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set

SESSION_DIR = Path.home() / ".ndp" / "sessions"
FILE_TOOLS = {"Read", "Edit", "Write", "MultiEdit"}
TURN_GAP_SECONDS = 30
MIN_HISTORY_TURNS = 3
LOOKAHEAD_TURNS = 5


@dataclass
class EvalSample:
    session_id: str
    turn_idx: int
    state: dict
    ground_truth: Set[str] = field(default_factory=set)


def load_sessions(session_dir: Path = SESSION_DIR) -> Dict[str, list]:
    sessions = {}
    for f in Path(session_dir).glob("*.jsonl"):
        events = []
        for line in f.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if events:
            sessions[f.stem] = events
    return sessions


def group_into_turns(events: list) -> List[list]:
    sorted_events = sorted(events, key=lambda e: e.get("ts", 0))
    turns: List[list] = []
    current: list = []

    for event in sorted_events:
        if event.get("event") == "turn_start":
            if current:
                turns.append(current)
            current = [event]
        elif current and (event.get("ts", 0) - current[-1].get("ts", 0)) > TURN_GAP_SECONDS:
            turns.append(current)
            current = [event]
        else:
            current.append(event)

    if current:
        turns.append(current)

    return turns


def files_in_turn(turn: list) -> Set[str]:
    paths: Set[str] = set()
    for event in turn:
        if event.get("tool") in FILE_TOOLS:
            for p in event.get("paths", []):
                if p:
                    paths.add(p)
    return paths


def build_samples(
    sessions: Dict[str, list],
    lookahead: int = LOOKAHEAD_TURNS,
    min_history: int = MIN_HISTORY_TURNS,
) -> List[EvalSample]:
    samples = []
    for session_id, events in sessions.items():
        turns = group_into_turns(events)
        if len(turns) < min_history + 1:
            continue

        for t in range(min_history, len(turns) - 1):
            seen_counts: Dict[str, int] = {}
            last_seen: Dict[str, int] = {}
            for i in range(t):
                for path in files_in_turn(turns[i]):
                    seen_counts[path] = seen_counts.get(path, 0) + 1
                    last_seen[path] = i

            state = {
                "accesses": [
                    {
                        "path": p,
                        "count": seen_counts[p],
                        "last_turn": last_seen[p],
                    }
                    for p in seen_counts
                ],
                "cwd": events[0].get("cwd", ""),
                "turn_idx": t,
            }

            ground_truth: Set[str] = set()
            for i in range(t + 1, min(t + 1 + lookahead, len(turns))):
                ground_truth.update(files_in_turn(turns[i]))

            if ground_truth:
                samples.append(EvalSample(session_id, t, state, ground_truth))

    return samples


def evaluate(predictor, samples: List[EvalSample], ks=(1, 3, 5)) -> Dict[str, float]:
    hits = {k: 0 for k in ks}
    n = len(samples)
    if n == 0:
        return {f"hit@{k}": 0.0 for k in ks}

    for sample in samples:
        preds = predictor.predict(sample.state)
        pred_paths = [p for p, _ in preds]
        for k in ks:
            if set(pred_paths[:k]) & sample.ground_truth:
                hits[k] += 1

    return {f"hit@{k}": round(hits[k] / n, 4) for k in ks}


def run_eval(predictor=None, session_dir: Path = SESSION_DIR, report: bool = True) -> dict:
    from eval.baseline import LastNPredictor

    if predictor is None:
        predictor = LastNPredictor(n=5)

    sessions = load_sessions(session_dir)
    if not sessions:
        print(f"No session logs found in {session_dir}")
        print("The collector hook should be writing to that directory.")
        print("If sessions are empty, check that the hook is installed:")
        print("  grep -A3 'ndp' ~/.claude/settings.json")
        return {}

    samples = build_samples(sessions)
    if not samples:
        print(f"Found {len(sessions)} sessions but no valid eval samples.")
        print("Need at least sessions with 4+ turns. Keep collecting.")
        return {}

    results = evaluate(predictor, samples)

    if report:
        name = type(predictor).__name__
        print(f"\n{name} — {len(samples)} samples from {len(sessions)} sessions")
        for metric, value in sorted(results.items()):
            print(f"  {metric}: {value:.3f}")

    return results


if __name__ == "__main__":
    session_dir = SESSION_DIR
    if len(sys.argv) > 2 and sys.argv[1] == "--session-dir":
        session_dir = Path(sys.argv[2])
    run_eval(session_dir=session_dir)
