"""
Synthetic session generator.

Produces realistic JSONL session logs that mimic three workflow patterns:
  1. test-driven   — edit foo.py → read foo_test.py → edit foo.py
  2. import-graph  — read a.ts → read b.ts (imported by a) → read c.ts
  3. cycle         — small file set touched repeatedly (recency-favorable)

These are the patterns the heuristic predictor SHOULD beat baseline on.
If it doesn't beat baseline on synthetic data with known structure, the
kernel itself is broken.

Usage:
    python3 -m eval.synthetic                  # generates 30 sessions
    python3 -m eval.synthetic --count 100      # generates 100 sessions
"""

import argparse
import json
import random
import time
import uuid
from pathlib import Path

OUTPUT_DIR = Path.home() / ".ndp" / "synthetic_sessions"

LANGS = {
    "py":  {"test_suffix": "_test.py",   "imports": []},
    "ts":  {"test_suffix": ".test.ts",   "imports": []},
    "tsx": {"test_suffix": ".test.tsx",  "imports": []},
}


def gen_test_driven_session(cwd: str, n_turns: int = 12) -> list:
    """Edit code, read test, edit code, read test. Predictor SHOULD win this."""
    files = [f"{cwd}/src/{w}.py" for w in ["auth", "user", "session", "payment"]]
    events = []
    ts = time.time()

    for turn in range(n_turns):
        ts += random.uniform(5, 30)
        events.append({
            "event": "turn_start", "ts": ts,
            "prompt": "fix the bug", "cwd": cwd,
        })

        source = random.choice(files)
        test = source.replace(".py", "_test.py")

        ts += random.uniform(1, 5)
        events.append({
            "event": "tool_use", "ts": ts, "tool": "Edit",
            "paths": [source], "cwd": cwd,
        })
        ts += random.uniform(1, 5)
        events.append({
            "event": "tool_use", "ts": ts, "tool": "Read",
            "paths": [test], "cwd": cwd,
        })
        if random.random() < 0.5:
            ts += random.uniform(1, 5)
            events.append({
                "event": "tool_use", "ts": ts, "tool": "Edit",
                "paths": [source], "cwd": cwd,
            })

    return events


def gen_import_graph_session(cwd: str, n_turns: int = 12) -> list:
    """File A imports B imports C. Reading A predicts B. Predictor SHOULD win."""
    modules = ["index", "api", "db", "auth", "utils"]
    events = []
    ts = time.time()

    for turn in range(n_turns):
        ts += random.uniform(5, 30)
        events.append({
            "event": "turn_start", "ts": ts,
            "prompt": "trace this", "cwd": cwd,
        })

        start = random.choice(modules)
        chain = [start, random.choice(modules), random.choice(modules)]
        for mod in chain:
            ts += random.uniform(1, 5)
            events.append({
                "event": "tool_use", "ts": ts, "tool": "Read",
                "paths": [f"{cwd}/src/{mod}.ts"], "cwd": cwd,
            })

    return events


def gen_cycle_session(cwd: str, n_turns: int = 12) -> list:
    """Small file set rotating. Baseline (recency) wins this one."""
    files = [f"{cwd}/src/main.go", f"{cwd}/src/util.go", f"{cwd}/src/types.go"]
    events = []
    ts = time.time()

    for turn in range(n_turns):
        ts += random.uniform(5, 30)
        events.append({
            "event": "turn_start", "ts": ts,
            "prompt": "iterate", "cwd": cwd,
        })

        for _ in range(random.randint(1, 3)):
            ts += random.uniform(1, 5)
            events.append({
                "event": "tool_use", "ts": ts,
                "tool": random.choice(["Read", "Edit"]),
                "paths": [random.choice(files)], "cwd": cwd,
            })

    return events


GENERATORS = [
    ("test_driven",  gen_test_driven_session),
    ("import_graph", gen_import_graph_session),
    ("cycle",        gen_cycle_session),
]


def write_session(events: list, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    sid = str(uuid.uuid4())
    path = out_dir / f"{sid}.jsonl"
    with open(path, "w") as f:
        for ev in events:
            ev["session_id"] = sid
            f.write(json.dumps(ev) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for existing in args.out.glob("*.jsonl"):
        existing.unlink()

    counts = {name: 0 for name, _ in GENERATORS}
    for i in range(args.count):
        name, gen = random.choice(GENERATORS)
        cwd = f"/synthetic/repo_{i % 5}"
        n_turns = random.randint(8, 20)
        events = gen(cwd, n_turns)
        write_session(events, args.out)
        counts[name] += 1

    print(f"Wrote {args.count} synthetic sessions to {args.out}")
    for name, count in counts.items():
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
