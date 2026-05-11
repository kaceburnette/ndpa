#!/usr/bin/env python3
"""
NDPA installer.

Sets up the .ndp directory, patches Claude Code settings to install
the collector hook and context injector, and indexes the current project.

Usage:
    python3 install.py
    python3 install.py --index /path/to/project
"""

import json
import os
import sys
from pathlib import Path

SETTINGS_PATH = Path.home() / ".claude" / "settings.json"
NDP_DIR = Path.home() / ".ndp"
INSTALL_DIR = Path(__file__).parent.resolve()

HOOK_CMD = f"python3 {INSTALL_DIR}/collector/hook.py"
INJECT_CMD = f"python3 {INSTALL_DIR}/collector/inject.py"


def patch_settings():
    if not SETTINGS_PATH.exists():
        print(f"Claude Code settings not found at {SETTINGS_PATH}")
        print("Install Claude Code first: https://claude.ai/code")
        sys.exit(1)

    with open(SETTINGS_PATH) as f:
        settings = json.load(f)

    hooks = settings.setdefault("hooks", {})

    # PostToolUse — collector
    post = hooks.setdefault("PostToolUse", [])
    hook_entry = {
        "matcher": "Read|Edit|Write|MultiEdit",
        "hooks": [{"type": "command", "command": HOOK_CMD, "timeout": 3}]
    }
    if not any(
        any(h.get("command") == HOOK_CMD for h in e.get("hooks", []))
        for e in post
    ):
        post.append(hook_entry)
        print("  + PostToolUse hook installed")
    else:
        print("  ✓ PostToolUse hook already present")

    # UserPromptSubmit — collector (turn markers)
    submit = hooks.setdefault("UserPromptSubmit", [])
    turn_entry = {
        "hooks": [{"type": "command", "command": HOOK_CMD, "timeout": 3}]
    }
    inject_entry = {
        "hooks": [{"type": "command", "command": INJECT_CMD, "timeout": 3}]
    }
    for label, entry in [("turn marker hook", turn_entry), ("inject hook", inject_entry)]:
        cmd = entry["hooks"][0]["command"]
        if not any(
            any(h.get("command") == cmd for h in e.get("hooks", []))
            for e in submit
        ):
            submit.append(entry)
            print(f"  + UserPromptSubmit {label} installed")
        else:
            print(f"  ✓ UserPromptSubmit {label} already present")

    with open(SETTINGS_PATH, "w") as f:
        json.dump(settings, f, indent=2)

    print()


def setup_dirs():
    (NDP_DIR / "sessions").mkdir(parents=True, exist_ok=True)
    (NDP_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)
    print(f"  + Data directory: {NDP_DIR}")


def index_project(path: Path):
    sys.path.insert(0, str(INSTALL_DIR))
    from ndp.store import NDPStore
    from ndp.indexer import scan_project
    store = NDPStore()
    print(f"  Indexing {path} ...")
    n = scan_project(path, store)
    print(f"  + {n} files indexed")


def main():
    print("\nNDPA Installer")
    print("==============")

    print("\n[1/3] Setting up data directory")
    setup_dirs()

    print("\n[2/3] Patching Claude Code settings")
    patch_settings()

    print("\n[3/3] Indexing project")
    index_path = Path.cwd()
    if "--index" in sys.argv:
        idx = sys.argv.index("--index")
        if idx + 1 < len(sys.argv):
            index_path = Path(sys.argv[idx + 1]).resolve()
    index_project(index_path)

    print("\nDone. NDPA is running.")
    print(f"\nCheck status anytime:")
    print(f"  python3 {INSTALL_DIR}/cli.py status")
    print(f"\nRun eval after a few sessions:")
    print(f"  python3 {INSTALL_DIR}/cli.py eval")
    print()


if __name__ == "__main__":
    main()
