#!/usr/bin/env python3
"""
UserPromptSubmit hook — injects staged hot-tier context before each message.

Reads ~/.ndp/context.md and returns it as additionalContext so the model
sees pre-staged file content without having to ask for it.
"""

import json
import sys
from pathlib import Path

CONTEXT_FILE = Path.home() / ".ndp" / "context.md"


def main():
    try:
        sys.stdin.read()
    except Exception:
        pass

    if not CONTEXT_FILE.exists():
        sys.exit(0)

    try:
        content = CONTEXT_FILE.read_text(errors="replace").strip()
    except Exception:
        sys.exit(0)

    if not content:
        sys.exit(0)

    print(json.dumps({
        "continue": True,
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": content,
        },
    }))
    sys.exit(0)


if __name__ == "__main__":
    main()
