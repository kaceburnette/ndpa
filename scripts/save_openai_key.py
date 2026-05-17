#!/usr/bin/env python3
"""Safely save OPENAI_API_KEY to .env without printing or deleting secrets."""

from __future__ import annotations

import getpass
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"


def main() -> None:
    key = getpass.getpass("Paste your OpenAI API key then press Enter: ").strip()
    if not key:
        raise SystemExit("No key entered.")

    try:
        from openai import OpenAI

        OpenAI(api_key=key).models.list()
    except ImportError:
        print("openai package not installed; saving without validation.", file=sys.stderr)
    except Exception as exc:
        raise SystemExit(f"OpenAI rejected that key; .env was not changed. {exc}") from exc

    lines: list[str] = []
    if ENV_FILE.exists():
        lines = [
            line
            for line in ENV_FILE.read_text(encoding="utf-8").splitlines()
            if not line.startswith("OPENAI_API_KEY=")
        ]

    lines.append(f"OPENAI_API_KEY={key}")
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ENV_FILE.chmod(stat.S_IRUSR | stat.S_IWUSR)
    print("Saved OPENAI_API_KEY to .env.")


if __name__ == "__main__":
    main()
