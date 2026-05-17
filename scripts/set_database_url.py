#!/usr/bin/env python3
"""Set DATABASE_URL for the clean Supabase project without echoing password."""

from __future__ import annotations

import getpass
from pathlib import Path
from urllib.parse import quote

PROJECT_REF = "ucertqzejhkdgvkvuxmp"
HOST = "aws-1-us-west-2.pooler.supabase.com"
PORT = 6543


def main() -> None:
    password = getpass.getpass("Supabase DB password: ")
    encoded = quote(password, safe="")
    database_url = (
        f"postgresql://postgres.{PROJECT_REF}:{encoded}"
        f"@{HOST}:{PORT}/postgres"
    )

    env_path = Path(".env")
    lines = []
    if env_path.exists():
        lines = [
            line
            for line in env_path.read_text(encoding="utf-8").splitlines()
            if not line.startswith("DATABASE_URL=")
        ]
    lines.append(f'DATABASE_URL="{database_url}"')
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(".env updated with DATABASE_URL for project", PROJECT_REF)
    print("Password was not printed.")


if __name__ == "__main__":
    main()
