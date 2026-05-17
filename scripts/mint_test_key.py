#!/usr/bin/env python3
"""Mint a local test API key for NDPA FastAPI smoke testing.

Writes NDPA_TEST_API_KEY=<token> to .env (chmod 600).
Prints only the SHA-256 hash and the INSERT SQL — raw token never printed.
"""
import hashlib
import os
import secrets
import stat
from pathlib import Path

ROOT = Path(__file__).parent.parent
ENV_FILE = ROOT / ".env"

token = secrets.token_hex(32)
key_hash = hashlib.sha256(token.encode()).hexdigest()

lines = []
if ENV_FILE.exists():
    existing = ENV_FILE.read_text()
    lines = [ln for ln in existing.splitlines() if not ln.startswith("NDPA_TEST_API_KEY=")]
lines.append(f"NDPA_TEST_API_KEY={token}")
ENV_FILE.write_text("\n".join(lines) + "\n")
ENV_FILE.chmod(stat.S_IRUSR | stat.S_IWUSR)

print(f"key_hash: {key_hash}")
print()
print("Run this INSERT via Supabase (execute_sql / psql):")
print(f"INSERT INTO public.ndp_api_keys (user_id, platform, key_hash)")
print(f"VALUES ('test_user', 'local', '{key_hash}');")
print()
print("Raw token written to .env as NDPA_TEST_API_KEY (not printed here).")
