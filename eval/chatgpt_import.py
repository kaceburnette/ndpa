#!/usr/bin/env python3
"""
Import ChatGPT conversation history into NDPA.

Usage:
    python3 -m eval.chatgpt_import --dir ~/Downloads/"chat gpt download"
    python3 -m eval.chatgpt_import --dir ~/Downloads/"chat gpt download" --dry-run
"""

import argparse
import glob
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "sdk" / "python"))


def extract_messages(conversation: dict) -> list[dict]:
    """Extract ordered user/assistant messages from a ChatGPT conversation."""
    mapping = conversation.get("mapping", {})
    messages = []

    for node in mapping.values():
        msg = node.get("message")
        if not msg:
            continue
        role = msg.get("author", {}).get("role", "")
        if role not in ("user", "assistant"):
            continue
        content = msg.get("content", {})
        parts = content.get("parts", [])
        text = " ".join(p for p in parts if isinstance(p, str)).strip()
        if not text:
            continue
        ts = msg.get("create_time") or conversation.get("create_time") or time.time()
        messages.append({"role": role, "content": text, "ts": float(ts)})

    messages.sort(key=lambda m: m["ts"])
    return messages


def load_conversations(data_dir: Path) -> list[dict]:
    convs = []
    for fpath in sorted(data_dir.glob("conversations-*.json")):
        with open(fpath) as f:
            batch = json.load(f)
        convs.extend(batch)
        print(f"  loaded {fpath.name}: {len(batch)} conversations")
    return convs


def import_to_ndpa(convs: list[dict], dry_run: bool = False):
    from ndp.config import load_config, is_configured
    if not is_configured():
        print("ERROR: run `python3 -m ndp.config` first")
        sys.exit(1)

    cfg = load_config()
    from ndpa import Client
    client = Client(api_key=cfg["api_key"], platform="chatgpt", async_send=False, timeout=10.0)

    imported = 0
    skipped = 0

    for conv in convs:
        msgs = extract_messages(conv)
        if len(msgs) < 2:
            skipped += 1
            continue

        session_id = f"chatgpt_{conv['id']}"
        events = [{"role": m["role"], "content": m["content"][:2000], "ts": m["ts"]} for m in msgs]

        if dry_run:
            print(f"  [dry] {session_id}: {len(events)} events — {conv.get('title','')[:60]}")
        else:
            try:
                client.log_events(session_id, events)
                imported += 1
            except Exception as e:
                print(f"  ERROR {session_id}: {e}")
                skipped += 1

        if imported % 50 == 0 and imported > 0:
            print(f"  {imported}/{len(convs)} imported...")

    return imported, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=str(Path.home() / "Downloads" / "chat gpt download"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.dir).expanduser()
    if not data_dir.exists():
        print(f"ERROR: directory not found: {data_dir}")
        sys.exit(1)

    print(f"Loading conversations from {data_dir}...")
    convs = load_conversations(data_dir)
    print(f"Total: {len(convs)} conversations\n")

    if args.dry_run:
        print("DRY RUN — no data will be sent\n")
        # show first 5
        for conv in convs[:5]:
            msgs = extract_messages(conv)
            print(f"  {conv['id'][:20]}... | {len(msgs)} msgs | {conv.get('title','')[:60]}")
        return

    print(f"Importing {len(convs)} conversations to NDPA...")
    start = time.time()
    imported, skipped = import_to_ndpa(convs, dry_run=False)
    elapsed = time.time() - start

    print(f"\nDone in {elapsed:.1f}s")
    print(f"  imported: {imported}")
    print(f"  skipped (< 2 msgs): {skipped}")


if __name__ == "__main__":
    main()
