"""
Local event queue with background flusher.

The hook writes events to a local JSONL queue (single fsync, <5ms) and returns.
A background daemon flushes the queue to the Ingestion API in batches.

Why: hook latency must be near-zero. Every tool call in Claude Code fires the
hook, so a 1-2s sync API call adds noticeable lag to every Read/Edit/Write.

Design:
- Queue file: ~/.ndp/queue/pending.jsonl (append-only, fsynced)
- Flusher: separate Python process spawned on demand, runs detached
- Idempotent flush: queue entries have UUIDs, server dedupes by uuid
- Crash-safe: queue survives across hook invocations; flusher restarts on failure
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
import subprocess
from pathlib import Path

QUEUE_DIR = Path.home() / ".ndp" / "queue"
PENDING_FILE = QUEUE_DIR / "pending.jsonl"
FLUSHER_PID_FILE = QUEUE_DIR / "flusher.pid"
FLUSHER_LOG = QUEUE_DIR / "flusher.log"

BATCH_SIZE = 50
FLUSH_INTERVAL_S = 2.0
MAX_PENDING_BYTES = 5 * 1024 * 1024  # 5MB cap on queue file


def _ensure_dir():
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)


def enqueue(session_id: str, events: list[dict], platform: str = "claude_code") -> None:
    """
    Append a batch to the queue. Returns in <5ms. Never raises.
    Each enqueue creates ONE line: {uuid, session_id, platform, events, ts}.
    """
    if not events:
        return
    try:
        _ensure_dir()
        entry = {
            "uuid": str(uuid.uuid4()),
            "session_id": session_id,
            "platform": platform,
            "events": events,
            "ts": time.time(),
        }
        with open(PENDING_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
            f.flush()
            os.fsync(f.fileno())
        _ensure_flusher_running()
    except Exception:
        pass


def _ensure_flusher_running() -> None:
    """Spawn the flusher if it's not already running. Idempotent."""
    try:
        if FLUSHER_PID_FILE.exists():
            pid = int(FLUSHER_PID_FILE.read_text().strip())
            try:
                os.kill(pid, 0)  # signal 0 = check if process exists
                return  # still running
            except ProcessLookupError:
                pass

        # Spawn detached
        script = Path(__file__).parent.parent / "ndp" / "queue.py"
        env = os.environ.copy()
        env["NDP_FLUSHER_MODE"] = "1"
        proc = subprocess.Popen(
            [sys.executable, str(script)],
            env=env,
            stdout=open(FLUSHER_LOG, "a"),
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        FLUSHER_PID_FILE.write_text(str(proc.pid))
    except Exception:
        pass


def _flush_once(client) -> int:
    """Read pending entries, post to API, truncate file. Returns count flushed."""
    if not PENDING_FILE.exists():
        return 0

    try:
        with open(PENDING_FILE, "r") as f:
            lines = f.readlines()
    except Exception:
        return 0

    if not lines:
        return 0

    flushed = 0
    failed_lines: list[str] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except Exception:
            continue  # drop malformed
        try:
            client.log_events(entry["session_id"], entry["events"])
            flushed += 1
        except Exception as e:
            message = str(e)
            # Permanent client/schema/auth failures should not hammer a local
            # dev server forever. The hook is best-effort dogfood telemetry;
            # losing malformed rows is safer than retry storms.
            if "NDPA API error 401" in message or "NDPA API error 422" in message:
                continue
            failed_lines.append(line)

    # Rewrite file with only failed entries
    try:
        with open(PENDING_FILE, "w") as f:
            for fl in failed_lines:
                f.write(fl + "\n")
    except Exception:
        pass

    return flushed


def _flusher_main():
    """Background process: flush pending events on an interval."""
    root = Path(__file__).parent.parent
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "sdk" / "python"))
    try:
        from ndp.config import load_config, is_configured
        if not is_configured():
            print(f"[{time.strftime('%H:%M:%S')}] not configured, exiting", flush=True)
            return
        cfg = load_config()
        base_url = cfg.get("api_base_url") or os.environ.get("NDPA_BASE_URL")
        if not base_url:
            print(
                f"[{time.strftime('%H:%M:%S')}] api_base_url not configured, exiting",
                flush=True,
            )
            return
        from ndpa import Client
        client = Client(
            api_key=cfg["api_key"],
            base_url=base_url,
            platform="claude_code",
            async_send=False,
            timeout=5.0,
        )
        print(f"[{time.strftime('%H:%M:%S')}] flusher started, pid={os.getpid()}", flush=True)
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] init failed: {e}", flush=True)
        return

    idle_cycles = 0
    while True:
        try:
            n = _flush_once(client)
            if n > 0:
                idle_cycles = 0
                print(f"[{time.strftime('%H:%M:%S')}] flushed {n} entries", flush=True)
            else:
                idle_cycles += 1
            # Exit after ~5min of idle to free resources
            if idle_cycles > 150:
                break
            time.sleep(FLUSH_INTERVAL_S)
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] flusher error: {e}", flush=True)
            time.sleep(FLUSH_INTERVAL_S)


def queue_stats() -> dict:
    """For monitoring: how many entries are pending?"""
    try:
        if not PENDING_FILE.exists():
            return {"pending": 0, "bytes": 0}
        size = PENDING_FILE.stat().st_size
        with open(PENDING_FILE) as f:
            lines = sum(1 for _ in f)
        return {"pending": lines, "bytes": size}
    except Exception:
        return {"pending": 0, "bytes": 0}


if __name__ == "__main__":
    if os.environ.get("NDP_FLUSHER_MODE") == "1":
        _flusher_main()
    else:
        print(json.dumps(queue_stats()))
