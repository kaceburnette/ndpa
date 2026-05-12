"""
.ndp portable bundle format.

A .ndp file is a zip archive — portable, self-contained, vendor-neutral.

Layout:
  manifest.json       — bundle metadata (version, user_hash, platform, counts)
  sessions/           — one JSON file per session, full event log
    {session_id}.json
  conversations/      — concatenated conversation text per session (optional)
    {session_id}.txt
  objects/            — NDP objects (file refs, tiers). NO content by default.
    {object_id}.json

Why a portable format:
  - Move behavioral memory between AI platforms
  - Backup before clearing chat history
  - Share anonymized sessions for research
  - Self-host: no vendor lock-in

Privacy:
  - manifest stores a SHA-256 hash of user_id, not the raw UUID
  - file contents are EXCLUDED unless --include-content is set
  - paths are kept (they're useful for the kernel) but no source code

CLI:
    python3 -m ndp.bundle export <session_id> [--out file.ndp] [--include-content]
    python3 -m ndp.bundle import <file.ndp>
    python3 -m ndp.bundle info <file.ndp>
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

BUNDLE_VERSION = "1"
SESSION_DIR = Path.home() / ".ndp" / "sessions"


def _short_hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def export_session(session_id: str, out_path: Path, include_content: bool = False) -> dict:
    """Export a single session to a .ndp bundle."""
    from ndp.config import load_config, is_configured
    if not is_configured():
        raise RuntimeError("Run `python3 -m ndp.config` first")

    cfg = load_config()
    from supabase import create_client
    client = create_client(cfg["supabase_url"], cfg["supabase_key"])
    user_id = cfg["user_id"]

    manifest = {
        "ndp_bundle_version": BUNDLE_VERSION,
        "exported_at": time.time(),
        "user_id_hash": _short_hash(user_id),
        "session_id": session_id,
        "platform": "claude_code",
        "include_content": include_content,
    }

    # Local JSONL events
    session_log = SESSION_DIR / f"{session_id}.jsonl"
    local_events = []
    if session_log.exists():
        for line in session_log.read_text(errors="replace").splitlines():
            try:
                local_events.append(json.loads(line))
            except Exception:
                continue

    # Remote events from Supabase
    try:
        sb_events = (
            client.table("ndp_events")
            .select("*")
            .eq("user_id", user_id)
            .eq("session_id", session_id)
            .order("ts", desc=False)
            .execute()
        ).data or []
    except Exception:
        sb_events = []

    # Conversation content
    conv_content = ""
    try:
        conv = (
            client.table("ndp_conversations")
            .select("content, started_at, platform")
            .eq("user_id", user_id)
            .eq("session_id", session_id)
            .maybe_single()
            .execute()
        )
        if conv and conv.data:
            conv_content = conv.data.get("content") or ""
            manifest["platform"] = conv.data.get("platform", manifest["platform"])
            manifest["started_at"] = conv.data.get("started_at")
    except Exception:
        pass

    # Objects (paths + scores; NEVER content unless include_content)
    try:
        objs_res = (
            client.table("ndp_objects")
            .select("*")
            .eq("user_id", user_id)
            .execute()
        )
        objects = objs_res.data or []
    except Exception:
        objects = []

    if not include_content:
        for o in objects:
            o.pop("content", None)

    manifest["counts"] = {
        "events_local": len(local_events),
        "events_remote": len(sb_events),
        "conversation_chars": len(conv_content),
        "objects": len(objects),
    }

    # Write bundle to disk
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_bundle_zip(out_path, manifest, session_id, local_events, sb_events, conv_content, objects)
    return manifest


def export_session_to_bytes(session_id: str, include_content: bool = False) -> tuple[bytes, dict]:
    """
    Same as export_session but returns the bundle as bytes.
    Used by TierBackend implementations that don't have a filesystem.
    """
    # Reuse the export logic, write to in-memory buffer instead of disk
    buf = io.BytesIO()
    # We need to replicate the manifest building — call export_session into a temp path
    # then read back. Simpler: do the inline build here.
    from ndp.config import load_config, is_configured
    if not is_configured():
        raise RuntimeError("Run `python3 -m ndp.config` first")
    cfg = load_config()
    from supabase import create_client
    client = create_client(cfg["supabase_url"], cfg["supabase_key"])
    user_id = cfg["user_id"]

    manifest = {
        "ndp_bundle_version": BUNDLE_VERSION,
        "exported_at": time.time(),
        "user_id_hash": _short_hash(user_id),
        "session_id": session_id,
        "platform": "claude_code",
        "include_content": include_content,
    }

    session_log = SESSION_DIR / f"{session_id}.jsonl"
    local_events = []
    if session_log.exists():
        for line in session_log.read_text(errors="replace").splitlines():
            try:
                local_events.append(json.loads(line))
            except Exception:
                continue

    try:
        sb_events = (client.table("ndp_events").select("*")
                     .eq("user_id", user_id).eq("session_id", session_id)
                     .order("ts", desc=False).execute()).data or []
    except Exception:
        sb_events = []

    conv_content = ""
    try:
        conv = (client.table("ndp_conversations")
                .select("content, started_at, platform")
                .eq("user_id", user_id).eq("session_id", session_id)
                .maybe_single().execute())
        if conv and conv.data:
            conv_content = conv.data.get("content") or ""
            manifest["platform"] = conv.data.get("platform", manifest["platform"])
            manifest["started_at"] = conv.data.get("started_at")
    except Exception:
        pass

    try:
        objs_res = (client.table("ndp_objects").select("*")
                    .eq("user_id", user_id).execute())
        objects = objs_res.data or []
    except Exception:
        objects = []
    if not include_content:
        for o in objects:
            o.pop("content", None)

    manifest["counts"] = {
        "events_local": len(local_events), "events_remote": len(sb_events),
        "conversation_chars": len(conv_content), "objects": len(objects),
    }

    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2, default=str))
        zf.writestr(f"sessions/{session_id}.json", json.dumps({
            "session_id": session_id, "local_events": local_events, "remote_events": sb_events,
        }, indent=2, default=str))
        if conv_content:
            zf.writestr(f"conversations/{session_id}.txt", conv_content)
        for obj in objects:
            obj_id = obj.get("id", _short_hash(obj.get("source_path", str(time.time()))))
            zf.writestr(f"objects/{obj_id}.json", json.dumps(obj, indent=2, default=str))

    return buf.getvalue(), manifest


def import_bundle_bytes(bundle_bytes: bytes) -> dict:
    """Import a .ndp bundle from raw bytes (no filesystem needed)."""
    from ndp.config import load_config, is_configured
    if not is_configured():
        raise RuntimeError("Run `python3 -m ndp.config` first")
    cfg = load_config()
    from supabase import create_client
    client = create_client(cfg["supabase_url"], cfg["supabase_key"])
    user_id = cfg["user_id"]

    stats = {"events_imported": 0, "objects_imported": 0, "conversations_imported": 0}
    buf = io.BytesIO(bundle_bytes)
    with zipfile.ZipFile(buf, "r") as zf:
        manifest = json.loads(zf.read("manifest.json"))
        for name in zf.namelist():
            try:
                if name.startswith("sessions/") and name.endswith(".json"):
                    sd = json.loads(zf.read(name))
                    sid = sd["session_id"]
                    rows = []
                    for ev in sd.get("remote_events", []):
                        rows.append({
                            "user_id": user_id, "session_id": sid,
                            "event_type": ev.get("event_type", "tool_use"),
                            "tool_name": ev.get("tool_name"),
                            "source_path": ev.get("source_path"),
                            "source_type": ev.get("source_type", "file"),
                            "ts": ev.get("ts"), "turn_idx": ev.get("turn_idx", 0),
                        })
                    if rows:
                        client.table("ndp_events").insert(rows).execute()
                        stats["events_imported"] += len(rows)
                elif name.startswith("conversations/") and name.endswith(".txt"):
                    sid = Path(name).stem
                    content = zf.read(name).decode("utf-8", errors="replace")
                    client.table("ndp_conversations").upsert({
                        "user_id": user_id, "session_id": sid, "content": content,
                        "platform": manifest.get("platform", "imported"),
                        "started_at": manifest.get("started_at") or time.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                    }).execute()
                    stats["conversations_imported"] += 1
                elif name.startswith("objects/") and name.endswith(".json"):
                    obj = json.loads(zf.read(name))
                    obj["user_id"] = user_id
                    obj.pop("id", None)
                    client.table("ndp_objects").upsert(obj).execute()
                    stats["objects_imported"] += 1
            except Exception:
                continue
    return {"manifest": manifest, "stats": stats}


def _write_bundle_zip(out_path, manifest, session_id, local_events, sb_events, conv_content, objects):
    """Write the zip to disk path. Shared by export_session."""
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2, default=str))
        zf.writestr(f"sessions/{session_id}.json", json.dumps({
            "session_id": session_id,
            "local_events": local_events,
            "remote_events": sb_events,
        }, indent=2, default=str))
        if conv_content:
            zf.writestr(f"conversations/{session_id}.txt", conv_content)
        for obj in objects:
            obj_id = obj.get("id", _short_hash(obj.get("source_path", str(time.time()))))
            zf.writestr(f"objects/{obj_id}.json", json.dumps(obj, indent=2, default=str))


def import_bundle(bundle_path: Path) -> dict:
    """Import a .ndp bundle into the local NDPA store."""
    from ndp.config import load_config, is_configured
    if not is_configured():
        raise RuntimeError("Run `python3 -m ndp.config` first")

    cfg = load_config()
    from supabase import create_client
    client = create_client(cfg["supabase_url"], cfg["supabase_key"])
    user_id = cfg["user_id"]

    bundle_path = Path(bundle_path)
    if not bundle_path.exists():
        raise FileNotFoundError(bundle_path)

    stats = {"events_imported": 0, "objects_imported": 0, "conversations_imported": 0}

    with zipfile.ZipFile(bundle_path, "r") as zf:
        manifest = json.loads(zf.read("manifest.json"))

        for name in zf.namelist():
            try:
                if name.startswith("sessions/") and name.endswith(".json"):
                    sd = json.loads(zf.read(name))
                    sid = sd["session_id"]
                    rows = []
                    for ev in sd.get("remote_events", []):
                        rows.append({
                            "user_id": user_id,
                            "session_id": sid,
                            "event_type": ev.get("event_type", "tool_use"),
                            "tool_name": ev.get("tool_name"),
                            "source_path": ev.get("source_path"),
                            "source_type": ev.get("source_type", "file"),
                            "ts": ev.get("ts"),
                            "turn_idx": ev.get("turn_idx", 0),
                        })
                    if rows:
                        client.table("ndp_events").insert(rows).execute()
                        stats["events_imported"] += len(rows)

                elif name.startswith("conversations/") and name.endswith(".txt"):
                    sid = Path(name).stem
                    content = zf.read(name).decode("utf-8", errors="replace")
                    client.table("ndp_conversations").upsert({
                        "user_id": user_id,
                        "session_id": sid,
                        "content": content,
                        "platform": manifest.get("platform", "imported"),
                        "started_at": manifest.get("started_at") or time.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                    }).execute()
                    stats["conversations_imported"] += 1

                elif name.startswith("objects/") and name.endswith(".json"):
                    obj = json.loads(zf.read(name))
                    obj["user_id"] = user_id
                    obj.pop("id", None)  # let DB regenerate
                    client.table("ndp_objects").upsert(obj).execute()
                    stats["objects_imported"] += 1
            except Exception:
                continue

    return {"manifest": manifest, "stats": stats}


def info(bundle_path: Path) -> dict:
    """Read manifest from a .ndp bundle without importing."""
    bundle_path = Path(bundle_path)
    with zipfile.ZipFile(bundle_path, "r") as zf:
        manifest = json.loads(zf.read("manifest.json"))
        manifest["file_count"] = len(zf.namelist())
        manifest["bundle_size_bytes"] = bundle_path.stat().st_size
        return manifest


def main():
    parser = argparse.ArgumentParser(prog="ndp.bundle", description="Export/import .ndp portable bundles")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_exp = sub.add_parser("export", help="Export a session to a .ndp bundle")
    p_exp.add_argument("session_id")
    p_exp.add_argument("--out", type=Path, default=None)
    p_exp.add_argument("--include-content", action="store_true",
                       help="Include file contents (default: paths/metadata only)")

    p_imp = sub.add_parser("import", help="Import a .ndp bundle")
    p_imp.add_argument("path", type=Path)

    p_info = sub.add_parser("info", help="Show bundle manifest")
    p_info.add_argument("path", type=Path)

    args = parser.parse_args()

    if args.cmd == "export":
        out = args.out or Path(f"{args.session_id}.ndp")
        manifest = export_session(args.session_id, out, include_content=args.include_content)
        print(f"Exported to {out}")
        print(json.dumps(manifest, indent=2, default=str))

    elif args.cmd == "import":
        result = import_bundle(args.path)
        print(json.dumps(result, indent=2, default=str))

    elif args.cmd == "info":
        print(json.dumps(info(args.path), indent=2, default=str))


if __name__ == "__main__":
    main()
