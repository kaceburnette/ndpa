# .ndp File Format

## Philosophy

Plain-readable. AI-native. Local-first. Every file in `~/.ndp/` should be inspectable with `cat`, `jq`, or a text editor without tooling.

## Storage layout

```
~/.ndp/
├── sessions/          — session event logs (one file per session)
│   ├── {session_id}.jsonl
│   └── ...
├── objects.jsonl      — append-only object store
├── index.db           — SQLite index for fast queries
└── checkpoints/       — workflow state snapshots (not yet implemented)
    └── {checkpoint_id}.json
```

## Session log format (`sessions/*.jsonl`)

One JSON object per line. Two event types:

### tool_use event
```json
{
  "event": "tool_use",
  "ts": 1715000000.123,
  "session_id": "abc123def456",
  "tool": "Read",
  "paths": ["/Users/me/project/auth.ts"],
  "cwd": "/Users/me/project"
}
```

### turn_start event
```json
{
  "event": "turn_start",
  "ts": 1715000001.000,
  "session_id": "abc123def456",
  "prompt": "fix the auth middleware bug",
  "cwd": "/Users/me/project"
}
```

## Object store format (`objects.jsonl`)

One serialized NDP object per line. Append-only. Index is rebuilt from this file.

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "type": "file",
  "source_path": "/Users/me/project/auth.ts",
  "source_type": "file",
  "summary": "JWT auth middleware",
  "tags": ["auth", "middleware", "typescript"],
  "relationships": [
    {"target_id": "...", "rel_type": "co_accessed_with"}
  ],
  "freshness": 1715000000.0,
  "token_estimate": 420,
  "prior_usefulness": 0.72,
  "access_history": [
    {"ts": 1715000000.0, "session_id": "abc123", "tool": "Edit"}
  ],
  "tier": "cold",
  "content": null
}
```

## SQLite index schema (`index.db`)

```sql
CREATE TABLE objects (
    id TEXT PRIMARY KEY,
    source_path TEXT,
    type TEXT,
    tier TEXT,
    freshness REAL,
    token_estimate INTEGER,
    prior_usefulness REAL,
    data JSON
);

CREATE INDEX idx_path ON objects(source_path);
CREATE INDEX idx_tier ON objects(tier);
CREATE INDEX idx_freshness ON objects(freshness);
```

## Inspecting the store

```bash
# Recent session events
tail -20 ~/.ndp/sessions/*.jsonl | python3 -m json.tool

# All hot-tier objects
sqlite3 ~/.ndp/index.db "SELECT source_path, prior_usefulness FROM objects WHERE tier='hot' ORDER BY prior_usefulness DESC"

# Session count
ls ~/.ndp/sessions/*.jsonl | wc -l
```
