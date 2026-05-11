# NDP Object Schema

An NDP object is a serializable unit of memory with provenance, tier assignment, and access history.

## Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID string | Globally unique identifier |
| `type` | enum | `file` \| `symbol` \| `chunk` \| `fact` \| `summary` |
| `source_path` | string | Absolute path to source (files) or qualified name (symbols) |
| `source_type` | enum | `file` \| `git` \| `semantic` \| `user` |
| `summary` | string | One-line human-readable description |
| `tags` | string[] | Free-form labels for grouping/filtering |
| `relationships` | Relationship[] | Edges to other NDP objects |
| `freshness` | unix timestamp | Time of last content modification |
| `token_estimate` | int | Estimated token count of `content` |
| `prior_usefulness` | float [0,1] | Decaying score: how useful this object has been historically |
| `access_history` | AccessEvent[] | Log of when/how this object was accessed |
| `tier` | enum | `hot` \| `warm` \| `cold` \| `evicted` |
| `content` | string \| null | Raw content — only present for `hot` tier objects |

## Sub-types

### Relationship
```json
{
  "target_id": "uuid",
  "rel_type": "imports | imported_by | co_accessed_with | defines | defined_in"
}
```

### AccessEvent
```json
{
  "ts": 1715000000.0,
  "session_id": "abc123",
  "tool": "Read | Edit | Write"
}
```

## v0 scope

v0 only implements `type: "file"`. Symbols, chunks, facts, and summaries are reserved for future versions.

## Tier semantics

| Tier | Meaning | Content present? |
|------|---------|------------------|
| `hot` | Injected directly into context | Yes |
| `warm` | Staged in memory, ready to promote | Optional |
| `cold` | Indexed, not staged | No |
| `evicted` | Removed from active memory | No |

Tier transitions are managed by the daemon based on kernel predictions.
