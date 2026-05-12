# NDPA Architecture

## What NDPA is

NDPA (Neural Data Prefetch Architecture) is **token prediction for the data
layer.** Transformers predict the next token. NDPA predicts the next *data* —
the files, conversations, and context an AI assistant will need on its next
turn — and stages it before the user asks.

This is not RAG. RAG retrieves on query (reactive). NDPA stages context
proactively from behavioral signal: which files you've touched, what topics
you've been discussing, what past sessions are about to become relevant.

## System diagram

```
                  ┌─────────────────────────┐
                  │   AI Platform / IDE     │
                  │   (Claude, ChatGPT,     │
                  │    Cursor, Continue)    │
                  └────────────┬────────────┘
                               │
                       events  │  predictions
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
  ┌───────────┐         ┌───────────┐         ┌───────────┐
  │ Collector │         │ Local     │         │ Prediction│
  │  (hook)   │────────▶│  queue    │         │  kernel   │
  │           │ enqueue │ (jsonl)   │         │           │
  └───────────┘         └─────┬─────┘         └─────┬─────┘
                              │                     │
                       flusher│                     │
                              ▼                     │
                       ┌─────────────────────┐      │
                       │  Ingestion API      │      │
                       │  /v1/events         │◀─────┘
                       └──────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │  Supabase Postgres  │
                       │  + Row-Level Sec    │
                       │  ndp_events         │
                       │  ndp_conversations  │
                       │  ndp_objects        │
                       │  ndp_api_keys       │
                       │  ndp_rate_limits    │
                       └──────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │ Predictions API     │
                       │ /v1/predictions     │
                       │ (BoW cosine +       │
                       │  recency decay)     │
                       └─────────────────────┘
```

## Components

### Collector (`collector/hook.py`)

A subprocess that platforms invoke on every tool use and user prompt. Captures:
- File paths touched (Read, Edit, Write, MultiEdit)
- User prompts (UserPromptSubmit)
- Session metadata (turn count, cwd)

Writes to a local queue in <50ms. **Never** blocks the host platform.

### Queue + flusher (`ndp/queue.py`)

Local append-only JSONL queue (`~/.ndp/queue/pending.jsonl`). A background
daemon flushes batches to the Ingestion API every 2 seconds. Self-restarting,
crash-safe. Pending events persist across reboots.

### Prediction kernel (`kernel/predictor.py`)

Pure-Python scoring engine. Four features, weighted sum:

| Feature              | Weight | What it scores                         |
|----------------------|--------|----------------------------------------|
| `recency`            | 0.35   | How recently a file was accessed       |
| `frequency`          | 0.25   | How often it appears in this session   |
| `extension_affinity` | 0.20   | Test/spec/type pairings (foo.ts↔foo.test.ts) |
| `import_proximity`   | 0.20   | Files imported by currently-hot files  |

No GPU. No LLM. No model weights to load. <100ms per `predict()` call.

### Ingestion API (`Edge function: events`)

`POST /v1/events` — bearer-auth, rate-limited (600 req/min/key), max 1000
events per call, max 2MB body. Writes to `ndp_events`, upserts session row,
appends to `ndp_conversations`.

### Predictions API (`Edge function: predictions`)

`POST /v1/predictions` — bearer-auth. Two modes:

1. **Query mode** (`session_id=""`, `query=text`): pure topic match (cosine
   on bag-of-words) across all past conversations. Returns top-K.
2. **Live mode** (`session_id=<active>`): blends topic match (0.7) + recency
   decay (0.3, exponential, 180-day half-life).

### Storage schema

```sql
ndp_events           -- every tool use + prompt, partitioned by user
ndp_sessions         -- one row per session, turn count, platform
ndp_conversations    -- concatenated conversation text per session
ndp_objects          -- file refs, tiers (hot/warm/cold), scores
ndp_api_keys         -- SHA-256 hash + user binding
ndp_rate_limits      -- per-key, per-minute, per-endpoint counters
```

All tables have **Row-Level Security** enabled. Service role bypasses RLS;
users never query directly.

## Latency budget

| Stage             | Target  | Actual |
|-------------------|---------|--------|
| Hook write        | <50ms   | 44ms   |
| Queue → API flush | <500ms  | ~300ms (background) |
| Predictions API   | <2s     | 1.2s for 600 conversations |
| Health check      | <300ms  | 174ms  |

## Privacy model

| Data                          | Stored locally | Stored remote | Opt-in to upload |
|-------------------------------|----------------|---------------|------------------|
| File paths                    | yes            | yes           | no (always sent) |
| Behavioral signal (counts/ts) | yes            | yes           | no (always sent) |
| Conversation text             | yes            | yes           | no (always sent) |
| **File contents**             | yes            | **no**        | **yes** (config)|

File contents are read locally for context staging but **never uploaded by
default**. Set `upload_file_contents: true` in `~/.ndp/config.json` to
opt in.

## Scaling characteristics

- **Per-user**: ~5KB/session in `ndp_events`, ~20KB/session in
  `ndp_conversations`. 1M sessions = ~25GB.
- **Predictions API**: O(N) over past conversations per request. BoW cosine,
  no embeddings, no vector index. Bottleneck is row fetch, not compute.
- **For 100M users**: shard by `user_id` (already partitioned in schema),
  add Postgres read replicas for predictions reads. Edge functions scale
  horizontally.

## Why no embeddings?

Embeddings work for retrieval. NDPA does prediction. The signal is *behavioral*
(what you touched, when, in what sequence), not *semantic* (what the words
mean). A BoW tokenizer + cosine catches enough of the topic signal to drive
the recency/frequency/affinity features, and runs in 5ms.

Embeddings would add cost (GPU inference) and latency (vector DB hop) for
marginal gains on a problem where the behavioral signal is the primary
predictor. We can add them later as an optional feature for platforms that
already pay the embedding cost.

## Why a separate prediction layer (not in-model)?

- **Cross-platform**: a single user spans ChatGPT, Claude, Cursor, Continue.
  No model sees their full behavior. NDPA does.
- **Stateless model, stateful memory**: keeps LLMs cheap and replaceable;
  memory accumulates separately.
- **Predict before inference**: the LLM doesn't have to spend tokens
  re-establishing context. NDPA stages it.
