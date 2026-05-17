# NDPA Architecture

## What NDPA is

NDPA (Neural Data Prefetch Architecture) is **token prediction for the data
layer.** Transformers predict the next token. NDPA predicts the next *data* —
the files, conversations, and context an AI assistant will need on its next
turn — and stages it before the user asks.

Predictive memory index for AI apps. Core returns handles/previews/scores;
Hydration fetches raw context only when needed.

This is not RAG. RAG retrieves on query (reactive). NDPA stages context
proactively from behavioral signal: which files you've touched, what topics
you've been discussing, what past sessions are about to become relevant.

NDPA ships as four separable layers:

| Layer | Responsibility | Metric |
|---|---|---|
| **Predict** | Pre-query staging from trajectory and behavioral signal | PQHR |
| **Memory** | Explicit-query retrieval over hot metadata | LongMemEval / LoCoMo hit@K |
| **Hydration** | Raw context fetch for selected handles | latency / bandwidth / storage efficiency |
| **Reasoning** | Optional LLM answer layer over retrieved/hydrated context | LongMemEval E2E LLM judge |

Those metrics are intentionally separate. Retrieval hit@K is not QA accuracy,
and Hydration is not a QA benchmark.

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
                       │  ndp_sessions       │
                       │  ndp_conversations  │
                       │  ndp_api_keys       │
                       │  ndp_usage_events   │
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

Embeddings are not required. No LLM or model weights are loaded in the core
scoring path.

### Ingestion API (`POST /events`)

Bearer-auth, rate-limited (600 req/min/key), max 1000 events per call. Accepts
raw text, writes it to the configured Hydration store, upserts session metadata,
and stores compact conversation metadata in Postgres.

### Predictions API (`POST /predictions`)

`POST /v1/predictions` — bearer-auth. Two modes:

1. **Query mode** (`session_id=""`, `query=text`): ranks hot metadata by
   compact terms, preview overlap, entities, dates, and recency.
2. **Live mode** (`session_id=<active>`): builds a trajectory query from recent
   previews and predicts relevant memory handles.

### Hydration API (`POST /hydrate`)

Fetches raw context from local/blob storage by `storage_key` or memory handle
after Predict or Memory has selected a small top-K candidate set.

### Storage schema

```sql
ndp_sessions         -- one row per session, turn count, platform
ndp_conversations    -- hot metadata, previews, handles, storage keys
ndp_api_keys         -- SHA-256 hash + user binding
ndp_usage_events     -- per-product usage telemetry
```

All tables have **Row-Level Security** enabled. Service role bypasses RLS;
users never query directly.

### Hosted metadata-first storage

Core architecture works. Hosted Core treats Postgres as the compact hot tier,
not the raw archive.
The default query path should fit in small rows:

| Field | Purpose |
|---|---|
| `session_id` / `memory_handle` | stable handle returned by Predict and Memory |
| `updated_at`, `platform`, `end_user_id` | tenant filtering and temporal scoring |
| `top_terms` | compact lexical index for candidate selection |
| `content_preview` | short preview for ranking and prompt hints |
| `storage_key` | pointer for Hydration |
| `content_bytes` | storage accounting and hydration budgeting |

Raw conversations, file contents, and `.ndp` bundles belong in blob/object/local
storage. Hydration fetches raw content by `storage_key` only after Predict or
Memory has selected a small top-K candidate set. Full-content `tsv_content` and
full raw-text GIN indexes are optional self-host experiments, not the hosted
default path.

## Latency budget

Current local FastAPI -> Supabase pooler cold prediction smoke is around
~700ms until the latency work improves it. Do not make low-latency hosted
claims yet. Predict latency should be reported as staged/warm latency; cold
Supabase pooler misses are fallback cost, not the hot-path claim.

| Stage | Current claim |
|---|---|
| Core architecture | works; returns handles/previews/scores |
| Predict staged/warm read | cache-hit latency after staging |
| Local FastAPI -> Supabase pooler cold miss | ~700ms smoke |
| Hydration | local-FS implemented; latency/storage benchmark pending |

## Privacy model

| Data                          | Stored locally/blob | Stored in Postgres |
|-------------------------------|---------------------|--------------------|
| File paths / behavioral signal| yes                 | compact metadata   |
| Conversation text             | yes                 | preview + terms only |
| File contents / raw bundles   | yes                 | storage pointer only |

Raw memory is hydrated by key only after Core selects relevant handles.

## Scaling characteristics

- **Per-user**: compact metadata rows in `ndp_sessions` and
  `ndp_conversations`; raw context is outside Postgres.
- **Predictions API**: candidate search over compact metadata. Embeddings are
  not required, and there is no LLM or full raw-text index in the hosted hot
  path.
- **For 100M users**: shard by `user_id` (already partitioned in schema),
  add Postgres read replicas for predictions reads. Edge functions scale
  horizontally.

## Why embeddings are not required

Embeddings work for retrieval. NDPA does prediction. The signal is *behavioral*
(what you touched, when, in what sequence), not *semantic* (what the words
mean). A BoW tokenizer + cosine catches enough of the topic signal to drive
the recency/frequency/affinity features, and runs in 5ms.

Embeddings can be useful for semantic retrieval, but they are not required for
NDPA Core. They add cost and latency when the primary Predict signal is
behavioral. They can remain an optional add-on for platforms that already pay
the embedding cost.

## Why a separate prediction layer (not in-model)?

- **Cross-platform**: a single user spans ChatGPT, Claude, Cursor, Continue.
  No model sees their full behavior. NDPA does.
- **Stateless model, stateful memory**: keeps LLMs cheap and replaceable;
  memory accumulates separately.
- **Predict before inference**: the LLM doesn't have to spend tokens
  re-establishing context. NDPA stages it.
