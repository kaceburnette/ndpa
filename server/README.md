# NDPA Predictions Server (FastAPI)

Replaces the Supabase Edge Function deployment. The production latency shape
is staged: cold prediction may still pay the hosted Postgres round trip, while
`/stage` precomputes handles and `/predictions` can return the warm result from
process memory.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/config`       | Public products, pricing, metrics, and billing readiness |
| `GET`  | `/health`       | Liveness probe |
| `GET`  | `/account`      | Authenticated account/key metadata |
| `GET`  | `/usage`        | Authenticated usage summary from `ndp_usage_events` |
| `POST` | `/stage`        | Precompute and cache top-K memory handles / previews / scores |
| `POST` | `/predictions`  | Top-K memory handles / previews / scores |
| `POST` | `/events`       | Ingest a batch of session events |
| `POST` | `/hydrate` | Fetch raw context for selected `session_id` / `storage_key` handles |
| `POST` | `/reasoning`    | Premium Memory -> Hydration -> OpenAI answer layer |
| `POST` | `/checkout`     | Authenticated billing handoff to configured Stripe payment link |
| `POST` | `/admin/api-keys` | Admin-only API key minting with `X-NDPA-Admin-Token` |

`/hydrate` supports local filesystem blobs via `NDPA_HYDRATION_ROOT` and
Cloudflare R2 via `NDPA_HYDRATION_BACKEND=s3`. The env names use S3 because R2
exposes an S3-compatible API. Use R2 for hosted deployments unless the API host
has a persistent mounted volume.

Same wire contract as the Edge Function — drop-in compatible with the SDK
when you change the base URL.

## Local dev

```bash
cd /Users/kaceburnette/Desktop/ndp
pip install -r server/requirements.txt
cp server/.env.example server/.env
# Fill DATABASE_URL in server/.env (Supabase pooler URL, port 6543)
export $(cat server/.env | xargs)
uvicorn server.main:app --reload --port 8000
```

Smoke test:

```bash
curl -s http://localhost:8000/health
# {"status":"ok","ts":...}
```

Launch env vars:

```bash
export DATABASE_URL='postgresql://postgres.PROJECT:PASS@...pooler.supabase.com:6543/postgres'
export NDPA_HYDRATION_ROOT='/var/lib/ndpa/blobs'
export NDPA_HYDRATION_BACKEND='local'                      # local or s3
export NDPA_PUBLIC_API_BASE_URL='https://api.your-domain.com'
export NDPA_CORS_ORIGINS='https://your-domain.com,https://www.your-domain.com'
export NDPA_ADMIN_TOKEN='long-random-admin-token'
export NDPA_STRIPE_PAYMENT_LINK='https://buy.stripe.com/...'  # optional until billing is live
export OPENAI_API_KEY='sk-...'                               # optional hosted Reasoning
export NDPA_REASONING_MODEL='gpt-5-mini'
```

For Cloudflare R2:

```bash
export NDPA_HYDRATION_BACKEND='s3'
export NDPA_S3_ENDPOINT='https://ACCOUNT_ID.r2.cloudflarestorage.com'
export NDPA_S3_BUCKET='ndpa-raw'
export NDPA_S3_REGION='auto'
export NDPA_S3_ACCESS_KEY_ID='...'
export NDPA_S3_SECRET_ACCESS_KEY='...'
```

Mint the first customer/test key through the API:

```bash
curl -s -X POST "$NDPA_BASE_URL/admin/api-keys" \
  -H "X-NDPA-Admin-Token: $NDPA_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test_user","platform":"local"}'
```

The response includes the raw `ndpa_live_...` token once. Store it immediately.

## Deploy to Fly.io (recommended)

```bash
cd /Users/kaceburnette/Desktop/ndp
fly launch --copy-config --name ndpa-api --region iad --no-deploy
fly secrets set DATABASE_URL='postgres://...'   # paste Supabase pooler URL
fly deploy
```

Verify after deploy:

```bash
curl -s https://ndpa-api.fly.dev/health
```

Point the SDK at the new URL:

```python
from ndpa import Client
c = Client(api_key="ndpa_...", base_url="https://ndpa-api.fly.dev")
```

Or set globally: `export NDPA_BASE_URL='https://ndpa-api.fly.dev'`.

## Why this is faster than the Edge Function

| Component | Edge Function | FastAPI on Fly |
|---|---|---|
| Cold start | 200–400ms (Deno isolate) | 0ms (persistent process) |
| Postgres connection | Per-request open | Pooled (asyncpg, 5–20) |
| Network user→region | depends on user | iad / Fly region close to users |
| Retrieval query | edge cold-start bound | compact metadata query, raw hydration after top-K |
| Process model | Stateless | Long-lived workers (2× uvicorn) |

Report cold miss, stage, warm cache-hit, and hydration latency separately.
Do not claim the cold hosted Postgres path is the voice hot path.

## Auth

API keys: bearer token in `Authorization` header. The server hashes the token
with SHA-256 and looks it up in `ndp_api_keys` by `key_hash`. Revoked keys
(non-null `revoked_at`) return 401.

## Rate limits

In-memory per-process token bucket:
- `/predictions`: 300 req/min/key
- `/stage`: 300 req/min/key
- `/events`: 600 req/min/key

Single-instance for v1. If you scale to multiple Fly machines, swap the
in-memory limiter for Redis.

## Database expectations

This server expects the clean metadata-only schema in
`db/migrations/20260514_clean_metadata_schema.sql`:

- `ndp_api_keys(user_id, platform, key_hash, revoked_at, ...)`
- `ndp_sessions(user_id, end_user_id, session_id, platform, turn_count, last_active)`
  with `(user_id, session_id)` unique
- `ndp_conversations(user_id, end_user_id, session_id, platform, top_terms,
  content_preview, storage_key, content_bytes, updated_at, created_at)`
- `ndp_usage_events(user_id, product, endpoint, units, bytes, latency_ms, created_at)`

`/events` accepts raw text, writes it to `NDPA_HYDRATION_ROOT`, and stores only
compact metadata in Postgres. `/stage` ranks over metadata and caches memory
handles, previews, scores, and storage keys for the next hot read.
`/predictions` checks that staged cache first, then falls back to metadata
ranking on cache miss. `/hydrate` fetches raw context from local storage or
Cloudflare R2 object storage only when explicitly requested.

Do **not** run `db/migrations/20260514_add_tsv_content_gin.sql` on hosted
Supabase. Materialized full-conversation `tsvector` plus a GIN index can exceed
free-tier limits and burn Disk IO quickly. Full-content search is an optional
self-host experiment, not the hosted default path.

## Hosted storage architecture

Hosted NDPA should not use Postgres as the raw memory warehouse. Postgres is
the hot metadata/index tier; raw conversation text belongs in object/blob
storage and is hydrated only for the final top-K candidates.

Product layers:

- **NDPA Predict**: trajectory / behavioral signal -> memory handles,
  previews, scores. Benchmark: PQHR.
- **NDPA Memory**: explicit query -> memory handles, previews, scores.
  Benchmark: LongMemEval / LoCoMo retrieval hit@K.
- **NDPA Hydration**: selected handles / storage keys -> raw full context.
  Benchmarks: latency, bandwidth, storage cost. Not a QA benchmark.
- **NDPA Reasoning**: query + retrieved/hydrated context -> answer. Benchmark:
  LongMemEval E2E LLM judge.

Default hosted shape:

- Postgres: session metadata, timestamps, compact terms, preview text,
  `storage_key`, `content_bytes`, usage/accounting, API keys.
- Blob/object storage: full raw conversation transcripts and portable memory
  payloads. Cloudflare R2 is the preferred hosted
  path; local FS is for development or hosts with persistent volumes.
- Embedded/self-hosted mode: local FS can hold raw content, with Postgres or
  SQLite storing the compact hot index.

Local FS hydration:

```bash
export NDPA_HYDRATION_ROOT=/var/lib/ndpa/blobs
```

Cloudflare R2 hydration:

```bash
export NDPA_HYDRATION_BACKEND=s3
export NDPA_S3_ENDPOINT=https://ACCOUNT_ID.r2.cloudflarestorage.com
export NDPA_S3_BUCKET=ndpa-raw
export NDPA_S3_REGION=auto
export NDPA_S3_ACCESS_KEY_ID=...
export NDPA_S3_SECRET_ACCESS_KEY=...
```

Then call `POST /hydrate` with direct `storage_keys` or with
`memory_handles`/`session_ids` that resolve to `ndp_conversations.storage_key`.

Reasoning:

```bash
curl -s -X POST "$NDPA_BASE_URL/reasoning" \
  -H "Authorization: Bearer $NDPA_TEST_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo","query":"What should I remember?","k":5}'
```

For BYOK Reasoning, omit `OPENAI_API_KEY` on the server and pass
`X-OpenAI-API-Key` on the request. NDPA does not store that provider key.

`db/migrations/20260514_add_hosted_metadata_columns.sql` defines the compact
metadata columns for that direction. Apply it only after the storage recovery
is complete and the database is back under cap.
