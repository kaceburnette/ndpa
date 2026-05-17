# NDPA Supabase Setup

Operator runbook. Not end-user docs.

## Current project

| Field       | Value                        |
|-------------|------------------------------|
| Project ref | `ucertqzejhkdgvkvuxmp`       |
| Region      | us-west-2                    |
| Status      | check dashboard before use   |
| Created     | 2026-05-14                   |

## Architecture constraints

- Postgres is **metadata and control plane only**. No raw conversation text in any column.
- Raw memory lives in local FS for dev or Cloudflare R2 for hosted (`NDPA_HYDRATION_BACKEND`). Fetched by `/hydrate` only.
- No `tsv_content`, no full-content GIN, no `ndp_events` raw event log.
- No benchmark imports into this project.
- No Supabase Storage buckets until explicitly approved.
- RLS is enabled on all `ndp_*` tables with no permissive policies.
  FastAPI connects via service-role `DATABASE_URL` which bypasses RLS by design.
  Direct supabase-js client access is default-deny.

## Required env vars

```
# Supabase Postgres — use Session Pooler (port 6543), not direct connection (5432)
DATABASE_URL=postgres://postgres.[ref]:[password]@aws-0-us-west-2.pooler.supabase.com:6543/postgres

# Blob storage root — gitignored, raw conversation text lives here in local mode
NDPA_HYDRATION_ROOT=memory/blobs          # default; override for production
NDPA_HYDRATION_BACKEND=local              # local or s3

# Cloudflare R2 mode. Env names use S3 because R2 exposes an S3-compatible API.
NDPA_S3_ENDPOINT=https://ACCOUNT_ID.r2.cloudflarestorage.com
NDPA_S3_BUCKET=ndpa-raw
NDPA_S3_REGION=auto
NDPA_S3_ACCESS_KEY_ID=...
NDPA_S3_SECRET_ACCESS_KEY=...

# Launch control plane
NDPA_PUBLIC_API_BASE_URL=https://api.your-domain.com
NDPA_CORS_ORIGINS=https://your-domain.com,https://www.your-domain.com
NDPA_ADMIN_TOKEN=long-random-admin-token
NDPA_STRIPE_PAYMENT_LINK=https://buy.stripe.com/...
OPENAI_API_KEY=sk-...                     # optional hosted Reasoning
```

Current local FastAPI -> Supabase pooler smoke is around ~700ms until latency
work improves it. Do not make low-latency hosted claims yet.

Get `DATABASE_URL` from: Supabase dashboard → NDPA project → Settings → Database → Connection Pooling → Session mode.

The FastAPI server sets `asyncpg` `statement_cache_size=0` for Supabase
pooler/PgBouncer compatibility.

## Bootstrap from scratch

1. Apply the canonical migration:
   ```bash
   # via Supabase MCP apply_migration or psql
   psql "$DATABASE_URL" -f db/migrations/20260514_clean_metadata_schema.sql
   ```
2. Mint a test API key:
   ```bash
   python3 scripts/mint_test_key.py
   # Prints SHA-256 hash + INSERT SQL. Run the INSERT against Supabase.
   # Raw token written to .env as NDPA_TEST_API_KEY.
   ```
3. Start the server:
   ```bash
   uvicorn server.main:app --port 8000
   ```

## Tables

| Table                  | Purpose                                  |
|------------------------|------------------------------------------|
| `ndp_api_keys`         | Bearer token → user_id auth lookup       |
| `ndp_sessions`         | Turn counter + last_active per session   |
| `ndp_conversations`    | Hot metadata: top_terms, preview, key    |
| `ndp_usage_events`     | Per-product usage telemetry              |

`ndp_idf` is intentionally absent. The offline `scripts/compute_idf.py` references it and will fail — do not run that script against this project until ndp_idf is re-added.

## Recovery procedure (if project becomes degraded)

1. Delete the project from the Supabase dashboard.
2. Create a new project (us-west-2 recommended to stay close to Fly deploys).
3. Update `DATABASE_URL` in local `.env` and Fly secrets.
4. Re-apply schema + re-seed key as above.
5. No data migration needed — raw blobs remain in `NDPA_HYDRATION_ROOT` and reconnect automatically via `storage_key`.
