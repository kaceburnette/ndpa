# Vercel Deploy Runbook

This repo can deploy as a single Vercel project:

- Landing page: `/`
- Console: `/console`
- FastAPI API: `/api/*`

The Vercel entrypoint is `api/index.py`, which mounts the production FastAPI
app from `server/main.py`. The static routes are configured in `vercel.json`.

## 1. Prepare Supabase

Apply the clean metadata schema only:

```bash
psql "$DATABASE_URL" -f db/migrations/20260514_clean_metadata_schema.sql
```

Do not apply `20260514_add_tsv_content_gin.sql` to hosted Supabase.

## 2. Prepare Cloudflare R2

Create one private Cloudflare R2 bucket for raw memory payloads. R2 speaks the
S3-compatible API, so the backend env vars are named `NDPA_S3_*`, but the
hosted launch target is R2.

In Cloudflare, create an R2 API token / access key and copy:

- endpoint
- bucket
- access key id
- secret access key

## 3. Set Vercel Environment Variables

Copy `.env.example` into the Vercel project environment settings and fill:

```bash
DATABASE_URL
NDPA_PUBLIC_API_BASE_URL=https://YOUR-DOMAIN/api
NDPA_CORS_ORIGINS=https://YOUR-DOMAIN,https://www.YOUR-DOMAIN
NDPA_ADMIN_TOKEN
NDPA_HYDRATION_BACKEND=s3
NDPA_S3_ENDPOINT
NDPA_S3_BUCKET
NDPA_S3_REGION=auto
NDPA_S3_ACCESS_KEY_ID
NDPA_S3_SECRET_ACCESS_KEY
OPENAI_API_KEY                  # optional hosted Reasoning
NDPA_STRIPE_PAYMENT_LINK        # optional checkout handoff
```

For R2 the endpoint looks like:

```text
https://ACCOUNT_ID.r2.cloudflarestorage.com
```

For Vercel/serverless, keep the pool small:

```bash
DB_POOL_MIN=1
DB_POOL_MAX=5
```

## 4. Deploy

```bash
vercel --prod
```

Or import the GitHub repo in the Vercel dashboard and deploy from `main`.

## 5. Smoke Test

Open:

```text
https://YOUR-DOMAIN/
https://YOUR-DOMAIN/console
https://YOUR-DOMAIN/api/config
https://YOUR-DOMAIN/api/health
```

Mint the first key from the console:

1. Open `/console`.
2. Go to `API Keys`.
3. Paste `NDPA_ADMIN_TOKEN`.
4. Enter a `user_id`.
5. Create the key.
6. Paste the returned `ndpa_live_...` key into the console header.

Then run the playground in this order:

1. `Log sample event`
2. `Run /stage`
3. `Run /predictions`
4. `Run /hydrate`
5. `Run /reasoning`
6. `Refresh usage`
7. `Open checkout` if `NDPA_STRIPE_PAYMENT_LINK` is set

## Important Latency Note

Vercel is enough for launch smoke and customer trials. The staged warm cache is
best on a long-running process because serverless instances are ephemeral. For
the lowest Predict latency, run the same FastAPI app on Fly/Render/Railway or a
dedicated host and set `NDPA_PUBLIC_API_BASE_URL` to that API URL.
