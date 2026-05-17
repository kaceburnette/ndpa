# NDPA Scaling Characteristics

Core architecture works: Core returns handles/previews/scores; Hydration
fetches raw context only when needed.

## Current latency status

Local FastAPI -> Supabase pooler cold prediction smoke is currently around
600-700ms. That is not the voice hot path. The product latency shape is:

1. `/stage` runs prediction before the user asks or between turns.
2. `/predictions` returns the staged handles/previews from process cache.
3. `/hydrate` fetches raw context only for selected handles.

Latest local smoke showed warm `/predictions` cache hits around 1-3ms wall
clock after the first cold miss. Keep reporting cold miss, stage cost, warm
hit, and hydration latency separately.

The table below is a legacy Supabase Edge Function scaling benchmark. Keep it
as historical evidence only until the metadata-only FastAPI path is rerun.

## Predictions API latency by corpus size

| Conversations | Wall P50 | Wall P95 | Wall P99 | Server P50 | Server P95 |
|---------------|----------|----------|----------|------------|------------|
| 50            | 734ms    | 837ms    | 1056ms   | 105ms      | 133ms      |
| 200           | 737ms    | 1064ms   | 1535ms   | 108ms      | 165ms      |
| 500           | 753ms    | 1250ms   | 1985ms   | 114ms      | 143ms      |
| 1000          | 729ms    | 1226ms   | 1767ms   | 115ms      | 170ms      |

**Key insight:** Server-side P50 is flat at ~110ms from 50 to 1000 conversations.
Latency does **not** scale with corpus size — it scales with query term overlap.

This is the GIN-indexed `top_terms` filter at work: we only cosine-score the
subset of conversations that share at least one top-50 term with the query.
For typical queries, that's 5-100 conversations out of millions.

## Why it scales

Without the inverted index: predictions would scan all N conversations per
request. At 100k conversations per user, this becomes a multi-second hot loop
in the Edge Function. Not viable.

With the inverted index:
```sql
SELECT ... FROM ndp_conversations
WHERE user_id = $1
  AND top_terms && $2     -- GIN index hit, O(log N) lookup
ORDER BY started_at DESC
LIMIT 200;
```

Postgres returns only conversations sharing at least one query term. Cosine
scoring runs on this filtered set, typically <100 rows regardless of total
corpus size.

## Wall clock vs server time

The cold hosted path includes network round trips to the API host and Postgres
pooler. NDPA Predict should hide that path by staging ahead. Warm staged reads
are the number to optimize for voice AI; cold misses remain important for
honest fallback reporting.

## Scaling profile for AI platforms

Targets, assuming self-hosted same-region deployment:

| Platform size                | Predictions QPS | Storage    | Self-host config         |
|------------------------------|-----------------|------------|--------------------------|
| 10k users, 100k conv/user    | ~500            | ~25GB      | Single Postgres, 2 edge fn replicas |
| 100k users, 1M conv/user     | ~5k             | ~250GB     | Postgres read replicas, edge fn auto-scale |
| 1M users, 5M conv/user       | ~50k            | ~5TB       | Partition by user_id, regional shards |
| 100M users (Anthropic/OpenAI scale) | ~5M       | ~500TB     | Per-region deployment, tiered storage |

## Ingestion latency (write path)

The local queue + background flusher (`ndp/queue.py`) decouples your hot
path from network I/O entirely:

| Operation                          | Latency |
|------------------------------------|---------|
| Hook write (local JSONL append)    | 5-10ms  |
| Hook total return                  | ~44ms   |
| Background flush to Ingestion API  | 200-500ms (off hot path) |
| Server-side event insert + index   | ~50ms   |

The hot path **never** waits for the network.

## Rate limits

Current configured limits per API key:

- **600 requests/minute** per endpoint
- **2 MB** max body size
- **1000 events** max per `/v1/events` request

At 600 req/min × 1000 events = 600k events/min per key. Sufficient for any
single user. For platform keys serving millions of end users: contact for
custom limits, or self-host (no limits).

## Cost characteristics

Order of magnitude, on managed Supabase:

| Component                   | Cost driver        | At 1M users  |
|-----------------------------|--------------------|--------------|
| Postgres storage            | ~250GB/user × 1M   | $2,500/mo    |
| Edge function invocations   | ~10/user/day × 1M  | $300/mo      |
| Bandwidth (event ingestion) | ~100KB/user/day    | $200/mo      |
| **Total**                   |                    | **~$3,000/mo** |

That's $0.003/user/month at 1M users for the memory layer. For comparison:
- GPT-4 input tokens: $5/1M tokens. A single user's pasted-context-per-session
  at 2k tokens × 5 sessions/day × 30 days = 300k tokens/month = **$1.50/user**.
- Eliminate even 10% of those redundant tokens via NDPA staging:
  **$0.15/user saved**, vs $0.003/user infrastructure cost. **50× ROI.**

## What we haven't benchmarked yet

- **100k+ conversations per user**: benchmark capped at 1000 for the public
  test (insertion time is the bottleneck — 5 minutes for 1000 conversations
  through the public API). Server-side latency should remain flat per
  index design but needs verification at 100k scale.
- **Concurrent QPS**: numbers above are sequential. Edge functions scale
  horizontally but contention on `ndp_rate_limits` increments could become
  a bottleneck at >10k concurrent writes/second. Plan: switch hot counters
  to Redis or use Postgres advisory locks.
- **Multi-tenant under heavy load**: `end_user_id` filter is indexed but
  cross-tenant fairness under saturation hasn't been tested.

Open issues for benchmarking these are tracked at
https://github.com/kaceburnette/ndpa/issues.
