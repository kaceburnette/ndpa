# NDPA Launch Drafts

Use the same claim discipline everywhere: NDPA has four layers, and each layer
has its own metric. Do not compare NDPA retrieval hit@K to competitor
full-stack QA numbers.

Core message: Predict-first memory for realtime AI apps. NDPA keeps likely
context hot before the user asks. Predict and Memory return
handles/previews/scores; Hydration fetches raw context only when needed;
Reasoning is the optional LLM answer layer.

## Launch Wiring Checklist

Backend:

- Apply `db/migrations/20260514_clean_metadata_schema.sql` to the clean
  Supabase project.
- Set `DATABASE_URL` to the Supabase pooler URL.
- Set `NDPA_HYDRATION_BACKEND=s3` for hosted Cloudflare R2 deploys and configure
  `NDPA_S3_ENDPOINT`, `NDPA_S3_BUCKET`, `NDPA_S3_REGION`,
  `NDPA_S3_ACCESS_KEY_ID`, and `NDPA_S3_SECRET_ACCESS_KEY`.
  The env names say S3 because R2 exposes an S3-compatible API.
- Set `NDPA_PUBLIC_API_BASE_URL`, `NDPA_CORS_ORIGINS`, and
  `NDPA_ADMIN_TOKEN`.
- Optional hosted Reasoning: set `OPENAI_API_KEY` and
  `NDPA_REASONING_MODEL=gpt-5-mini`.
- Optional billing: set `NDPA_STRIPE_PAYMENT_LINK`.
- Deploy FastAPI and verify `/health`, `/config`, `/admin/api-keys`,
  `/events`, `/stage`, `/predictions`, `/hydrate`, `/reasoning`, `/usage`,
  and `/checkout`.

Frontend:

- Deploy with Vercel using `docs/VERCEL_DEPLOY.md`.
- Vercel serves landing/console plus FastAPI under `/api/*`.
- Verify `/console` opens the console route.
- Paste API base URL and a minted key into the console, then run the
  playground calls.

Do not use Supabase as the raw memory warehouse. Postgres is metadata and
control plane only; hosted raw memory belongs in Cloudflare R2.

## HN post

**Title options**

- Show HN: NDPA, predict-first memory for AI agents
- Show HN: I built a memory layer that predicts context before the user asks
- Show HN: Token prediction for the data layer

**Post**

I built NDPA, an open-source predictive memory index for AI apps.

Most memory systems are reactive: the user asks, the app searches, then the
model gets context. NDPA adds a pre-query layer. It watches prior session
trajectory and stages likely context before the next request exists.

The architecture is four layers:

- Predict: pre-query staging, measured by PQHR.
- Memory: explicit-query retrieval over hot metadata, measured by hit@K.
- Hydration: raw context fetch for selected handles, measured by
  latency/storage.
- Reasoning: optional LLM answer layer, measured by E2E LLM judge.

Current verified numbers:

- Predict adaptive W=10 public PQHR@5: 85.9%.
- Predict fixed W=7 multi-signal public PQHR@5: 84.6%.
- Memory metadata-only retrieval: 90.6% LongMemEval retrieval@5 and 85.1%
  LoCoMo retrieval@5.
- Hydration: local-FS optional raw context fetch, p50 0.49ms / p95 0.88ms
  in local API smoke; not a QA benchmark.
- LongMemEval-S E2E LLM judge: 85.2% current repaired Reasoning run.

The hosted architecture is metadata-first: Postgres stores handles, previews,
scores, top terms, timestamps, and storage keys. Raw conversations live in
Cloudflare R2 and are hydrated only after top-K selection.
Local FastAPI -> Supabase pooler smoke is currently around ~700ms until the
latency work improves it.
Predict latency should be described as staged/warm latency, not cold Supabase
pooler latency.

Repo: https://github.com/kaceburnette/ndpa

I would especially like feedback from people building voice agents, coding
agents, and long-term chat platforms.

## X thread

1. I open-sourced NDPA: predict-forward memory for AI agents.

2. Transformers predict the next token. NDPA predicts the next data: the past
conversation, file, or context an agent is likely to need next.

3. Most AI memory is reactive: query -> search -> retrieve. NDPA adds a layer
before that: Predict.

4. The product model is four layers:
Predict -> Memory -> Hydration -> Reasoning.

5. Metrics stay separated:
PQHR for Predict.
hit@K for Memory.
latency/storage for Hydration.
E2E LLM judge for Reasoning.

6. Verified numbers:
Predict adaptive W=10 public PQHR@5: 85.9%.
Predict fixed W=7 multi-signal public PQHR@5: 84.6%.
Memory metadata-only retrieval: 90.6% LongMemEval retrieval@5 and 85.1% LoCoMo
retrieval@5.
Hydration: local-FS optional raw context fetch, p50 0.49ms / p95 0.88ms in
local API smoke; not QA.
LongMemEval-S E2E LLM judge: 85.2%.

7. Retrieval hit@K is not a competitor QA claim. Different metric. Different
layer.

8. NDPA Core is metadata-first: Postgres keeps handles, previews, top terms,
scores, and storage keys hot. Raw context lives in blob storage and hydrates
only after top-K selection.

9. Local FastAPI -> Supabase pooler smoke is currently around ~700ms. The
architecture works; latency work is still in progress.

10. Predict latency is staged/warm latency. Do not claim cold Supabase is fast.

11. This matters most where latency is brutal: voice AI, code agents, ambient
assistants, and high-volume chat platforms.

12. Repo: https://github.com/kaceburnette/ndpa

## Voice AI outreach

Subject: Predict-forward memory for voice agents

I'm building NDPA, an open-source memory layer that stages likely context
before the user asks.

For voice AI, the useful distinction is latency: reactive memory waits for an
utterance, searches, and then injects context. NDPA adds a Predict layer that
can pre-stage likely context between turns, then use Memory/Hydration only
when the current utterance needs exact detail.

Verified numbers:

- Predict: 85.9% adaptive W=10 public PQHR@5; 84.6% fixed W=7 multi-signal public PQHR@5.
- Memory: metadata-only retrieval is 90.6% LongMemEval retrieval@5 and 85.1% LoCoMo retrieval@5.
- Hydration: local-FS optional raw context fetch is p50 0.49ms / p95 0.88ms in local API smoke; not a QA benchmark.
- Reasoning: 85.2% LongMemEval-S E2E LLM judge.

The hosted design is metadata-first: Postgres stores handles/previews/scores
and raw transcripts hydrate from object storage only after selection.
Local FastAPI -> Supabase pooler smoke is currently around ~700ms until the
latency work improves it.
Predict latency should be framed as staged/warm latency; cold Supabase is not
the fast path.

Repo: https://github.com/kaceburnette/ndpa

Worth a quick technical look for your latency-sensitive memory path?

## Code agent outreach

Subject: Predict-forward context for coding agents

I'm open-sourcing NDPA, a memory layer that predicts which files,
conversations, and prior sessions an agent will need next.

For coding agents, the pitch is simple: don't wait until the model asks for a
file or context trail. Log tool use and prompts, stage likely context before
the next turn, and keep raw hydration optional.

Metrics are separated by layer:

- Predict: PQHR.
- Memory: LongMemEval/LoCoMo retrieval hit@K.
- Hydration: latency/storage.
- Reasoning: optional E2E LLM judge.

Current verified numbers: 85.9% adaptive W=10 public PQHR@5, 84.6% fixed W=7
multi-signal public PQHR@5, Memory metadata-only retrieval at 90.6%
LongMemEval retrieval@5 and 85.1% LoCoMo retrieval@5, and 85.2% LongMemEval-S
E2E LLM judge. Hydration is local-FS optional raw context fetch at p50 0.49ms
/ p95 0.88ms in local API smoke, not a QA benchmark.

Repo: https://github.com/kaceburnette/ndpa

I would value feedback on the code-agent integration surface.

## Chat platform outreach

Subject: Cross-session memory without putting raw archives in the hot path

I'm building NDPA, a predict-forward memory layer for AI chat platforms.

NDPA separates memory into four layers:

- Predict: stage likely context before a query exists.
- Memory: retrieve handles/previews/scores for explicit queries.
- Hydration: fetch raw context only for selected handles.
- Reasoning: optional answer generation over retrieved/hydrated context.

The Core architecture is metadata-first. Postgres stores compact metadata:
handles, previews, scores, top terms, timestamps, storage keys. Raw
conversations live in Cloudflare R2 and hydrate only when selected.
Local FastAPI -> Supabase pooler smoke is currently around ~700ms until the
latency work improves it.
Predict latency should be framed as staged/warm latency; cold Supabase is not
the fast path.

Verified numbers: 85.9% adaptive W=10 public PQHR@5, 84.6% fixed W=7
multi-signal public PQHR@5, Memory metadata-only retrieval at 90.6%
LongMemEval retrieval@5 and 85.1% LoCoMo retrieval@5, 85.2% LongMemEval-S E2E
LLM judge. Hydration is local-FS optional raw context fetch at p50 0.49ms /
p95 0.88ms in local API smoke, not a QA benchmark.

Repo: https://github.com/kaceburnette/ndpa

This should be especially relevant if your users repeat context across long
conversation histories or across multiple chat surfaces.
