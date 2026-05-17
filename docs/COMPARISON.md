# NDPA vs the Memory Landscape

NDPA sits one layer above vector stores and conventional memory systems. It
has four product layers, and each layer needs its own comparison.

Predictive memory index for AI apps. Core returns handles/previews/scores;
Hydration fetches raw context only when needed.

| NDPA layer | What it does | Correct benchmark |
|---|---|---|
| **Predict** | Stages context from trajectory before a query exists | PQHR |
| **Memory** | Retrieves memory handles/previews/scores for an explicit query | LongMemEval / LoCoMo retrieval hit@K |
| **Hydration** | Fetches raw context for selected handles | Latency / cost / storage efficiency |
| **Reasoning** | Answers questions over retrieved/hydrated context | LongMemEval E2E LLM judge |

Do not compare retrieval hit@K to competitor full-stack QA accuracy. Those
are different metrics. Do not compare Hydration to QA systems either;
Hydration is a storage/context-fetch step, not an answer layer.

## Side-by-Side

| Capability | NDPA | Mem0 / Zep / Letta / LangMem | OpenAI / Claude Memory |
|---|---|---|---|
| Predict-forward staging before query | Yes | Not generally published as a standalone API/benchmark | No public standalone benchmark |
| Reactive retrieval | Yes | Yes | Product-internal |
| Raw context hydration | Yes | Varies | Product-internal |
| Optional E2E QA layer | Yes | Yes | Product-internal |
| Cross-platform memory | Yes | Usually app/framework scoped | Provider scoped |
| Portable `.ndp` bundle | Yes | No common portable format | No |
| No LLM in retrieval hot path | Yes for Predict/Memory/Hydration | Usually no; many use embeddings/graphs/LLM extraction | Not disclosed |
| Self-hostable | Yes | Varies | No |

## Architecture Difference

NDPA Hosted Core is metadata-first. Postgres stores compact hot data:
handles, previews, timestamps, top terms, scores, storage keys, and byte
counts. Raw conversations and `.ndp` bundles live in blob/object/local storage
and are fetched by Hydration only after top-K selection.

That means NDPA's default hosted path is not a full raw-text warehouse and does
not require a materialized full-content GIN index. Self-hosters can experiment
with full-text indexes, embeddings, or fact extraction, but those belong to
their chosen Memory/Reasoning layer and should be measured under the matching
metric.

## What NDPA Is Good At

- Predicting/staging context before the user asks, measured by PQHR.
- Returning memory handles through compact metadata retrieval, measured by
  LongMemEval and LoCoMo retrieval hit@K. Predict latency claims should refer
  to staged/warm reads, not cold Supabase pooler misses.
- Hydrating raw text from blob storage only after top-K selection.
- Portable memory bundles via `.ndp`.
- Optional E2E QA on top of retrieval when customers need answer generation.

## What NDPA Is Not Claiming

- We do not claim NDPA Reasoning beats Mem0, OMEGA, ByteRover, or MemPalace
  on full-stack QA.
- We do not claim NDPA retrieval hit@K is directly comparable to competitor
  full-stack QA accuracy.
- We do not claim NDPA Hydration is a QA benchmark. It is measured on latency,
  bandwidth, and storage cost.
- We do not claim embeddings are always worse. Embeddings may help semantic
  paraphrase recall; NDPA's core advantage is cost, portability, and
  predict-forward staging.

## Current NDPA Numbers

| Layer | Metric | Result | Notes |
|---|---:|---:|---|
| Predict | PQHR@5 | 85.9% | adaptive W=10 public LongMemEval-S; fixed W=7 multi-signal is 84.6% |
| Memory | LongMemEval retrieval@5 | 90.6% | latest metadata-only run |
| Memory | LoCoMo retrieval@5 | 85.1% | latest local metadata-only run |
| Hydration | local-FS raw context fetch | 0.49ms p50 / 0.88ms p95 | local API smoke; not QA |
| Reasoning | LongMemEval-S E2E LLM judge | 85.2% | current repaired full 500-row run; optional premium E2E layer |

Older in-house file/conversation/novel-topic evals remain useful sanity
checks for Predict, but PQHR is the public benchmark to cite first.

## When To Use What

| If you need... | Use |
|---|---|
| Stage relevant context before a query exists | NDPA Predict |
| Get memory handles/previews for an explicit query | NDPA Memory |
| Fetch raw full context for selected handles | NDPA Hydration |
| Answers over retrieved memories | NDPA Reasoning or another QA layer |
| Heavy semantic paraphrase search | Mem0/Zep/vector search, or an embedding add-on |
| Agent reflection / summarization workflows | Letta/Zep-style systems |

NDPA can compose with those systems. For example, use NDPA Predict for
pre-staging and Mem0/Zep for fact extraction if that is the product shape.
