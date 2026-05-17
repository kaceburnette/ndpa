# NDPA — Neural Data Prefetch Architecture

**Token prediction for the data layer.**

Transformers predict the next token. NDPA predicts the next *data* — the
files, conversations, and context an AI assistant will need on its next
turn — and stages it before the user asks.

Predictive memory index for AI apps. Core returns handles/previews/scores;
Hydration fetches raw context only when needed.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/badge/pip-ndpa-blue.svg)](sdk/python)
[![npm](https://img.shields.io/badge/npm-ndpa-blue.svg)](sdk/typescript)

---

## What it does

Most AI memory tools are reactive: you ask, they search, they return
(Mem0, Zep, MemGPT, LangMem, OpenAI Memory, Claude Memory). That's RAG with a
nicer name.

NDPA has four layers. **Predict** watches behavioral signal and pre-stages
context before a query exists. **Memory** does lightweight explicit-query
retrieval over hot metadata. **Hydration** fetches raw context only for selected
memory handles. **Reasoning** is optional E2E QA over retrieved/hydrated
context.

| Reactive retrieval (everything else) | Predictive staging (NDPA) |
|--------------------------------------|---------------------------|
| You ask → it searches → returns      | Behavioral signal → context staged |
| Big context, hope something fits     | Small targeted context    |
| Embedding pipeline + vector DB + GPU | Compact metadata query + blob hydration |
| Bound to one AI platform             | Cross-platform (ChatGPT → Claude, etc.) |

---

## Eval numbers

Older in-house predict-forward evals on real data, reproducible via
`python3 -m eval.conversation_eval` and `python3 -m eval.novel_topic_eval`.
These predate the public PQHR benchmark and are reported as NDPA Predict
sanity checks, not Memory retrieval or Reasoning QA numbers.

| Eval                                            | Baseline | NDPA   | Lift  |
|-------------------------------------------------|----------|--------|-------|
| File prediction hit@5 (122 samples)             | 31.1%    | 47.5%  | +16.4 |
| Conversation continuity hit@5 (494 samples)     | 31.6%    | 85.2%  | +53.6 |
| **Novel topic prediction hit@5 (480 samples)**  | **17.9%** | **35.4%** | **+17.5** |

The **novel topic** number is the honest one — predicting topics introduced
later in a conversation that don't appear in its history. This is closest
to "predict context the user is about to need but hasn't asked for yet."

## PQHR — The benchmark NDPA owns

**Pre-Query Hit Rate (PQHR)** is NDPA's own benchmark for predict-forward
memory. Unlike LongMemEval and LoCoMo which test reactive retrieval
("given a question, find the answer"), PQHR tests anticipation:

> Given **only the user's prior session trajectory** — no current question,
> no current conversation, no explicit input — can the system stage the
> right past context that will become relevant for their next session?

This is the metric for voice AI, ambient assistants, agentic loops, and any
predict-forward use case. Vector DBs and embedding-based memory systems
**can't be scored on this** because they require an explicit query.

### Public dataset (LongMemEval-S, fully reproducible)

500 synthetic users, 18,773 PQHR samples, BM25 labeler.

| Predictor | Window | pqhr@5 |
|-----------|--------|--------|
| Random (sanity check) | W=7 | 68.7% |
| Recency-only baseline | W=7 | 68.0% |
| **NDPA fixed multi-signal** | **W=7** | **84.6%** |
| **NDPA adaptive multi-signal** | **W=10** | **85.9%** |

**Lift @5: +17.9 pts over random for adaptive W=10; +16.6 pts over recency
for fixed W=7.**

> ⚠️ random@5 is 68.7% on this corpus — small candidate pool per synthetic
> user. Always show random + recency alongside NDPA. The @1 column is the
> more honest headline.

Reproduce: `python3 -m eval.pre_query_hit_rate --public`

### Author's real-user corpus (internal validation, less externally verifiable)

651 samples from 668-conversation multi-platform real-user history.

| Predictor                | pqhr@1 | pqhr@3 | pqhr@5 |
|--------------------------|--------|--------|--------|
| Random (sanity check)    | 0.018  | 0.023  | 0.049  |
| Recency-only baseline    | 0.128  | 0.238  | 0.321  |
| NDPA heuristic           | 0.200  | 0.326  | 0.424  |
| **NDPA pure topic**      | **0.263** | **0.518** | **0.645** |

**Lift @5: +32.4 pts over recency, +59.6 over random.** The harder candidate
pool (more diverse history) collapses random@5 to 4.9% — much stronger lift
signal than the public dataset. Provided as additional evidence; corpus is
private so this is not externally reproducible.

**BM25 is the labeler, not a competitor.** Predictors are scored against
BM25's top-5 picks for each target. BoW cosine (predictor) and BM25 (labeler)
are different formulas on different inputs — independent.

Spec: [docs/PQHR.md](docs/PQHR.md). Leaderboard: [docs/PQHR_LEADERBOARD.md](docs/PQHR_LEADERBOARD.md).

---

## External Benchmarks

NDPA ships as four product layers. The benchmarks below report the measured
layers separately. **Retrieval hit@K and end-to-end QA are different metrics
and are not combined into a single headline.** Hydration is a storage/context
fetch step, not a QA benchmark.

| Product layer       | What it tests                       | Benchmark              |
|---------------------|-------------------------------------|------------------------|
| **NDPA Predict**    | Pre-query staging from trajectory / behavioral signal | PQHR |
| **NDPA Memory**     | Explicit-query retrieval over hot metadata | LongMemEval / LoCoMo hit@K |
| **NDPA Hydration**  | Raw context fetch for selected memory handles | Latency / cost / storage efficiency |
| **NDPA Reasoning**  | Optional answer layer over retrieved/hydrated context | LongMemEval E2E (LLM judge) |

Current Hydration status: local FS is implemented as an optional raw context
fetch layer. Latest local API smoke: `/hydrate` p50 0.49ms, p95 0.88ms.
Object-storage latency is still pending. Hydration is not a QA benchmark.

### Retrieval (NDPA Memory) — hit@5 on the maintained public datasets

| System            | LongMemEval retrieval@5 | LoCoMo retrieval@5 | Method                          | Cost / 1M queries     |
|-------------------|-------------------------|--------------------|---------------------------------|-----------------------|
| **NDPA Memory (local metadata-only)** | **90.6%** | **85.1%** | lexical metadata + temporal/entity/multi-hop/MMR boosts | Postgres hot tier + blob storage |

These are the latest verified metadata-only retrieval numbers from
`eval/metadata_retrieval_results.json` and `eval/locomo_results.json`.

Most competing memory systems publish full-stack QA accuracy, not retrieval
hit@K. Putting their full-stack numbers in this column would be a mixed-metric
comparison. **NDPA retrieval hit@K is not a claim about competitor full-stack
QA systems.** Our E2E results are reported separately in the next section.

### End-to-end QA (NDPA Reasoning, optional) — LongMemEval-S, LLM judge

NDPA Reasoning is an optional premium layer that can run over retrieved and
hydrated context for question types that need answer generation. All numbers
below use LLM-as-judge grading; the latest full repaired 500-row run used
`gpt-5-mini` generation and `gpt-4.1-mini` judging as recorded in the result
JSON files.

| Pipeline                                         | LongMemEval E2E (LLM judge) | Notes |
|--------------------------------------------------|-----------------------------|-------|
| Baseline E2E (no extraction, prior prompt)       | 46.0%                       | first regrade of original raw-context pipeline |
| Patched raw E2E (no extraction, prompt fixes)    | 47.6%                       | reader prompt fixes only |
| Routed Reasoning (tested variant)                | 49.8%                       | preference-question routing; improves preference behavior but lowers overall |
| Earlier Reasoning extraction run                 | 51.8%                       | historical best before full-context repair |
| **Reasoning repaired full run (current best)**   | **85.2%**                   | full 500-row LongMemEval-S run; blank generation caps repaired |

The 85.2% repaired full run is the current best E2E number. The routed 49.8%
variant is **not** the headline number; it is retained for methodological
transparency because it improved one preference-handling path while reducing
overall accuracy.

We do not directly compare these numbers to MemPalace, OMEGA, Mem0, or
similar full-stack memory systems. Those systems do LLM fact extraction at
ingestion time, which is a different architectural choice with a different
cost profile. NDPA Reasoning is the optional E2E layer for customers who
want stronger answer accuracy on top of NDPA Memory retrieval without
changing the ingest model.

### What we claim, plainly

1. **Retrieval quality** for NDPA Memory: latest metadata-only runs show 90.6%
   LongMemEval hit@5 and 85.1% LoCoMo hit@5.
2. **Predict-forward staging** for NDPA Predict: PQHR is a category competing
   memory systems do not benchmark on because they require an explicit
   query.
3. **End-to-end QA** for NDPA Reasoning: 85.2% LongMemEval-S LLM-judge
   accuracy on the current repaired full run. We do not claim 90% yet and we
   do not claim parity with every full-stack incumbent on QA.

For high-volume platforms — chat apps, voice AI, code agents, ambient
assistants — retrieval cost and latency dominate. Predict latency claims refer
to staged/warm reads; cold Local FastAPI -> Supabase pooler smoke is around
~700ms and should not be described as fast. NDPA Hydration fetches raw context
only when needed.
For applications that also need answer accuracy on top, NDPA Reasoning is the
optional LLM-priced layer.

### LongMemEval

Latest metadata-only Memory retrieval run.

`python3 -m eval.metadata_retrieval_eval --benchmark longmemeval` on `xiaowu0162/longmemeval-cleaned`
(`longmemeval_s_cleaned.json`), scored as evidence-session recall:

| Scope | n | hit@1 | hit@3 | hit@5 |
|-------|---:|------:|------:|------:|
| Overall | 500 | 77.6% | 86.8% | **90.6%** |

LongMemEval category rows should stay out of launch copy until the current
metadata-only runner records per-category values.

The LongMemEval paper reports retrieval baselines such as BM25, Contriever,
GTE, and Stella. It does not publish Mem0/Zep/MemGPT category numbers in the
benchmark paper, so no head-to-head claim is made here.

### LoCoMo

Latest local metadata-only Memory retrieval run.

`python3 -m eval.locomo_runner` on the maintained public
`snap-research/locomo` `locomo10.json`, scored as evidence-session recall:

| Category | n | hit@1 | hit@3 | hit@5 |
|----------|---:|------:|------:|------:|
| Overall | 1,982 | 61.5% | 79.0% | **85.1%** |
| Adversarial | 446 | 63.9% | 80.5% | 86.1% |
| Open-domain | 841 | 62.0% | 78.6% | 84.8% |
| Temporal | 321 | 75.4% | 88.2% | 91.9% |
| Single-hop | 282 | 47.5% | 75.9% | 85.1% |
| Multi-hop | 92 | 39.1% | 53.3% | 59.8% |

### LMSYS-Chat-1M

`python3 -m eval.lmsys_validation` is blocked without HuggingFace
authentication: `lmsys/lmsys-chat-1m` is a gated dataset. The runner writes
`eval/lmsys_results.json` with `status: "blocked"` instead of inventing a
number. Run it with an authenticated HuggingFace session or pass a local
JSON/JSONL sample via `--input`.

### Benchmark Workstream

NDPA now reports separate measurement lanes: PQHR for Predict,
LongMemEval/LoCoMo hit@K for Memory, latency/cost/storage efficiency for
Hydration, and LongMemEval E2E LLM-judge for Reasoning. Next benchmark work
should improve those lanes without mixing their claims:

1. **Competitor PQHR baselines**: run Mem0, Zep, Letta, and simple
   vector/BM25 baselines on PQHR using the same trajectory-only input policy.
2. **Retrieval quality ablations**: multi-hop query splitting, date-aware
   boosting, entity-aware boosting, and MMR diversity on LongMemEval/LoCoMo
   hit@K.
3. **Reasoning QA ablations**: compare raw context, extraction, and routed
   variants with LLM-as-judge, while reporting them only as NDPA Reasoning
   E2E numbers.
4. **Harder PQHR corpora**: add public datasets where random@5 is lower and
   candidate pools are larger, so the predict-forward signal is easier to
   scrutinize.
5. **TF-IDF + bigrams + better stopwords**: low-risk retrieval quality
   improvements expected to push hit@5 into the mid-to-high 80s on
   LongMemEval. Single PR, ablatable.
6. **Cross-encoder reranker experiment**: top-50 BoW retrieval to top-5
   reranked. Report any gains under the affected lane only, with latency and
   cost attached.

---

## 60-second quickstart

```bash
git clone https://github.com/kaceburnette/ndpa
cd ndpa
python3 demo/demo.py
```

That's it. Runs locally, no network needed, shows real prediction numbers
on synthetic data.

To use it in your AI app:

```python
pip install ndpa
```

```python
from ndpa import Client
c = Client(api_key="ndpa_...", platform="myapp")

# Log every turn
c.log_turn(session_id, "user", user_msg)
c.log_turn(session_id, "assistant", reply)

# Before generating the next response, get predicted context
result = c.get_predictions(session_id, k=3)
context = "\n".join(p["content"][:500] for p in result["predictions"])
# Pass `context` as system prompt to your LLM
```

5 lines. Zero embeddings. Sub-2-second predictions over 600+ conversations.

---

## How it works

```
User message → Hook captures event in <50ms
            → Background queue posts to Ingestion API
            → Behavioral kernel scores past conversations
              + files by recency, frequency, topic overlap
            → Predictions API returns top-K context
            → Stage in system prompt BEFORE generation
```

The kernel:
- **Recency** (0.35) — how recently accessed
- **Frequency** (0.25) — how often in this session
- **Extension affinity** (0.20) — test/spec/type pairings
- **Import proximity** (0.20) — files imported by currently-hot files

Plus BoW cosine topic match for conversation prediction. Embeddings are not
required, and there is no LLM in the hot path. Runs in <2s on 600+
conversations.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full system diagram.

---

## Hosted metadata-first architecture

The core architecture works: Postgres is the hot metadata/index tier, not the
raw memory warehouse. Default Predict and Memory responses return handles,
previews, scores, timestamps, and storage keys:

```
Postgres hot tier: session_id, started_at, platform, top_terms,
                   content_preview, storage_key, content_bytes, scores
Blob/object/local storage: raw conversations, full files, .ndp bundles
```

That shape keeps the hot path cheap and quota-safe. **Predict** and
**Memory** select likely context from compact metadata. **Hydration** fetches
raw context by `storage_key` only after top-K selection. **Reasoning** is an
optional LLM layer over retrieved/hydrated context. Full raw-text Postgres
indexes such as materialized `tsv_content` are optional self-hosting
experiments, not the hosted/default path.

Current local FastAPI -> Supabase pooler smoke is around ~700ms until the
latency work improves it. Do not make low-latency hosted claims yet.
Predict latency should be reported as staged/warm latency; cold Supabase
pooler misses are fallback cost, not the hot-path claim.

---

## What's in the box

- **Collector** (`collector/hook.py`) — Claude Code reference integration
- **Kernel** (`kernel/predictor.py`) — heuristic prediction engine
- **Queue** (`ndp/queue.py`) — local JSONL queue + background flusher
- **Bundle** (`ndp/bundle.py`) — `.ndp` portable format (export/import)
- **APIs** — `events`, `predictions`, `health` (Supabase Edge Functions)
- **Python SDK** (`sdk/python/`) — zero dependencies
- **TypeScript SDK** (`sdk/typescript/`) — zero dependencies
- **Eval harness** (`eval/`) — file + conversation + novel-topic benchmarks
- **ChatGPT importer** (`eval/chatgpt_import.py`) — bulk import history
- **Demo** (`demo/demo.py`) — one-command reproducible demo

---

## The `.ndp` format

Portable session bundles. Move your behavioral memory between platforms.

```bash
python3 -m ndp.bundle export <session_id>           # → session.ndp
python3 -m ndp.bundle info session.ndp              # show manifest
python3 -m ndp.bundle import session.ndp            # restore
```

A `.ndp` file is a zip with `manifest.json`, `sessions/`, `conversations/`,
`objects/`. No vendor lock-in. Standard you can hand off, archive, or share.

---

## Privacy

| Data                       | Stored locally | Uploaded by default |
|----------------------------|----------------|---------------------|
| File paths                 | yes            | yes                 |
| Tool names, timestamps     | yes            | yes                 |
| Conversation text          | yes            | yes                 |
| **File contents**          | yes            | **NO** (opt-in)     |

Set `upload_file_contents: true` in `~/.ndp/config.json` to opt in.

Full details: [docs/SECURITY.md](docs/SECURITY.md).

---

## Production characteristics

- **Hook latency**: 44ms (queue write, never blocks)
- **API rate limit**: 600 req/min/key
- **Body limit**: 2MB, 1000 events/call
- **Predictions**: 1.2s for 600 conversations
- **Health endpoint**: `GET /v1/health`
- **Hosted storage**: compact metadata in Postgres, raw conversations in blob storage
- **Self-hostable**: Postgres hot tier + local/object storage

---

## For platform engineers

Wiring NDPA into your AI product:

- [docs/INTEGRATION.md](docs/INTEGRATION.md) — 30s integration, patterns, API contracts
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — system diagram, scaling, latency
- [docs/VERCEL_DEPLOY.md](docs/VERCEL_DEPLOY.md) — one-project Vercel deploy runbook
- [docs/VOICE_AI.md](docs/VOICE_AI.md) — voice AI integration guide
- [docs/SECURITY.md](docs/SECURITY.md) — threat model, GDPR, self-host
- [docs/COMPARISON.md](docs/COMPARISON.md) — vs Mem0, Zep, MemGPT, OpenAI Memory

---

## License

MIT. Use it, fork it, sell it. Just don't claim you invented it.

## Why

I built this in a week because nobody else had. The big labs are circling the
idea in 2026 workshop papers. Mem0/Zep/Letta are all reactive. The core
architecture demonstrates behavioral prediction across sessions for AI
context.

Open sourced before any of them shipped it.

If you're at an AI company and this would save your users time and your
inference $: integrate it. The SDK is 5 lines. If you want this architecture
in your stack, fork the repo or talk to me.
