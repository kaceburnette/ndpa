# NDPA — Neural Data Prefetch Architecture

**A predictive memory layer for AI. Stop retrieving. Start predicting.**

> Every memory system today is reactive. You ask, it fetches.
> NDPA learns what you'll need next — and stages it before you ask.

---

## The problem

Every AI memory system on the market (Mem0, Zep, LangMem, MemGPT) does the same thing: store facts, retrieve by vector similarity when queried. It's RAG dressed up as memory. **Reactive.** Expensive. Wasteful — most retrieved context is irrelevant.

Context windows keep growing because we're stuffing them with everything that *might* be relevant. The model burns compute on noise.

## The idea

What if memory worked like a CPU cache instead of a search engine?

NDPA watches behavioral signal — which conversations you return to, what files follow what, what topics cluster, when sessions branch — and **predicts the data you'll need before you ask for it**. By the time the AI starts generating, the right context is already staged.

Not vector similarity. Not keyword search. **Behavioral prediction on a conversation graph.**

| Reactive retrieval (today) | Predictive prefetch (NDPA) |
| --- | --- |
| You ask → it searches → returns | It learns your patterns → stages context |
| Big context, hope something's relevant | Small context, mostly relevant |
| Pay tokens × params per call | Pay near-zero CPU per prediction |
| Bigger model = more context = more $ | Smaller targeted context = less $ |

## Status

**Early. Working. Open.**

- Cloud schema live (Supabase Postgres + RLS)
- Ingestion API live (Supabase Edge Function)
- Python + TypeScript SDKs published
- Prediction kernel running, evaluable
- File-prediction eval: **+8.3 pts hit@5 over the recency baseline** on real Claude Code sessions
- Conversation-level eval scaffolded, awaiting more data
- Conversations cross-platform via SDK (any AI loop can plug in)

This is signal, not proof. The architecture is unoccupied territory and the early numbers point the right way.

## Architecture

```
   ┌──────────────────────────────────────────────────────────────┐
   │  Any AI platform (Claude / OpenAI / Gemini / local / yours)   │
   │                                                                │
   │     conversation loop ──► ndpa SDK ─► POST /v1/events          │
   │                                          │                     │
   └──────────────────────────────────────────┼─────────────────────┘
                                              ▼
                              ┌───────────────────────────────┐
                              │  NDPA Ingestion API           │
                              │  (Supabase Edge Function)     │
                              └───────────────┬───────────────┘
                                              ▼
                              ┌───────────────────────────────┐
                              │  ndp_events / ndp_conversations│
                              │  ndp_sessions / ndp_objects    │
                              │  (Postgres, RLS per user)      │
                              └───────────────┬───────────────┘
                                              ▼
                              ┌───────────────────────────────┐
                              │  Prediction kernel             │
                              │  (recency, frequency, affinity,│
                              │   import / topic proximity)    │
                              └───────────────┬───────────────┘
                                              ▼
                              ┌───────────────────────────────┐
                              │  Staged context returned to    │
                              │  the calling AI before query   │
                              └───────────────────────────────┘
```

## Quickstart

### Python

```bash
pip install ndpa
```

```python
from ndpa import Client

client = Client(api_key="ndpa_...", platform="my_app")

client.log_exchange(
    session_id="chat_42",
    user_message="What's the migration status?",
    assistant_message="80% done, blocking on schema review.",
)
```

### TypeScript / JavaScript

```bash
npm install ndpa
```

```typescript
import { Client } from "ndpa";

const client = new Client({ apiKey: process.env.NDPA_API_KEY!, platform: "my_app" });

await client.logExchange(
  "chat_42",
  "What's the migration status?",
  "80% done, blocking on schema review.",
);
```

Both SDKs send asynchronously by default — your AI loop never waits on the network.

## How it works under the hood

Three things working together:

1. **Collector** — captures conversation turns + tool uses via the SDK (or a Claude Code hook reference integration in this repo).
2. **Storage** — every event lands in Postgres, scoped by user with RLS. Conversations stored verbatim, not chunked.
3. **Prediction kernel** — scores candidates on behavioral features. No LLM calls in the hot path, target <100ms per prediction.

## Repo layout

```
ndp/              core schema, store, compiler
kernel/           prediction engine
collector/        Claude Code reference hook
eval/             eval harness + baselines + synthetic generator
sdk/python/       ndpa Python SDK
sdk/typescript/   ndpa TS/JS SDK
spec/             schema, format, eval protocol
```

## Eval

```bash
python3 -m eval.harness            # file-level eval
python3 -m eval.conversation_eval  # conversation-level eval
```

Current numbers:

```
File-level (84 samples, 6 sessions):
              hit@1   hit@3   hit@5
baseline      0.250   0.393   0.441
heuristic     0.143   0.381   0.524
lift@5: +8.3 pts
```

The kernel beats baseline at staging (hit@5) and loses at single-prediction (hit@1) — which is fine: NDPA's job is to stage a candidate set, not to pick the one perfect file. Numbers tighten as session count grows. Conversation eval is data-starved until more sessions accumulate.

## Roadmap

- [x] Cloud storage with RLS
- [x] Ingestion API (platform-agnostic)
- [x] Python + TypeScript SDKs
- [x] File-prediction eval beats baseline
- [ ] Conversation-prediction eval beats baseline (data-starved)
- [ ] Learned weights (gradient-boosted model on top of heuristic)
- [ ] Warm migration (ingest existing AI history on signup)
- [ ] Redis hot-tier cache for serving predictions
- [ ] Real-time prediction API (`GET /v1/predictions`)
- [ ] Multimodal objects (image, audio, video chunks)

## License

MIT.

## Citation

If you use NDPA in research or product, cite as:

```
Burnette, K. (2026). NDPA: Neural Data Prefetch Architecture —
a predictive memory layer for AI conversations.
https://github.com/kaceburnette/ndp
```
