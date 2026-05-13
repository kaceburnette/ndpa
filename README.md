# NDPA — Neural Data Prefetch Architecture

**Token prediction for the data layer.**

Transformers predict the next token. NDPA predicts the next *data* — the
files, conversations, and context an AI assistant will need on its next
turn — and stages it before the user asks.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/badge/pip-ndpa-blue.svg)](sdk/python)
[![npm](https://img.shields.io/badge/npm-ndpa-blue.svg)](sdk/typescript)

---

## What it does

Every AI memory tool today is reactive: you ask, it searches, it returns
(Mem0, Zep, MemGPT, LangMem, OpenAI Memory, Claude Memory). That's RAG with a
nicer name.

NDPA is **predictive**: it watches behavioral signal — which files you
touched, what topics you've been discussing, what past sessions are about to
become relevant — and pre-stages the right context before you ask.

| Reactive retrieval (everything else) | Predictive staging (NDPA) |
|--------------------------------------|---------------------------|
| You ask → it searches → returns      | Behavioral signal → context staged |
| Big context, hope something fits     | Small targeted context    |
| Embeddings + vector DB + GPU         | BoW cosine, single Postgres query |
| Bound to one AI platform             | Cross-platform (ChatGPT → Claude, etc.) |

---

## Eval numbers

Real data, reproducible (`python3 -m eval.conversation_eval`, `python3 -m eval.novel_topic_eval`):

| Eval                                            | Baseline | NDPA   | Lift  |
|-------------------------------------------------|----------|--------|-------|
| File prediction hit@5 (122 samples)             | 31.1%    | 47.5%  | +16.4 |
| Conversation continuity hit@5 (494 samples)     | 31.6%    | 85.2%  | +53.6 |
| **Novel topic prediction hit@5 (480 samples)**  | **17.9%** | **35.4%** | **+17.5** |

The **novel topic** number is the honest one — predicting topics introduced
later in a conversation that don't appear in its history. This is closest
to "predict context the user is about to need but hasn't asked for yet."

## External Benchmarks

Full retrieval runs against external memory benchmarks:

### LongMemEval

`python3 -m eval.longmemeval_runner` on `xiaowu0162/longmemeval-cleaned`
(`longmemeval_s_cleaned.json`), scored as evidence-session recall:

| Category | n | hit@1 | hit@3 | hit@5 |
|----------|---:|------:|------:|------:|
| Overall | 500 | 56.8% | 75.8% | 80.4% |
| Single-session assistant | 56 | 87.5% | 92.9% | 94.6% |
| Knowledge update | 72 | 70.8% | 93.1% | 95.8% |
| Temporal reasoning | 127 | 58.3% | 77.2% | 81.9% |
| Multi-session | 121 | 55.4% | 80.2% | 84.3% |
| Single-session user | 64 | 46.9% | 68.8% | 76.6% |
| Abstention | 30 | 26.7% | 40.0% | 53.3% |
| Single-session preference | 30 | 16.7% | 30.0% | 30.0% |

The LongMemEval paper reports retrieval baselines such as BM25, Contriever,
GTE, and Stella. It does not publish Mem0/Zep/MemGPT category numbers in the
benchmark paper, so no head-to-head claim is made here.

### LoCoMo

`python3 -m eval.locomo_runner` on the maintained public
`snap-research/locomo` `locomo10.json`, scored as evidence-session recall:

| Category | n | hit@1 | hit@3 | hit@5 |
|----------|---:|------:|------:|------:|
| Overall | 1,982 | 35.2% | 58.6% | 68.2% |
| Adversarial | 446 | 39.7% | 63.7% | 74.2% |
| Open-domain | 841 | 39.1% | 62.8% | 71.0% |
| Temporal | 321 | 29.0% | 50.5% | 59.8% |
| Single-hop | 282 | 27.3% | 53.2% | 66.3% |
| Multi-hop | 92 | 22.8% | 40.2% | 47.8% |

### LMSYS-Chat-1M

`python3 -m eval.lmsys_validation` is blocked without HuggingFace
authentication: `lmsys/lmsys-chat-1m` is a gated dataset. The runner writes
`eval/lmsys_results.json` with `status: "blocked"` instead of inventing a
number. Run it with an authenticated HuggingFace session or pass a local
JSON/JSONL sample via `--input`.

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

Plus BoW cosine topic match for conversation prediction. No GPU. No LLM in
the hot path. Runs in <2s on 600+ conversations.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full system diagram.

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
- **Self-hostable**: any Postgres + edge function runtime

---

## For platform engineers

Wiring NDPA into your AI product:

- [docs/INTEGRATION.md](docs/INTEGRATION.md) — 30s integration, patterns, API contracts
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — system diagram, scaling, latency
- [docs/SECURITY.md](docs/SECURITY.md) — threat model, GDPR, self-host
- [docs/COMPARISON.md](docs/COMPARISON.md) — vs Mem0, Zep, MemGPT, OpenAI Memory

---

## License

MIT. Use it, fork it, sell it. Just don't claim you invented it.

## Why

I built this in a week because nobody else had. The big labs are circling the
idea in 2026 workshop papers. Mem0/Zep/Letta are all reactive. Nothing in
production does behavioral prediction across sessions for AI context.

Open sourced before any of them shipped it.

If you're at an AI company and this would save your users time and your
inference $: integrate it. The SDK is 5 lines. If you want to ship this
faster than your competitors, fork the repo or talk to me.
