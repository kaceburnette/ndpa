# NDPA vs the Memory Landscape

NDPA isn't competing with vector stores. It sits one layer up. Most existing
"AI memory" tools are reactive — they retrieve when asked. NDPA is predictive
— it stages context before the user asks.

## Side-by-side

|                          | NDPA          | Mem0   | Zep    | MemGPT/Letta | LangMem | OpenAI Memory | Claude Memory |
|--------------------------|---------------|--------|--------|--------------|---------|---------------|---------------|
| **Retrieval model**      | Predictive    | Reactive | Reactive | Reactive  | Reactive | Reactive | Reactive |
| **Cross-platform**       | Yes (any LLM) | Single | Single | Single | Single | OpenAI only | Anthropic only |
| **Behavioral signal**    | Yes (file + conv) | No | No | No | No | No | No |
| **Cold start**           | Onboard via .ndp import or static analysis | Empty | Empty | Empty | Empty | Empty | Empty |
| **Self-host**            | Yes (Supabase) | Yes | Yes | Yes | Yes | No | No |
| **Open source**          | MIT | Apache | Apache | Apache | MIT | No | No |
| **Embeddings required?** | No | Yes | Yes | Yes | Yes | Unknown | Unknown |
| **Latency at retrieval** | <2s (no GPU) | 200-500ms (embed + vector hop) | similar | similar | similar | TBD | TBD |
| **Portable format**      | .ndp bundle | No | No | No | No | No | No |
| **GDPR delete**          | One SQL stmt | Yes | Yes | Yes | Yes | Manual | Manual |
| **License cost**         | Free/MIT | Free OSS / paid hosted | Paid hosted | Free OSS | Free OSS | Bundled w/ Plus | Bundled w/ Pro |

## What NDPA is good at

- **Predicting context before the user asks**, from behavioral patterns
- **Cross-platform memory** (your ChatGPT history informs your Claude
  session)
- **Fast retrieval** — no embeddings, no vector DB, single Postgres query
- **Open + portable** — `.ndp` file format means no vendor lock-in
- **Cheap to run** — BoW cosine + recency decay. Pennies per million events.

## What NDPA is NOT good at (yet)

- **Semantic similarity on very specific phrases** — embeddings beat BoW
  on "find the conversation where I mentioned a particular concept by
  paraphrase." If you need that, add embeddings as a layer (we may ship
  this as optional).
- **Sub-second predictions** at scale — we're at 1-2s for 600+
  conversations. Embedding-indexed vector search is faster for very large
  corpora (10M+ conversations per user, which doesn't exist yet).
- **Conversational memory with summarization** — Zep specializes in
  collapsing long conversations into structured memory. NDPA stores raw
  conversation text and predicts at retrieval time.

## When to use what

| If you need...                                             | Use         |
|------------------------------------------------------------|-------------|
| Predict the next file the user will open                  | **NDPA**    |
| Stage relevant past chats before the user asks            | **NDPA**    |
| Cross-platform memory (multi-LLM)                         | **NDPA**    |
| Semantic search on facts ("when did I mention X?")        | Mem0 / Zep  |
| Long-term agent memory with reflection / summarization    | Letta       |
| Per-user persistent facts inside ChatGPT                  | OpenAI Memory |
| Per-user persistent facts inside Claude                   | Claude Memory |

NDPA composes with the others. You can run NDPA for predictive staging
**and** Mem0 for fact retrieval. They solve different problems.

## How the eval numbers compare

NDPA's published numbers (from `eval/`):

| Eval                                           | Baseline | NDPA   | Lift  |
|------------------------------------------------|----------|--------|-------|
| File prediction hit@5 (122 samples, 13 sess.) | 31.1%    | 47.5%  | +16.4 |
| Conversation continuity hit@5 (494 samples)    | 31.6%    | 85.2%  | +53.6 |
| **Novel topic prediction hit@5 (480 samples)** | **17.9%** | **35.4%** | **+17.5** |

The "novel topic" eval is the strictest: predicting topics introduced in
the *future* of a conversation that don't appear in its *history*. This is
the closest analog to "predict context the user is about to need but
hasn't asked for yet."

Mem0, Zep, and Letta publish their own benchmarks on their own tasks (fact
recall, multi-session QA). They're solving different problems, so a direct
benchmark on the same dataset hasn't been run. We welcome PRs that wire
them into our eval harness.
