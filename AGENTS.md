# Agent handoff — NDPA next steps

For Codex, Claude, or any agent picking up this repo. Read this first.

## What NDPA is

Predict-forward memory layer for AI. Predicts what context an AI will need
next and stages it BEFORE the user asks. Tiered storage (hot/cold) with
predictive promotion. MIT licensed. Live on github.com/kaceburnette/ndpa.

Read `README.md` and `docs/ARCHITECTURE.md` for the full picture.

## What's done (as of commit 91f54fb)

- Edge functions deployed (events, predictions v5 with inverted index, health)
- Python + TypeScript SDKs at v0.2.0, publish-ready
- 4 internal evals (file, conversation, novel-topic, held-out future)
- Tiered storage with LocalFS backend + S3 stub + predictive_promote
- `.ndp` portable bundle format (export/import working)
- Hook with local queue + background flusher (44ms hot path)
- Reference integrations: OpenAI Chat, Anthropic Messages, LangChain
- 657 of the author's ChatGPT conversations imported and queryable
- Docs: ARCHITECTURE, SECURITY, INTEGRATION, COMPARISON, SCALING, TIERED_STORAGE
- Demo: `python3 demo/demo.py` runs end-to-end with no setup
- Multi-tenant support via `end_user_id` parameter

## Best published number

**71.8% hit@5 on held-out future predictions** (`eval/held_out_future_eval.py`,
180-day holdout, 156 samples, topic-only predictor). Baseline = 0.6%.

Mid-range numbers:
- File prediction: 47.5% hit@5 (+16.4 pts over baseline)
- Conversation continuity: 85.2% hit@5 (+53.6 pts) — partially circular metric
- Novel topic prediction: 35.4% hit@5 (+17.5 pts) — strict
- Held-out future: 71.8% hit@5 (+71.2 pts) — strictest, gold-standard

## What to build next (priority order)

### 1. LongMemEval runner — HIGHEST PRIORITY

The standard memory benchmark. Mem0, Zep, MemGPT have all published on it.
If NDPA gets competitive numbers, we have apples-to-apples comparison to
the funded incumbents.

- Paper: arxiv 2410.10813 "LongMemEval: Benchmarking Chat Assistants
  on Long-Term Interactive Memory"
- Dataset on HuggingFace (search "longmemeval")
- 500 test questions across 5 categories: single-session recall,
  multi-session reasoning, temporal reasoning, knowledge updates, abstention
- Implementation: load dataset → for each test question, ingest the
  prior conversation history into NDPA, then call get_predictions
  with the test question as query → score whether the right past
  conversation surfaces in top-K
- Save results to `eval/longmemeval_results.json`
- Update README with comparison table

### 2. LoCoMo runner

Long Conversation Memory benchmark from Snap/USC. Tests retrieval AND
reasoning over 9k-token average conversations.

- Paper: arxiv 2402.17753
- Dataset: 50 multi-session conversations
- Same pattern: ingest → predict → score

### 3. LMSYS-Chat-1M validation

Run our existing `eval/conversation_eval.py` and `eval/held_out_future_eval.py`
on the public LMSYS-Chat-1M dataset (1M real ChatGPT conversations from
HuggingFace). Removes "only tested on author's data" criticism.

- Write `eval/lmsys_import.py` modeled on `eval/chatgpt_import.py`
- Pick a single random user's worth of conversations (LMSYS has user IDs)
- Import via Ingestion API with `platform="lmsys"`
- Run held-out future eval on it
- Publish the number

### 4. S3 backend implementation

Currently `S3Backend` in `ndp/tiered.py` is a stub. Implement it:

- Don't add boto3 as a dependency — make it optional
- Subclass example is in `docs/TIERED_STORAGE.md`
- Test against MinIO or a real S3 bucket
- Add `tests/test_s3_backend.py` with mock S3

### 5. Publish SDKs

- `cd sdk/python && python3 -m build && twine upload dist/*`
  (needs PyPI token)
- `cd sdk/typescript && npm publish`
  (needs npm token)

User will run these locally with their own tokens. Don't attempt unless
explicitly asked.

## Project goals (for context)

The user is shipping to the military in 5 days (as of 2026-05-12). The
goal is to have NDPA as defensibly good as possible by then so they can:

1. Open source it widely
2. Present it to AI companies / get acquired
3. Have it work as production infrastructure

Big AI companies they're targeting: Anthropic, OpenAI, Google. Also
YC AI companies (Lindy, Decagon, Cursor, Retell, etc.) as developer
adoption channel.

## Code style notes

- Python: stdlib only where reasonable, no premature abstractions
- TypeScript: zero runtime dependencies in the SDK
- Match what's already in the repo, small focused changes
- No tests are required but don't break the demo

## How the user works

- Direct feedback, calls out being oblivious or undersized
- Hates over-explanation. Wants results not narration.
- Will tell you when you're wrong. Take it seriously.
- Trusts you to act on durable changes. Don't ask permission for small things.

## What NOT to do

- Don't claim numbers we haven't measured
- Don't add features beyond what's listed above
- Don't refactor working code unless required for a new feature
- Don't publish to PyPI/npm without explicit confirmation
- Don't add embeddings/vector search as a dependency (the architecture wins
  precisely because we don't need them)

## Quick verification

To verify everything still works after your changes:

```bash
python3 demo/demo.py              # should print eval numbers, no errors
python3 -m eval.held_out_future_eval --holdout-days 180  # should print 71.8% hit@5
curl https://izackuempnhgoojtbgvi.supabase.co/functions/v1/health  # should return healthy
```

If those three pass, the system is intact.
