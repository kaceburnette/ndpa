# NDPA Goal Chunks

Use these as separate `/goal` inputs. Each chunk is scoped to stay compact and executable.

## Current Architecture Decision

NDPA now ships as four layers:

1. **NDPA Predict** — pre-query staging from trajectory; benchmarked by PQHR.
2. **NDPA Memory** — lightweight explicit-query retrieval over hot metadata; benchmarked by LongMemEval/LoCoMo retrieval hit@K.
3. **NDPA Hydration** — optional raw context fetch for selected memory handles; measured by latency, bandwidth, and storage efficiency.
4. **NDPA Reasoning** — optional LLM answer layer over retrieved/hydrated context; benchmarked by LongMemEval E2E LLM judge.

Storage rule: Postgres is the hot metadata/index tier. Raw conversations,
events, and `.ndp` bundles belong in blob/object/local storage. Hosted Core
must not depend on full raw-text GIN indexes or materialized `tsv_content`.

## Goal 1: Benchmark Discipline + E2E Recovery

Lock benchmark split discipline and recover e2e score credibility.

Tasks:
1. Define train/validation/test split rules for LoCoMo, LongMemEval, and PQHR.
2. Add LLM judge regrade for e2e QA.
3. Audit e2e misses into: retrieval miss, reader miss, grader false negative, multi-hop miss, temporal miss, abstention miss, duplicate-context issue.
4. Fix reader prompt for concise answers, abstention, temporal reasoning, preference inference, and context-only behavior.
5. Adapt context budget by query type.
6. Re-run e2e with judge and report retrieval hit@K plus judged answer accuracy.

Success:
- E2E score is judge-graded.
- Failure categories are documented.
- No final number is tuned on the test set.

## Goal 2: NDPA Memory Retrieval Quality

Improve lightweight metadata-first retrieval for LoCoMo and LongMemEval.

Tasks:
1. Keep `tsv_content` optional; hosted/free-tier path must work without it.
2. Implement compact metadata retrieval over `top_terms`, `content_preview`, timestamps, entities, and storage handles.
3. Implement multi-hop query splitting.
4. Implement deterministic query expansion: stemming, plurals, aliases.
5. Add date-aware retrieval boosting.
6. Add entity/keyphrase-aware retrieval boosting.
7. Add MMR diversity reranking.
8. Add cross-session context expansion for high-confidence hits.
9. Add adaptive K for broad, temporal, and multi-hop queries.
10. Implement hybrid retrieval score: topic, term overlap, entity match, date match, recency, continuity, diversity.
11. Re-run LongMemEval and LoCoMo retrieval with per-category breakdown.

Success:
- LoCoMo and LongMemEval retrieval improve or regressions are explained.
- Multi-hop and temporal categories are explicitly tracked.
- Default retrieval returns handles/previews/scores, not full raw archives.

## Goal 2A: Supabase Recovery + Hosted Storage Shape

Recover the Supabase project from the failed full-text index/backfill and lock
hosted storage into the lightweight/tiered architecture.

Tasks:
1. Stop all Supabase writes, imports, and backfills until recovery completes.
2. When Supabase accepts SQL, inspect table/index/database sizes.
3. Drop `ndp_conversations_tsv_idx`, `ndp_conversations_tsv_update`, `ndp_update_tsv_content`, and `ndp_conversations.tsv_content`.
4. Measure database size after dropping full-text objects.
5. If still over cap, identify benchmark/import rows by platform/end_user_id before deleting anything.
6. Delete only confirmed synthetic benchmark/import data if needed; never delete user/prod data without explicit confirmation.
7. Apply only compact hosted metadata columns after recovery: `top_terms`, `content_preview`, `storage_key`, `content_bytes`, and similar small fields.
8. Do not use Supabase Storage buckets while the project is over quota/degraded.
9. Update docs so full-content `tsv_content` + GIN is marked optional and not free-tier/default.

Success:
- Supabase is under quota and accepting reads/writes.
- Hosted mode uses compact metadata only.
- No full-content backfill or full raw-text GIN index remains in the default path.

## Goal 3: NDPA Predict + PQHR Quality

Improve predict-forward staging and harden PQHR.

Tasks:
1. Add multi-signal trajectory features: topic frequency, explicit refs, code/file paths, repeated-topic weighting.
2. Add adaptive trajectory windows: 3, 5, 7, 10, 15.
3. Add lightweight entity/keyphrase tracking across sessions.
4. Add temporal coherence scoring.
5. Implement hybrid PQHR scoring: topic, recency, trajectory frequency, entity overlap, temporal coherence.
6. Add MRR and NDCG to PQHR.
7. Run PQHR on real-user corpus and public LongMemEval synthetic users.
8. Always report random and recency beside NDPA.
9. Document negative ablations: TF-IDF and ms-marco cross-encoder.

Success:
- PQHR docs clearly state BM25 is the ground-truth labeler.
- Public and real-user results are separated.
- Random/recency baselines are visible everywhere.

## Goal 4: Competitor + DIY Baselines

Produce defensible external comparisons.

Tasks:
1. Run Mem0 PQHR baseline with same trajectory input and scoring.
2. Run Zep PQHR baseline with same methodology.
3. Run Letta PQHR baseline with same methodology.
4. Run Pinecone/vector DIY PQHR baseline.
5. Run embedding candidate retrieval ablation with clear caveats.
6. Publish leaderboard rows only for verified runs.
7. Label adapted reactive systems honestly: trajectory-as-query, not native prediction.

Success:
- Leaderboard includes NDPA, random, recency, and verified competitor baselines.
- No competitor claim compares different metrics as if identical.

## Goal 5: Hosted Core API Infrastructure

Ship the hosted API path.

Tasks:
1. Run FastAPI locally against recovered Supabase using metadata fallback.
2. Expose ingest, predictions, retrieval, health, and usage endpoints.
3. Expose `/stage` so Predict pays cold metadata ranking before the hot path.
4. Make `/predictions` cache-first for staged handles/previews/scores.
5. Ensure default prediction/retrieval responses return handles, previews, scores, and storage keys.
6. Add optional hydration API shape without making raw hydration part of the core hot path.
7. Configure DATABASE_URL and required env without committing secrets.
8. Add API key generation with ndpa_ keys, hashed storage, and project scope.
9. Require API key auth on public endpoints.
10. Add free-tier enforcement and usage counters.
11. Add request size limits and timeout caps.
12. Re-run latency benchmark against local endpoint first.
13. Report cold miss, stage cost, warm staged hit, and hydration latency separately.
14. Choose hosted provider after local smoke; Fly is optional and not a blocker.
15. Update SDK default base URL with env override.

Success:
- Hosted API works with SDK.
- Auth is required.
- Latency and failure rates are measured.
- Core latency is measured separately from optional hydration/reasoning.
- Warm staged Predict latency is measured separately from cold Supabase latency.

## Goal 6: Billing + Console

Make the product sellable.

Tasks:
1. Add usage metering for events, predictions, retrievals, storage, latency, and errors.
2. Add Stripe billing or checkout path.
3. Build minimal console: signup/login, API keys, usage, billing, quickstart.
4. Add plan/free-tier display.
5. Add upgrade path.

Success:
- A customer can create/get a key, use the API, see usage, and pay or upgrade.

## Goal 7: SDKs + Docs + Landing

Make adoption obvious.

Tasks:
1. Publish Python SDK.
2. Publish TypeScript SDK.
3. Write hosted quickstart.
4. Write voice AI guide.
5. Write coding-agent guide.
6. Write Express/Node guide.
7. Update landing page with Predict, Memory, Hydration, and Reasoning.
8. Explain deployment modes: hosted metadata-first, embedded/local FS, self-host, object storage.
9. Update README benchmark section: PQHR, retrieval hit@K, e2e QA, hydration latency/storage, infra cost.
10. Explain that Hydration is a storage/context-fetch layer, not a QA benchmark.

Success:
- A developer can understand and integrate NDPA from docs alone.

## Goal 8: Embedded Kernel Moat

Build embedded prediction/retrieval libraries.

Tasks:
1. Build Python embedded kernel with in-memory, SQLite, and Postgres backends.
2. Match server scoring behavior.
3. Publish ndpa-kernel.
4. Build JS/TS embedded kernel with practical backend support.
5. Publish @ndpa/kernel.
6. Add embedded Python and Node quickstarts.

Success:
- Customers can run NDPA in-process without hosted API latency.

## Goal 9: Production Hardening

Add scale and enterprise readiness.

Tasks:
1. Precompute compact metadata stats per user.
2. Add tiered candidate scoring over hot metadata.
3. Add optional Postgres scoring functions only when they do not require raw full-text indexes.
4. Add hydration cache for top-K raw context fetches.
5. Add Redis/Upstash cache when hosted traffic justifies it.
6. Add observability for latency, errors, usage anomalies, storage growth, and hydration misses.
7. Add public status page.
8. Add self-host Docker bundle.
9. Add open leaderboard submission harness.

Success:
- Product is observable, scalable, and credible for serious users.

## Goal 10: Launch

Launch and convert attention.

Tasks:
1. Prepare claim memo: public/private/proxy-scored/needs-caveat.
2. Prepare HN post.
3. Prepare X thread.
4. Prepare outreach to voice AI companies.
5. Prepare outreach to code-agent companies.
6. Prepare outreach to AI chat platforms.
7. Launch, monitor, respond, and fix only blockers.

Success:
- Launch narrative is sharp: predictive data layer for AI agents.
- Inbound users can sign up, get a key, and use the product.

## Later Research

Defer until launch/product path is working.

Tasks:
1. Personalized prediction model.
2. Conversation-domain reranker.
3. Reinforcement learning from clicks/usage.
4. SimHash/MinHash acceleration.
5. NDPA Reasoning improvements.
6. Hydration storage adapters and cache policy research.
7. PQHR/predict-forward memory paper.
