# PQHR Leaderboard

[Pre-Query Hit Rate (PQHR)](PQHR.md) measures predict-forward memory: can a
system stage relevant past context from session trajectory alone, with no
explicit query?

PQHR is the benchmark for **NDPA Predict**. Reactive retrieval is benchmarked
separately on LongMemEval and LoCoMo hit@K (NDPA Memory). Raw context fetch is
**NDPA Hydration** and is measured by latency, cost, and storage efficiency,
not QA. End-to-end QA is benchmarked separately on LongMemEval E2E with an LLM
judge (NDPA Reasoning, optional / routed). PQHR does not collapse those metrics
into a single number.

To submit: open a PR adding a row below. Provide reproducible evidence.
Honest setup details required.

---

## Dataset 1: Public synthetic users (LongMemEval-S, fully reproducible)

500 synthetic users from the LongMemEval public dataset, 18,773 PQHR samples,
trajectory window = 7, ground truth = BM25 top-5. Latest NDPA PQHR run was
verified locally on 2026-05-14 and saved to
`eval/pre_query_hit_rate_results.json`.

**Always cite random and recency baselines alongside.** Hit@5 has a high
floor on this corpus because each user's candidate pool is small (~30–40
past conversations) and the truth set is top-5.

| Rank | System | Version | pqhr@1 | pqhr@3 | pqhr@5 | Notes |
|-----:|--------|---------|-------:|-------:|-------:|-------|
|   1  | **NDPA (multi-signal)** | v11 | 0.368 | 0.688 | **0.846** | Topic frequency + refs + keyphrases + temporal coherence, BM25 labeler |
|   2  | NDPA (heuristic) | v11 | 0.405 | 0.697 | 0.842 | 0.4 recency + 0.6 topic |
|   3  | NDPA (pure topic) | v11 | **0.405** | **0.697** | 0.841 | BoW cosine predictor, BM25 labeler |
|   4  | local lexical vector proxy | local-hash-512 | 0.434 | 0.695 | 0.796 | Hashed lexical vector cosine; no embedding API required |
|   5  | local BM25 trajectory query | local | 0.386 | 0.689 | 0.779 | BM25 over past conversations using trajectory as query |
|   6  | local TF-IDF | local | 0.253 | 0.594 | 0.749 | TF-IDF cosine over past conversations using trajectory as query |
|  —   | Random baseline (sanity) | — | 0.205 | 0.505 | 0.688 | Uniform sample |
|  —   | Recency baseline | — | 0.209 | 0.503 | 0.681 | Most-recent-K |

**Lift over recency @5: +16.5 points. Lift over random @5: +15.8 points.**

Latest verified NDPA ranking metrics:

| System | mrr@5 | ndcg@5 |
|--------|------:|-------:|
| NDPA (multi-signal) | 0.543 | 0.353 |
| NDPA (heuristic) | **0.565** | 0.362 |
| NDPA (pure topic) | **0.565** | **0.362** |
| Random baseline (sanity) | 0.374 | 0.212 |
| Recency baseline | 0.375 | 0.210 |

**Caveat (honest):** random@5 is 68.7% on this corpus, so the absolute hit@5
numbers look stronger than the underlying signal. Look at the @1 column for a
cleaner discrimination: pure-topic NDPA 40.5% vs random 20.5% =
**1.98× lift**, which is the more defensible headline.

Adaptive trajectory windows for NDPA multi-signal:

| Window | Samples | pqhr@1 | pqhr@3 | pqhr@5 | mrr@5 | ndcg@5 |
|-------:|--------:|-------:|-------:|-------:|------:|-------:|
| 3 | 18,773 | 0.313 | 0.616 | 0.828 | 0.490 | 0.324 |
| 5 | 18,773 | 0.347 | 0.656 | 0.834 | 0.522 | 0.338 |
| 7 | 18,773 | 0.368 | 0.688 | 0.846 | 0.543 | 0.353 |
| 10 | 18,773 | **0.399** | 0.726 | **0.859** | **0.573** | **0.378** |
| 15 | 16,285 | 0.384 | **0.728** | 0.855 | 0.564 | 0.360 |

Best public multi-signal window by pqhr@5: **10**.

Reproduce: `python3 -m eval.pre_query_hit_rate --public --adaptive-windows`

---

## Dataset 2: Author's real-user corpus (internal validation, less externally verifiable)

651 PQHR samples from the author's 668-conversation real-user corpus across
multiple platforms. Provided as evidence on a harder candidate pool, not as
the primary reproducibility claim.

| Rank | System | Version | pqhr@1 | pqhr@3 | pqhr@5 | Notes |
|-----:|--------|---------|-------:|-------:|-------:|-------|
|   1  | **NDPA (pure topic)** | v10 | **0.263** | **0.518** | **0.645** | BoW cosine predictor, BM25 labeler |
|   2  | NDPA (heuristic) | v10 | 0.200 | 0.326 | 0.424 | 0.4 recency + 0.6 topic |
|  —   | Recency baseline | — | 0.128 | 0.238 | 0.321 | Most-recent-K |
|  —   | Random baseline (sanity) | — | 0.018 | 0.023 | 0.049 | Uniform sample |

**Lift over recency @5: +32.4 points. Lift over random @5: +59.6 points.**

Why this corpus is harder: candidate pool is larger and more diverse
(multi-platform coherent history), so random@5 collapses to 4.9% instead of
68.7%. The lift signal is much stronger here.

This corpus is private. The author does not publish the conversation content.
The aggregate benchmark numbers above are provided as additional evidence;
treat them as a secondary signal next to the public dataset.

---

## On BM25 in this benchmark

BM25 is **the ground-truth labeler**, not a competitor baseline. For each
target conversation $C_i$, we compute BM25 scores against all prior
conversations and treat the top-5 as the "should have been staged" set.

The predictor (BoW cosine on trajectory) is then scored against those labels.
BM25 and BoW cosine are different formulas on different inputs (target
conversation vs. trajectory), so a predictor that games BoW cosine does not
automatically game BM25. That is the independence claim — not "we beat BM25
as a competitor."

---

## Submitted by other systems

*No external submissions yet. Be the first.*

If you build a memory system, run PQHR on your data and submit:
1. Fork the repo
2. Add a row above with your numbers AND the random + recency baselines on the same corpus
3. Include reproduction script or data sample in the PR
4. Open PR titled `pqhr: submit {SYSTEM_NAME}`

## What disqualifies a submission

- Systems that require an explicit query to retrieve are not measuring the
  same thing. PQHR specifically tests staging from trajectory alone.
- Cherry-picked subsets without methodology disclosure.
- Submissions that omit random and recency baselines on the same corpus.
- Non-reproducible numbers without source code or sample data.

## What we welcome

- Honest numbers, even if worse than baseline. The point is to map the field.
- Novel predictor architectures (different kernel, different signal sources).
- Methodological critiques (the spec is open for revision).
- Dataset contributions — especially corpora where random@5 is low (harder
  signal).
