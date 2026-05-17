# Benchmark Split Discipline

This document defines train / validation / test splits for the external
benchmarks NDPA reports on. The rule is simple:

> **Hyperparameters and predictor weights are tuned on the validation split.
> The test split is reported and never used to make tuning decisions.**

If a test split number moves because of an explicit tuning decision, that
decision must be re-validated by re-splitting (or by acknowledging the leak
honestly in the writeup).

Each benchmark below corresponds to a different NDPA product layer:

| Benchmark         | Product layer       | Metric reported |
|-------------------|---------------------|------------------|
| PQHR              | NDPA Predict        | pqhr@K           |
| LongMemEval hit@K | NDPA Memory         | retrieval hit@K  |
| LoCoMo hit@K      | NDPA Memory         | retrieval hit@K  |
| Hydration smoke   | NDPA Hydration      | latency / cost / storage efficiency |
| LongMemEval E2E   | NDPA Reasoning      | LLM-judge QA accuracy |

Retrieval hit@K, Hydration latency/cost, and end-to-end QA are never combined
into a single number. Hydration is not a QA benchmark.

---

## PQHR (Pre-Query Hit Rate)

PQHR runs on two corpora. Both follow the same rule.

### Public corpus — LongMemEval synthetic users

- **Source:** `xiaowu0162/longmemeval-cleaned` (LongMemEval-S, 500 synthetic users).
- **Sample count after building PQHR samples:** 18,773 (trajectory window = 7,
  BM25 ground truth, minimum past-conversation requirement applied).
- **Split (deterministic by question_id sort):**
  - **Validation:** first 100 synthetic users (≈ 3,700 samples). Use for grid
    search, prompt tuning, ablations.
  - **Test:** remaining 400 synthetic users (≈ 15,073 samples). Reported
    publicly. **No tuning decisions made on these samples.**
- The published `pqhr@5 = 0.841` is computed on ALL 18,773 samples for the
  legacy reference (predates split discipline). After this document lands,
  subsequent published numbers must be on test-only and clearly labeled.

### Real-user corpus (private)

- **Source:** author's personal multi-platform conversation history.
- **Sample count:** 651 PQHR samples from 668 conversations.
- **Split (chronological by `started_at`):**
  - **Validation:** first 70% of samples chronologically.
  - **Test:** last 30% chronologically. Reported as internal-validation
    secondary evidence in `docs/PQHR_LEADERBOARD.md`.
- Chronological split (not random) because PQHR measures predicting forward
  in time. A random split would leak future context into trajectory queries.

---

## LongMemEval (reactive retrieval, hit@K and end-to-end QA)

- **Source:** `xiaowu0162/longmemeval-cleaned` (LongMemEval-S, 500 questions).
- **Split:**
  - **Validation:** the larger LongMemEval-M and LongMemEval-XL datasets
    (different question pools, different haystacks). Tune retrieval params
    here.
  - **Test:** all 500 LongMemEval-S questions. Reported publicly.
- LongMemEval-S is small enough that random subsplitting would be noisy.
  Using the M/XL splits as validation gives more headroom for tuning and
  keeps S as a clean test set.

---

## LoCoMo (long conversational memory)

- **Source:** `snap-research/locomo` (`locomo10.json`, 1,982 evidence questions).
- **Split (deterministic by conversation_id):**
  - **Validation:** conversations 1–3 (≈ 30% of questions, by sample count).
  - **Test:** conversations 4–10 (≈ 70% of questions).
- Split by conversation_id (not by question) because retrieval state on a
  conversation contaminates if questions from the same conversation appear in
  both splits.

---

## Reading the leaderboard after this lands

Until each benchmark's test-only run is recomputed, published numbers in
`docs/PQHR_LEADERBOARD.md` and `README.md` are on the full sample set (legacy).
Once test-only numbers are computed, they become the primary headline and the
full-sample number stays as a reproducibility check (always reproducible by
`python3 -m eval.pre_query_hit_rate --public`).

---

## What this protects against

- **Inflated headline numbers from hyperparameter overfitting.** Without a
  validation set, every grid search effectively leaks information into the
  reported number.
- **Skeptic critique on launch.** "Did you tune on the test set?" — answer is
  documented here and verifiable in code.
- **Drift after launch.** If we tune a new feature in goal-mode while at
  military and the test number jumps, the rule says we re-validate before
  publishing.

---

## What this does NOT protect against

- **Selection of which benchmarks to publish.** We chose LongMemEval, LoCoMo,
  and PQHR. There are other public memory benchmarks (LMSYS-Chat-1M,
  ShareGPT-derived, MemoryBench) where we may score differently. This
  document does not commit us to running all of them.
- **Public corpus characteristics.** LongMemEval synthetic users have a high
  random@5 baseline (68.7%) which inflates absolute numbers; we document this
  in `docs/PQHR.md`, but it's an artifact of the corpus, not of the split.
- **Dataset leakage at the corpus level.** If LongMemEval haystacks were
  scraped from the same source as our future training data for a learned
  reranker, that's leakage outside this document's scope.
