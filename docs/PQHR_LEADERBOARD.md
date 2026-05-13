# PQHR Leaderboard

[Pre-Query Hit Rate (PQHR)](PQHR.md) measures predict-forward memory: can a
system stage relevant past context from session trajectory alone, with no
explicit query?

To submit: open a PR adding a row below. Provide reproducible evidence.
Honest setup details required.

## Current rankings (pqhr@5)

| Rank | System | Version | pqhr@1 | pqhr@3 | pqhr@5 | Dataset | Notes |
|-----:|--------|---------|-------:|-------:|-------:|---------|-------|
| 1 | **NDPA (pure topic)** | v8 | 0.174 | 0.442 | **0.571** | real-user corpus (651 samples, 668 convs) | BoW + TF-IDF + BM25, no embeddings |
| 2 | NDPA (heuristic) | v8 | 0.183 | 0.310 | 0.476 | real-user corpus (651 samples) | 0.4 recency + 0.6 topic |
| 3 | Recency baseline | — | 0.178 | 0.283 | 0.381 | real-user corpus (651 samples) | Most-recent-K-conversations |
| — | Random baseline | — | 0.020 | 0.028 | 0.065 | real-user corpus (651 samples) | Sanity check |

**Lift over recency baseline @5: +19.0 percentage points.**

Methodology note: scored on COHERENT single-user conversation history
(real user's 668 conversations across multiple platforms). Running PQHR
across a synthetic mix of 500 unrelated synthetic users gives a lower
score (~31%) because predictions can't generalize across unrelated
users — that's intentional. PQHR measures per-user predict-forward
quality, which is the production use case.

## Submitted by other systems

*No external submissions yet. Be the first.*

If you build a memory system, run PQHR on your data and submit:
1. Fork the repo
2. Add a row above with your numbers
3. Include reproduction script or data sample in the PR
4. Open PR titled `pqhr: submit {SYSTEM_NAME}`

## What disqualifies a submission

- Systems that require an explicit query to retrieve are not measuring the
  same thing. PQHR specifically tests staging from trajectory alone.
- Cherry-picked subsets without methodology disclosure.
- Non-reproducible numbers without source code or sample data.

## What we welcome

- Honest numbers, even if worse than baseline. The point is to map the field.
- Novel predictor architectures (different kernel, different signal sources).
- Methodological critiques (the spec is open for revision).
- Dataset contributions (more reference corpora improves the benchmark).
