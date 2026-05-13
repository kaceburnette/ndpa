# PQHR Leaderboard

[Pre-Query Hit Rate (PQHR)](PQHR.md) measures predict-forward memory: can a
system stage relevant past context from session trajectory alone, with no
explicit query?

To submit: open a PR adding a row below. Provide reproducible evidence.
Honest setup details required.

## Current rankings (pqhr@5)

| Rank | System | Version | pqhr@1 | pqhr@3 | pqhr@5 | Dataset | Notes |
|-----:|--------|---------|-------:|-------:|-------:|---------|-------|
| 1 | **NDPA (pure topic)** | v8 | 0.139 | 0.354 | **0.487** | reference corpus (973 samples) | BoW + TF-IDF + BM25, no embeddings |
| 2 | NDPA (heuristic) | v8 | 0.145 | 0.262 | 0.416 | reference corpus (973 samples) | 0.4 recency + 0.6 topic |
| 3 | Recency baseline | — | 0.133 | 0.220 | 0.302 | reference corpus (973 samples) | Most-recent-K-conversations |
| — | Random baseline | — | 0.018 | 0.023 | 0.048 | reference corpus (973 samples) | Sanity check |

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
