# Pre-Query Hit Rate (PQHR)

**A benchmark for predict-forward memory systems. PQHR is the benchmark for
NDPA Predict (pre-query staging from trajectory).** Reactive retrieval —
NDPA Memory — is measured separately on LongMemEval and LoCoMo hit@K.
Hydration is the raw-context fetch step and is measured by latency, cost, and
storage efficiency. End-to-end QA — NDPA Reasoning — is measured on
LongMemEval E2E with an LLM judge. PQHR does not combine those metrics.

Most published memory benchmarks (LongMemEval, LoCoMo, MTEB-Memory) test
*reactive retrieval*: given a question, did the system surface the right past
context? This is what RAG-style memory systems do.

PQHR tests something fundamentally different: **predictive staging**.

> Given only a user's prior session trajectory — no current question, no
> current conversation, no explicit input — can the system stage the
> relevant past context that will become useful in their NEXT session?

This is the metric for predict-forward memory: the ability to anticipate
context *before the user asks for it*. Voice AI, live agents, latency-critical
applications, and ambient AI assistants all need this. Vector DBs and
embedding-based memory systems can't be scored on it because they require an
explicit query.

---

## Methodology

**Setup:**
1. Sort a user's conversations chronologically.
2. For each conversation $C_i$ (where $i \geq W$ and there are $\geq P$ past
   conversations):
   - **Trajectory** = concatenated content of $C_{i-W} \ldots C_{i-1}$ (the
     last $W$ conversations before $C_i$).
   - The predictor sees **only the trajectory**. It never sees $C_i$ itself or
     anything after.
3. **Ground truth** = top-$N$ past conversations whose content best matches
   $C_i$'s full content, scored by **BM25** (K1=1.5, B=0.75). This is what
   *should* have been staged. BM25 is independent from the BoW cosine
   predictor — no circularity.
4. **Predict** top-$k$ past conversations using only the trajectory. The
   current NDPA predictors use trajectory-only lexical/topic signals; they
   never see $C_i$.
5. **Score**: hit@k, MRR@5, and NDCG@5 against BM25 ground truth.

**Default parameters:**
- Trajectory window $W = 7$ (last 7 conversations)
- Ground truth top-$N = 5$ (BM25)
- Predicted top-$k \in \{1, 3, 5\}$ (BoW cosine)
- Minimum past conversations $P = 10$ (need history to evaluate)
- Minimum future conversation length: 200 characters (skip trivial)
- Adaptive trajectory window experiment: $W \in \{3, 5, 7, 10, 15\}$
- Predictor-side topic cap: top 250 trajectory terms for tractable full
  public runs. BM25 labels still use full target/past conversation content.

**Independence (why this is not circular):**
The predictor scores using BoW cosine. The ground truth is built using BM25.
These are different scoring methods on different inputs:
- Predictor input: $W$-conversation trajectory → score each past conversation
- Ground truth input: target conversation $C_i$ → score each past conversation
A predictor that gamed BoW cosine would not automatically game BM25.

**Reproducibility:**
- Run via `python3 -m eval.pre_query_hit_rate`
- Source: `eval/pre_query_hit_rate.py` (MIT)
- Stateless: anyone with conversation history can reproduce.

---

## Why it's different from existing benchmarks

| Benchmark | Predictor sees | Tests |
|-----------|---------------|-------|
| LongMemEval | Question + haystack | Reactive retrieval + reasoning |
| LoCoMo | Question + conversation | Reactive retrieval + QA |
| MTEB-Memory | Query + document corpus | Semantic similarity |
| **PQHR** | **Only prior trajectory** | **Predictive staging** |

The first three answer "did you find what was asked for?" PQHR answers
"can you anticipate what will be asked for?"

---

## Why this matters for AI memory

In production, the gap between reactive retrieval and predictive staging is:

| Reactive retrieval (vector DBs, Mem0, Zep) | Predictive staging (NDPA) |
|---|---|
| User asks → embed query → vector search → return | Trajectory tracked → stage context → context ready when user types |
| 300-800ms per query | 0ms perceived (staged ahead of time) |
| Bound to query existence | Works without a query |
| Requires explicit input | Works on behavioral signal alone |

For voice AI, agentic loops, ambient assistants, and any application where
the user expects sub-second response: predictive staging is structurally
different. PQHR measures it.

---

## Baselines

PQHR ships with four reference predictors and one labeler.

**Predictors (compared on the leaderboard):**

1. **Random** — pick K past conversations uniformly at random. Sanity check.
2. **Recency-only** — return the K most recent past conversations. No topic
   awareness. The "dumbest sensible" predictor.
3. **Heuristic** — blend $0.4 \cdot$ recency $+ 0.6 \cdot$ topic. NDPA's
   live runtime predictor.
4. **Pure topic** — pure cosine similarity over BoW. NDPA's
   `predictive_promote` path (used for cold-tier promotion).
5. **Multi-signal** — trajectory-only hybrid signal: topic frequency,
   explicit refs/file paths/code refs, lightweight keyphrase overlap, and
   temporal coherence.

**Labeler (defines ground truth, NOT a competitor):**

- **BM25** — produces the top-5 "should-have-been-staged" set for each
  target conversation. Predictors are scored against these labels.
  BM25 is not on the leaderboard. It is the ruler we measure against.

**Why this isn't circular:** the predictor input is the trajectory (last 7
prior conversations); the labeler input is the target conversation itself.
BoW cosine and BM25 are also different formulas. A predictor that games BoW
cosine does not automatically game BM25.

---

## Reference numbers (public, reproducible)

Run on **LongMemEval public dataset** (500 synthetic users, 18,773 PQHR
samples, trajectory window = 7, BM25 labeler). Verified locally on
2026-05-14 with `python3 -m eval.pre_query_hit_rate --public
--adaptive-windows --output eval/pre_query_hit_rate_results.json`.

| Predictor | pqhr@1 | pqhr@3 | pqhr@5 | mrr@5 | ndcg@5 |
|-----------|--------|--------|--------|-------|--------|
| Random (sanity) | 0.2045 | 0.5053 | 0.6877 | 0.3743 | 0.2116 |
| Recency-only | 0.2088 | 0.5029 | 0.6808 | 0.3750 | 0.2098 |
| Heuristic (0.4 recency + 0.6 topic) | 0.4053 | 0.6966 | 0.8416 | **0.5646** | 0.3621 |
| Pure topic (NDPA cold-promote) | **0.4054** | **0.6968** | 0.8414 | **0.5646** | **0.3621** |
| **Multi-signal (NDPA)** | 0.3680 | 0.6879 | **0.8458** | 0.5435 | 0.3533 |

**Lift @5: +16.5 pts over recency, +15.8 pts over random for multi-signal.**
**Lift @1: pure topic 40.5% vs random 20.5% = 1.98×.**

**Caveat (honest):** random@5 is 68.7% on this corpus because each synthetic
user has a small candidate pool (~30–40 past conversations) and the truth set
is top-5. Always cite random + recency next to NDPA numbers. Look at @1 for
the cleaner discrimination signal.

Adaptive window result for multi-signal:

| Window | Samples | pqhr@1 | pqhr@3 | pqhr@5 | mrr@5 | ndcg@5 |
|-------:|--------:|-------:|-------:|-------:|------:|-------:|
| 3 | 18,773 | 0.3133 | 0.6164 | 0.8277 | 0.4901 | 0.3244 |
| 5 | 18,773 | 0.3471 | 0.6562 | 0.8341 | 0.5218 | 0.3379 |
| 7 | 18,773 | 0.3680 | 0.6879 | 0.8458 | 0.5435 | 0.3533 |
| 10 | 18,773 | **0.3990** | 0.7261 | **0.8592** | **0.5727** | **0.3776** |
| 15 | 16,285 | 0.3838 | **0.7277** | 0.8547 | 0.5638 | 0.3602 |

Best public multi-signal window by pqhr@5: **10**.

Reproduce: `python3 -m eval.pre_query_hit_rate --public --adaptive-windows`

Optional semantic/embedding ablations are accuracy experiments only. The
default PQHR command remains no-embedding; do not replace the default reported
predictor unless an explicit ablation run wins and is documented with random
and recency baselines on the same corpus.

For a harder corpus where random@5 collapses to 4.9% and NDPA's lift jumps to
+59.6 pts, see the [author's real-user corpus](PQHR_LEADERBOARD.md#dataset-2-authors-real-user-corpus-internal-validation-less-externally-verifiable) results (internal validation, not publicly reproducible).

---

## Submitting a score

If you build a memory system that attempts predict-forward staging, run PQHR
on your data and submit a PR to this repo with:

1. Your system name, version, and architecture summary
2. The dataset you ran on (or share it)
3. Your reported pqhr@1, pqhr@3, pqhr@5
4. Reproduction instructions
5. Any deviations from default methodology

NDPA's leaderboard lives in `docs/PQHR_LEADERBOARD.md`. Be honest about
your setup; the eval is fully open-source and reproducible.

---

## Limitations

- **Single user assumption**: PQHR scores per-user. Cross-user generalization
  isn't measured (and probably shouldn't be — memory is per-user).
- **Lexical ground truth**: BM25 ground truth favors lexical overlap. Future
  work: cross-encoder or LLM-judge ground truth for semantic match.
- **Trajectory window sensitivity**: larger $W$ gives more signal but skews
  toward recent patterns. Default $W=7$ is a balance (validated empirically
  on the public dataset).
- **Cold start**: a user with $< P$ conversations can't be evaluated. Predict-
  forward memory needs history to learn from.

These are honest limitations, not hidden. PR welcome on improvements.

---

## Citation

If you use PQHR in a paper or product comparison:

```bibtex
@misc{ndpa-pqhr-2026,
  title  = {Pre-Query Hit Rate (PQHR): A Benchmark for Predict-Forward
            Memory Systems},
  author = {Burnette, Kace},
  year   = {2026},
  url    = {https://github.com/kaceburnette/ndpa/blob/main/docs/PQHR.md}
}
```
