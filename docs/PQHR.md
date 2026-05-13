# Pre-Query Hit Rate (PQHR)

**A benchmark for predict-forward memory systems.**

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
   $C_i$'s full content (cosine similarity over bag-of-words). This is what
   *should* have been staged.
4. **Predict** top-$k$ past conversations using only the trajectory.
5. **Score**: hit@k against ground truth.

**Default parameters:**
- Trajectory window $W = 3$ (last 3 conversations)
- Ground truth top-$N = 5$
- Predicted top-$k \in \{1, 3, 5\}$
- Minimum past conversations $P = 10$ (need history to evaluate)
- Minimum future conversation length: 200 characters (skip trivial)

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

PQHR ships with four reference predictors:

1. **Random** — pick K past conversations uniformly at random. Sanity check.
2. **Recency-only** — return the K most recent past conversations. No topic
   awareness. The "dumbest sensible" predictor.
3. **Heuristic** — blend $0.4 \cdot$ recency $+ 0.6 \cdot$ topic. NDPA's
   live production predictor.
4. **Pure topic** — pure cosine similarity over BoW. NDPA's
   `predictive_promote` path (used for cold-tier promotion).

---

## Reference numbers

Run on the NDPA project's reference corpus (993 conversations, 973 PQHR
samples, trajectory window = 3):

| Predictor | pqhr@1 | pqhr@3 | pqhr@5 |
|-----------|--------|--------|--------|
| Random | 0.0175 | 0.0226 | 0.0483 |
| Recency-only | 0.1326 | 0.2199 | 0.3022 |
| Heuristic (NDPA live) | 0.1449 | 0.2621 | 0.4162 |
| **Pure topic (NDPA cold-promote)** | **0.1387** | **0.3535** | **0.4872** |

**Lift over recency baseline at @5: +18.5 percentage points.**

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
- **BoW ground truth**: ground truth uses BoW cosine, which favors lexical
  overlap. Future work: ground truth could use LLM-judge for semantic match.
- **Trajectory window sensitivity**: larger $W$ gives more signal but skews
  toward recent patterns. Default $W=3$ is a balance.
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
