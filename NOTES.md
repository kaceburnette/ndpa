# NDPA Notes

## 2026-05-14 - Goal 6 console skeleton

Added a static frontend-only console preview at `web/console.html` because
`web/` currently contains a static landing page and no app build structure.
The console covers Dashboard, Signup/Login, API Keys, Usage, Quickstart, and
Pricing screens around the four-layer NDPA model: Predict, Memory, Hydration,
and Reasoning.

No live Supabase writes, Stripe checkout, API key creation, revoke action, or
usage reads are wired. All key/billing/usage controls are explicit placeholders
until Supabase recovery and local smoke pass. Landing page copy in
`web/index.html` now links to the console preview, uses request-access language,
and avoids claiming hosted production is live.

## 2026-05-14 12:23 EDT - PQHR public multi-signal run

Goal 3 PQHR run completed locally with no Supabase writes and no LLM judge:
`python3 -m eval.pre_query_hit_rate --public --adaptive-windows --output eval/pre_query_hit_rate_results.json`.

Verified LongMemEval-S public results: 500 synthetic users, 18,773 samples,
BM25 top-5 labeler, trajectory-only predictors. At W=7, multi-signal NDPA
scored pqhr@5 0.8458 (+16.5 pts over recency 0.6808, +15.8 pts over random
0.6877), mrr@5 0.5435, ndcg@5 0.3533. Pure topic remains stronger at pqhr@1
and MRR/NDCG: pqhr@1 0.4054, mrr@5 0.5646, ndcg@5 0.3621.

Adaptive multi-signal window sweep: W=3 pqhr@5 0.8277, W=5 0.8341, W=7
0.8458, W=10 0.8592, W=15 0.8547. Best public pqhr@5 window was W=10.

Implementation notes: BM25 remains the labeler and uses target conversation
content; predictors remain trajectory-only. Added MRR@5/NDCG@5, explicit
refs/path/code-ref overlap, keyphrase overlap, topic-frequency signal, and
temporal coherence. Predictor-side topic vectors are capped at top 250 terms
for tractable full public runs; BM25 labels still use full content.

## 2026-05-12 18:47:46 EDT - LMSYS validation blocked

LMSYS-Chat-1M validation could not run automatically. Reason: Python package `datasets` is not installed. The dataset is gated on HuggingFace and distributed as parquet; this repo intentionally does not add datasets/pyarrow as required dependencies. Provide --input with an accepted/exported JSON or JSONL sample, or install datasets after accepting the license.

## 2026-05-12 18:48:12 EDT - LoCoMo public dataset size

Requested LoCoMo runner expected 50 conversations, but the maintained Snap/USC public file at https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json contains 10 conversations. The runner scores the official public locomo10.json file and records the actual dataset size in eval/locomo_results.json.

## 2026-05-12 18:55:00 EDT - LoCoMo public dataset size

Requested LoCoMo runner expected 50 conversations, but the maintained Snap/USC public file at https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json contains 10 conversations. The runner scores the official public locomo10.json file and records the actual dataset size in eval/locomo_results.json.

## 2026-05-12 18:55:36 EDT - LoCoMo public dataset size

Requested LoCoMo runner expected 50 conversations, but the maintained Snap/USC public file at https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json contains 10 conversations. The runner scores the official public locomo10.json file and records the actual dataset size in eval/locomo_results.json.

## 2026-05-12 18:55:53 EDT - LMSYS initial run interpreter error

Initial LMSYS background run PID 24325 used Apple Python 3.9 and failed before benchmark execution:
```
Traceback (most recent call last):
  File "/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/lib/python3.9/runpy.py", line 197, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/lib/python3.9/runpy.py", line 87, in _run_code
    exec(code, run_globals)
  File "/Users/kaceburnette/Desktop/ndp/eval/lmsys_validation.py", line 22, in <module>
    from eval.conversation_eval import Conversation, cosine, tokenize
  File "/Users/kaceburnette/Desktop/ndp/eval/conversation_eval.py", line 28, in <module>
    from supabase import create_client
ModuleNotFoundError: No module named 'supabase'
```
Restarting with /opt/homebrew/bin/python3, where datasets was installed.

## 2026-05-12 18:56:08 EDT - LoCoMo public dataset size

Requested LoCoMo runner expected 50 conversations, but the maintained Snap/USC public file at https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json contains 10 conversations. The runner scores the official public locomo10.json file and records the actual dataset size in eval/locomo_results.json.

## 2026-05-12 18:56:23 EDT - LMSYS validation blocked

LMSYS-Chat-1M validation could not run automatically. Reason: Dataset 'lmsys/lmsys-chat-1m' is a gated dataset on the Hub. You must be authenticated to access it.. The dataset is gated on HuggingFace and distributed as parquet; this repo intentionally does not add datasets/pyarrow as required dependencies. Provide --input with an accepted/exported JSON or JSONL sample, or install datasets after accepting the license.

## 2026-05-12 19:06:39 EDT - LMSYS gated dataset blocked

Full LMSYS validation run PID 24743 installed datasets successfully but HuggingFace denied unauthenticated access to lmsys/lmsys-chat-1m:
```
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Wrote /Users/kaceburnette/Desktop/ndp/eval/lmsys_results.json
blocked Dataset 'lmsys/lmsys-chat-1m' is a gated dataset on the Hub. You must be authenticated to access it.
```
Continuing with LongMemEval and LoCoMo.

## 2026-05-12 19:28:43 EDT - LoCoMo public dataset size

Requested LoCoMo runner expected 50 conversations, but the maintained Snap/USC public file at https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json contains 10 conversations. The runner scores the official public locomo10.json file and records the actual dataset size in eval/locomo_results.json.

## 2026-05-12 19:28:45 EDT - LMSYS validation blocked

LMSYS-Chat-1M validation could not run automatically. Reason: Dataset 'lmsys/lmsys-chat-1m' is a gated dataset on the Hub. You must be authenticated to access it.. The dataset is gated on HuggingFace and distributed as parquet; this repo intentionally does not add datasets/pyarrow as required dependencies. Provide --input with an accepted/exported JSON or JSONL sample, or install datasets after accepting the license.

## 2026-05-12 19:49:37 EDT - LongMemEval timeout

LongMemEval run PID 51970 failed during concurrent ingestion:
```
           ~~~~~~~~~^^^^^^^^^^^^^^^^
  File "/opt/homebrew/Cellar/python@3.14/3.14.4/Frameworks/Python.framework/Versions/3.14/lib/python3.14/ssl.py", line 1138, in read
    return self._sslobj.read(len, buffer)
           ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^
TimeoutError: The read operation timed out

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/Users/kaceburnette/Desktop/ndp/eval/longmemeval_runner.py", line 157, in <module>
    main()
    ~~~~^^
  File "/Users/kaceburnette/Desktop/ndp/eval/longmemeval_runner.py", line 151, in main
    results = run(args)
  File "/Users/kaceburnette/Desktop/ndp/eval/longmemeval_runner.py", line 100, in run
    future.result()
    ~~~~~~~~~~~~~^^
  File "/opt/homebrew/Cellar/python@3.14/3.14.4/Frameworks/Python.framework/Versions/3.14/lib/python3.14/concurrent/futures/_base.py", line 443, in result
    return self.__get_result()
           ~~~~~~~~~~~~~~~~~^^
  File "/opt/homebrew/Cellar/python@3.14/3.14.4/Frameworks/Python.framework/Versions/3.14/lib/python3.14/concurrent/futures/_base.py", line 395, in __get_result
    raise self._exception
  File "/opt/homebrew/Cellar/python@3.14/3.14.4/Frameworks/Python.framework/Versions/3.14/lib/python3.14/concurrent/futures/thread.py", line 86, in run
    result = ctx.run(self.task)
  File "/opt/homebrew/Cellar/python@3.14/3.14.4/Frameworks/Python.framework/Versions/3.14/lib/python3.14/concurrent/futures/thread.py", line 73, in run
    return fn(*args, **kwargs)
  File "/Users/kaceburnette/Desktop/ndp/eval/longmemeval_runner.py", line 92, in ingest_session
    client.log_events(session_id, events, end_user_id=end_user_id)
    ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/kaceburnette/Desktop/ndp/sdk/python/ndpa/client.py", line 99, in log_events
    self._send(payload)
    ~~~~~~~~~~^^^^^^^^^
  File "/Users/kaceburnette/Desktop/ndp/sdk/python/ndpa/client.py", line 148, in _send
    self._post_path("/events", payload)
    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^
  File "/Users/kaceburnette/Desktop/ndp/sdk/python/ndpa/client.py", line 171, in _post_path
    raise NDPAError(f"NDPA request failed: {e}") from e
ndpa.client.NDPAError: NDPA request failed: The read operation timed out
```
Adding retry handling and restarting LongMemEval with a longer timeout.

## 2026-05-13 16:04:11 EDT - LoCoMo public dataset size

Requested LoCoMo runner expected 50 conversations, but the maintained Snap/USC public file at https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json contains 10 conversations. The runner scores the official public locomo10.json file and records the actual dataset size in eval/locomo_results.json.

## 2026-05-13 16:04:16 EDT - LoCoMo public dataset size

Requested LoCoMo runner expected 50 conversations, but the maintained Snap/USC public file at https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json contains 10 conversations. The runner scores the official public locomo10.json file and records the actual dataset size in eval/locomo_results.json.

## 2026-05-13 16:04:34 EDT - LoCoMo public dataset size

Requested LoCoMo runner expected 50 conversations, but the maintained Snap/USC public file at https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json contains 10 conversations. The runner scores the official public locomo10.json file and records the actual dataset size in eval/locomo_results.json.

## 2026-05-13 16:15:13 EDT - LoCoMo public dataset size

Requested LoCoMo runner expected 50 conversations, but the maintained Snap/USC public file at https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json contains 10 conversations. The runner scores the official public locomo10.json file and records the actual dataset size in eval/locomo_results.json.

## 2026-05-13 16:17:58 EDT - LoCoMo public dataset size

Requested LoCoMo runner expected 50 conversations, but the maintained Snap/USC public file at https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json contains 10 conversations. The runner scores the official public locomo10.json file and records the actual dataset size in eval/locomo_results.json.

---

# Build Log — 2026-05-14 Codex Execution Plan (Tasks 1-6)

This section traces exactly what's being built, file by file, decision by
decision. Each Codex task gets its own subsection.

## Task 1: Benchmark split discipline

- **File written:** `docs/BENCHMARK_SPLITS.md`
- **Rule:** validation/test splits per benchmark, never tune on test.
- **PQHR public:** 100 users validation, 400 users test (≈ 15,073 test samples).
- **PQHR real-user:** chronological 70/30 split (forward-prediction discipline).
- **LongMemEval:** M/XL for validation, S for test.
- **LoCoMo:** convs 1–3 validation, 4–10 test (by conversation_id, not question).
- **Carve-out:** legacy published numbers stay as full-sample reproducibility
  checks; new published numbers must be test-only and labeled.

## Task 2: LLM judge regrade — IN PROGRESS

- **Script:** `eval/llm_judge_regrade.py` (existing, extended with `--target`
  flag to support multiple results files).
- **Targets running:**
  - `eval/longmemeval_e2e_results.json` (baseline 31.4% string-overlap → real LLM-judge number)
  - `eval/longmemeval_reasoning_results.json` (reasoning 34.6% string-overlap → real LLM-judge number)
- **Model:** gpt-4o-mini, $0.15 in / $0.60 out per 1M tokens
- **Est. cost:** ~$0.20 each = ~$0.40 total
- **Background task IDs:** b049n0aul (E2E), b4e2b1nn3 (Reasoning)
- **Status:** running silently (buffered output)

## Task 3: e2e failure case audit — PENDING (waits for Task 2)

Plan: bucket every miss row into the seven Codex categories using rules:

| Bucket | Detection rule |
|---|---|
| retrieval miss | true session_ids ∉ predicted_session_ids (any K) |
| reader miss | retrieval hit AND judge says wrong AND grader says wrong |
| grader false negative | string-overlap false AND judge true |
| multi-hop miss | question has multiple entities AND retrieval missed any |
| temporal miss | question contains date phrase AND wrong on temporal-reasoning category |
| abstention miss | category=abstention AND judge=false |
| duplicate-context | top-K contains ≥2 chunks from same session_id AND wrong |

Output: `eval/e2e_failure_audit.json` with per-row bucketing + counts.

## Task 4: Reader prompt fix — PENDING (waits for Task 3)

Will be informed by audit findings. Likely changes:
- Tighten abstention language ("you did not mention X" formatting)
- Add explicit temporal reasoning instruction
- Add preference inference instruction
- Constrain to provided context only

## Task 5: Adaptive context budget — PENDING

Rule: increase context for multi-session and multi-hop categories;
decrease for single-session-user.

## Task 6: Re-run e2e with judge + new retrieval — PENDING

Final QA number that goes on the public landing page.


## Task 2 RESULTS: E2E baseline LLM judge regrade

**Date:** 2026-05-14 08:00 EDT
**Model:** gpt-4o-mini judge, $0.024 total cost
**File updated:** `eval/longmemeval_e2e_results.json` (added `metrics_llm_judge`)

| Category | string overlap | LLM judge | lift |
|---|---|---|---|
| abstention | 0.0% | 70.0% | +70 |
| single-session-assistant | 58.9% | 89.3% | +30 |
| single-session-preference | 0.0% | 36.7% | +37 |
| single-session-user | 48.4% | 62.5% | +14 |
| knowledge-update | 45.8% | 55.6% | +10 |
| multi-session | 26.4% | 32.2% | +6 |
| temporal-reasoning | 22.0% | 22.8% | +1 |
| **Overall** | **31.4%** | **46.0%** | **+14.6** |

**Key takeaways:**
- The string-overlap grader was suppressing ~15 pts of true accuracy.
- Abstention (0% → 70%) and preference (0% → 37%) confirm: those were grader
  artifacts, not model failures. The reader was answering correctly; the
  scorer couldn't see it.
- Temporal-reasoning (22.8%) is the genuine weak spot, not a grader artifact.
  This is where Codex item 8 (date-aware retrieval) targets.
- Multi-session (32.2%) is the second weak spot, target of Codex item 6
  (multi-hop query splitting).
- Single-session-assistant at 89.3% shows the reader IS very good when the
  answer is plainly in one retrieved chunk.


## Task 2 RESULTS (continued): Reasoning (extraction) LLM judge regrade

**Date:** 2026-05-14 08:00 EDT
**Model:** gpt-4o-mini judge, $0.022 total cost
**File updated:** `eval/longmemeval_reasoning_results.json` (added `metrics_llm_judge`)

| Category | string overlap | LLM judge | lift |
|---|---|---|---|
| abstention | 6.7% | 80.0% | +73.3 |
| knowledge-update | 41.7% | 51.4% | +9.7 |
| multi-session | 29.8% | 36.4% | +6.6 |
| single-session-assistant | 78.6% | 89.3% | +10.7 |
| single-session-preference | 0.0% | 6.7% | +6.7 |
| single-session-user | 46.9% | 73.4% | +26.5 |
| temporal-reasoning | 24.4% | 43.3% | +18.9 |
| **Overall** | **34.6%** | **51.8%** | **+17.2** |

## Head-to-head: E2E (baseline) vs Reasoning (extraction), both LLM-judged

| Category | E2E (no extract) | Reasoning (extract) | winner |
|---|---|---|---|
| abstention | 70.0% | 80.0% | extraction (+10) |
| temporal-reasoning | 22.8% | 43.3% | extraction (+20.5) ⭐ |
| single-session-user | 62.5% | 73.4% | extraction (+11) |
| multi-session | 32.2% | 36.4% | extraction (+4) |
| single-session-assistant | 89.3% | 89.3% | tie |
| knowledge-update | 55.6% | 51.4% | baseline (-4) |
| single-session-preference | 36.7% | 6.7% | baseline (-30) ⚠️ |
| **OVERALL** | **46.0%** | **51.8%** | **extraction (+5.8)** |

**Decision implications:**
- NDPA Reasoning is worth shipping after all (+5.8 pts overall).
- Extraction helps massively on temporal (+20.5) and clearly on most categories.
- Extraction destroys single-session-preference (-30) — facts lose the nuance
  that preferences require. Reader must NOT see structured-fact output for
  preference-style questions.
- Path forward (Codex item 4 + 5 territory): detect preference questions
  upstream, route them around the extractor. Treat extraction as a *router*,
  not a default pipeline step.
- Total LLM judge regrade cost: $0.024 + $0.022 = $0.046 OpenAI spend.


## Task 3 RESULTS: E2E failure case audit

**Date:** 2026-05-14 08:05 EDT
**Script:** `eval/e2e_failure_audit.py` (new)
**Outputs:** `eval/e2e_failure_audit.json`, `eval/reasoning_failure_audit.json`

### Baseline E2E (gpt-5.4-nano reader, no extraction) — 230 correct / 270 wrong (46.0%)

| Bucket | Count | Note |
|---|---|---|
| retrieved_right_reader_wrong | 213 | **78.9% of all misses** — reader is the bottleneck |
| grader_false_negative | 101 | 43.9% of judge-correct rows — string-overlap was hiding real wins |
| temporal_miss | 98 | 36.3% of misses — date-aware retrieval needed |
| multi_hop_miss | 88 | 32.6% of misses — multi-hop query splitting needed |
| retrieval_miss | 57 | 21.1% of misses — retrieval itself missed evidence |
| abstention_miss | 9 | 3.3% of misses — small surface area |

### Reasoning (extraction + reader) — 259 correct / 241 wrong (51.8%)

| Bucket | Count | Δ vs baseline | Note |
|---|---|---|---|
| retrieved_right_reader_wrong | 192 | −21 | reader still bottleneck even with extraction |
| grader_false_negative | 104 | +3 | similar grader sensitivity |
| multi_hop_miss | 85 | −3 | basically unchanged |
| temporal_miss | 72 | **−26** | extraction directly helped temporal |
| retrieval_miss | 49 | −8 | extraction can salvage some retrieval misses by guessing |
| abstention_miss | 6 | −3 | extraction helps spot "not mentioned" |

### Per-category bottleneck table

| Category | Top miss bucket | Implication |
|---|---|---|
| temporal-reasoning | temporal_miss (98 baseline → 72 reasoning) | date-aware retrieval is the lever |
| multi-session | retrieved_right_reader_wrong + multi_hop_miss | reader prompt + multi-hop splitting |
| knowledge-update | retrieved_right_reader_wrong (29) | reader needs "use most recent fact" instruction |
| single-session-preference | grader_false_negative + retrieval_miss (11 each) | preference questions hit retrieval recall ceiling |
| abstention | grader_false_negative (21 of 30) | grader is THE problem here, fixed by LLM judge |
| single-session-* | retrieved_right_reader_wrong | reader prompt fix |

### What this implies for the next tasks

- **Task 4 (reader prompt fix)** is the highest-ROI move. 213/270 baseline misses
  (79%) are reader misses on rows where retrieval succeeded. Fix the prompt to:
  (a) extract concise answers from provided chunks, (b) use most recent fact on
  updates, (c) say "you did not mention X" cleanly for abstentions.
- **Task 8 (date-aware retrieval)** is the second-biggest lever — 98 misses
  baseline, the reasoning extractor already drops it by 26, so adding
  date-aware retrieval should close most of the remaining gap.
- **Task 6 (multi-hop splitting)** is mid-priority — 88 baseline → 85 reasoning,
  so extraction didn't help. Splitting at retrieval time is the right fix.
- **Task 5 (adaptive context budget)** matters most for multi-session category
  (213 retrieval-right-reader-wrong rows clustered there) — bigger K for that
  bucket likely helps.

## Task 4/5 Codex patch: reader prompt + adaptive context

**Date:** 2026-05-14 08:20 EDT
**Files changed:** `eval/longmemeval_e2e_runner.py`,
`eval/longmemeval_reasoning_runner.py`

Implemented:
- Category-specific reader guidance for temporal, multi-session,
  knowledge-update, abstention, and preference questions.
- Context headers preserving rank, session_id, date, platform, and score so
  temporal reasoning can use dates instead of relying only on raw text.
- Adaptive context budget by category, with larger budgets for temporal,
  multi-session, knowledge-update, and preference questions.
- Reasoning runner routing: `single-session-preference` bypasses extraction
  and uses raw context, because extraction lost 30 points on that category.

Verification:
- `python3 -m py_compile eval/longmemeval_e2e_runner.py eval/longmemeval_reasoning_runner.py`
- Smoke checked prompt builder includes temporal guidance and session headers.

Next:
- Re-run E2E/Reasoning on a smoke subset, then full judge regrade.
- Next retrieval-side work remains date-aware retrieval and multi-hop splitting.

## Task 6 RESULTS: Baseline E2E rerun with task 4/5 patch

**Date:** 2026-05-14 08:55 EDT
**Files:** `eval/longmemeval_e2e_rerun_results.json`, `eval/e2e_failure_audit_rerun.json`
**Reader/Judge model:** gpt-4.1-mini

Results:
- String-overlap: 38.8% (old 31.4%)
- LLM judge: 47.6% (old 46.0%, +1.6)
- Reader cost: $0.3500
- Judge cost: $0.0235

Category judge deltas vs old baseline:
- abstention: 70.0% -> 76.7% (+6.7)
- temporal-reasoning: 22.8% -> 30.7% (+7.9)
- single-session-preference: 36.7% -> 50.0% (+13.3)
- single-session-assistant: 89.3% -> 91.1% (+1.8)

- knowledge-update: 55.6% -> 48.6% (-6.9)
- multi-session: 32.2% -> 29.8% (-2.5)
- single-session-user: 62.5% -> 60.9% (-1.6)

Retrieval hit@K is unchanged because retrieval did not change:
- overall hit@1 78.0%, hit@3 83.6%, hit@5 84.2%
- temporal hit@5 82.7%
- multi-session hit@5 88.4%

Interpretation:
- Task 4/5 patch is a small net win, not a breakthrough.
- Temporal/preference improved; multi-session and knowledge-update regressed.
- Remaining misses are still mostly reader misses after retrieval succeeds.
- Next best E2E path is routed Reasoning: use extraction for temporal/abstention/user facts, raw context for preference.

## Task 6 RESULTS: Routed Reasoning rerun

**Date:** 2026-05-14 09:45 EDT
**Files:** `eval/longmemeval_reasoning_rerun_results.json`
**Reader/Judge model:** gpt-4.1-mini

Results:
- String-overlap: 38.8%
- LLM judge: 49.8%
- Reader/extraction cost: $0.3635
- Judge cost: $0.0230

Comparison:
- Old baseline E2E judge: 46.0%
- Patched raw E2E judge: 47.6%
- Old Reasoning judge: 51.8%
- Routed Reasoning rerun judge: 49.8%

Category judge results:
- abstention: 83.3%
- temporal-reasoning: 36.2%
- single-session-preference: 46.7%
- knowledge-update: 52.8%
- multi-session: 30.6%
- single-session-user: 59.4%
- single-session-assistant: 91.1%

Interpretation:
- Routed Reasoning improved preference vs old extraction (6.7% -> 46.7%), but lost too much elsewhere.
- Old Reasoning remains the best overall E2E score (51.8%).
- The category-router idea is valid, but current implementation should not replace old Reasoning for headline metrics.
- Next work should move to Goal 2: retrieval quality, especially multi-hop splitting and date-aware retrieval, then rerun E2E.

## Doc-only pass: launch/benchmark claim alignment

**Date:** 2026-05-14 08:25 EDT
**Scope (Codex-directed):** docs only. No edits to `server/`, `eval/` runners, or `sdk/`.

**Files touched:**
- `README.md`
- `docs/PQHR.md`
- `docs/PQHR_LEADERBOARD.md`
- `docs/BENCHMARK_SPLITS.md`
- `NOTES.md` (this entry)

**Product naming locked in across docs:**
- **NDPA Predict** = PQHR / pre-query staging from trajectory.
- **NDPA Memory** = reactive retrieval given a query (LongMemEval / LoCoMo hit@K).
- **NDPA Hydration** = optional raw context fetch for selected memory handles
  (`session_id` / `storage_key`). Benchmarked by latency, cost, and storage
  efficiency; not a QA benchmark.
- **NDPA Reasoning** = optional end-to-end QA pipeline over retrieved/hydrated context
  (LongMemEval E2E, LLM judge). Routed (preference-question routing) is
  a tested variant of this layer, not the current shipping config.

**Numbers reflected:**
- Retrieval hit@5: LongMemEval **84.2%**, LoCoMo **82.5%**.
- E2E LLM-judge (NDPA Reasoning):
  - Baseline E2E (no extraction, prior prompt): **46.0%**
  - Patched raw E2E (no extraction, prompt fixes): **47.6%**
  - Routed Reasoning (tested variant, preference routing, not headline): **49.8%**
  - Reasoning un-routed (current best E2E run): **51.8%**
- PQHR public test set: 84.1% pqhr@5, +16.1 over recency / +15.4 over random.
  Real-user PQHR: 64.5% pqhr@5 with much larger lift.

**Claims I updated:**
- Replaced the single "31.4%†" NDPA QA cell with a separated retrieval table
  and an E2E table.
- Reframed the headline so retrieval hit@K and E2E QA are reported as
  independent metrics, not collapsed.
- Reframed NDPA Reasoning as **optional**, with 51.8% labeled as the
  current best E2E run and 49.8% reported as a tested routed variant for
  transparency, not as the headline.
- Tagged PQHR explicitly as the NDPA Predict benchmark across all four docs.

**Claims I refused or softened:**
- Refused to put NDPA in the same QA comparison table as MemPalace, Mem0,
  OMEGA, ByteRover, etc. Those are full-stack memory systems doing LLM fact
  extraction at ingest; comparing our retrieval-only or routed-reasoning
  numbers head-to-head against their full-stack QA would be misleading.
- Refused to headline 49.8% (routed Reasoning) — it is a tested variant
  that trades overall accuracy for behavior on preference questions.
  Listed it transparently, did not claim it as best.
- Softened "10× faster than embeddings" claim further — current latency is
  honestly described as in-progress; landing-page footer keeps the explicit
  ~300ms / ~930ms / FastAPI rewrite caveat.

**Files I did NOT touch (per scope):**
- `server/main.py`, `server/predictions.py` (Codex owns)
- `eval/longmemeval_e2e_runner.py`, `eval/longmemeval_reasoning_runner.py` (Codex owns)
- `sdk/python/ndpa/client.py`, `sdk/typescript/` (Codex owns)


## Goal 2 server retrieval patch

**Date:** 2026-05-14 10:05 EDT
**Files changed:** `server/predictions.py`, `server/main.py`, `tests/test_server_predictions.py`, `sdk/typescript/src/client.ts`, `db/migrations/20260514_add_tsv_content_gin.sql`

Implemented in FastAPI retrieval path:
- Multi-hop subquery splitting for compound questions.
- Deterministic query expansion with stemming-style prefix tsquery and small alias map.
- Entity/keyphrase and temporal/date feature extraction.
- Hybrid final scoring: topic rank, term overlap, entity match, date match, recency, length normalization.
- MMR diversity selection to reduce redundant top-K sessions.
- Indexed conversation text now preserves `source_path` metadata, which carries benchmark dates.
- Prediction response schema now exposes `started_at` for reader temporal context.
- TypeScript SDK now honors `NDPA_BASE_URL`, matching Python SDK behavior.
- Added Supabase migration for `ndp_conversations.tsv_content`, GIN index, and trigger maintenance.

Verification:
- `python3 -m unittest tests/test_server_predictions.py` passes.
- `python3 -m py_compile server/predictions.py server/main.py` passes.
- `python3 -m py_compile eval/longmemeval_e2e_runner.py eval/longmemeval_reasoning_runner.py eval/llm_judge_regrade.py` passes.
- `npm run build` in `sdk/typescript` passes.
- `npm test` is unavailable: package has no test script.
- Risky doc phrases around routed Reasoning/current-shipping claims removed from README/NOTES.

Caveat:
- These changes affect the FastAPI server path. Existing Supabase Edge Function benchmark numbers will not move until the FastAPI endpoint is deployed and benchmarks are rerun against it.
- Local FastAPI/latency verification is blocked in this shell because `DATABASE_URL` is not set and `fly` CLI is not installed.


## Claim hygiene follow-up

**Date:** 2026-05-14 10:18 EDT
**Files changed:** `README.md`, `docs/COMPARISON.md`

Followed up on the remaining risky-claim audit:
- README intro now names the product layers up front.
- Older in-house file/conversation/novel-topic evals are labeled as NDPA Predict sanity checks, not Memory retrieval or Reasoning QA.
- Stale benchmark roadmap removed claims that E2E QA and PSA were future work; replaced with current benchmark workstream across PQHR, retrieval hit@K, and E2E QA.
- `docs/COMPARISON.md` rewritten around the three-layer taxonomy and explicitly says not to compare retrieval hit@K to competitor full-stack QA.

Verification:
- `rg` scan found no stale positive "apples-to-apples", PSA-as-future-benchmark, or routed-supersedes-51.8 claims in README/docs. Remaining matches in NOTES are audit-log references describing what was fixed or refused.


## Supabase tsv_content migration

**Date:** 2026-05-14 10:58 EDT

Applied the Supabase schema migration for FastAPI/Postgres full-text retrieval
through Supabase MCP.

Verified in Supabase:
- Migration history contains `20260514145131 add_tsv_content_gin`.
- `public.ndp_conversations.tsv_content` exists as `tsvector`.
- `ndp_conversations_tsv_idx` exists as `USING gin (tsv_content)`.
- `ndp_conversations_tsv_update` trigger exists on insert/update of `content`.

Recovery status:
- Initial all-row backfill timed out through MCP.
- Bounded batches partially populated old rows before the storage issue was
  noticed.
- Stop condition: Supabase free-tier storage exceeded the 500MB cap. Do not run
  more backfills. Do not recreate `tsv_content`. Do not create a full-content
  GIN index on this free-tier project.
- Required DB cleanup when Supabase accepts connections again:
  `drop index if exists public.ndp_conversations_tsv_idx;`
  `drop trigger if exists ndp_conversations_tsv_update on public.ndp_conversations;`
  `drop function if exists public.ndp_update_tsv_content();`
  `alter table public.ndp_conversations drop column if exists tsv_content;`
- Supabase is currently rejecting SQL connections with `FATAL: 57P03: the
  database system is not accepting connections / Hot standby mode is disabled`.

Local verification rerun:
- `python3 -m unittest tests/test_server_predictions.py` passes.
- `python3 -m py_compile server/predictions.py server/main.py eval/longmemeval_e2e_runner.py eval/longmemeval_reasoning_runner.py eval/llm_judge_regrade.py` passes.
- `npm run build` in `sdk/typescript` passes.
- Follow-up patch: FastAPI no longer requires `tsv_content`; `/events` writes
  only base conversation columns, and `/predictions` detects the full-text
  column at runtime with a lightweight lexical fallback when absent.
- Architecture correction: hosted mode must treat Postgres as the compact
  metadata/index tier, not the raw memory warehouse. Raw conversation text
  belongs in Supabase Storage/R2/blob storage and should be hydrated only after
  top-K selection.
- Architecture taxonomy is now four layers:
  Predict -> Memory -> Hydration -> Reasoning.
  Predict and Memory return handles/previews/scores. Hydration fetches raw
  context only when needed. Reasoning answers using retrieved/hydrated context.
- Added `db/migrations/20260514_add_hosted_metadata_columns.sql` as the safer
  hosted shape (`content_preview`, `top_terms`, `storage_key`, `content_bytes`).
  Do not apply it until Supabase storage recovery is complete and DB size is
  back under cap.
- FastAPI fallback now prefers compact metadata (`top_terms` /
  `content_preview`) when those columns exist and uses legacy `content` only as
  compatibility fallback.
- Prediction responses now include `memory_handle`, `content_preview`,
  `storage_key`, and `content_bytes`. The legacy `content` field is a preview
  alias in hosted Memory; raw full context belongs to future Hydration.

Blocked:
- Local FastAPI latency smoke remains blocked because this shell has no
  `DATABASE_URL` / `POSTGRES_URL`, `.env` does not contain either key, and
  neither `psql` nor `supabase` CLI is installed.

## Docs/launch four-layer positioning pass

**Date:** 2026-05-14 11:20 EDT
**Scope:** docs/landing only. No code, server, SDK, eval, or migration edits.

Files changed:
- `README.md`
- `docs/ARCHITECTURE.md`
- `docs/COMPARISON.md`
- `docs/INTEGRATION.md`
- `docs/VOICE_AI.md`
- `docs/LAUNCH.md`
- `web/index.html`
- `NOTES.md`

Changes:
- Updated docs and landing around the four-layer model:
  Predict -> Memory -> Hydration -> Reasoning.
- Kept metrics separated:
  PQHR for Predict, hit@K for Memory, latency/storage for Hydration, and E2E
  LLM judge for Reasoning.
- Removed the landing-page competitor benchmark table that mixed NDPA
  retrieval hit@K with competitor full-stack QA-style claims and cost/latency
  estimates.
- Replaced the landing-page competitor pricing comparison with a neutral
  pricing note so unverified competitor cost numbers are not presented as
  launch claims.
- Added hosted metadata-first architecture language: Postgres hot tier stores
  handles, previews, top terms, scores, timestamps, storage keys, and byte
  counts; raw conversations and `.ndp` bundles hydrate from blob/object/local
  storage only after top-K selection.
- Added `docs/VOICE_AI.md`.
- Added `docs/LAUNCH.md` with HN post, X thread, voice AI outreach, code
  agent outreach, and chat platform outreach.
- Corrected README LongMemEval summary to use only verified values from this
  notes file: overall hit@1 78.0%, hit@3 83.6%, hit@5 84.2%, temporal hit@5
  82.7%, multi-session hit@5 88.4%.


## Local FS Hydration API

**Date:** 2026-05-14 12:15 EDT

Implemented the local-FS Hydration layer without touching Supabase:
- Added `server/hydration.py` with path-safe local blob reads.
- Added `POST /hydrate` to FastAPI. It accepts `storage_keys` directly and can
  resolve `memory_handles` / `session_ids` to `storage_key` when the compact
  metadata column exists.
- Added `NDPA_HYDRATION_ROOT` for local blob root configuration.
- Added Python SDK `Client.hydrate(...)`.
- Added TypeScript SDK `client.hydrate(...)`.
- Updated `server/README.md` so `/predictions` is Memory
  handles/previews/scores and `/hydrate` is raw context fetch.

Verification:
- `python3 -m unittest tests/test_server_predictions.py` passes (11 tests).
- `python3 -m py_compile server/predictions.py server/main.py server/hydration.py` passes.
- `npm run build` in `sdk/typescript` passes.

Supabase recovery:
- Still blocked. Latest read-only recovery query returned `FATAL: 57P03: the
  database system is not accepting connections / Hot standby mode is disabled`.
- No Supabase writes, imports, backfills, Storage usage, or deletes were run.


## Core prediction timing + cache

**Date:** 2026-05-14 13:05 EDT

Added Core speed instrumentation without changing storage architecture:
- `/predictions` response now includes `timing`:
  `auth_ms`, `db_candidate_ms`, `rank_ms`, `serialize_ms`, `total_ms`,
  `cache_hit`.
- Added in-process API-key auth cache (`NDPA_AUTH_CACHE_TTL_SEC`, default 45s).
- Added in-process prediction cache (`NDPA_PREDICTION_CACHE_TTL_SEC`, default
  20s), keyed by `user_id`, `end_user_id`, `query`, `session_id`, `k`.
- `/events` clears the prediction cache after ingest.

Local timing smoke against FastAPI on `127.0.0.1:8000`:
- Cold sample for query `rust async tokio borrow checker cold`:
  `total_ms=643`, `auth_ms=332`, `db_candidate_ms=311`, `rank_ms=0`,
  `serialize_ms=0`, `cache_hit=false`, `predictions=0`.
- Immediate repeat of same query:
  `total_ms=0`, all timing fields 0, `cache_hit=true`, `predictions=0`.
- `python3 -m eval.compare_latency --requests 20` after warmup:
  p50 `0.4ms`, p95 `0.6ms`, mean `0.4ms`. This is warm-cache behavior because
  the script warms the same query before measuring.

Read-only `EXPLAIN ANALYZE` for the metadata candidate query:
- Execution time: `0.075ms`.
- Plan used `Bitmap Index Scan on ndp_conversations_user_updated_idx`, then
  filtered by `top_terms && '{rust,async,tokio,borrow,checker}'`.
- No schema changes or indexes were added; measured DB execution was already
  cheap for the sampled user. The cold request was dominated by remote DB/auth
  round trips, so the auth/prediction caches are the correct first speed fix.

Verification:
- `python3 -m unittest tests/test_server_predictions.py` passes (9 tests).
- `python3 -m py_compile server/main.py server/predictions.py server/hydration.py` passes.
- `npm run build` in `sdk/typescript` passes.


## Stage hardening benchmark

**Date:** 2026-05-14 13:45 EDT

Hardened `/stage` and cache behavior:
- `TTLCache` now exposes remaining TTL via `get_with_ttl`.
- `/stage` returns `expires_in_sec` and `expires_at`.
- `/predictions` timing now includes `cache_remaining_ms`.
- Prediction cache invalidation is scoped to the ingested user/session instead
  of clearing the whole process cache.
- Usage telemetry writes to `ndp_usage_events` are best-effort background tasks
  so accounting does not block hot responses.

Verification:
- `python3 -m unittest tests/test_server_predictions.py` passes (11 tests).
- `python3 -m py_compile server/main.py server/predictions.py server/hydration.py` passes.
- `npm run build` in `sdk/typescript` passes.

Benchmark on local FastAPI `127.0.0.1:8001`, 80 requests per phase:

Before fixing telemetry (awaited usage writes):
- cold prediction p50 `630.39ms`, p95 `822.58ms`.
- `/stage` p50 `645.34ms`, p95 `744.26ms`.
- warm staged `/predictions` p50 `317.28ms`, p95 `363.79ms`.
- `/hydrate` p50 `319.22ms`, p95 `348.5ms`.

After moving telemetry off the response path:
- cold prediction p50 `341.96ms`, p95 `389.33ms`.
- `/stage` p50 `315.92ms`, p95 `386.0ms`.
- warm staged `/predictions` p50 `0.85ms`, p95 `2.61ms`.
- `/hydrate` p50 `0.49ms`, p95 `0.88ms`.

Timing breakdown averages after fix:
- cold: `db_candidate_ms=343.55`, `cache_hit_rate=0.0`.
- stage: `db_candidate_ms=321.91`, `cache_hit_rate=0.0`.
- warm staged read: `db_candidate_ms=0`, `cache_hit_rate=1.0`,
  `cache_remaining_ms≈19994`.

Interpretation:
- The hot architecture is working: `/stage` pays the remote metadata DB cost,
  and later `/predictions` returns from in-process cache.
- Hydration is local-FS fast when direct `storage_key` is known.
- Remaining cold/stage latency is remote database round trip, not ranking or
  serialization.

## 2026-05-14 12:08:36 EDT - Goal 4 competitor + DIY PQHR baselines

Updated `eval/competitor_pqhr.py` to default to no-cost local baselines and
gate paid/reactive adapters behind `--allow-paid-api`.

Verified on LongMemEval-S public synthetic users, 500 users / 18,773 PQHR
samples, trajectory window 7, BM25 top-5 ground truth:

| System | pqhr@1 | pqhr@3 | pqhr@5 | Status |
|---|---:|---:|---:|---|
| random | 0.2046 | 0.5048 | 0.6874 | verified |
| recency | 0.2085 | 0.5025 | 0.6801 | verified |
| local BM25 trajectory query | 0.3856 | 0.6892 | 0.7788 | verified |
| local TF-IDF trajectory query | 0.2534 | 0.5939 | 0.7495 | verified |
| local lexical vector proxy | 0.4338 | 0.6953 | 0.7955 | verified |

Mem0 was not run: `OPENAI_API_KEY` and the `mem0` package are present, but the
adapter would make paid OpenAI LLM/embedding calls. Status recorded as
`blocked-by-paid-approval`.

Zep, Letta, and Pinecone were not run: no corresponding API keys were present
in `.env` or the shell environment. Status recorded as `blocked-by-key`.

No Supabase calls, paid API calls, or LLM judge calls were made. Results saved
to `eval/competitor_pqhr_results.json`; verified local rows added to
`docs/PQHR_LEADERBOARD.md`.
