# NDPA — Neural Data Prefetch Architecture

A memory substrate that predicts-forward instead of retrieving-on-demand.

## What this is

A file format, prediction kernel, and daemon that stage AI context proactively based on workflow trajectory rather than waiting for queries.

**Core idea:** Given file access history through turn T, predict which files will be needed at turns T+1..T+5 and stage them before they're requested.

**Target:** Beat a last-N-files baseline on hit@5, measured on real Claude Code session logs.

## What this is not (v0)

- Not a vector database or semantic search system
- Not a RAG replacement — it's orthogonal (prefetch ≠ retrieve)
- Not solving hardware memory bandwidth
- Not infinite context
- The kernel is **not implemented yet** — it requires real session log data to train and tune

## Architecture

```
Session logs (~/.ndp/sessions/*.jsonl)
        ↓
   Eval harness  ←→  Predictor (kernel/baseline)
        ↓
    hit@k score
```

```
Runtime (not yet built):
Workflow state → Kernel → tier assignments → Daemon → hot/warm/cold tiers
                                                           ↓
                                                   Context compiler
                                                           ↓
                                                   token-budgeted packet
```

## Directory layout

```
spec/          — schema, format, eval protocol specs
ndp/           — Python package: schema, storage, compiler
kernel/        — prediction engine (stub until data exists)
eval/          — harness, metrics, baseline predictor
collector/     — PostToolUse hook that logs session data
```

## Getting started

The collector hook should already be installed in your global Claude Code settings. Verify:

```bash
ls ~/.ndp/sessions/
```

If sessions are appearing there, data is collecting. Once you have 10+ sessions, run the eval:

```bash
python3 -m eval.harness
```

## Eval protocol

**Ground truth:** A file is "used" at turn T if its path appears as an argument to `Read`, `Edit`, or `Write` within turns T+1..T+5.

**Baseline:** Last-N-files (most recently accessed files before turn T).

**Success bar (v0):** 20%+ improvement in hit@5 over the baseline on accumulated session logs.

## What this is not yet

- [ ] Trained/tuned prediction kernel (needs data)
- [ ] Running daemon
- [ ] Tier state machine
- [ ] Context compiler integration with Claude Code
- [ ] Cross-session checkpoint/restore
- [ ] Co-access matrix from accumulated sessions
