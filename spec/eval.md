# Evaluation Protocol

## Ground truth definition

> A file is "used" at turn T if its absolute path appears as an argument to `Read`, `Edit`, or `Write` within turns T+1 through T+5 of the same session.

`Bash` calls are excluded — shell commands are too noisy and file paths are not reliably extractable.

## Metric: hit@k

For each evaluation position (session S, turn T):
- Generate top-k predicted file paths using the predictor
- Check whether `intersection(top_k_predictions, ground_truth) ≠ ∅`
- hit@k score = fraction of positions where this is true

Reported at k = 1, 3, 5.

## Baseline: last-N-files

Returns the N most recently accessed files before turn T, ranked by recency.

This is the floor. Any prediction system worth shipping must beat it.

## Data requirements

- Minimum: 10 sessions with 8+ turns each to produce ~50 eval samples
- Meaningful: 50+ sessions for kernel training and held-out evaluation
- Split: 80/20 train/eval; never tune on eval split

## Turn grouping

Events are grouped into turns using `turn_start` markers in the session log. If markers are absent, events within a 30-second window are treated as the same turn.

## Minimum history

Evaluation positions require at least 3 prior turns of history. Positions with fewer than 3 prior turns are excluded.

## Success bar (v0)

A predictor earns the v0 label if it achieves:
- hit@5 ≥ baseline + 0.20 on held-out eval sessions
- hit@3 ≥ baseline + 0.15

No partial credit for "close." Either the number is there or it isn't.

## Running the eval

```bash
# Baseline only (sanity check that logs are collecting)
python3 -m eval.harness

# With a custom predictor
python3 -c "
from eval.harness import run_eval
from kernel.predictor import NullPredictor
run_eval(NullPredictor())
"
```

## What we're not measuring

- Latency (prediction must be <100ms but this is architectural, not measured here)
- Token savings (downstream metric, not v0)
- Cross-session transfer (future work)
