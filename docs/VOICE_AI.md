# NDPA for Voice AI

Voice agents cannot hide memory latency behind a long UI pause. NDPA is useful
because the Predict layer can stage likely context before the user asks, while
the Memory and Hydration layers stay available when an explicit query arrives.

Current local FastAPI -> Supabase pooler smoke is around ~700ms until latency
work improves it. Use Predict to pre-stage between turns rather than blocking
on a user-visible retrieval path.

## Layer mapping

| Layer | Voice use | Metric |
|---|---|---|
| **Predict** | Pre-stage likely caller/customer/session context from prior trajectory | PQHR |
| **Memory** | Retrieve handles/previews when the current utterance names a topic | LongMemEval / LoCoMo hit@K |
| **Hydration** | Fetch raw transcript or CRM notes for selected handles | latency / bandwidth / storage |
| **Reasoning** | Optional answer synthesis over retrieved/hydrated context | E2E LLM judge |

Keep those numbers separate. PQHR says whether NDPA staged the right context
before a query. hit@K says whether Memory retrieved the right evidence handle
after a query. Hydration is a fetch/storage measurement. Reasoning is the only
QA metric.

## Integration pattern

1. Log each call turn asynchronously with `log_turn` or batched `log_events`.
2. Between turns, call `get_predictions(session_id, k=3)` without waiting for
   the next user utterance.
3. Put returned `content_preview`, `memory_handle`, and `storage_key` into your
   voice agent state.
4. If the next utterance needs full detail, hydrate only the selected
   `storage_key`.
5. Let your voice model answer from staged/hydrated context, or skip Reasoning
   and pass the memory preview directly as background context.

```python
from ndpa import Client

ndpa = Client(api_key="ndpa_...", platform="voice")

ndpa.log_turn(call_id, "user", transcript_chunk)
ndpa.log_turn(call_id, "assistant", assistant_reply)

prefetch = ndpa.get_predictions(call_id, k=3)
staged = [
    {
        "handle": p.get("memory_handle") or p["session_id"],
        "preview": p.get("content_preview") or p.get("content", ""),
        "storage_key": p.get("storage_key"),
        "score": p["score"],
    }
    for p in prefetch["predictions"]
]
```

## Hosted storage shape

Hosted NDPA keeps compact metadata hot in Postgres:

- `session_id`
- `started_at`
- `platform`
- `top_terms`
- `content_preview`
- `storage_key`
- `content_bytes`

Raw transcripts and `.ndp` bundles belong in blob/object/local storage.
Local-FS Hydration is implemented; latency/storage benchmarking is pending.
Hydration fetches raw context by handle only after Predict or Memory has
selected a small candidate set. Do not build a voice integration around full
raw-text Postgres scans or a required full-content GIN index.

## Prompting guidance

Use staged memories as background, not as user-visible claims:

```text
Relevant prior context may include:
- {preview}

Use this only if it helps answer the caller's current request.
If the caller contradicts it, trust the caller.
```

For low-latency calls, use Predict + Memory in the hot path and reserve
Hydration + Reasoning for cases where the caller asks for exact historical
detail.
