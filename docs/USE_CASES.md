# NDPA Use Cases

NDPA is a four-layer memory system:

| Layer | Job |
|---|---|
| Predict | Stage likely context before a query exists |
| Memory | Retrieve handles, previews, and scores for an explicit query |
| Hydration | Fetch raw context for selected handles |
| Reasoning | Optionally answer over retrieved or hydrated context |

## Voice AI

Voice agents cannot hide memory latency behind a long user pause. NDPA Predict
can stage likely context between turns or before a call starts, while Memory
and Hydration stay available when the caller asks for exact history.

Use NDPA for:

- Returning caller history handles before the model speaks.
- Hydrating exact transcript snippets only when needed.
- Keeping raw call transcripts out of the hot metadata path.
- Running Reasoning only when the voice stack needs answer synthesis.

Measure Predict with PQHR, Memory with hit@K, Hydration with latency and bytes,
and Reasoning with answer accuracy. Do not combine those into one score.

## Chat Apps

Chat products need continuity across sessions without stuffing every old
conversation into the prompt. NDPA can keep compact memory metadata hot and
fetch raw messages only after the app selects relevant handles.

Use NDPA for:

- "Pick up where we left off" context.
- User preference and project continuity.
- Session summaries with storage keys for exact Hydration.
- Optional Reasoning over selected memory context.

The core cost advantage is that Predict and Memory do not require an LLM or
embedding call per request.

## Code Agents

Coding agents repeatedly need repository, issue, decision, and debugging
context. NDPA Predict can stage likely files or prior sessions from recent
trajectory; Memory can retrieve explicit handles when the agent names a symbol,
file, bug, or task.

Use NDPA for:

- Prior implementation decisions and debugging history.
- File and symbol context handles.
- Project-specific preferences and constraints.
- Hydrating full logs or prior transcripts only after top-K selection.

For code agents, Hydration should be byte-aware because logs and transcripts
can be much larger than ordinary chat snippets.

## Support and CRM

Support and CRM workflows need the right customer history without pulling a
full account archive into every model call. NDPA keeps compact metadata in the
hot path and fetches raw notes, tickets, or transcripts only when selected.

Use NDPA for:

- Pre-staging likely account context before an agent opens a thread.
- Retrieving recent tickets, preferences, or unresolved issues by handle.
- Hydrating exact CRM notes or transcript sections for final response drafting.
- Passing selected context into an existing support LLM or human-agent UI.

For regulated or enterprise deployments, price storage retention, Hydration
bytes, and Reasoning separately from core Predict and Memory requests.
