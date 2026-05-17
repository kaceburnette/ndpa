# NDPA Integration Guide

For platform engineers wiring NDPA into an AI assistant, IDE, or chat app.

## 30-second integration

```typescript
import { Client } from "ndpa";

const ndpa = new Client({ apiKey: process.env.NDPA_API_KEY });

// On every user message:
await ndpa.logTurn(sessionId, "user", userMessage);

// Before generating a response:
const { predictions } = await ndpa.getPredictions(sessionId, { k: 3 });
const stagedContext = predictions
  .map(p => p.content_preview || p.content)
  .join("\n");

// Pass stagedContext into your LLM call as system context.
```

That's it. 5 lines. Works with any LLM.

Predictive memory index for AI apps. Core returns handles/previews/scores;
Hydration fetches raw context only when needed.

## What integrating gives you

**Without NDPA** (your AI app today):
- User starts a new session
- AI has zero memory of past sessions
- User has to re-establish context every time
- Or you pay for cross-session retrieval via embeddings + vector DB

**With NDPA**:
- Every user message + tool call → NDPA logs it (one async call)
- On next turn, you query NDPA for relevant past context
- NDPA returns top-K past conversations/files, scored by behavior + topic
- Inject into your system prompt; AI responds with context already loaded

## Integration patterns

### Pattern 1: Pure conversation memory (ChatGPT/Claude style)

For a chat app where conversations are the main signal:

```python
from ndpa import Client
client = Client(api_key=API_KEY, platform="my_chat_app")

# Log each exchange
client.log_exchange(session_id, user_msg, assistant_msg)

# Before responding to a new message:
client.stage(session_id, query=new_user_msg, k=3)
result = client.get_predictions(session_id, query=new_user_msg, k=3)
context = "\n".join(p["content"][:500] for p in result["predictions"])
```

Cost: 2-3 API calls per turn depending on whether you stage. The cold
FastAPI -> Supabase pooler path can be hundreds of ms, so latency-sensitive
apps should call `/stage` before generation or between turns and then read
`/predictions` from the warm staged cache.

Hosted responses should be treated as Memory handles plus previews. Use
`memory_handle` / `session_id`, `content_preview`, `score`, and `storage_key`
in the hot path. Fetch raw context through Hydration only when the selected
handle needs full detail.

### Pattern 2: Agent / coding tool (Cursor/Continue style)

For a tool that fires events on file operations:

```python
# Each tool invocation:
client.log_events(session_id, [
    {"role": "tool", "tool_name": "read_file", "source_path": path, "ts": time.time()}
])

# Before next user message arrives, stage context:
client.stage(session_id, k=5)
predictions = client.get_predictions(session_id, k=5)
# Use file kernel predictions: predictions[i]["source_path"] is what to pre-load
```

### Pattern 3: Multi-platform sync

User has accounts on multiple platforms. Each platform reports under its
own `platform` tag:

```python
ChatGPTClient = Client(api_key=KEY, platform="chatgpt")
ClaudeClient = Client(api_key=KEY, platform="claude")
```

Same `user_id` (derived from key) → predictions span platforms. ChatGPT
session signal informs Claude session predictions.

## Platforms with reference integrations

- **Claude Code**: `collector/hook.py` (PostToolUse + UserPromptSubmit hooks)
- **ChatGPT export**: `eval/chatgpt_import.py` (bulk import from data export)
- **Generic Python**: `sdk/python/examples/basic.py`
- **Generic TypeScript**: `sdk/typescript/examples/basic.ts`

## SDK reference

### Python

```python
pip install ndpa
```

```python
from ndpa import Client

c = Client(api_key="ndpa_...", platform="myapp")
c.log_turn(session_id, role, content)           # single turn
c.log_events(session_id, events)                # batch up to 1000
c.log_exchange(session_id, user, assistant)     # convenience for chat apps
c.stage(session_id, query=None, k=5)            # precompute warm result
c.get_predictions(session_id, query=None, k=5)  # returns top-K past context
c.hydrate(memory_handles=[...])                 # fetch raw context if needed
c.reasoning("question", session_id=session_id)  # premium answer layer
c.get_config()                                  # public prices/metrics
c.get_usage(days=30)                            # usage summary
c.checkout()                                    # billing handoff
```

### TypeScript

```bash
npm install ndpa
```

```typescript
import { Client } from "ndpa";

const c = new Client({ apiKey: "ndpa_...", platform: "myapp" });
await c.logTurn(sessionId, role, content);
await c.logEvents(sessionId, events);
await c.logExchange(sessionId, userMsg, assistantMsg);
await c.stage(sessionId, { query, k: 5 });
const { predictions } = await c.getPredictions(sessionId, { query, k: 5 });
const { contexts } = await c.hydrate({ memoryHandles: [predictions[0].memory_handle!] });
const answer = await c.reasoning(query, { sessionId, k: 5 });
const usage = await c.getUsage({ days: 30 });
```

Both SDKs are **zero dependencies** and **async-by-default**. Event posting
is fire-and-forget; latency to your hot path is <10ms.

## Self-hosting

For platforms that want to run NDPA on their own infra:

1. Create a Postgres/Supabase project with the clean metadata schema
2. Store raw memory in local FS, S3, R2, or another blob/object store
3. Deploy the FastAPI server in `server/`
4. Point SDKs at your URL:

```python
client = Client(api_key=KEY, base_url="https://api.your-domain.com")
```

## API contracts

### `GET /config`

Public products, pricing, metrics, and whether billing is configured.

### `GET /account`

Authenticated metadata for the current API key.

### `GET /usage`

Authenticated usage summary from `ndp_usage_events`.

### `POST /events`

```json
{
  "session_id": "uuid-or-string",
  "platform": "claude_code",
  "events": [
    {"role": "user", "content": "...", "ts": 1700000000.0},
    {"role": "tool", "tool_name": "Read", "source_path": "/x.py", "ts": 1700000001.0}
  ]
}
```

Response: `200 { inserted: N, invalidated_predictions: N }`.
Errors: `400` (validation), `401` (auth), `413` (size), `429` (rate limit), `500`.

### `POST /stage`

Same request shape as `/predictions`. Runs the cold metadata ranking now,
stores the result in the staged cache, and returns `{staged_id, predictions,
latency_ms, expires_in_sec}`.

### `POST /predictions`

```json
{
  "session_id": "current-session-or-empty-string",
  "query": "optional override text",
  "k": 5
}
```

Response:
```json
{
  "predictions": [
    {
      "session_id": "uuid",
      "platform": "chatgpt",
      "score": 0.92,
      "recency_score": 0.85,
      "topic_score": 0.95,
      "memory_handle": "uuid",
      "content_preview": "short preview for hot path use",
      "storage_key": "conversations/user/session.ndp",
      "content_bytes": 12048,
      "content": "legacy preview alias",
      "started_at": "2025-08-12T..."
    }
  ],
  "candidates_considered": 668,
  "k": 5
}
```

### `POST /hydrate`

```json
{
  "memory_handles": ["session_123"],
  "storage_keys": ["local://..."],
  "end_user_id": "optional"
}
```

Response: `{ "contexts": [...], "latency_ms": 1 }`.

### `POST /reasoning`

Premium answer layer: Memory -> optional Hydration -> LLM answer.

```json
{
  "session_id": "current-session-or-empty-string",
  "query": "What should the assistant remember?",
  "k": 5,
  "hydrate": true
}
```

Hosted mode uses `OPENAI_API_KEY` configured on the NDPA API. BYOK mode sends
the customer key as `X-OpenAI-API-Key`; NDPA does not store it.

### `POST /checkout`

Authenticated billing handoff. Returns a configured Stripe payment link when
`NDPA_STRIPE_PAYMENT_LINK` is set.

### `POST /admin/api-keys`

Admin-only key minting. Requires `X-NDPA-Admin-Token`.

### `GET /health`

Returns `{status: "ok", ts: ...}`. No auth.

## Rate limits

Current in-process limits:

- `/events`: 600 req/min/key
- `/stage`: 300 req/min/key
- `/predictions`: 300 req/min/key
- `/hydrate`: 120 req/min/key
- `/reasoning`: 60 req/min/key

When exceeded: `429`. For multi-machine deployments, replace the in-memory
limiter with Redis or another shared counter.

For enterprise volume (>10k req/min), contact for a custom limit.

## Common questions

**Q: Will this slow down my AI app?**
A: Event posting is async. For latency-sensitive apps, call `/stage` before
the answer is needed. Then `/predictions` returns the staged handles from the
warm cache; use `/hydrate` only for selected raw context.

**Q: What if my user is offline?**
A: SDK queues locally. Auto-flushes when network returns.

**Q: Can I use NDPA without the SDK?**
A: Yes. HTTPS + Bearer auth. Any language.

**Q: How do I handle GDPR deletion?**
A: One SQL query: `DELETE FROM ndp_* WHERE user_id = $1`. See SECURITY.md.

**Q: How does pricing work?**
A: Preview pricing is in `docs/PRICING.md`: Predict $0.03/1K, Memory
$0.10/1K, Hydration $0.05/1K plus bytes, Reasoning BYOK $5/1K, hosted
Reasoning $40/1K.
