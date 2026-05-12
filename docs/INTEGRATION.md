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
const stagedContext = predictions.map(p => p.content).join("\n");

// Pass stagedContext into your LLM call as system context.
```

That's it. 5 lines. Works with any LLM.

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
result = client.get_predictions(session_id, query=new_user_msg, k=3)
context = "\n".join(p["content"][:500] for p in result["predictions"])
```

Cost: 2 API calls per turn. Latency: <300ms each, both async.

### Pattern 2: Agent / coding tool (Cursor/Continue style)

For a tool that fires events on file operations:

```python
# Each tool invocation:
client.log_events(session_id, [
    {"role": "tool", "tool_name": "read_file", "source_path": path, "ts": time.time()}
])

# Before next user message arrives, async-prefetch context:
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
c.get_predictions(session_id, query=None, k=5)  # returns top-K past context
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
const { predictions } = await c.getPredictions(sessionId, { query, k: 5 });
```

Both SDKs are **zero dependencies** and **async-by-default**. Event posting
is fire-and-forget; latency to your hot path is <10ms.

## Self-hosting

For platforms that want to run NDPA on their own infra:

1. Create a Supabase project (or use any Postgres + edge function runtime)
2. Run migrations: `supabase/migrations/*.sql`
3. Deploy edge functions: `supabase/functions/{events,predictions,health}`
4. Point SDKs at your URL:

```python
client = Client(api_key=KEY, base_url="https://your-domain.com/v1")
```

## API contracts

### `POST /v1/events`

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

Response: `200 { ok: true, received: N, session_id, turn_count }`.
Errors: `400` (validation), `401` (auth), `413` (size), `429` (rate limit), `500`.

### `POST /v1/predictions`

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
      "content": "first 5000 chars",
      "started_at": "2025-08-12T..."
    }
  ],
  "candidates_considered": 668,
  "k": 5
}
```

### `GET /v1/health`

Returns `{healthy: true|false, db: "healthy", latency_ms}`. No auth.

## Rate limits

600 requests per minute per API key, per endpoint. Headers on every response:

- `x-ratelimit-limit: 600`
- `x-ratelimit-remaining: 599`

When exceeded: `429` with `{ error: "rate_limit_exceeded" }`. Retry after the
next minute boundary.

For enterprise volume (>10k req/min), contact for a custom limit.

## Common questions

**Q: Will this slow down my AI app?**
A: No. Event posting is async (returns in <10ms). Predictions are <2s but
you call them before generation, not during.

**Q: What if my user is offline?**
A: SDK queues locally. Auto-flushes when network returns.

**Q: Can I use NDPA without the SDK?**
A: Yes. HTTPS + Bearer auth. Any language.

**Q: How do I handle GDPR deletion?**
A: One SQL query: `DELETE FROM ndp_* WHERE user_id = $1`. See SECURITY.md.

**Q: How does pricing work?**
A: Currently free / open source. Hosted enterprise plan available for
multi-tenant deployments. See pricing page.
