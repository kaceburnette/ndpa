# NDPA reference integrations

Paste-ready wrappers for the major AI APIs. Drop into your platform's
request loop, add `ndpa_key` and `session_id`, and the user's relevant
past context is staged into every call.

## What's here

| File                              | Wraps                          | Lines |
|-----------------------------------|--------------------------------|-------|
| `openai_chat.py`                  | OpenAI Chat Completions        | ~100  |
| `anthropic_messages.py`           | Anthropic Messages API         | ~95   |
| `langchain_memory.py`             | LangChain BaseChatMemory       | ~110  |

## Quick OpenAI example

```python
from openai import OpenAI
from integrations.openai_chat import predict_then_complete

openai = OpenAI()
response = predict_then_complete(
    openai,
    ndpa_key="ndpa_your_key",
    session_id=f"chat_{user.id}",
    end_user_id=user.id,             # multi-tenant: scope to your user
    messages=[{"role": "user", "content": user_input}],
    model="gpt-4o",
)
```

NDPA stages relevant past conversation context (titles only, ~50 tokens)
as a leading system message. The user's exchange gets logged back to NDPA
for future predictions.

## Quick Anthropic example

```python
from anthropic import Anthropic
from integrations.anthropic_messages import predict_then_send

anthropic = Anthropic()
response = predict_then_send(
    anthropic,
    ndpa_key="ndpa_your_key",
    session_id=f"chat_{user.id}",
    end_user_id=user.id,
    messages=[{"role": "user", "content": user_input}],
    model="claude-sonnet-4-6",
    max_tokens=1024,
)
```

## LangChain example

```python
from integrations.langchain_memory import NDPAMemory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

chain = ConversationChain(
    llm=ChatOpenAI(),
    memory=NDPAMemory(
        ndpa_key="ndpa_your_key",
        session_id="chat_xyz",
        end_user_id="user_123",
    ),
)
chain.predict(input="continue our last discussion")
```

## Multi-tenant model

For platforms (you have many users): use ONE NDPA `api_key` per platform and
pass `end_user_id` to scope events/predictions to your user. NDPA partitions
storage and prediction queries by `(your_user_id, end_user_id)`.

For single-user apps: omit `end_user_id`; events scope to your single key.

## How it composes with existing memory

NDPA stages **predicted context**. If you also use Mem0/Zep/etc. for fact
retrieval, run both: NDPA's leading system message + your retrieved facts.
Different memory types, complementary use cases.

## Latency

- Predictions call: 100ms–1.5s (depending on user history size)
- Event logging: async fire-and-forget (<10ms hot path)
- Adds one round-trip per turn. Worth it for the context lift.

For latency-critical apps, fetch predictions in parallel with prompt
preparation:

```python
import asyncio
preds_task = asyncio.create_task(ndpa.get_predictions(...))
# ... do other prep ...
preds = await preds_task
```
