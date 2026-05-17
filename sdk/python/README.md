# ndpa - Python SDK

Predict-forward memory for AI apps. The SDK exposes the four NDPA layers:
Predict, Memory, Hydration, and Reasoning.

## Install

```bash
pip install ndpa
```

## Quickstart

```python
from ndpa import Client

client = Client(api_key="ndpa_your_key", platform="my_app")

# Log a single turn
client.log_turn(
    session_id="chat_abc",
    role="user",
    content="What's the status of the migration?",
)

# Log an exchange
client.log_exchange(
    session_id="chat_abc",
    user_message="What's the status?",
    assistant_message="The migration is 80% done.",
)

# Log tool calls
client.log_turn(
    session_id="chat_abc",
    role="tool",
    tool_name="Read",
    source_path="/repo/migration.sql",
)

# Predict: pre-stage likely context before the next turn
client.stage("chat_abc", query="What's the migration status?", k=5)

# Memory: retrieve handles/previews/scores
memory = client.get_predictions("chat_abc", query="What's the migration status?", k=5)
first = memory["predictions"][0]

# Hydration: fetch raw context only when needed
raw = client.hydrate(memory_handles=[first["memory_handle"]])

# Reasoning: optional premium answer layer
answer = client.reasoning("What's the migration status?", session_id="chat_abc", k=5)
```

By default the client sends events in a background thread — your AI loop never waits on the network.

## Control Plane

```python
client.get_config()       # public products, pricing, metrics
client.get_account()      # current API key metadata
client.get_usage(days=30) # usage summary
client.checkout()         # billing handoff if configured
```

Admin key minting:

```python
client.admin_create_api_key(
    admin_token="...",
    user_id="customer_workspace",
    platform="my_app",
)
```

## Pricing Shape

- Predict: $0.03 / 1K staged prediction requests
- Memory: $0.10 / 1K retrieval requests
- Hydration: $0.05 / 1K fetches plus bytes
- Reasoning BYOK: $5 / 1K orchestration calls
- Reasoning Hosted: $40 / 1K answer-generation calls

## License

MIT
