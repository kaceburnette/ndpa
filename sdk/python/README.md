# ndpa — Python SDK

Predictive memory layer for AI. **Stop retrieving. Start predicting.**

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
```

By default the client sends events in a background thread — your AI loop never waits on the network.

## License

MIT
