# ndpa — TypeScript / JavaScript SDK

Predictive memory layer for AI. **Stop retrieving. Start predicting.**

## Install

```bash
npm install ndpa
```

## Quickstart

```typescript
import { Client } from "ndpa";

const client = new Client({
  apiKey: process.env.NDPA_API_KEY!,
  platform: "my_app",
});

// Single turn
await client.logTurn("chat_abc", "user", "What's the migration status?");

// Full exchange
await client.logExchange(
  "chat_abc",
  "What's the status?",
  "The migration is 80% done.",
);

// Tool call
await client.logTurn("chat_abc", "tool", undefined, {
  tool_name: "Read",
  source_path: "/repo/migration.sql",
});
```

Events are sent asynchronously by default — your AI loop never waits on the network.

## License

MIT
