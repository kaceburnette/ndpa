# ndpa - TypeScript / JavaScript SDK

Predict-forward memory for AI apps. The SDK exposes the four NDPA layers:
Predict, Memory, Hydration, and Reasoning.

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

// Predict: pre-stage likely context before the next turn
await client.stage("chat_abc", {
  query: "What's the migration status?",
  k: 5,
});

// Memory: retrieve handles/previews/scores
const memory = await client.getPredictions("chat_abc", {
  query: "What's the migration status?",
  k: 5,
});
const first = memory.predictions[0];

// Hydration: fetch raw context only when needed
if (first?.memory_handle) {
  await client.hydrate({ memoryHandles: [first.memory_handle] });
}

// Reasoning: optional premium answer layer
const answer = await client.reasoning("What's the migration status?", {
  sessionId: "chat_abc",
  k: 5,
});
```

Events are sent asynchronously by default — your AI loop never waits on the network.

## Control Plane

```typescript
await client.getConfig();          // public products, pricing, metrics
await client.getAccount();         // current API key metadata
await client.getUsage({ days: 30 }); // usage summary
await client.checkout();           // billing handoff if configured
```

Admin key minting:

```typescript
await client.adminCreateApiKey({
  adminToken: process.env.NDPA_ADMIN_TOKEN!,
  userId: "customer_workspace",
  platform: "my_app",
});
```

## Pricing Shape

- Predict: $0.03 / 1K staged prediction requests
- Memory: $0.10 / 1K retrieval requests
- Hydration: $0.05 / 1K fetches plus bytes
- Reasoning BYOK: $5 / 1K orchestration calls
- Reasoning Hosted: $40 / 1K answer-generation calls

## License

MIT
