import { Client } from "../src/index";
import { randomUUID } from "crypto";

async function main() {
  const apiKey = process.env.NDPA_API_KEY;
  if (!apiKey) throw new Error("Set NDPA_API_KEY");

  const sessionId = `example_${randomUUID().slice(0, 8)}`;
  const client = new Client({
    apiKey,
    platform: "example_ts",
    asyncSend: false,
  });

  await client.logExchange(
    sessionId,
    "Hello from the TypeScript SDK.",
    "Got it via TS.",
  );

  await client.logTurn(sessionId, "tool", undefined, {
    tool_name: "Read",
    source_path: "/example/file.ts",
  });

  console.log(`Sent events for session ${sessionId}`);
  const staged = await client.stage(sessionId, {
    query: "What example context is ready?",
    k: 3,
  });
  console.log("staged", staged.latency_ms, "ms");

  const predictions = await client.getPredictions(sessionId, {
    query: "What example context is ready?",
    k: 3,
  });
  console.log("predictions", predictions.predictions.length);

  const firstHandle = predictions.predictions[0]?.memory_handle;
  if (firstHandle) {
    const hydrated = await client.hydrate({ memoryHandles: [firstHandle] });
    console.log("hydrated", hydrated.contexts.length);
  }
}

main().catch(console.error);
