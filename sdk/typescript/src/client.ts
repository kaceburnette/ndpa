// NDPA TypeScript client — drop-in instrumentation for any AI conversation loop.

const DEFAULT_BASE_URL = "https://izackuempnhgoojtbgvi.supabase.co/functions/v1";
const DEFAULT_TIMEOUT_MS = 5000;
const BATCH_SIZE_LIMIT = 1000;

export type Role = "user" | "assistant" | "tool" | "system";

export interface Event {
  role: Role;
  content?: string;
  tool_name?: string;
  source_path?: string;
  ts?: number;
}

export interface ClientOptions {
  apiKey: string;
  baseUrl?: string;
  platform?: string;
  timeoutMs?: number;
  asyncSend?: boolean;
  onError?: (err: Error) => void;
}

export class NDPAError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "NDPAError";
  }
}

export class Client {
  private apiKey: string;
  private baseUrl: string;
  private platform?: string;
  private timeoutMs: number;
  private asyncSend: boolean;
  private onError?: (err: Error) => void;

  constructor(opts: ClientOptions) {
    if (!opts.apiKey) throw new Error("apiKey is required");
    this.apiKey = opts.apiKey;
    this.baseUrl = (opts.baseUrl ?? DEFAULT_BASE_URL).replace(/\/$/, "");
    this.platform = opts.platform;
    this.timeoutMs = opts.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    this.asyncSend = opts.asyncSend ?? true;
    this.onError = opts.onError;
  }

  async logTurn(
    sessionId: string,
    role: Role,
    content?: string,
    extras: { tool_name?: string; source_path?: string; ts?: number } = {},
  ): Promise<void> {
    const event: Event = { role, ts: extras.ts ?? Date.now() / 1000 };
    if (content !== undefined) event.content = content;
    if (extras.tool_name !== undefined) event.tool_name = extras.tool_name;
    if (extras.source_path !== undefined) event.source_path = extras.source_path;
    await this.logEvents(sessionId, [event]);
  }

  async logExchange(
    sessionId: string,
    userMessage: string,
    assistantMessage: string,
    ts?: number,
  ): Promise<void> {
    const now = ts ?? Date.now() / 1000;
    await this.logEvents(sessionId, [
      { role: "user", content: userMessage, ts: now },
      { role: "assistant", content: assistantMessage, ts: now + 0.001 },
    ]);
  }

  /**
   * Fetch the top-K past conversations relevant to the current session.
   * This is the read side of NDPA — what the AI should know before responding.
   */
  async getPredictions(
    sessionId: string = "",
    options: { query?: string; k?: number } = {},
  ): Promise<{
    predictions: Array<{
      session_id: string;
      platform: string;
      score: number;
      recency_score: number;
      topic_score: number;
      content: string;
      token_estimate: number;
      started_at: string;
    }>;
    candidates_considered?: number;
    reason?: string;
  }> {
    const payload: Record<string, unknown> = {
      session_id: sessionId,
      k: options.k ?? 5,
    };
    if (options.query !== undefined) payload.query = options.query;
    const result = await this.postPath("/predictions", payload);
    return (result as any) || { predictions: [] };
  }

  async logEvents(sessionId: string, events: Event[]): Promise<void> {
    if (!events || events.length === 0) return;
    for (let i = 0; i < events.length; i += BATCH_SIZE_LIMIT) {
      const chunk = events.slice(i, i + BATCH_SIZE_LIMIT);
      const payload: Record<string, unknown> = { session_id: sessionId, events: chunk };
      if (this.platform) payload.platform = this.platform;
      if (this.asyncSend) {
        void this.postPath("/events", payload).catch((e) => this.onError?.(e as Error));
      } else {
        await this.postPath("/events", payload);
      }
    }
  }

  private async postPath(path: string, payload: unknown): Promise<unknown> {
    const ctrl = new AbortController();
    const timeoutHandle = setTimeout(() => ctrl.abort(), this.timeoutMs);
    try {
      const resp = await fetch(`${this.baseUrl}${path}`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${this.apiKey}`,
          "Content-Type": "application/json",
          "User-Agent": "ndpa-typescript/0.1.0",
        },
        body: JSON.stringify(payload),
        signal: ctrl.signal,
      });
      if (!resp.ok) {
        const body = await resp.text();
        throw new NDPAError(`NDPA API error ${resp.status}: ${body}`);
      }
      return await resp.json();
    } finally {
      clearTimeout(timeoutHandle);
    }
  }
}
