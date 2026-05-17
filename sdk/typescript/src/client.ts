// NDPA TypeScript client — drop-in instrumentation for any AI conversation loop.

declare const process:
  | { env?: { NDPA_BASE_URL?: string } }
  | undefined;

const DEFAULT_API_URL = "http://localhost:8000";
const DEFAULT_BASE_URL = process?.env?.NDPA_BASE_URL ?? DEFAULT_API_URL;
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

export interface HydratedContext {
  memory_handle?: string | null;
  session_id?: string | null;
  storage_key: string;
  content: string;
  content_bytes: number;
  found: boolean;
  source: string;
  error?: string | null;
}

export interface PredictionTiming {
  auth_ms?: number;
  db_candidate_ms?: number;
  rank_ms?: number;
  serialize_ms?: number;
  total_ms?: number;
  cache_hit?: boolean;
  retrieve_ms?: number;
  hydrate_ms?: number;
  llm_ms?: number;
  byok?: boolean;
}

export interface ReasoningResponse {
  answer: string;
  model: string;
  mode: "hosted" | "byok" | string;
  latency_ms: number;
  contexts_used: number;
  memory_handles: string[];
  timing?: PredictionTiming;
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
   *
   * For platforms (multi-tenant integrations): pass `endUserId` to scope
   * predictions to one of YOUR users.
   */
  async getPredictions(
    sessionId: string = "",
    options: { query?: string; k?: number; endUserId?: string } = {},
  ): Promise<{
    predictions: Array<{
      memory_handle?: string;
      session_id: string;
      platform: string;
      score: number;
      recency_score: number;
      topic_score: number;
      content_preview?: string;
      storage_key?: string | null;
      content_bytes?: number | null;
      content: string;
      token_estimate: number;
      started_at: string;
    }>;
    candidates_considered?: number;
    elapsed_ms?: number;
    latency_ms?: number;
    timing?: PredictionTiming;
    reason?: string;
  }> {
    const payload: Record<string, unknown> = {
      session_id: sessionId,
      k: options.k ?? 5,
    };
    if (options.query !== undefined) payload.query = options.query;
    if (options.endUserId !== undefined) payload.end_user_id = options.endUserId;
    const result = await this.postPath("/predictions", payload);
    return (result as any) || { predictions: [] };
  }

  /**
   * Precompute and cache top-K memory handles for a later hot read.
   * Call this between turns, before a voice call starts, or from a background
   * worker. A later getPredictions() call with the same key returns from the
   * staged cache.
   */
  async stage(
    sessionId: string = "",
    options: { query?: string; k?: number; endUserId?: string } = {},
  ): Promise<{
    staged_id: string;
    predictions: Array<{
      memory_handle?: string;
      session_id: string;
      platform: string;
      score: number;
      recency_score: number;
      topic_score: number;
      content_preview?: string;
      storage_key?: string | null;
      content_bytes?: number | null;
      content: string;
      token_estimate: number;
      started_at: string;
    }>;
    latency_ms?: number;
    mode?: string;
    expires_in_sec?: number;
    expires_at?: number;
    timing?: PredictionTiming;
  }> {
    const payload: Record<string, unknown> = {
      session_id: sessionId,
      k: options.k ?? 5,
    };
    if (options.query !== undefined) payload.query = options.query;
    if (options.endUserId !== undefined) payload.end_user_id = options.endUserId;
    const result = await this.postPath("/stage", payload);
    return (result as any) || { predictions: [] };
  }

  /**
   * Fetch raw context for selected memory handles/storage keys.
   * Hydration is separate from Memory retrieval and should only be called for
   * the top handles that actually need full context.
   */
  async hydrate(
    options: {
      memoryHandles?: string[];
      sessionIds?: string[];
      storageKeys?: string[];
      endUserId?: string;
    } = {},
  ): Promise<{ contexts: HydratedContext[]; latency_ms?: number }> {
    const payload: Record<string, unknown> = {
      memory_handles: options.memoryHandles ?? [],
      session_ids: options.sessionIds ?? [],
      storage_keys: options.storageKeys ?? [],
    };
    if (options.endUserId !== undefined) payload.end_user_id = options.endUserId;
    const result = await this.postPath("/hydrate", payload);
    return (result as any) || { contexts: [] };
  }

  /**
   * Premium answer layer: Memory -> optional Hydration -> LLM answer.
   *
   * Pass `openaiApiKey` for BYOK mode. It is sent as X-OpenAI-API-Key and is
   * not stored by the NDPA API.
   */
  async reasoning(
    query: string,
    options: {
      sessionId?: string;
      k?: number;
      endUserId?: string;
      model?: string;
      contextCharLimit?: number;
      hydrate?: boolean;
      openaiApiKey?: string;
    } = {},
  ): Promise<ReasoningResponse> {
    const payload: Record<string, unknown> = {
      query,
      session_id: options.sessionId ?? "",
      k: options.k ?? 5,
      hydrate: options.hydrate ?? true,
    };
    if (options.endUserId !== undefined) payload.end_user_id = options.endUserId;
    if (options.model !== undefined) payload.model = options.model;
    if (options.contextCharLimit !== undefined) payload.context_char_limit = options.contextCharLimit;
    const headers = options.openaiApiKey ? { "X-OpenAI-API-Key": options.openaiApiKey } : undefined;
    return (await this.postPath("/reasoning", payload, { extraHeaders: headers })) as ReasoningResponse;
  }

  async getConfig(): Promise<Record<string, unknown>> {
    return (await this.getPath("/config", { auth: false })) as Record<string, unknown>;
  }

  async getAccount(): Promise<Record<string, unknown>> {
    return (await this.getPath("/account")) as Record<string, unknown>;
  }

  async getUsage(options: { days?: number } = {}): Promise<Record<string, unknown>> {
    return (await this.getPath(`/usage?days=${options.days ?? 30}`)) as Record<string, unknown>;
  }

  async checkout(): Promise<Record<string, unknown>> {
    return (await this.postPath("/checkout", {})) as Record<string, unknown>;
  }

  async adminCreateApiKey(options: {
    adminToken: string;
    userId: string;
    platform?: string;
  }): Promise<Record<string, unknown>> {
    const payload: Record<string, unknown> = {
      user_id: options.userId,
      platform: options.platform ?? null,
    };
    return (await this.postPath("/admin/api-keys", payload, {
      auth: false,
      extraHeaders: { "X-NDPA-Admin-Token": options.adminToken },
    })) as Record<string, unknown>;
  }

  async logEvents(
    sessionId: string,
    events: Event[],
    options: { endUserId?: string } = {},
  ): Promise<void> {
    if (!events || events.length === 0) return;
    for (let i = 0; i < events.length; i += BATCH_SIZE_LIMIT) {
      const chunk = events.slice(i, i + BATCH_SIZE_LIMIT);
      const payload: Record<string, unknown> = { session_id: sessionId, events: chunk };
      if (this.platform) payload.platform = this.platform;
      if (options.endUserId !== undefined) payload.end_user_id = options.endUserId;
      if (this.asyncSend) {
        void this.postPath("/events", payload).catch((e) => this.onError?.(e as Error));
      } else {
        await this.postPath("/events", payload);
      }
    }
  }

  private async getPath(
    path: string,
    options: { auth?: boolean; extraHeaders?: Record<string, string> } = {},
  ): Promise<unknown> {
    const ctrl = new AbortController();
    const timeoutHandle = setTimeout(() => ctrl.abort(), this.timeoutMs);
    const headers: Record<string, string> = {
      "User-Agent": "ndpa-typescript/0.1.0",
      ...(options.extraHeaders ?? {}),
    };
    if (options.auth !== false) headers.Authorization = `Bearer ${this.apiKey}`;
    try {
      const resp = await fetch(`${this.baseUrl}${path}`, {
        method: "GET",
        headers,
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

  private async postPath(
    path: string,
    payload: unknown,
    options: { auth?: boolean; extraHeaders?: Record<string, string> } = {},
  ): Promise<unknown> {
    const ctrl = new AbortController();
    const timeoutHandle = setTimeout(() => ctrl.abort(), this.timeoutMs);
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      "User-Agent": "ndpa-typescript/0.1.0",
      ...(options.extraHeaders ?? {}),
    };
    if (options.auth !== false) headers.Authorization = `Bearer ${this.apiKey}`;
    try {
      const resp = await fetch(`${this.baseUrl}${path}`, {
        method: "POST",
        headers,
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
