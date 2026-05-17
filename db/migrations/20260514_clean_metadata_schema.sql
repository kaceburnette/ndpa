-- NDPA clean metadata schema. Postgres is metadata/control plane only.
-- Raw conversation text lives in local FS / R2 / S3, fetched via /hydrate.
-- No tsv_content, no full-content GIN, no raw event log in Postgres.

CREATE TABLE IF NOT EXISTS public.ndp_api_keys (
  id          uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id     text        NOT NULL,
  platform    text,
  key_hash    text        NOT NULL UNIQUE,
  revoked_at  timestamptz,
  created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ndp_api_keys_user_id_idx ON public.ndp_api_keys (user_id);

CREATE TABLE IF NOT EXISTS public.ndp_sessions (
  user_id      text        NOT NULL,
  end_user_id  text,
  session_id   text        NOT NULL,
  platform     text,
  turn_count   integer     NOT NULL DEFAULT 0,
  last_active  timestamptz,
  created_at   timestamptz NOT NULL DEFAULT now(),
  UNIQUE (user_id, session_id)
);
CREATE INDEX IF NOT EXISTS ndp_sessions_user_enduser_idx ON public.ndp_sessions (user_id, end_user_id);
CREATE INDEX IF NOT EXISTS ndp_sessions_user_active_idx  ON public.ndp_sessions (user_id, last_active DESC);

CREATE TABLE IF NOT EXISTS public.ndp_conversations (
  user_id          text        NOT NULL,
  end_user_id      text,
  session_id       text        NOT NULL,
  platform         text,
  top_terms        text[]      NOT NULL DEFAULT '{}',
  content_preview  text,
  storage_key      text,
  content_bytes    integer     NOT NULL DEFAULT 0,
  updated_at       timestamptz NOT NULL DEFAULT now(),
  created_at       timestamptz NOT NULL DEFAULT now(),
  UNIQUE (user_id, session_id)
);
CREATE INDEX IF NOT EXISTS ndp_conversations_user_enduser_idx ON public.ndp_conversations (user_id, end_user_id);
CREATE INDEX IF NOT EXISTS ndp_conversations_user_updated_idx ON public.ndp_conversations (user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS ndp_conversations_top_terms_idx    ON public.ndp_conversations USING GIN (top_terms);

CREATE TABLE IF NOT EXISTS public.ndp_usage_events (
  id          uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id     text        NOT NULL,
  product     text        NOT NULL,  -- predict | memory | hydration | reasoning
  endpoint    text,
  units       integer     NOT NULL DEFAULT 1,
  bytes       integer     NOT NULL DEFAULT 0,
  latency_ms  integer,
  created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ndp_usage_events_user_created_idx    ON public.ndp_usage_events (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS ndp_usage_events_product_created_idx ON public.ndp_usage_events (product, created_at DESC);

ALTER TABLE public.ndp_api_keys      ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.ndp_sessions      ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.ndp_conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.ndp_usage_events  ENABLE ROW LEVEL SECURITY;

-- Intentionally NO permissive policies. Public access goes through FastAPI
-- with bearer-token auth over the service-role DATABASE_URL, which bypasses
-- RLS. Direct supabase-js anon/authenticated access is default-deny.
