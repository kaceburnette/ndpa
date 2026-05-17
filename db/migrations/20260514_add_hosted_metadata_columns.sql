-- Hosted-mode metadata shape for NDPA.
--
-- This is the storage direction for hosted deployments:
-- - Postgres keeps compact hot metadata/index fields.
-- - Raw full conversation text lives in blob/object storage.
-- - Retrieval scores metadata first and hydrates raw content only for top-K.
--
-- This migration is intentionally separate from the full-content tsvector/GIN
-- migration. It does not create a full-text index over raw conversation text.

ALTER TABLE public.ndp_conversations
  ADD COLUMN IF NOT EXISTS content_preview text,
  ADD COLUMN IF NOT EXISTS top_terms text[],
  ADD COLUMN IF NOT EXISTS storage_key text,
  ADD COLUMN IF NOT EXISTS content_bytes integer;

CREATE INDEX IF NOT EXISTS ndp_conversations_top_terms_idx
  ON public.ndp_conversations USING GIN (top_terms);

CREATE INDEX IF NOT EXISTS ndp_conversations_scope_updated_idx
  ON public.ndp_conversations (user_id, end_user_id, updated_at DESC);

-- Intentionally no all-row backfill here. Backfill compact metadata in small
-- batches only after storage recovery is complete.
