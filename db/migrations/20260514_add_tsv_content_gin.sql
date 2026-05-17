-- Add and maintain full-text index support for FastAPI/Postgres retrieval.
--
-- WARNING: Do not run this on free-tier Supabase projects. Materialized
-- full-conversation tsvector plus a GIN index can exceed the 500MB database
-- limit quickly. The FastAPI server supports a lightweight fallback when this
-- column is absent. Reserve this migration for larger Postgres instances.
--
-- Safe to run more than once on a database with enough storage headroom.

ALTER TABLE ndp_conversations
  ADD COLUMN IF NOT EXISTS tsv_content tsvector;

UPDATE ndp_conversations
SET tsv_content = to_tsvector('english', COALESCE(content, ''))
WHERE tsv_content IS NULL;

CREATE INDEX IF NOT EXISTS ndp_conversations_tsv_idx
  ON ndp_conversations USING GIN (tsv_content);

CREATE OR REPLACE FUNCTION ndp_update_tsv_content()
RETURNS trigger AS $$
BEGIN
  NEW.tsv_content := to_tsvector('english', COALESCE(NEW.content, ''));
  RETURN NEW;
END
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS ndp_conversations_tsv_update ON ndp_conversations;

CREATE TRIGGER ndp_conversations_tsv_update
BEFORE INSERT OR UPDATE OF content ON ndp_conversations
FOR EACH ROW
EXECUTE FUNCTION ndp_update_tsv_content();
