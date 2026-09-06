-- Keep Date Night data behind the FastAPI session boundary.
--
-- Date Night intentionally does not use Supabase Auth. The browser never talks
-- to PostgREST directly; FastAPI authenticates the opaque HttpOnly session,
-- checks room membership, then connects with the postgres DATABASE_URL role.
-- RLS therefore remains deny-by-default (no client policies) and the two Data
-- API roles receive no table or sequence privileges.

ALTER TABLE public.date_night_accounts ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.date_night_bootstrap ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.date_night_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.date_night_rooms ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.date_night_memberships ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.date_night_invites ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.date_night_messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.date_night_ai_cache ENABLE ROW LEVEL SECURITY;

-- No browser identity is mapped into PostgreSQL, so any client-facing policy
-- would be misleading at best and a data leak at worst. Remove policies left
-- by dashboard experiments or earlier deployments.
DO $migration$
DECLARE
  existing_policy record;
BEGIN
  FOR existing_policy IN
    SELECT schemaname, tablename, policyname
    FROM pg_policies
    WHERE schemaname = 'public'
      AND tablename = ANY (ARRAY[
        'date_night_accounts',
        'date_night_bootstrap',
        'date_night_sessions',
        'date_night_rooms',
        'date_night_memberships',
        'date_night_invites',
        'date_night_messages',
        'date_night_ai_cache'
      ])
  LOOP
    EXECUTE format(
      'DROP POLICY %I ON %I.%I',
      existing_policy.policyname,
      existing_policy.schemaname,
      existing_policy.tablename
    );
  END LOOP;
END
$migration$;

-- This event-trigger function runs internally after DDL. Browser-facing roles
-- never need to invoke it through PostgREST.
REVOKE EXECUTE ON FUNCTION public.rls_auto_enable() FROM PUBLIC, anon, authenticated;

-- PUBLIC grants would flow through to every login role, including Supabase's
-- anon/authenticated roles, so remove them as well as any explicit grants.
REVOKE ALL PRIVILEGES ON TABLE
  public.date_night_accounts,
  public.date_night_bootstrap,
  public.date_night_sessions,
  public.date_night_rooms,
  public.date_night_memberships,
  public.date_night_invites,
  public.date_night_messages,
  public.date_night_ai_cache
FROM PUBLIC;

REVOKE ALL PRIVILEGES ON SEQUENCE public.date_night_messages_id_seq FROM PUBLIC;

-- These roles exist on Supabase, but the guards keep this migration portable
-- to plain PostgreSQL and local test databases.
DO $migration$
DECLARE
  api_role text;
BEGIN
  FOREACH api_role IN ARRAY ARRAY['anon', 'authenticated']
  LOOP
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = api_role) THEN
      EXECUTE format(
        'REVOKE ALL PRIVILEGES ON TABLE public.date_night_accounts, public.date_night_bootstrap, public.date_night_sessions, public.date_night_rooms, public.date_night_memberships, public.date_night_invites, public.date_night_messages, public.date_night_ai_cache FROM %I',
        api_role
      );
      EXECUTE format(
        'REVOKE ALL PRIVILEGES ON SEQUENCE public.date_night_messages_id_seq FROM %I',
        api_role
      );
    END IF;
  END LOOP;
END
$migration$;

COMMENT ON TABLE public.date_night_accounts IS
  'Server-only Date Night data; access is mediated by FastAPI HttpOnly sessions.';
COMMENT ON TABLE public.date_night_sessions IS
  'Server-only opaque session hashes; never expose through the Supabase Data API.';
COMMENT ON TABLE public.date_night_rooms IS
  'Server-only Date Night rooms; FastAPI enforces membership before access.';
COMMENT ON TABLE public.date_night_memberships IS
  'Server-only room authorization records used by FastAPI member checks.';
COMMENT ON TABLE public.date_night_invites IS
  'Server-only one-time invite hashes; never expose through the Supabase Data API.';
COMMENT ON TABLE public.date_night_messages IS
  'Server-only private couple messages; FastAPI enforces room membership.';
COMMENT ON TABLE public.date_night_ai_cache IS
  'Server-only room-scoped AI response cache.';
