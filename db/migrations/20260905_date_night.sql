-- Private, invite-only rooms for the Date Night app.
-- All access is mediated by the FastAPI app; direct Data API access stays denied.

CREATE TABLE IF NOT EXISTS public.date_night_accounts (
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  email         text NOT NULL,
  display_name  text NOT NULL,
  password_hash text NOT NULL,
  created_at    timestamptz NOT NULL DEFAULT now(),
  UNIQUE (email)
);

CREATE TABLE IF NOT EXISTS public.date_night_bootstrap (
  singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
  owner_id  uuid NOT NULL REFERENCES public.date_night_accounts(id),
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.date_night_sessions (
  token_hash text PRIMARY KEY,
  account_id uuid NOT NULL REFERENCES public.date_night_accounts(id) ON DELETE CASCADE,
  expires_at timestamptz NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS date_night_sessions_account_idx
  ON public.date_night_sessions (account_id);

CREATE TABLE IF NOT EXISTS public.date_night_rooms (
  id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name        text NOT NULL,
  owner_id    uuid NOT NULL REFERENCES public.date_night_accounts(id),
  created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.date_night_memberships (
  room_id    uuid NOT NULL REFERENCES public.date_night_rooms(id) ON DELETE CASCADE,
  account_id uuid NOT NULL REFERENCES public.date_night_accounts(id) ON DELETE CASCADE,
  joined_at  timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (room_id, account_id)
);
CREATE INDEX IF NOT EXISTS date_night_memberships_account_idx
  ON public.date_night_memberships (account_id);

CREATE TABLE IF NOT EXISTS public.date_night_invites (
  token_hash  text PRIMARY KEY,
  room_id     uuid NOT NULL REFERENCES public.date_night_rooms(id) ON DELETE CASCADE,
  created_by  uuid NOT NULL REFERENCES public.date_night_accounts(id),
  claimed_by  uuid REFERENCES public.date_night_accounts(id),
  expires_at  timestamptz NOT NULL,
  created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.date_night_messages (
  id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  room_id     uuid NOT NULL REFERENCES public.date_night_rooms(id) ON DELETE CASCADE,
  sender_id   uuid REFERENCES public.date_night_accounts(id) ON DELETE SET NULL,
  sender_name text NOT NULL,
  kind        text NOT NULL DEFAULT 'human' CHECK (kind IN ('human', 'assistant')),
  body        text NOT NULL CHECK (char_length(body) BETWEEN 1 AND 4000),
  created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS date_night_messages_room_created_idx
  ON public.date_night_messages (room_id, created_at, id);

ALTER TABLE public.date_night_accounts ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.date_night_bootstrap ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.date_night_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.date_night_rooms ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.date_night_memberships ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.date_night_invites ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.date_night_messages ENABLE ROW LEVEL SECURITY;
