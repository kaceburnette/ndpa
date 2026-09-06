-- Couple activity preferences and a durable cache for assistant responses.

ALTER TABLE public.date_night_rooms
  ADD COLUMN IF NOT EXISTS ai_mode text NOT NULL DEFAULT 'quiet'
    CHECK (ai_mode IN ('quiet', 'cohost')),
  ADD COLUMN IF NOT EXISTS activity_mode text NOT NULL DEFAULT 'watch'
    CHECK (activity_mode IN ('watch', 'food', 'date', 'game', 'talk')),
  ADD COLUMN IF NOT EXISTS date_style text NOT NULL DEFAULT 'long_distance'
    CHECK (date_style IN ('long_distance', 'in_person', 'flexible'));

CREATE TABLE IF NOT EXISTS public.date_night_ai_cache (
  room_id       uuid NOT NULL REFERENCES public.date_night_rooms(id) ON DELETE CASCADE,
  cache_key     text NOT NULL,
  response_body text NOT NULL,
  message_id    bigint REFERENCES public.date_night_messages(id) ON DELETE SET NULL,
  expires_at    timestamptz NOT NULL,
  created_at    timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (room_id, cache_key)
);
CREATE INDEX IF NOT EXISTS date_night_ai_cache_expiry_idx
  ON public.date_night_ai_cache (expires_at);

ALTER TABLE public.date_night_ai_cache ENABLE ROW LEVEL SECURITY;
