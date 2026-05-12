# NDPA Security & Privacy

## TL;DR for security review

| Concern                          | Status                              |
|----------------------------------|-------------------------------------|
| Auth                             | Bearer API keys, SHA-256 hash storage |
| Row-Level Security on user data  | Enabled on all tables               |
| File contents uploaded?          | **No, opt-in only**                 |
| TLS in transit                   | All traffic via HTTPS (Supabase Edge) |
| At-rest encryption               | Postgres + Supabase default         |
| Rate limiting                    | 600 req/min/key, per endpoint       |
| Key revocation                   | Single-row update, immediate        |
| Audit log                        | Every event timestamped + user-bound |
| GDPR right-to-erasure            | Single `DELETE WHERE user_id` query |
| SOC2                             | Inherited from Supabase backbone    |

## Authentication

API keys are generated as `ndpa_` + 32 bytes of `secrets.token_urlsafe`. The
**raw key is shown to the user exactly once**. The server only stores a
SHA-256 hash in `ndp_api_keys.key_hash`.

Every API call requires `Authorization: Bearer ndpa_...`. Lookup by hash;
invalid or revoked keys return `401`.

```sql
CREATE TABLE ndp_api_keys (
  id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id     uuid NOT NULL,
  key_hash    text NOT NULL UNIQUE,    -- SHA-256, never the raw key
  key_prefix  text NOT NULL,           -- first 12 chars, for display only
  platform    text,
  created_at  timestamptz DEFAULT now(),
  last_used_at timestamptz,
  revoked_at  timestamptz
);
```

## Row-Level Security

Every table is RLS-enabled. User data is bound to `user_id` (UUID). Service
role bypasses RLS for edge functions; end users never query the database
directly.

Tables protected:
- `ndp_events`
- `ndp_sessions`
- `ndp_conversations`
- `ndp_objects`
- `ndp_api_keys`
- `ndp_rate_limits` (service-only)

## What gets uploaded

**Always uploaded** (necessary for predictions):
- File paths (filenames + directory)
- Tool names (Read, Edit, etc.)
- Timestamps + turn indices
- Conversation text (user/assistant messages)
- Session metadata

**Never uploaded by default**:
- File contents (source code)
- Local environment variables
- File system structure beyond accessed paths

To opt in to file content sync (for richer hot-tier staging):

```json
// ~/.ndp/config.json
{
  "upload_file_contents": true
}
```

Even with this flag, only files the user accessed are uploaded — NDPA never
crawls the file system.

## What does NOT get logged

- API keys (only hashes)
- Local file system contents (opt-in only)
- Network requests / DNS / external traffic
- Anything outside the AI session loop

## Data residency

Default deployment: Supabase project in **us-west-2 (Oregon)**. Self-hosting
supported via:

```bash
NDPA_BASE_URL=https://your-supabase-project.supabase.co/functions/v1
```

For EU residency or on-prem: deploy your own Supabase instance, run the
schema migrations (`supabase/migrations/`), deploy the edge functions
(`supabase/functions/`). Point SDKs at your URL. No code changes needed.

## Right to erasure (GDPR / CCPA)

A single SQL command deletes all data for one user:

```sql
DELETE FROM ndp_events           WHERE user_id = $1;
DELETE FROM ndp_sessions         WHERE user_id = $1;
DELETE FROM ndp_conversations    WHERE user_id = $1;
DELETE FROM ndp_objects          WHERE user_id = $1;
DELETE FROM ndp_api_keys         WHERE user_id = $1;
```

Or run `python3 -m ndp.bundle export <session> --out backup.ndp` first, then
delete. The `.ndp` file is yours to keep, share, or re-import elsewhere.

## Threat model

| Threat                                 | Mitigation                            |
|----------------------------------------|---------------------------------------|
| API key leak                           | Revoke via `UPDATE ndp_api_keys SET revoked_at = now()`. Key is hashed, raw never stored. |
| User A reading user B's data           | RLS on every table, service role only in edge functions |
| DoS via flooding events                | Rate limit 600 req/min/key, 1000 events/call, 2MB body cap |
| Cross-tenant prompt injection          | Predictions API filtered by `user_id`. Past conversations from another user are unreachable. |
| Subdomain takeover                     | Edge functions deployed on Supabase-managed domain only |
| MITM                                   | TLS-only (HSTS), no HTTP endpoints |
| Malicious .ndp import                  | `import` revalidates schema, drops malformed entries, rebinds `user_id` |

## What we don't do (yet)

- **No customer-managed encryption keys (CMEK)** — using Supabase default
  at-rest encryption. Available for enterprise self-host.
- **No PII redaction** — conversation text is stored as-is. If your data
  contains PII you don't want stored, either redact upstream or set
  `upload_conversation_text: false` (planned).
- **No HIPAA compliance** — Supabase is not HIPAA-certified by default.
  Self-host on HIPAA-compliant Postgres for healthcare use cases.
