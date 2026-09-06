import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "db/migrations/20260906_date_night_server_only_rls.sql"


class DateNightDatabaseIsolationTests(unittest.TestCase):
    def test_every_date_night_table_is_deny_by_default(self) -> None:
        sql = MIGRATION.read_text(encoding="utf-8")
        tables = {
            "date_night_accounts",
            "date_night_bootstrap",
            "date_night_sessions",
            "date_night_rooms",
            "date_night_memberships",
            "date_night_invites",
            "date_night_messages",
            "date_night_ai_cache",
        }

        for table in tables:
            self.assertIn(
                f"ALTER TABLE public.{table} ENABLE ROW LEVEL SECURITY;",
                sql,
            )
            self.assertIn(f"public.{table}", sql)

        self.assertIn("FROM PUBLIC;", sql)
        self.assertIn("ARRAY['anon', 'authenticated']", sql)
        self.assertIn(
            "REVOKE EXECUTE ON FUNCTION public.rls_auto_enable() FROM PUBLIC, anon, authenticated;",
            sql,
        )
        self.assertIn("DROP POLICY %I", sql)
        self.assertNotIn("CREATE POLICY", sql.upper())
        self.assertNotIn("FORCE ROW LEVEL SECURITY", sql.upper())


if __name__ == "__main__":
    unittest.main()
