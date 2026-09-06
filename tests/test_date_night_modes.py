import unittest

from server.main import (
    _date_night_ai_cache_key,
    _date_night_assistant_prompt,
    app,
)


class DateNightModeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = {
            "ai_mode": "quiet",
            "activity_mode": "watch",
            "date_style": "long_distance",
        }

    def test_cache_key_tracks_human_context_settings_and_freshness(self) -> None:
        base = _date_night_ai_cache_key(
            "Kace: comedy",
            self.settings,
            fresh=False,
            invocation="picker",
        )
        self.assertEqual(
            base,
            _date_night_ai_cache_key(
                "Kace: comedy",
                dict(reversed(list(self.settings.items()))),
                fresh=False,
                invocation="picker",
            ),
        )
        self.assertNotEqual(
            base,
            _date_night_ai_cache_key(
                "Kace: comedy\nMorgan: action",
                self.settings,
                fresh=False,
                invocation="picker",
            ),
        )
        self.assertNotEqual(
            base,
            _date_night_ai_cache_key(
                "Kace: comedy",
                self.settings,
                fresh=True,
                invocation="picker",
            ),
        )

    def test_prompt_adapts_to_activity_and_date_style(self) -> None:
        settings = {**self.settings, "activity_mode": "game", "date_style": "in_person"}
        query, instructions = _date_night_assistant_prompt(settings, fresh=False, mention=False)

        self.assertIn("game", query)
        self.assertIn("in person", instructions)
        self.assertIn("exactly three", instructions)

    def test_mention_prompt_is_short_and_direct(self) -> None:
        query, instructions = _date_night_assistant_prompt(self.settings, fresh=False, mention=True)

        self.assertIn("@DateNight", query)
        self.assertIn("1-3 short sentences", instructions)

    def test_settings_and_mention_routes_are_registered(self) -> None:
        routes = {
            (route.path, method)
            for route in app.routes
            for method in getattr(route, "methods", set())
        }
        self.assertIn(("/date-night/rooms/{room_id}/settings", "PATCH"), routes)
        self.assertIn(("/date-night/rooms/{room_id}/assistant/mention", "POST"), routes)


if __name__ == "__main__":
    unittest.main()
