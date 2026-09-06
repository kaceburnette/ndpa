import unittest

from fastapi import HTTPException, Response

from server.main import (
    _date_night_email,
    _date_night_password,
    _date_night_password_hash,
    _date_night_password_matches,
    _set_date_night_cookie,
    app,
)


class DateNightAuthTests(unittest.TestCase):
    def test_password_hash_is_salted_and_one_way(self) -> None:
        password = "a-valid-long-password"
        first = _date_night_password_hash(password)
        second = _date_night_password_hash(password)

        self.assertNotEqual(first, second)
        self.assertNotIn(password, first)
        self.assertTrue(_date_night_password_matches(password, first))
        self.assertFalse(_date_night_password_matches("a-different-password", first))

    def test_password_length_bounds(self) -> None:
        self.assertEqual(_date_night_password("x" * 12), "x" * 12)
        self.assertEqual(_date_night_password("x" * 128), "x" * 128)
        for invalid in ("x" * 11, "x" * 129):
            with self.assertRaises(HTTPException) as raised:
                _date_night_password(invalid)
            self.assertEqual(raised.exception.status_code, 422)

    def test_email_is_normalized(self) -> None:
        self.assertEqual(_date_night_email("  Person@Example.COM "), "person@example.com")

    def test_session_cookie_is_http_only_and_secure(self) -> None:
        response = Response()
        _set_date_night_cookie(response, "opaque-token")
        cookie = response.headers["set-cookie"].lower()

        self.assertIn("httponly", cookie)
        self.assertIn("secure", cookie)
        self.assertIn("samesite=lax", cookie)

    def test_signup_and_claim_routes_are_registered(self) -> None:
        post_paths = {
            route.path
            for route in app.routes
            if "POST" in getattr(route, "methods", set())
        }
        self.assertIn("/date-night/signup", post_paths)
        self.assertIn("/date-night/claim", post_paths)
        self.assertIn("/date-night/signin", post_paths)


if __name__ == "__main__":
    unittest.main()
