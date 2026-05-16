import asyncio
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

from server.main import (
    TTLCache,
    _invalidate_prediction_scope,
    _extract_response_text,
    _mint_api_key,
    _prediction_cache_key,
    _public_metrics,
    _public_pricing,
    _stage_id,
)
from server.hydration import _s3_object_url, hydrate_blob, hydrate_local_blob
from server.predictions import (
    _expand_query,
    _fetch_metadata_candidates,
    _mmr_select,
    _prediction_payload,
    _query_features,
)


class FakeConn:
    def __init__(self):
        self.queries = []

    async def fetch(self, query, *args):
        self.queries.append(query)
        return [
            {
                "session_id": "s1",
                "content": "Postgres metadata preview",
                "storage_key": "mem/s1.json",
                "content_bytes": 4096,
                "platform": "test",
                "updated_epoch": 1.0,
                "updated_at": datetime(2026, 5, 14, tzinfo=timezone.utc),
                "topic_score": 1.0,
            }
        ]


class ServerPredictionHelpersTest(unittest.TestCase):
    def test_ttl_cache_expires_and_clears(self):
        cache = TTLCache(0.01)
        cache.set("k", {"v": 1})
        self.assertEqual(cache.get("k"), {"v": 1})
        item = cache.get_with_ttl("k")
        self.assertIsNotNone(item)
        self.assertGreater(item[1], 0)
        time.sleep(0.02)
        self.assertIsNone(cache.get("k"))
        cache.set("k", {"v": 2})
        cache.clear()
        self.assertIsNone(cache.get("k"))

    def test_prediction_invalidation_is_scoped(self):
        cache = TTLCache(60)
        scoped = _prediction_cache_key("platform-a", "user-1", "query", "session-1", 5)
        live = _prediction_cache_key("platform-a", "user-1", "", "session-1", 5)
        other_session = _prediction_cache_key("platform-a", "user-1", "query", "session-2", 5)
        other_user = _prediction_cache_key("platform-a", "user-2", "query", "session-1", 5)
        for key in (scoped, live, other_session, other_user):
            cache.set(key, "value")

        removed = _invalidate_prediction_scope(
            cache,
            user_id="platform-a",
            end_user_id="user-1",
            session_id="session-1",
        )

        self.assertEqual(removed, 2)
        self.assertIsNone(cache.get(scoped))
        self.assertIsNone(cache.get(live))
        self.assertEqual(cache.get(other_session), "value")
        self.assertEqual(cache.get(other_user), "value")

    def test_stage_id_is_stable_and_scoped(self):
        key = _prediction_cache_key("platform-a", "user-1", "Postgres memory", "", 5)
        same = _prediction_cache_key("platform-a", "user-1", "Postgres memory", "", 5)
        other_user = _prediction_cache_key("platform-a", "user-2", "Postgres memory", "", 5)
        self.assertEqual(_stage_id(key), _stage_id(same))
        self.assertNotEqual(_stage_id(key), _stage_id(other_user))

    def test_public_pricing_keeps_reasoning_premium(self):
        prices = {p.product: p.price_usd for p in _public_pricing()}
        self.assertEqual(prices["predict"], 0.03)
        self.assertEqual(prices["memory"], 0.10)
        self.assertEqual(prices["reasoning_hosted"], 40.00)
        self.assertGreater(prices["reasoning_hosted"], prices["memory"])
        self.assertEqual(_public_metrics()["reasoning_longmemeval_s_e2e"], "85.2%")

    def test_mint_api_key_returns_hashable_live_token(self):
        token, key_hash = _mint_api_key()
        self.assertTrue(token.startswith("ndpa_live_"))
        self.assertEqual(len(key_hash), 64)

    def test_extract_response_text_supports_output_text_and_content_parts(self):
        self.assertEqual(_extract_response_text({"output_text": " hello "}), "hello")
        payload = {
            "output": [
                {"content": [{"type": "output_text", "text": "a"}, {"text": "b"}]},
            ]
        }
        self.assertEqual(_extract_response_text(payload), "a\nb")

    def test_expand_query_removes_question_words_and_adds_synonyms(self):
        expanded = _expand_query("Where did I buy the restaurant gift?")
        self.assertNotIn("where", expanded)
        self.assertIn("buy", expanded)
        self.assertIn("bought", expanded)
        self.assertIn("restaurant", expanded)
        self.assertIn("dinner", expanded)

    def test_query_features_detect_temporal_and_entities(self):
        features = _query_features("When did I update Postgres in March 2025?")
        self.assertTrue(features["is_temporal"])
        self.assertIn("postgres", features["entities"])
        self.assertIn("march", features["dates"])
        self.assertIn("2025", features["dates"])

    def test_mmr_select_penalizes_redundant_candidates(self):
        ranked = [
            {"_score": 1.0, "_terms": {"postgres", "index", "query"}, "session_id": "a"},
            {"_score": 0.99, "_terms": {"postgres", "index", "query"}, "session_id": "b"},
            {"_score": 0.9, "_terms": {"rome", "dinner"}, "session_id": "c"},
        ]
        selected = _mmr_select(ranked, 2)
        self.assertEqual([r["session_id"] for r in selected], ["a", "c"])

    def test_prediction_payload_returns_handle_and_preview(self):
        payload = _prediction_payload({
            "session_id": "s1",
            "content": "x" * 2500,
            "platform": "test",
            "storage_key": "mem/s1.json",
            "content_bytes": 4096,
            "updated_at": datetime(2026, 5, 14, tzinfo=timezone.utc),
            "_score": 0.9,
            "_topic_norm": 0.8,
            "_recency": 0.7,
        })
        self.assertEqual(payload["memory_handle"], "s1")
        self.assertEqual(payload["session_id"], "s1")
        self.assertEqual(payload["storage_key"], "mem/s1.json")
        self.assertEqual(payload["content_bytes"], 4096)
        self.assertEqual(len(payload["content_preview"]), 2000)
        self.assertEqual(payload["content"], payload["content_preview"])

    def test_metadata_fallback_uses_top_terms_not_raw_full_text_index(self):
        conn = FakeConn()
        rows = asyncio.run(_fetch_metadata_candidates(
            conn,
            "Postgres metadata",
            ["postgres", "metadata"],
            "user",
            None,
            "",
        ))
        self.assertEqual(rows[0]["session_id"], "s1")
        self.assertTrue(any("top_terms &&" in q for q in conn.queries))
        self.assertFalse(any("tsv_content" in q for q in conn.queries))

    def test_hydrate_reads_local_fs_blob(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mem" / "s1.txt"
            path.parent.mkdir()
            path.write_text("raw context", encoding="utf-8")

            item = hydrate_local_blob("mem/s1.txt", tmp)

        self.assertTrue(item["found"])
        self.assertEqual(item["content"], "raw context")
        self.assertEqual(item["content_bytes"], len("raw context"))
        self.assertEqual(item["source"], "local_fs")

    def test_hydrate_missing_blob_returns_clean_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            item = hydrate_local_blob("missing.txt", tmp)

        self.assertFalse(item["found"])
        self.assertEqual(item["content"], "")
        self.assertEqual(item["error"], "not_found")

    def test_hydrate_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            item = hydrate_local_blob("../secret.txt", tmp)

        self.assertFalse(item["found"])
        self.assertEqual(item["error"], "invalid_storage_key")

    def test_hydrate_blob_reports_missing_s3_config(self):
        item = hydrate_blob(
            "user/session.txt",
            backend="s3",
            root="unused",
            s3={"endpoint": "", "bucket": "", "access_key": "", "secret_key": ""},
        )
        self.assertFalse(item["found"])
        self.assertEqual(item["source"], "s3")
        self.assertEqual(item["error"], "missing_s3_config")

    def test_s3_object_url_uses_path_style_and_quotes_key(self):
        url = _s3_object_url(
            {"endpoint": "https://example.r2.cloudflarestorage.com/", "bucket": "ndpa raw"},
            "user one/session.txt",
        )
        self.assertEqual(
            url,
            "https://example.r2.cloudflarestorage.com/ndpa%20raw/user%20one/session.txt",
        )


if __name__ == "__main__":
    unittest.main()
