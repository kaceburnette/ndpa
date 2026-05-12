"""
NDP object store — pure Supabase.

Scoring uses a single batch query (get_all) so we pay one network
round-trip per hook invocation, not one per file. Silent-fail
everywhere — a Supabase error must never crash the hook.
"""

import time
import threading
from typing import Optional

from .schema import NDPObject


def _client():
    from .config import load_config
    from supabase import create_client
    cfg = load_config()
    return create_client(cfg["supabase_url"], cfg["supabase_key"]), cfg["user_id"]


def _row_to_obj(row: dict) -> NDPObject:
    return NDPObject(
        id=row["id"],
        source_type=row.get("source_type", "file"),
        source_path=row.get("source_path", ""),
        tier=row.get("tier", "cold"),
        freshness=row.get("freshness", time.time()),
        token_estimate=row.get("token_estimate", 0),
        prior_usefulness=row.get("prior_usefulness", 0.5),
        summary=row.get("summary", ""),
        tags=row.get("tags") or [],
        relationships=row.get("relationships") or [],
        access_history=row.get("access_history") or [],
    )


def _async(fn, *args):
    t = threading.Thread(target=fn, args=args, daemon=True)
    t.start()
    return t


def _obj_to_row(obj: NDPObject, user_id: str) -> dict:
    return {
        "id": obj.id,
        "user_id": user_id,
        "source_type": obj.source_type,
        "source_path": obj.source_path,
        "tier": obj.tier,
        "freshness": obj.freshness,
        "token_estimate": obj.token_estimate,
        "prior_usefulness": obj.prior_usefulness,
        "summary": obj.summary,
        "tags": obj.tags,
        "relationships": obj.relationships,
        "access_history": obj.access_history,
    }


class NDPStore:
    def upsert(self, obj: NDPObject):
        """Async write — never blocks the hook. Prefer upsert_many for batches."""
        def _write():
            try:
                client, user_id = _client()
                client.table("ndp_objects").upsert(_obj_to_row(obj, user_id)).execute()
            except Exception:
                pass
        _async(_write)

    def upsert_many(self, objs: list):
        """Single batched upsert in a background thread. Use this from hot paths."""
        if not objs:
            return
        def _write():
            try:
                client, user_id = _client()
                rows = [_obj_to_row(o, user_id) for o in objs]
                client.table("ndp_objects").upsert(rows).execute()
            except Exception:
                pass
        _async(_write)

    def get_all(self) -> dict:
        """Single batch query: returns {source_path: NDPObject}. One round-trip."""
        try:
            client, user_id = _client()
            result = (
                client.table("ndp_objects")
                .select("*")
                .eq("user_id", user_id)
                .execute()
            )
            return {r["source_path"]: _row_to_obj(r) for r in result.data}
        except Exception:
            return {}

    def get_by_path(self, path: str) -> Optional[NDPObject]:
        try:
            client, user_id = _client()
            result = (
                client.table("ndp_objects")
                .select("*")
                .eq("user_id", user_id)
                .eq("source_path", path)
                .limit(1)
                .execute()
            )
            if result.data:
                return _row_to_obj(result.data[0])
        except Exception:
            pass
        return None

    def get_by_tier(self, tier: str) -> list:
        try:
            client, user_id = _client()
            result = (
                client.table("ndp_objects")
                .select("*")
                .eq("user_id", user_id)
                .eq("tier", tier)
                .order("prior_usefulness", desc=True)
                .execute()
            )
            return [_row_to_obj(r) for r in result.data]
        except Exception:
            return []

    def set_tier(self, obj_id: str, tier: str):
        try:
            client, user_id = _client()
            client.table("ndp_objects").update({"tier": tier}).eq("id", obj_id).eq("user_id", user_id).execute()
        except Exception:
            pass

    def all_paths(self) -> list:
        try:
            client, user_id = _client()
            result = (
                client.table("ndp_objects")
                .select("source_path")
                .eq("user_id", user_id)
                .execute()
            )
            return [r["source_path"] for r in result.data]
        except Exception:
            return []

    def stats(self) -> dict:
        try:
            client, user_id = _client()
            result = (
                client.table("ndp_objects")
                .select("tier")
                .eq("user_id", user_id)
                .execute()
            )
            by_tier: dict = {}
            for r in result.data:
                by_tier[r["tier"]] = by_tier.get(r["tier"], 0) + 1
            return {"total": len(result.data), "by_tier": by_tier}
        except Exception:
            return {"total": 0, "by_tier": {}}
