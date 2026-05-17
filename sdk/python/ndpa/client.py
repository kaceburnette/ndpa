"""
NDPA Python client.

Drop-in instrumentation for any AI conversation loop.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Iterable, Optional

# Default endpoint. Override at runtime by either:
#   1. Passing base_url= to Client()
#   2. Setting NDPA_BASE_URL in your environment
# Hosted Supabase Edge Functions are no longer the default architecture.
# The SDK defaults to local FastAPI for development until a hosted Core
# endpoint is configured.
_DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_BASE_URL = os.environ.get("NDPA_BASE_URL", _DEFAULT_API_URL)
DEFAULT_TIMEOUT = 5.0
BATCH_SIZE_LIMIT = 1000


class NDPAError(Exception):
    """Raised when the NDPA API returns an error or is unreachable."""


class Client:
    """
    Lightweight NDPA ingestion client.

    Args:
        api_key:   Your NDPA API key (starts with `ndpa_`).
        base_url:  Override the API endpoint (for self-hosted deployments).
        platform:  Optional platform name attached to all events.
        timeout:   Per-request timeout in seconds.
        async_send: If True, send events in a background thread (fire-and-forget).
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        platform: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        async_send: bool = True,
    ):
        if not api_key:
            raise ValueError("api_key is required")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.platform = platform
        self.timeout = timeout
        self.async_send = async_send

    # ── Public surface ────────────────────────────────────────────────────────

    def log_turn(
        self,
        session_id: str,
        role: str,
        content: Optional[str] = None,
        *,
        tool_name: Optional[str] = None,
        source_path: Optional[str] = None,
        ts: Optional[float] = None,
    ) -> None:
        """Log a single turn. The most common entry point."""
        event = {"role": role, "ts": ts if ts is not None else time.time()}
        if content is not None:
            event["content"] = content
        if tool_name is not None:
            event["tool_name"] = tool_name
        if source_path is not None:
            event["source_path"] = source_path
        self.log_events(session_id, [event])

    def log_events(
        self,
        session_id: str,
        events: Iterable[dict],
        *,
        end_user_id: Optional[str] = None,
    ) -> None:
        """
        Send a batch of events for one session. Up to 1000 per call.

        For platforms (multi-tenant integrations): pass `end_user_id` to scope
        events to one of YOUR users. Your platform key authenticates the call;
        end_user_id partitions storage and prediction queries.
        """
        evs = list(events)
        if not evs:
            return
        for i in range(0, len(evs), BATCH_SIZE_LIMIT):
            chunk = evs[i : i + BATCH_SIZE_LIMIT]
            payload: dict[str, Any] = {"session_id": session_id, "events": chunk}
            if self.platform:
                payload["platform"] = self.platform
            if end_user_id is not None:
                payload["end_user_id"] = end_user_id
            self._send(payload)

    def log_exchange(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        *,
        ts: Optional[float] = None,
    ) -> None:
        """Convenience: log a full user→assistant exchange in one call."""
        now = ts if ts is not None else time.time()
        self.log_events(session_id, [
            {"role": "user", "content": user_message, "ts": now},
            {"role": "assistant", "content": assistant_message, "ts": now + 0.001},
        ])

    def get_predictions(
        self,
        session_id: str = "",
        *,
        query: Optional[str] = None,
        k: int = 5,
        end_user_id: Optional[str] = None,
    ) -> dict:
        """
        Get top-K relevant past conversations for this session.

        Returns a dict with `predictions` (list of {memory_handle, session_id,
        content_preview, score, ...}). This is the hot read side of NDPA:
        fast memory handles/previews/scores. Fetch raw full context through
        Hydration when needed.

        For platforms (multi-tenant integrations): pass `end_user_id` to scope
        predictions to one of YOUR users.
        """
        payload: dict[str, Any] = {"session_id": session_id, "k": int(k)}
        if query is not None:
            payload["query"] = query
        if end_user_id is not None:
            payload["end_user_id"] = end_user_id
        resp = self._post_path("/predictions", payload, force_sync=True)
        return resp or {"predictions": []}

    def stage(
        self,
        session_id: str = "",
        *,
        query: Optional[str] = None,
        k: int = 5,
        end_user_id: Optional[str] = None,
    ) -> dict:
        """
        Precompute and cache top-K memory handles for a later hot read.

        Call this between turns, before a voice call starts, or from a
        background worker. A later get_predictions() call with the same
        session/query/end_user_id/k returns from the staged cache.
        """
        payload: dict[str, Any] = {"session_id": session_id, "k": int(k)}
        if query is not None:
            payload["query"] = query
        if end_user_id is not None:
            payload["end_user_id"] = end_user_id
        resp = self._post_path("/stage", payload, force_sync=True)
        return resp or {"predictions": []}

    def hydrate(
        self,
        *,
        memory_handles: Optional[Iterable[str]] = None,
        session_ids: Optional[Iterable[str]] = None,
        storage_keys: Optional[Iterable[str]] = None,
        end_user_id: Optional[str] = None,
    ) -> dict:
        """
        Fetch raw context for selected memory handles/storage keys.

        Hydration is separate from Memory retrieval: call get_predictions()
        first for handles/previews/scores, then hydrate only the selected
        handles that need raw full context.
        """
        payload: dict[str, Any] = {
            "memory_handles": list(memory_handles or []),
            "session_ids": list(session_ids or []),
            "storage_keys": list(storage_keys or []),
        }
        if end_user_id is not None:
            payload["end_user_id"] = end_user_id
        resp = self._post_path("/hydrate", payload, force_sync=True)
        return resp or {"contexts": []}

    def reasoning(
        self,
        query: str,
        *,
        session_id: str = "",
        k: int = 5,
        end_user_id: Optional[str] = None,
        model: Optional[str] = None,
        context_char_limit: Optional[int] = None,
        hydrate: bool = True,
        openai_api_key: Optional[str] = None,
    ) -> dict:
        """
        Premium answer layer: Memory -> optional Hydration -> LLM answer.

        If `openai_api_key` is supplied, NDPA runs in BYOK mode for this call.
        The key is sent only as an `X-OpenAI-API-Key` request header.
        """
        payload: dict[str, Any] = {
            "query": query,
            "session_id": session_id,
            "k": int(k),
            "hydrate": bool(hydrate),
        }
        if end_user_id is not None:
            payload["end_user_id"] = end_user_id
        if model is not None:
            payload["model"] = model
        if context_char_limit is not None:
            payload["context_char_limit"] = int(context_char_limit)
        headers = {"X-OpenAI-API-Key": openai_api_key} if openai_api_key else None
        resp = self._post_path("/reasoning", payload, force_sync=True, extra_headers=headers)
        return resp or {"answer": ""}

    def get_config(self) -> dict:
        """Fetch public product/pricing/metric config."""
        return self._get_path("/config", auth=False) or {}

    def get_account(self) -> dict:
        """Fetch authenticated account metadata for the current API key."""
        return self._get_path("/account", auth=True) or {}

    def get_usage(self, *, days: int = 30) -> dict:
        """Fetch usage summary for the current API key."""
        return self._get_path(f"/usage?days={int(days)}", auth=True) or {"products": []}

    def checkout(self) -> dict:
        """Create or fetch the configured billing handoff URL."""
        return self._post_path("/checkout", {}, force_sync=True) or {}

    def admin_create_api_key(
        self,
        *,
        admin_token: str,
        user_id: str,
        platform: Optional[str] = None,
    ) -> dict:
        """
        Mint an NDPA API key. Requires server-side NDPA_ADMIN_TOKEN.

        The raw key is returned once by the server.
        """
        payload = {"user_id": user_id, "platform": platform}
        return self._post_path(
            "/admin/api-keys",
            payload,
            force_sync=True,
            auth=False,
            extra_headers={"X-NDPA-Admin-Token": admin_token},
        ) or {}

    # ── Internals ─────────────────────────────────────────────────────────────

    def _send(self, payload: dict) -> None:
        if self.async_send:
            t = threading.Thread(target=self._post_path, args=("/events", payload), daemon=True)
            t.start()
        else:
            self._post_path("/events", payload)

    def _get_path(
        self,
        path: str,
        *,
        auth: bool = True,
        extra_headers: Optional[dict[str, str]] = None,
    ) -> Optional[dict]:
        headers = {
            "User-Agent": "ndpa-python/0.1.0",
        }
        if auth:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if extra_headers:
            headers.update(extra_headers)
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            method="GET",
            headers=headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise NDPAError(f"NDPA API error {e.code}: {body}") from e
        except Exception as e:
            raise NDPAError(f"NDPA request failed: {e}") from e

    def _post_path(
        self,
        path: str,
        payload: dict,
        *,
        force_sync: bool = False,
        auth: bool = True,
        extra_headers: Optional[dict[str, str]] = None,
    ) -> Optional[dict]:
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "ndpa-python/0.1.0",
        }
        if auth:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if extra_headers:
            headers.update(extra_headers)
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            method="POST",
            headers=headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            if force_sync or not self.async_send:
                raise NDPAError(f"NDPA API error {e.code}: {body}") from e
        except Exception as e:
            if force_sync or not self.async_send:
                raise NDPAError(f"NDPA request failed: {e}") from e
        return None
