"""
NDPA Python client.

Drop-in instrumentation for any AI conversation loop.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Iterable, Optional

DEFAULT_BASE_URL = "https://izackuempnhgoojtbgvi.supabase.co/functions/v1"
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

    def log_events(self, session_id: str, events: Iterable[dict]) -> None:
        """Send a batch of events for one session. Up to 1000 per call."""
        evs = list(events)
        if not evs:
            return
        for i in range(0, len(evs), BATCH_SIZE_LIMIT):
            chunk = evs[i : i + BATCH_SIZE_LIMIT]
            payload = {"session_id": session_id, "events": chunk}
            if self.platform:
                payload["platform"] = self.platform
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
    ) -> dict:
        """
        Get top-K relevant past conversations for this session.

        Returns a dict with `predictions` (list of {session_id, content, score, ...}).
        This is the read side of NDPA — what the AI should know before responding.
        """
        payload: dict[str, Any] = {"session_id": session_id, "k": int(k)}
        if query is not None:
            payload["query"] = query
        resp = self._post_path("/predictions", payload, force_sync=True)
        return resp or {"predictions": []}

    # ── Internals ─────────────────────────────────────────────────────────────

    def _send(self, payload: dict) -> None:
        if self.async_send:
            t = threading.Thread(target=self._post_path, args=("/events", payload), daemon=True)
            t.start()
        else:
            self._post_path("/events", payload)

    def _post_path(self, path: str, payload: dict, *, force_sync: bool = False) -> Optional[dict]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "ndpa-python/0.1.0",
            },
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
