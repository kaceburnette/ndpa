"""
OpenAI Chat Completions integration.

Drop-in wrapper that adds NDPA prediction to any openai.chat.completions.create
call. Stages relevant past conversation context as a system message before each
request.

Usage:
    from integrations.openai_chat import predict_then_complete
    from openai import OpenAI

    client = OpenAI()
    response = predict_then_complete(
        client,
        ndpa_key="ndpa_...",
        session_id="user_123_chat_abc",
        end_user_id="user_123",
        messages=[{"role": "user", "content": "What did we discuss last week?"}],
        model="gpt-4o",
    )

What this does:
  1. Calls NDPA predictions API with the user's latest message as query
  2. Inserts top-K past context as a system message at the front of `messages`
  3. Logs the new exchange back to NDPA after completion (async, non-blocking)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "sdk" / "python"))
from ndpa import Client  # noqa: E402


def stage_context(
    ndpa: Client,
    session_id: str,
    user_message: str,
    *,
    end_user_id: str | None = None,
    k: int = 3,
) -> str | None:
    """Fetch relevant past conversations and return a system-message string."""
    result = ndpa.get_predictions(
        session_id,
        query=user_message,
        k=k,
        end_user_id=end_user_id,
    )
    preds = result.get("predictions", [])
    if not preds:
        return None

    lines = ["Relevant context from this user's past conversations:"]
    for p in preds:
        date = (p.get("started_at") or "")[:10]
        platform = p.get("platform") or "chat"
        # First user message as a one-line summary
        content = p.get("content", "")
        title = ""
        for chunk in content.split("[user]")[1:]:
            cand = chunk.strip().split("\n")[0][:160].strip()
            if cand and not cand.startswith("[tool]") and len(cand) > 15:
                title = cand
                break
        if title:
            lines.append(f"- [{date}|{platform}] {title}")
    return "\n".join(lines)


def predict_then_complete(
    openai_client: Any,
    *,
    ndpa_key: str,
    session_id: str,
    messages: list[dict[str, str]],
    model: str = "gpt-4o",
    end_user_id: str | None = None,
    k: int = 3,
    **completion_kwargs: Any,
) -> Any:
    """
    Wrapper around openai_client.chat.completions.create that adds NDPA
    context staging before the call and logs the exchange after.

    `messages` must follow the OpenAI format. The user's most recent message
    is used as the prediction query.
    """
    ndpa = Client(api_key=ndpa_key, platform="openai_chat", async_send=True)

    # Find last user message
    last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    if last_user:
        context_msg = stage_context(
            ndpa, session_id, last_user["content"],
            end_user_id=end_user_id, k=k,
        )
        if context_msg:
            messages = [{"role": "system", "content": context_msg}] + messages

    # Forward to OpenAI
    response = openai_client.chat.completions.create(
        model=model, messages=messages, **completion_kwargs,
    )

    # Log the exchange back to NDPA (async)
    if last_user and response.choices:
        assistant_msg = response.choices[0].message.content or ""
        ndpa.log_exchange(
            session_id, last_user["content"], assistant_msg,
            # log_exchange doesn't accept end_user_id; use log_events for that
        )
        if end_user_id:
            # Re-log with end_user_id via log_events for proper scoping
            import time
            ts = time.time()
            ndpa.log_events(session_id, [
                {"role": "user", "content": last_user["content"], "ts": ts},
                {"role": "assistant", "content": assistant_msg, "ts": ts + 0.001},
            ], end_user_id=end_user_id)

    return response


if __name__ == "__main__":
    print("Reference integration — see docstring for usage.")
    print("Requires: pip install openai ndpa")
