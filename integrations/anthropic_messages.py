"""
Anthropic Messages API integration.

Drop-in wrapper that adds NDPA prediction to any anthropic.messages.create call.
Stages relevant past conversation context as a system parameter before each request.

Usage:
    from integrations.anthropic_messages import predict_then_send
    from anthropic import Anthropic

    client = Anthropic()
    response = predict_then_send(
        client,
        ndpa_key="ndpa_...",
        session_id="user_123_chat_abc",
        end_user_id="user_123",
        messages=[{"role": "user", "content": "Pick up where we left off"}],
        model="claude-sonnet-4-6",
        max_tokens=1024,
    )
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "sdk" / "python"))
from ndpa import Client  # noqa: E402


def stage_context_anthropic(
    ndpa: Client,
    session_id: str,
    user_message: str,
    *,
    end_user_id: str | None = None,
    k: int = 3,
) -> str | None:
    """Fetch relevant past conversations and return a system-prompt string."""
    result = ndpa.get_predictions(
        session_id,
        query=user_message,
        k=k,
        end_user_id=end_user_id,
    )
    preds = result.get("predictions", [])
    if not preds:
        return None

    lines = ["Context from this user's past conversations across platforms:"]
    for p in preds:
        date = (p.get("started_at") or "")[:10]
        platform = p.get("platform") or "chat"
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


def predict_then_send(
    anthropic_client: Any,
    *,
    ndpa_key: str,
    session_id: str,
    messages: list[dict[str, str]],
    model: str = "claude-sonnet-4-6",
    end_user_id: str | None = None,
    k: int = 3,
    system: str | None = None,
    **kwargs: Any,
) -> Any:
    """
    Wrapper around anthropic_client.messages.create with NDPA context staging.
    `system` (if provided) is appended after the NDPA-staged context.
    """
    ndpa = Client(api_key=ndpa_key, platform="anthropic_messages", async_send=True)

    last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    if last_user:
        context_msg = stage_context_anthropic(
            ndpa, session_id, last_user["content"],
            end_user_id=end_user_id, k=k,
        )
        if context_msg:
            system = context_msg + ("\n\n" + system if system else "")

    response = anthropic_client.messages.create(
        model=model,
        messages=messages,
        system=system or "",
        **kwargs,
    )

    # Log the exchange back to NDPA
    if last_user and response.content:
        # Anthropic returns a list of content blocks
        assistant_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                assistant_text += block.text
        ts = time.time()
        ndpa.log_events(session_id, [
            {"role": "user", "content": last_user["content"], "ts": ts},
            {"role": "assistant", "content": assistant_text, "ts": ts + 0.001},
        ], end_user_id=end_user_id)

    return response


if __name__ == "__main__":
    print("Reference integration — see docstring for usage.")
    print("Requires: pip install anthropic ndpa")
