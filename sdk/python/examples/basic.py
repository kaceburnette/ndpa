"""
Minimal NDPA usage example.
Requires NDPA_API_KEY env var.
"""

import os
import uuid

from ndpa import Client


def main():
    api_key = os.environ.get("NDPA_API_KEY")
    if not api_key:
        raise SystemExit("Set NDPA_API_KEY")

    session_id = f"example_{uuid.uuid4().hex[:8]}"
    client = Client(api_key=api_key, platform="example", async_send=False)

    client.log_exchange(
        session_id=session_id,
        user_message="Hello from the example.",
        assistant_message="Got it.",
    )

    client.log_turn(
        session_id=session_id,
        role="tool",
        tool_name="Read",
        source_path="/example/file.py",
    )

    print(f"Sent events for session {session_id}")


if __name__ == "__main__":
    main()
