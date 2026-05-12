"""
LangChain memory class backed by NDPA.

Drop-in replacement for LangChain's ConversationBufferMemory et al, but the
context loaded into each turn is PREDICTED from behavioral signal, not just
the recent buffer.

Usage:
    from integrations.langchain_memory import NDPAMemory
    from langchain.chains import ConversationChain
    from langchain_openai import ChatOpenAI

    memory = NDPAMemory(
        ndpa_key="ndpa_...",
        session_id="chat_xyz",
        end_user_id="user_123",  # optional, for multi-tenant platforms
    )
    chain = ConversationChain(llm=ChatOpenAI(), memory=memory)
    chain.predict(input="continue our last discussion about ML training")
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "sdk" / "python"))
from ndpa import Client  # noqa: E402


# Soft import — works whether or not the user has langchain installed
try:
    from langchain.memory.chat_memory import BaseChatMemory
    from langchain.schema.messages import AIMessage, HumanMessage
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

    class BaseChatMemory:  # type: ignore
        """Stub when langchain is not installed."""
        pass

    class HumanMessage:  # type: ignore
        def __init__(self, content: str):
            self.content = content

    class AIMessage:  # type: ignore
        def __init__(self, content: str):
            self.content = content


class NDPAMemory(BaseChatMemory):
    """
    LangChain memory class powered by NDPA predictions.

    On every chain call:
      1. Calls NDPA predictions API with the input as query
      2. Returns the staged context as the memory variable
      3. Logs the new exchange to NDPA after the chain runs
    """

    memory_key: str = "history"
    return_messages: bool = False

    def __init__(
        self,
        ndpa_key: str,
        session_id: str,
        end_user_id: str | None = None,
        k: int = 3,
        platform: str = "langchain",
        **kwargs: Any,
    ):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("Install langchain to use NDPAMemory: pip install langchain")
        super().__init__(**kwargs)
        self._ndpa = Client(api_key=ndpa_key, platform=platform, async_send=True)
        self._session_id = session_id
        self._end_user_id = end_user_id
        self._k = k

    @property
    def memory_variables(self) -> list[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        # Use chain input as query
        query = inputs.get("input", "") or ""
        result = self._ndpa.get_predictions(
            self._session_id,
            query=query,
            k=self._k,
            end_user_id=self._end_user_id,
        )
        preds = result.get("predictions", [])
        if not preds:
            return {self.memory_key: ""}

        lines = []
        for p in preds:
            date = (p.get("started_at") or "")[:10]
            content = p.get("content", "")
            title = ""
            for chunk in content.split("[user]")[1:]:
                cand = chunk.strip().split("\n")[0][:160].strip()
                if cand and not cand.startswith("[tool]") and len(cand) > 15:
                    title = cand
                    break
            if title:
                lines.append(f"[{date}] {title}")
        return {self.memory_key: "\n".join(lines)}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        # Log the exchange after the chain runs
        ts = time.time()
        user_msg = str(inputs.get("input", ""))
        ai_msg = str(outputs.get("output", outputs.get("response", "")))
        self._ndpa.log_events(self._session_id, [
            {"role": "user", "content": user_msg, "ts": ts},
            {"role": "assistant", "content": ai_msg, "ts": ts + 0.001},
        ], end_user_id=self._end_user_id)

    def clear(self) -> None:
        # NDPA memory persists across sessions — clear is intentionally a no-op
        pass


if __name__ == "__main__":
    print("Reference integration — see docstring for usage.")
    print("Requires: pip install langchain ndpa")
