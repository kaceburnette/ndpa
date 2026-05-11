import time
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NDPObject:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = "file"
    source_path: str = ""
    source_type: str = "file"
    summary: str = ""
    tags: list = field(default_factory=list)
    relationships: list = field(default_factory=list)
    freshness: float = field(default_factory=time.time)
    token_estimate: int = 0
    prior_usefulness: float = 0.5
    access_history: list = field(default_factory=list)
    tier: str = "cold"
    content: Optional[str] = None

    def record_access(self, session_id: str, tool: str):
        self.access_history.append({
            "ts": time.time(),
            "session_id": session_id,
            "tool": tool,
        })
        self._decay_usefulness()

    def _decay_usefulness(self):
        if not self.access_history:
            return
        last_ts = self.access_history[-1]["ts"]
        age_days = (time.time() - last_ts) / 86400
        count = len(self.access_history)
        self.prior_usefulness = min(1.0, (count * 0.1 + 0.4) * (0.95 ** age_days))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "source_path": self.source_path,
            "source_type": self.source_type,
            "summary": self.summary,
            "tags": self.tags,
            "relationships": self.relationships,
            "freshness": self.freshness,
            "token_estimate": self.token_estimate,
            "prior_usefulness": self.prior_usefulness,
            "access_history": self.access_history,
            "tier": self.tier,
            "content": self.content,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NDPObject":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
