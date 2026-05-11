"""
Checkpoint — serialize and restore tier state across sessions.
"""

import json
import time
from pathlib import Path
from typing import Optional

from .schema import NDPObject
from .store import NDPStore

CHECKPOINT_DIR = Path.home() / ".ndp" / "checkpoints"


def save(store: NDPStore, name: Optional[str] = None, session_log: Optional[list] = None) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    cp_name = name or f"checkpoint_{int(time.time())}"
    cp_path = CHECKPOINT_DIR / f"{cp_name}.json"

    cp_path.write_text(json.dumps({
        "name": cp_name,
        "created_at": time.time(),
        "hot": [o.to_dict() for o in store.get_by_tier("hot")],
        "warm": [o.to_dict() for o in store.get_by_tier("warm")],
        "session_log": session_log or [],
        "stats": store.stats(),
    }, indent=2))
    return cp_path


def restore(store: NDPStore, cp_path: Path) -> dict:
    data = json.loads(cp_path.read_text())
    count = 0
    for obj_dict in data.get("hot", []) + data.get("warm", []):
        store.upsert(NDPObject.from_dict(obj_dict))
        count += 1
    return {
        "restored": count,
        "checkpoint": data.get("name"),
        "created_at": data.get("created_at"),
    }


def list_all() -> list:
    if not CHECKPOINT_DIR.exists():
        return []
    return sorted(CHECKPOINT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
