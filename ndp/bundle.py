"""
.ndp bundle — portable package of NDP objects.

Layout:
  project.ndp/
  ├── manifest.json      project metadata
  ├── objects/           one .md file per NDP object (human-readable)
  ├── index.db           SQLite index
  └── sessions/          session JSONL logs (optional, for portability)
"""

import json
import time
from pathlib import Path
from typing import Optional

from .schema import NDPObject
from .store import NDPStore


class NDPBundle:
    VERSION = "0.1.0"

    def __init__(self, path: Path):
        self.path = Path(path)
        self.manifest_path = self.path / "manifest.json"
        self.objects_dir = self.path / "objects"
        self.db_path = self.path / "index.db"
        self.sessions_dir = self.path / "sessions"

    # --- lifecycle ---

    def create(self, name: str, cwd: str = "") -> "NDPBundle":
        self.path.mkdir(parents=True, exist_ok=True)
        self.objects_dir.mkdir(exist_ok=True)
        self._write_manifest(name, cwd)
        return self

    def _write_manifest(self, name: str, cwd: str):
        self.manifest_path.write_text(json.dumps({
            "name": name,
            "version": self.VERSION,
            "created_at": time.time(),
            "cwd": cwd,
        }, indent=2))

    @classmethod
    def load(cls, path: Path) -> "NDPBundle":
        b = cls(path)
        if not b.manifest_path.exists():
            raise FileNotFoundError(f"Not a .ndp bundle: {path}")
        return b

    # --- accessors ---

    def store(self) -> NDPStore:
        return NDPStore(db_path=self.db_path)

    def manifest(self) -> dict:
        return json.loads(self.manifest_path.read_text()) if self.manifest_path.exists() else {}

    # --- export / import ---

    def export_objects(self, store: Optional[NDPStore] = None):
        """Write all objects as human-readable markdown files."""
        s = store or self.store()
        self.objects_dir.mkdir(exist_ok=True)
        for obj in s.get_by_tier("hot") + s.get_by_tier("warm") + s.get_by_tier("cold"):
            self._export_one(obj)

    def _export_one(self, obj: NDPObject):
        safe = obj.source_path.replace("/", "_").replace("\\", "_").lstrip("_")
        out = self.objects_dir / f"{obj.id[:8]}_{safe}.md"
        lines = [
            "---",
            f"id: {obj.id}",
            f"type: {obj.type}",
            f"source_path: {obj.source_path}",
            f"tier: {obj.tier}",
            f"freshness: {obj.freshness:.0f}",
            f"token_estimate: {obj.token_estimate}",
            f"prior_usefulness: {obj.prior_usefulness:.3f}",
            f"tags: [{', '.join(obj.tags)}]",
            "---",
            "",
            obj.summary,
        ]
        if obj.content:
            lines += ["", "```", obj.content[:3000], "```"]
        out.write_text("\n".join(lines))

    def pack_sessions(self, session_dir: Optional[Path] = None):
        """Copy session logs into the bundle."""
        import shutil
        src = session_dir or (Path.home() / ".ndp" / "sessions")
        if not src.exists():
            return
        self.sessions_dir.mkdir(exist_ok=True)
        for f in src.glob("*.jsonl"):
            shutil.copy2(f, self.sessions_dir / f.name)
