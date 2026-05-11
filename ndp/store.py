import json
import sqlite3
from pathlib import Path
from typing import Optional

from .schema import NDPObject

NDP_DIR = Path.home() / ".ndp"


class NDPStore:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or NDP_DIR / "index.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS objects (
                    id TEXT PRIMARY KEY,
                    source_path TEXT,
                    type TEXT,
                    tier TEXT,
                    freshness REAL,
                    token_estimate INTEGER,
                    prior_usefulness REAL,
                    data JSON
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_path ON objects(source_path)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tier ON objects(tier)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_freshness ON objects(freshness)")

    def upsert(self, obj: NDPObject):
        # Content is never persisted — always read fresh from disk when staging.
        # Keeps the DB as a pure metadata index regardless of project size.
        d = obj.to_dict()
        d["content"] = None
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO objects VALUES (?,?,?,?,?,?,?,?)",
                (
                    obj.id, obj.source_path, obj.type, obj.tier,
                    obj.freshness, obj.token_estimate, obj.prior_usefulness,
                    json.dumps(d),
                ),
            )

    def get_by_path(self, path: str) -> Optional[NDPObject]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT data FROM objects WHERE source_path = ?", (path,)
            ).fetchone()
        return NDPObject.from_dict(json.loads(row[0])) if row else None

    def get_by_tier(self, tier: str) -> list:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT data FROM objects WHERE tier = ? ORDER BY prior_usefulness DESC",
                (tier,),
            ).fetchall()
        return [NDPObject.from_dict(json.loads(r[0])) for r in rows]

    def set_tier(self, obj_id: str, tier: str):
        with self._conn() as conn:
            conn.execute(
                "UPDATE objects SET tier = ?, data = json_set(data, '$.tier', ?) WHERE id = ?",
                (tier, tier, obj_id),
            )

    def all_paths(self) -> list:
        with self._conn() as conn:
            return [r[0] for r in conn.execute("SELECT source_path FROM objects").fetchall()]

    def stats(self) -> dict:
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM objects").fetchone()[0]
            by_tier = dict(
                conn.execute(
                    "SELECT tier, COUNT(*) FROM objects GROUP BY tier"
                ).fetchall()
            )
        return {"total": total, "by_tier": by_tier}
