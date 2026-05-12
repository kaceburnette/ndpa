"""
Tiered storage for NDPA.

The Postgres backend is the HOT tier — fast, indexed, queryable. But hot
storage is expensive at scale. NDPA's bigger win is that it can intelligently
move conversations between tiers BEFORE the user needs them:

  Hot tier   (Postgres / Supabase)   ← queryable, indexed, fast
  Cold tier  (blob storage: S3/GCS/  ← cheap, slow to fetch
              local fs / anything)

The predictive kernel decides what to PROMOTE from cold → hot, so the
user never sees cold-tier latency. They get cold-tier prices with hot-tier
speed.

This module is the abstraction layer. Storage backends implement TierBackend.
The promote/demote functions use any backend. The predictive promotion job
runs the kernel against cold-tier metadata.

Backends shipped:
  - LocalFSBackend  — stores .ndp bundles in a local directory (default, works
                      out of the box, fine for prototypes + dev)
  - S3Backend (stub) — interface defined; subclass and implement read/write
                      with boto3 or any S3-compatible client

Usage:
    from ndp.tiered import LocalFSBackend, demote, promote, predictive_promote

    backend = LocalFSBackend(Path("/var/ndpa/cold"))

    # Move an old session out of Postgres into cold storage
    demote("session_abc", backend)

    # Pull it back into Postgres when needed
    promote("session_abc", backend)

    # Let the kernel decide what to promote
    predictive_promote(backend, k=5, query_context="rust async tokio")
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Iterator, Optional

# Same tokenizer config as the predictions edge function — keep these in sync.
_STOP = set(("the a an and or but if then else for of to in on with at by from "
             "as is was are be been being have has had do does did this that "
             "these those it its they them their").split())
_WORD = re.compile(r"[a-zA-Z]{3,}")
_TOP_TERMS_K = 50


def _top_terms(text: str, k: int = _TOP_TERMS_K) -> list[str]:
    counts: Counter = Counter()
    for w in _WORD.findall(text or ""):
        w = w.lower()
        if w not in _STOP:
            counts[w] += 1
    return [w for w, _ in counts.most_common(k)]


# ─── Backend interface ──────────────────────────────────────────────────────

class TierBackend(ABC):
    """
    Storage backend for cold-tier .ndp bundles. Implement these methods to
    plug NDPA into any object store.
    """

    @abstractmethod
    def read_bundle(self, session_id: str) -> bytes:
        """Return the .ndp bundle bytes for a session. Raises FileNotFoundError if absent."""
        ...

    @abstractmethod
    def write_bundle(self, session_id: str, data: bytes, metadata: Optional[dict] = None) -> None:
        """Store a bundle. Metadata is small JSON sidecar (top_terms, started_at, platform)."""
        ...

    @abstractmethod
    def delete_bundle(self, session_id: str) -> None:
        """Remove a bundle. Idempotent."""
        ...

    @abstractmethod
    def list_bundles(self) -> Iterator[str]:
        """Yield session_ids of all stored bundles."""
        ...

    @abstractmethod
    def read_metadata(self, session_id: str) -> Optional[dict]:
        """Return the metadata sidecar (top_terms, etc.) without reading the full bundle."""
        ...

    def exists(self, session_id: str) -> bool:
        try:
            return self.read_metadata(session_id) is not None
        except Exception:
            return False


# ─── Local filesystem backend (works out of the box) ────────────────────────

class LocalFSBackend(TierBackend):
    """
    Stores bundles on the local filesystem. Suitable for dev, single-machine
    deployments, or self-hosted clusters with shared storage (NFS, EFS, etc).

    Layout:
        {root}/bundles/{session_id}.ndp        — the bundle itself
        {root}/metadata/{session_id}.json      — sidecar for cheap manifest reads
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.bundles_dir = self.root / "bundles"
        self.metadata_dir = self.root / "metadata"
        self.bundles_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def _bundle_path(self, session_id: str) -> Path:
        safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", session_id)
        return self.bundles_dir / f"{safe}.ndp"

    def _metadata_path(self, session_id: str) -> Path:
        safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", session_id)
        return self.metadata_dir / f"{safe}.json"

    def read_bundle(self, session_id: str) -> bytes:
        return self._bundle_path(session_id).read_bytes()

    def write_bundle(self, session_id: str, data: bytes, metadata: Optional[dict] = None) -> None:
        self._bundle_path(session_id).write_bytes(data)
        if metadata is not None:
            self._metadata_path(session_id).write_text(json.dumps(metadata))

    def delete_bundle(self, session_id: str) -> None:
        self._bundle_path(session_id).unlink(missing_ok=True)
        self._metadata_path(session_id).unlink(missing_ok=True)

    def list_bundles(self) -> Iterator[str]:
        for p in self.bundles_dir.glob("*.ndp"):
            yield p.stem

    def read_metadata(self, session_id: str) -> Optional[dict]:
        p = self._metadata_path(session_id)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except Exception:
            return None


# ─── S3 backend (stub — subclass to plug in boto3 or any S3 client) ────────

class S3Backend(TierBackend):
    """
    Stub S3 backend. To actually use it, install boto3 and subclass this:

        class MyS3Backend(S3Backend):
            def __init__(self, bucket, prefix=""):
                import boto3
                self.s3 = boto3.client("s3")
                self.bucket = bucket
                self.prefix = prefix.rstrip("/")
            # implement the four abstract methods using self.s3.get_object,
            # put_object, delete_object, list_objects_v2 ...

    NDPA does not depend on boto3. Bring your own client to keep the package
    zero-dependency and platform-agnostic.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "S3Backend is a stub. Subclass and implement the four TierBackend "
            "methods with your S3 client of choice (boto3, aioboto3, anything)."
        )

    def read_bundle(self, session_id: str) -> bytes:
        raise NotImplementedError

    def write_bundle(self, session_id: str, data: bytes, metadata: Optional[dict] = None) -> None:
        raise NotImplementedError

    def delete_bundle(self, session_id: str) -> None:
        raise NotImplementedError

    def list_bundles(self) -> Iterator[str]:
        raise NotImplementedError

    def read_metadata(self, session_id: str) -> Optional[dict]:
        raise NotImplementedError


# ─── Promote / demote operations ────────────────────────────────────────────

def demote(session_id: str, backend: TierBackend, *, delete_from_hot: bool = True) -> dict:
    """
    Move a session from Postgres (hot) to a cold backend.

    1. Export session to .ndp bundle (in memory, no disk needed)
    2. Build small metadata sidecar (top_terms, started_at, platform)
    3. Write bundle + metadata to backend
    4. Optionally delete from Postgres
    """
    from ndp.bundle import export_session_to_bytes
    from ndp.config import load_config

    bundle_bytes, manifest = export_session_to_bytes(session_id)

    # Build metadata for cheap-to-read sidecar
    metadata = {
        "session_id": session_id,
        "platform": manifest.get("platform"),
        "started_at": manifest.get("started_at"),
        "bundle_bytes": len(bundle_bytes),
    }

    # Pull top_terms from the current hot row before deleting
    cfg = load_config()
    from supabase import create_client
    client = create_client(cfg["supabase_url"], cfg["supabase_key"])
    try:
        row = (client.table("ndp_conversations")
               .select("top_terms, content")
               .eq("user_id", cfg["user_id"]).eq("session_id", session_id)
               .maybe_single().execute())
        if row and row.data:
            metadata["top_terms"] = row.data.get("top_terms") or _top_terms(row.data.get("content", ""))
    except Exception:
        metadata["top_terms"] = []

    backend.write_bundle(session_id, bundle_bytes, metadata=metadata)

    if delete_from_hot:
        user_id = cfg["user_id"]
        try:
            client.table("ndp_events").delete().eq("user_id", user_id).eq("session_id", session_id).execute()
            client.table("ndp_conversations").delete().eq("user_id", user_id).eq("session_id", session_id).execute()
            client.table("ndp_sessions").delete().eq("user_id", user_id).eq("session_id", session_id).execute()
        except Exception as e:
            return {"demoted": True, "deleted_hot": False, "error": str(e), "bytes": len(bundle_bytes)}

    return {"demoted": True, "deleted_hot": delete_from_hot, "bytes": len(bundle_bytes)}


def promote(session_id: str, backend: TierBackend) -> dict:
    """
    Move a session from cold backend back to Postgres (hot).

    1. Read bundle from backend
    2. Import bundle into ndp_events, ndp_conversations, ndp_objects
    3. Bundle remains in cold (idempotent — same data, two tiers)

    Returns import stats. Caller can choose to delete from cold after.
    """
    from ndp.bundle import import_bundle_bytes
    try:
        data = backend.read_bundle(session_id)
    except FileNotFoundError:
        return {"promoted": False, "reason": "not_in_cold"}
    result = import_bundle_bytes(data)
    result["promoted"] = True
    return result


# ─── Predictive promotion ──────────────────────────────────────────────────

def predictive_promote(
    backend: TierBackend,
    query_context: str,
    k: int = 5,
    max_scan: int = 1000,
) -> list[dict]:
    """
    Run the prediction kernel against cold-tier metadata. Promote top-K
    matches BEFORE the user asks.

    This is the operation that makes the architecture sing: cold storage
    holds millions of conversations cheaply; this function picks the few
    that are about to matter and pre-warms them into Postgres.

    Args:
        backend: cold tier storage
        query_context: trajectory text (recent prompts, current direction)
        k: how many bundles to promote
        max_scan: cap on how many cold-tier metadata sidecars to read

    Returns the list of promoted sessions with their topic scores.
    """
    q_bow = _bow(query_context)
    if not q_bow:
        return []

    scored: list[tuple[str, float, list[str]]] = []
    scanned = 0
    for session_id in backend.list_bundles():
        if scanned >= max_scan:
            break
        scanned += 1
        meta = backend.read_metadata(session_id)
        if not meta:
            continue
        terms = meta.get("top_terms") or []
        if not terms:
            continue
        # Quick BoW score against the metadata's top_terms only (no bundle read)
        score = _score_terms(terms, q_bow)
        if score > 0:
            scored.append((session_id, score, terms))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:k]

    promoted = []
    for sid, score, _terms in top:
        result = promote(sid, backend)
        result["session_id"] = sid
        result["topic_score"] = score
        promoted.append(result)

    return promoted


def _bow(text: str) -> Counter:
    bow: Counter = Counter()
    for w in _WORD.findall(text or ""):
        w = w.lower()
        if w not in _STOP:
            bow[w] += 1
    return bow


def _score_terms(terms: list[str], query_bow: Counter) -> float:
    """Cosine-like score: how many query terms appear in conversation top_terms."""
    overlap = sum(1 for t in terms if t in query_bow)
    if not terms or not query_bow:
        return 0.0
    return overlap / (len(terms) ** 0.5 * len(query_bow) ** 0.5)


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(prog="ndp.tiered",
                                     description="Move sessions between hot (Postgres) and cold tiers")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dem = sub.add_parser("demote", help="Move session from Postgres → cold tier")
    p_dem.add_argument("session_id")
    p_dem.add_argument("--cold-dir", default=str(Path.home() / ".ndp" / "cold"))
    p_dem.add_argument("--keep-hot", action="store_true",
                       help="Don't delete from Postgres after demotion")

    p_pro = sub.add_parser("promote", help="Move session from cold tier → Postgres")
    p_pro.add_argument("session_id")
    p_pro.add_argument("--cold-dir", default=str(Path.home() / ".ndp" / "cold"))

    p_pp = sub.add_parser("predict", help="Predict + promote top-K from cold based on query")
    p_pp.add_argument("query")
    p_pp.add_argument("--cold-dir", default=str(Path.home() / ".ndp" / "cold"))
    p_pp.add_argument("--k", type=int, default=5)

    p_ls = sub.add_parser("list", help="List bundles in cold tier")
    p_ls.add_argument("--cold-dir", default=str(Path.home() / ".ndp" / "cold"))

    args = parser.parse_args()
    backend = LocalFSBackend(Path(args.cold_dir))

    if args.cmd == "demote":
        result = demote(args.session_id, backend, delete_from_hot=not args.keep_hot)
        print(json.dumps(result, indent=2))
    elif args.cmd == "promote":
        result = promote(args.session_id, backend)
        print(json.dumps(result, indent=2, default=str))
    elif args.cmd == "predict":
        results = predictive_promote(backend, args.query, k=args.k)
        print(json.dumps(results, indent=2, default=str))
    elif args.cmd == "list":
        for sid in backend.list_bundles():
            meta = backend.read_metadata(sid) or {}
            print(f"  {sid}  platform={meta.get('platform')}  terms={len(meta.get('top_terms') or [])}")


if __name__ == "__main__":
    main()
