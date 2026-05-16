"""Local filesystem hydration and write for raw NDPA context blobs.

Hydration is deliberately separate from Memory retrieval. Memory returns
handles, previews, and scores; Hydration fetches raw context only for selected
handles/storage keys.
"""

from __future__ import annotations

import re
import datetime as _dt
import hashlib
import hmac
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_\-]")


def write_local_blob(user_id: str, session_id: str, content: str, root: str | Path) -> tuple[str, int]:
    """Append content to <root>/<user_id>/<session_id>.txt.

    Returns (storage_key, total_bytes_after_append). storage_key is the path
    relative to root and safe to store in ndp_conversations.storage_key.
    """
    safe_user = _SAFE_ID_RE.sub("_", user_id)[:64]
    safe_session = _SAFE_ID_RE.sub("_", session_id)[:128]
    if not safe_user or not safe_session:
        raise ValueError(f"invalid user_id or session_id: {user_id!r}, {session_id!r}")

    base = Path(root).expanduser().resolve()
    user_dir = base / safe_user
    user_dir.mkdir(parents=True, exist_ok=True)

    path = user_dir / f"{safe_session}.txt"
    with path.open("a", encoding="utf-8") as f:
        f.write(content)
        f.write("\n")

    total_bytes = path.stat().st_size
    storage_key = f"{safe_user}/{safe_session}.txt"
    return storage_key, total_bytes


def hydrate_local_blob(storage_key: str, root: str | Path) -> dict:
    """Return raw context for one local-FS storage key without path traversal."""
    if not storage_key or storage_key.startswith("/"):
        return _missing(storage_key, "invalid_storage_key")

    base = Path(root).expanduser().resolve()
    path = (base / storage_key).resolve()
    try:
        path.relative_to(base)
    except ValueError:
        return _missing(storage_key, "invalid_storage_key")

    if not path.is_file():
        return _missing(storage_key, "not_found")

    content = path.read_text(encoding="utf-8")
    return {
        "storage_key": storage_key,
        "content": content,
        "content_bytes": len(content.encode("utf-8")),
        "found": True,
        "source": "local_fs",
    }


def write_blob(
    user_id: str,
    session_id: str,
    content: str,
    *,
    backend: str,
    root: str | Path,
    s3: dict[str, str] | None = None,
) -> tuple[str, int]:
    if backend == "local":
        return write_local_blob(user_id, session_id, content, root)
    if backend == "s3":
        if not s3:
            raise ValueError("missing s3 config")
        return write_s3_blob(user_id, session_id, content, s3)
    raise ValueError(f"unsupported hydration backend: {backend}")


def hydrate_blob(
    storage_key: str,
    *,
    backend: str,
    root: str | Path,
    s3: dict[str, str] | None = None,
) -> dict:
    if backend == "local":
        return hydrate_local_blob(storage_key, root)
    if backend == "s3":
        if not s3:
            return _missing(storage_key, "missing_s3_config", source="s3")
        return hydrate_s3_blob(storage_key, s3)
    return _missing(storage_key, "unsupported_backend", source=backend)


def write_s3_blob(user_id: str, session_id: str, content: str, config: dict[str, str]) -> tuple[str, int]:
    """Append content to an S3/R2 object without adding a runtime dependency.

    This is intentionally simple for launch: read existing object if present,
    append the new text, then PUT the full object. High-write customers should
    move to segment objects or a queue before concurrent writes matter.
    """
    safe_user = _SAFE_ID_RE.sub("_", user_id)[:64]
    safe_session = _SAFE_ID_RE.sub("_", session_id)[:128]
    if not safe_user or not safe_session:
        raise ValueError(f"invalid user_id or session_id: {user_id!r}, {session_id!r}")
    _require_s3_config(config)
    key = f"{safe_user}/{safe_session}.txt"
    existing = ""
    current = hydrate_s3_blob(key, config)
    if current.get("found"):
        existing = str(current.get("content") or "")
    body_text = f"{existing}{content}\n"
    body = body_text.encode("utf-8")
    url = _s3_object_url(config, key)
    headers = _signed_s3_headers("PUT", url, body, config, {"content-type": "text/plain; charset=utf-8"})
    req = urllib.request.Request(url, data=body, method="PUT", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20):
            pass
    except Exception as exc:
        raise RuntimeError(f"s3 put failed for {key}: {exc}") from exc
    return key, len(body)


def hydrate_s3_blob(storage_key: str, config: dict[str, str]) -> dict:
    if not storage_key or storage_key.startswith("/") or ".." in Path(storage_key).parts:
        return _missing(storage_key, "invalid_storage_key", source="s3")
    try:
        _require_s3_config(config)
    except ValueError:
        return _missing(storage_key, "missing_s3_config", source="s3")
    url = _s3_object_url(config, storage_key)
    headers = _signed_s3_headers("GET", url, b"", config, {})
    req = urllib.request.Request(url, method="GET", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return _missing(storage_key, "not_found", source="s3")
        return _missing(storage_key, f"http_{exc.code}", source="s3")
    except Exception:
        return _missing(storage_key, "request_failed", source="s3")
    content = raw.decode("utf-8", errors="replace")
    return {
        "storage_key": storage_key,
        "content": content,
        "content_bytes": len(raw),
        "found": True,
        "source": "s3",
    }


def _s3_object_url(config: dict[str, str], key: str) -> str:
    endpoint = config["endpoint"].rstrip("/")
    bucket = config["bucket"].strip("/")
    quoted_key = "/".join(urllib.parse.quote(part, safe="") for part in key.split("/"))
    return f"{endpoint}/{urllib.parse.quote(bucket, safe='')}/{quoted_key}"


def _require_s3_config(config: dict[str, str]) -> None:
    required = ["endpoint", "bucket", "access_key", "secret_key"]
    missing = [name for name in required if not config.get(name)]
    if missing:
        raise ValueError(f"missing S3 config: {', '.join(missing)}")


def _signed_s3_headers(
    method: str,
    url: str,
    body: bytes,
    config: dict[str, str],
    extra_headers: dict[str, str],
) -> dict[str, str]:
    access_key = config["access_key"]
    secret_key = config["secret_key"]
    region = config.get("region") or "auto"
    parsed = urllib.parse.urlparse(url)
    now = _dt.datetime.now(_dt.timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")
    payload_hash = hashlib.sha256(body).hexdigest()
    headers: dict[str, str] = {
        "host": parsed.netloc,
        "x-amz-content-sha256": payload_hash,
        "x-amz-date": amz_date,
        **{k.lower(): v for k, v in extra_headers.items()},
    }
    signed_headers = ";".join(sorted(headers))
    canonical_headers = "".join(f"{k}:{headers[k].strip()}\n" for k in sorted(headers))
    canonical_uri = parsed.path or "/"
    canonical_query = parsed.query
    canonical_request = "\n".join([
        method,
        canonical_uri,
        canonical_query,
        canonical_headers,
        signed_headers,
        payload_hash,
    ])
    scope = f"{date_stamp}/{region}/s3/aws4_request"
    string_to_sign = "\n".join([
        "AWS4-HMAC-SHA256",
        amz_date,
        scope,
        hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
    ])
    signing_key = _s3_signing_key(secret_key, date_stamp, region)
    signature = hmac.new(signing_key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    headers["authorization"] = (
        "AWS4-HMAC-SHA256 "
        f"Credential={access_key}/{scope}, "
        f"SignedHeaders={signed_headers}, "
        f"Signature={signature}"
    )
    return {k.title(): v for k, v in headers.items()}


def _s3_signing_key(secret_key: str, date_stamp: str, region: str) -> bytes:
    def sign(key: bytes, msg: str) -> bytes:
        return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

    k_date = sign(("AWS4" + secret_key).encode("utf-8"), date_stamp)
    k_region = sign(k_date, region)
    k_service = sign(k_region, "s3")
    return sign(k_service, "aws4_request")


def _missing(storage_key: str, error: str, *, source: str = "local_fs") -> dict:
    return {
        "storage_key": storage_key,
        "content": "",
        "content_bytes": 0,
        "found": False,
        "error": error,
        "source": source,
    }
