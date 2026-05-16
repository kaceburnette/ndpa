"""
NDPA FastAPI server — predictions and events endpoints.

Replaces the Supabase Edge Function deployment. Cuts wall P50 from ~1s to
target sub-150ms by eliminating cold starts and using persistent connection
pooling.

Endpoints (drop-in compatible with the SDK):
  POST /predictions   — top-K relevant past conversations
  POST /events        — ingest one batch of events
  GET  /health        — liveness probe

Auth: bearer token matched against ndp_api_keys.key_hash (SHA-256).
Rate limit: 600 req/min/key (events), 300 req/min/key (predictions).

Run locally:
    pip install fastapi 'uvicorn[standard]' asyncpg
    export DATABASE_URL='postgres://...'
    uvicorn server.main:app --reload --port 8000

Deploy:
    fly launch  (see server/fly.toml)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import secrets
import time
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from typing import Any

import asyncpg
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from server.hydration import hydrate_blob, write_blob
from server.predictions import predict_topk
from server.rate_limit import RateLimiter

DB_DSN = os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL", "")
POOL_MIN = int(os.environ.get("DB_POOL_MIN", "5"))
POOL_MAX = int(os.environ.get("DB_POOL_MAX", "20"))
HYDRATION_ROOT = os.environ.get("NDPA_HYDRATION_ROOT", "memory/blobs")
HYDRATION_BACKEND = os.environ.get("NDPA_HYDRATION_BACKEND", "local").lower()
S3_CONFIG = {
    "endpoint": os.environ.get("NDPA_S3_ENDPOINT", ""),
    "bucket": os.environ.get("NDPA_S3_BUCKET", ""),
    "region": os.environ.get("NDPA_S3_REGION", "auto"),
    "access_key": os.environ.get("NDPA_S3_ACCESS_KEY_ID", ""),
    "secret_key": os.environ.get("NDPA_S3_SECRET_ACCESS_KEY", ""),
}
AUTH_CACHE_TTL_SEC = float(os.environ.get("NDPA_AUTH_CACHE_TTL_SEC", "45"))
PREDICTION_CACHE_TTL_SEC = float(os.environ.get("NDPA_PREDICTION_CACHE_TTL_SEC", "20"))
PUBLIC_API_BASE_URL = os.environ.get("NDPA_PUBLIC_API_BASE_URL", "")
STRIPE_PAYMENT_LINK = os.environ.get("NDPA_STRIPE_PAYMENT_LINK", "")
ADMIN_TOKEN = os.environ.get("NDPA_ADMIN_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
REASONING_MODEL = os.environ.get("NDPA_REASONING_MODEL", "gpt-5-mini")
REASONING_CONTEXT_CHARS = int(os.environ.get("NDPA_REASONING_CONTEXT_CHARS", "12000"))
CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("NDPA_CORS_ORIGINS", "*").split(",")
    if origin.strip()
]
_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+-]{1,}")
_STOP = {
    "the", "a", "an", "and", "or", "but", "for", "of", "to", "in", "on",
    "with", "at", "by", "from", "as", "is", "was", "are", "be", "been",
    "have", "has", "had", "do", "does", "did", "this", "that", "it", "its",
    "they", "them", "their", "what", "which", "who", "can", "could", "would",
    "should", "about", "you", "your", "i", "me", "my",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pool = await _create_pool() if DB_DSN else None
    app.state.rate_limiter = RateLimiter()
    app.state.auth_cache = TTLCache(AUTH_CACHE_TTL_SEC)
    app.state.prediction_cache = TTLCache(PREDICTION_CACHE_TTL_SEC)
    app.state.pool_lock = asyncio.Lock()
    yield
    pool = getattr(app.state, "pool", None)
    if pool is not None:
        await pool.close()


app = FastAPI(title="NDPA Predictions API", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS or ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-NDPA-Admin-Token", "X-OpenAI-API-Key"],
)


async def _create_pool() -> asyncpg.Pool:
    if not DB_DSN:
        raise HTTPException(
            status_code=503,
            detail=(
                "DATABASE_URL not set. Get the Supabase Postgres connection "
                "string from your project's Database -> Connection Pooling page."
            ),
        )
    return await asyncpg.create_pool(
        DB_DSN,
        min_size=POOL_MIN,
        max_size=POOL_MAX,
        command_timeout=10.0,
        max_inactive_connection_lifetime=300,
        # Supabase pooler/PgBouncer does not preserve prepared statements
        # across transaction-pooled connections.
        statement_cache_size=0,
    )


async def _get_pool(request: Request) -> asyncpg.Pool:
    pool = getattr(request.app.state, "pool", None)
    if pool is not None:
        return pool
    lock = getattr(request.app.state, "pool_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        request.app.state.pool_lock = lock
    async with lock:
        pool = getattr(request.app.state, "pool", None)
        if pool is None:
            pool = await _create_pool()
            request.app.state.pool = pool
    return pool


def _get_rate_limiter(request: Request) -> RateLimiter:
    limiter = getattr(request.app.state, "rate_limiter", None)
    if limiter is None:
        limiter = RateLimiter()
        request.app.state.rate_limiter = limiter
    return limiter


def _get_ttl_cache(request: Request, name: str, ttl: float) -> "TTLCache":
    cache = getattr(request.app.state, name, None)
    if cache is None:
        cache = TTLCache(ttl)
        setattr(request.app.state, name, cache)
    return cache


# ── Models ──────────────────────────────────────────────────────────────────


class PredictionsRequest(BaseModel):
    session_id: str = ""
    k: int = 5
    query: str | None = None
    end_user_id: str | None = None


class StageRequest(BaseModel):
    session_id: str = ""
    k: int = 5
    query: str | None = None
    end_user_id: str | None = None


class Prediction(BaseModel):
    memory_handle: str | None = None
    session_id: str
    content_preview: str | None = None
    content: str
    score: float
    topic_score: float | None = None
    recency_score: float | None = None
    platform: str | None = None
    storage_key: str | None = None
    content_bytes: int | None = None
    started_at: str | None = None


class PredictionsResponse(BaseModel):
    predictions: list[Prediction]
    latency_ms: int
    mode: str  # "query" | "live"
    timing: dict[str, int | bool] | None = None


class StageResponse(BaseModel):
    staged_id: str
    predictions: list[Prediction]
    latency_ms: int
    mode: str
    expires_in_sec: int
    expires_at: int
    timing: dict[str, int | bool] | None = None


class Event(BaseModel):
    role: str
    content: str
    ts: float | None = None
    source_path: str | None = None


class EventsRequest(BaseModel):
    session_id: str
    events: list[Event] = Field(..., max_length=1000)
    platform: str | None = None
    end_user_id: str | None = None


class HydrateRequest(BaseModel):
    memory_handles: list[str] = Field(default_factory=list, max_length=50)
    session_ids: list[str] = Field(default_factory=list, max_length=50)
    storage_keys: list[str] = Field(default_factory=list, max_length=50)
    end_user_id: str | None = None


class HydratedContext(BaseModel):
    memory_handle: str | None = None
    session_id: str | None = None
    storage_key: str
    content: str
    content_bytes: int
    found: bool
    source: str
    error: str | None = None


class HydrateResponse(BaseModel):
    contexts: list[HydratedContext]
    latency_ms: int


class ProductPrice(BaseModel):
    product: str
    unit: str
    price_usd: float
    note: str | None = None


class PublicConfigResponse(BaseModel):
    api_base_url: str
    products: list[str]
    pricing: list[ProductPrice]
    metrics: dict[str, str]
    billing_configured: bool


class AccountResponse(BaseModel):
    user_id: str
    platform: str | None = None
    products: list[str]
    rate_limits: dict[str, str]


class UsageProductSummary(BaseModel):
    product: str
    units: int
    bytes: int
    calls: int
    avg_latency_ms: float | None = None


class UsageResponse(BaseModel):
    days: int
    products: list[UsageProductSummary]


class AdminCreateKeyRequest(BaseModel):
    user_id: str
    platform: str | None = None


class AdminCreateKeyResponse(BaseModel):
    api_key: str
    user_id: str
    platform: str | None = None
    key_hash: str


class CheckoutResponse(BaseModel):
    url: str | None = None
    status: str
    message: str


class ReasoningRequest(BaseModel):
    query: str
    session_id: str = ""
    k: int = 5
    end_user_id: str | None = None
    model: str | None = None
    context_char_limit: int | None = None
    hydrate: bool = True


class ReasoningResponse(BaseModel):
    answer: str
    model: str
    mode: str
    latency_ms: int
    contexts_used: int
    memory_handles: list[str]
    timing: dict[str, int | bool] | None = None


# ── Auth ────────────────────────────────────────────────────────────────────


class TTLCache:
    def __init__(self, ttl_sec: float):
        self.ttl_sec = ttl_sec
        self._items: dict[Any, tuple[float, Any]] = {}

    def get(self, key: Any) -> Any | None:
        item = self.get_with_ttl(key)
        if item is None:
            return None
        return item[0]

    def get_with_ttl(self, key: Any) -> tuple[Any, float] | None:
        item = self._items.get(key)
        if item is None:
            return None
        expires_at, value = item
        remaining = expires_at - time.monotonic()
        if remaining <= 0:
            self._items.pop(key, None)
            return None
        return value, remaining

    def set(self, key: Any, value: Any) -> float:
        expires_at = time.monotonic() + self.ttl_sec
        self._items[key] = (expires_at, value)
        return expires_at

    def delete_where(self, predicate) -> int:
        keys = [key for key in self._items if predicate(key)]
        for key in keys:
            self._items.pop(key, None)
        return len(keys)

    def clear(self) -> None:
        self._items.clear()


async def authenticate(pool: asyncpg.Pool, authorization: str | None, cache: TTLCache | None = None) -> dict:
    """
    Validate the bearer token against ndp_api_keys. Returns the key row
    {user_id, platform, ...}. 401s if invalid.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = authorization[7:].strip()
    if not token:
        raise HTTPException(status_code=401, detail="empty bearer token")
    h = hashlib.sha256(token.encode("utf-8")).hexdigest()
    if cache is not None:
        cached = cache.get(h)
        if cached is not None:
            return dict(cached)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT user_id, platform, revoked_at FROM ndp_api_keys WHERE key_hash = $1",
            h,
        )
    if not row:
        raise HTTPException(status_code=401, detail="invalid api key")
    if row["revoked_at"] is not None:
        raise HTTPException(status_code=401, detail="api key revoked")
    auth = dict(row)
    if cache is not None:
        cache.set(h, auth)
    return auth


def _prediction_cache_key(
    user_id: str,
    end_user_id: str | None,
    query: str | None,
    session_id: str,
    k: int,
) -> tuple[str, str | None, str, str, int]:
    return (user_id, end_user_id, query or "", session_id or "", int(k))


def _stage_id(cache_key: tuple[str, str | None, str, str, int]) -> str:
    raw = "\x1f".join("" if part is None else str(part) for part in cache_key)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _invalidate_prediction_scope(
    cache: TTLCache,
    *,
    user_id: str,
    end_user_id: str | None,
    session_id: str,
) -> int:
    def matches(key: Any) -> bool:
        if not isinstance(key, tuple) or len(key) != 5:
            return False
        key_user, key_end_user, _query, key_session, _k = key
        if key_user != user_id:
            return False
        if end_user_id is not None and key_end_user != end_user_id:
            return False
        return key_session in ("", session_id or "")

    return cache.delete_where(matches)


async def _record_usage(
    pool: asyncpg.Pool,
    *,
    user_id: str,
    product: str,
    endpoint: str,
    units: int = 1,
    bytes_count: int = 0,
    latency_ms: int | None = None,
) -> None:
    """Best-effort usage telemetry. Never fail the API path for telemetry."""
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO ndp_usage_events (user_id, product, endpoint, units, bytes, latency_ms)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                user_id, product, endpoint, int(units), int(bytes_count), latency_ms,
            )
    except Exception:
        return


def _record_usage_later(pool: asyncpg.Pool, **kwargs: Any) -> None:
    try:
        asyncio.create_task(_record_usage(pool, **kwargs))
    except RuntimeError:
        return


def _public_pricing() -> list[ProductPrice]:
    return [
        ProductPrice(product="predict", unit="1K staged prediction requests", price_usd=0.03),
        ProductPrice(product="memory", unit="1K retrieval requests", price_usd=0.10),
        ProductPrice(
            product="hydration",
            unit="1K fetches",
            price_usd=0.05,
            note="$0.15 / GB returned and $0.03 / GB-month stored in preview",
        ),
        ProductPrice(
            product="reasoning_byok",
            unit="1K orchestration calls",
            price_usd=5.00,
            note="customer pays model provider directly",
        ),
        ProductPrice(
            product="reasoning_hosted",
            unit="1K hosted answer calls",
            price_usd=40.00,
            note="standard token budget; large-context overages quoted separately",
        ),
    ]


def _public_metrics() -> dict[str, str]:
    return {
        "predict_pqhr_adaptive_w10": "85.9%",
        "predict_pqhr_fixed_w7": "84.6%",
        "memory_longmemeval_retrieval_at_5": "90.6%",
        "memory_locomo_retrieval_at_5": "85.1%",
        "hydration_local_api_p50": "0.49ms",
        "hydration_local_api_p95": "0.88ms",
        "reasoning_longmemeval_s_e2e": "85.2%",
    }


def _mint_api_key() -> tuple[str, str]:
    token = f"ndpa_live_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return token, key_hash


def _extract_response_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"].strip()
    parts: list[str] = []
    for item in payload.get("output", []) or []:
        for content in item.get("content", []) or []:
            text = content.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts).strip()


def _call_openai_response(
    *,
    api_key: str,
    model: str,
    query: str,
    context: str,
) -> str:
    body = {
        "model": model,
        "instructions": (
            "You are NDPA Reasoning. Answer using only the retrieved memory "
            "context. If the context is insufficient, say what is missing. "
            "Be concise and preserve dates, names, numbers, and user preferences."
        ),
        "input": (
            "Retrieved memory context:\n"
            f"{context}\n\n"
            "Question:\n"
            f"{query}\n\n"
            "Answer:"
        ),
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise HTTPException(status_code=502, detail=f"OpenAI error {exc.code}: {detail}") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI request failed: {exc}") from exc
    text = _extract_response_text(payload)
    if not text:
        raise HTTPException(status_code=502, detail="OpenAI returned no text")
    return text


# ── Endpoints ───────────────────────────────────────────────────────────────


@app.get("/config", response_model=PublicConfigResponse)
async def public_config() -> PublicConfigResponse:
    return PublicConfigResponse(
        api_base_url=PUBLIC_API_BASE_URL or "http://localhost:8000",
        products=["predict", "memory", "hydration", "reasoning_byok", "reasoning_hosted"],
        pricing=_public_pricing(),
        metrics=_public_metrics(),
        billing_configured=bool(STRIPE_PAYMENT_LINK),
    )


@app.get("/health")
async def health(request: Request) -> dict:
    pool = await _get_pool(request)
    async with pool.acquire() as conn:
        await conn.fetchval("SELECT 1")
    return {"status": "ok", "ts": int(time.time())}


@app.get("/account", response_model=AccountResponse)
async def account(
    request: Request,
    authorization: str = Header(default=""),
) -> AccountResponse:
    pool = await _get_pool(request)
    auth_cache = _get_ttl_cache(request, "auth_cache", AUTH_CACHE_TTL_SEC)
    auth = await authenticate(pool, authorization, auth_cache)
    return AccountResponse(
        user_id=auth["user_id"],
        platform=auth.get("platform"),
        products=["predict", "memory", "hydration", "reasoning_byok", "reasoning_hosted"],
        rate_limits={
            "events": "600/min/key",
            "stage": "300/min/key",
            "predictions": "300/min/key",
            "hydrate": "120/min/key",
        },
    )


@app.get("/usage", response_model=UsageResponse)
async def usage(
    request: Request,
    days: int = 30,
    authorization: str = Header(default=""),
) -> UsageResponse:
    pool = await _get_pool(request)
    auth_cache = _get_ttl_cache(request, "auth_cache", AUTH_CACHE_TTL_SEC)
    auth = await authenticate(pool, authorization, auth_cache)
    safe_days = max(1, min(int(days), 90))
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
              product,
              COALESCE(SUM(units), 0)::bigint AS units,
              COALESCE(SUM(bytes), 0)::bigint AS bytes,
              COUNT(*)::bigint AS calls,
              AVG(latency_ms)::float AS avg_latency_ms
            FROM ndp_usage_events
            WHERE user_id = $1
              AND created_at >= NOW() - ($2::int * INTERVAL '1 day')
            GROUP BY product
            ORDER BY product
            """,
            auth["user_id"],
            safe_days,
        )
    return UsageResponse(
        days=safe_days,
        products=[
            UsageProductSummary(
                product=r["product"],
                units=int(r["units"] or 0),
                bytes=int(r["bytes"] or 0),
                calls=int(r["calls"] or 0),
                avg_latency_ms=round(float(r["avg_latency_ms"]), 2)
                if r["avg_latency_ms"] is not None else None,
            )
            for r in rows
        ],
    )


@app.post("/admin/api-keys", response_model=AdminCreateKeyResponse)
async def admin_create_api_key(
    req: AdminCreateKeyRequest,
    request: Request,
    x_ndpa_admin_token: str = Header(default=""),
) -> AdminCreateKeyResponse:
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=503, detail="admin key minting is not configured")
    if not secrets.compare_digest(x_ndpa_admin_token, ADMIN_TOKEN):
        raise HTTPException(status_code=401, detail="invalid admin token")
    token, key_hash = _mint_api_key()
    pool = await _get_pool(request)
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO ndp_api_keys (user_id, platform, key_hash)
            VALUES ($1, $2, $3)
            """,
            req.user_id,
            req.platform,
            key_hash,
        )
    return AdminCreateKeyResponse(
        api_key=token,
        user_id=req.user_id,
        platform=req.platform,
        key_hash=key_hash,
    )


@app.post("/checkout", response_model=CheckoutResponse)
async def checkout(
    request: Request,
    authorization: str = Header(default=""),
) -> CheckoutResponse:
    pool = await _get_pool(request)
    auth_cache = _get_ttl_cache(request, "auth_cache", AUTH_CACHE_TTL_SEC)
    await authenticate(pool, authorization, auth_cache)
    if not STRIPE_PAYMENT_LINK:
        return CheckoutResponse(
            status="billing_not_configured",
            message="Stripe payment link is not configured. Set NDPA_STRIPE_PAYMENT_LINK.",
        )
    return CheckoutResponse(
        url=STRIPE_PAYMENT_LINK,
        status="ready",
        message="Open this URL to start the Stripe checkout flow.",
    )


@app.post("/predictions", response_model=PredictionsResponse)
async def predictions(
    req: PredictionsRequest,
    request: Request,
    authorization: str = Header(default=""),
) -> PredictionsResponse:
    pool = await _get_pool(request)
    auth_cache = _get_ttl_cache(request, "auth_cache", AUTH_CACHE_TTL_SEC)
    prediction_cache = _get_ttl_cache(request, "prediction_cache", PREDICTION_CACHE_TTL_SEC)
    total_t0 = time.monotonic()
    auth_t0 = time.monotonic()
    auth = await authenticate(pool, authorization, auth_cache)
    auth_ms = int((time.monotonic() - auth_t0) * 1000)

    rl = _get_rate_limiter(request)
    rl.check(f"pred:{auth['user_id']}", limit=300, window_sec=60)

    cache_key = _prediction_cache_key(auth["user_id"], req.end_user_id, req.query, req.session_id, req.k)
    cached_item = prediction_cache.get_with_ttl(cache_key)
    if cached_item is None:
        rows, mode, pred_timing = await predict_topk(
            pool,
            user_id=auth["user_id"],
            end_user_id=req.end_user_id,
            session_id=req.session_id,
            query=req.query,
            k=req.k,
        )
        prediction_cache.set(cache_key, (rows, mode))
        cache_hit = False
        cache_remaining_ms = int(PREDICTION_CACHE_TTL_SEC * 1000)
    else:
        cached, remaining = cached_item
        rows, mode = cached
        pred_timing = {"db_candidate_ms": 0, "rank_ms": 0}
        cache_hit = True
        cache_remaining_ms = int(remaining * 1000)

    serialize_t0 = time.monotonic()
    predictions_out = [Prediction(**r) for r in rows]
    serialize_ms = int((time.monotonic() - serialize_t0) * 1000)
    elapsed_ms = int((time.monotonic() - total_t0) * 1000)
    bytes_out = sum(int(r.get("content_bytes") or 0) for r in rows)
    timing: dict[str, int | bool] = {
        "auth_ms": auth_ms,
        "db_candidate_ms": int(pred_timing.get("db_candidate_ms", 0)),
        "rank_ms": int(pred_timing.get("rank_ms", 0)),
        "serialize_ms": serialize_ms,
        "total_ms": elapsed_ms,
        "cache_hit": cache_hit,
        "cache_remaining_ms": cache_remaining_ms,
    }
    _record_usage_later(
        pool,
        user_id=auth["user_id"],
        product="memory",
        endpoint="/predictions",
        units=len(rows),
        bytes_count=bytes_out,
        latency_ms=elapsed_ms,
    )
    return PredictionsResponse(
        predictions=predictions_out,
        latency_ms=elapsed_ms,
        mode=mode,
        timing=timing,
    )


@app.post("/stage", response_model=StageResponse)
async def stage(
    req: StageRequest,
    request: Request,
    authorization: str = Header(default=""),
) -> StageResponse:
    """
    Precompute and cache a prediction result for the next hot read.

    This is the product surface for predict-forward latency: callers run
    /stage between turns, before a call starts, or in a background worker.
    The later /predictions call with the same key returns from memory without
    a Supabase candidate query.
    """
    pool = await _get_pool(request)
    auth_cache = _get_ttl_cache(request, "auth_cache", AUTH_CACHE_TTL_SEC)
    prediction_cache = _get_ttl_cache(request, "prediction_cache", PREDICTION_CACHE_TTL_SEC)
    total_t0 = time.monotonic()
    auth_t0 = time.monotonic()
    auth = await authenticate(pool, authorization, auth_cache)
    auth_ms = int((time.monotonic() - auth_t0) * 1000)

    rl = _get_rate_limiter(request)
    rl.check(f"stage:{auth['user_id']}", limit=300, window_sec=60)

    rows, mode, pred_timing = await predict_topk(
        pool,
        user_id=auth["user_id"],
        end_user_id=req.end_user_id,
        session_id=req.session_id,
        query=req.query,
        k=req.k,
    )
    cache_key = _prediction_cache_key(auth["user_id"], req.end_user_id, req.query, req.session_id, req.k)
    prediction_cache.set(cache_key, (rows, mode))

    serialize_t0 = time.monotonic()
    predictions_out = [Prediction(**r) for r in rows]
    serialize_ms = int((time.monotonic() - serialize_t0) * 1000)
    elapsed_ms = int((time.monotonic() - total_t0) * 1000)
    expires_in_sec = int(math.ceil(PREDICTION_CACHE_TTL_SEC))
    expires_at = int(time.time() + PREDICTION_CACHE_TTL_SEC)
    bytes_out = sum(int(r.get("content_bytes") or 0) for r in rows)
    timing: dict[str, int | bool] = {
        "auth_ms": auth_ms,
        "db_candidate_ms": int(pred_timing.get("db_candidate_ms", 0)),
        "rank_ms": int(pred_timing.get("rank_ms", 0)),
        "serialize_ms": serialize_ms,
        "total_ms": elapsed_ms,
        "cache_hit": False,
        "cache_remaining_ms": int(PREDICTION_CACHE_TTL_SEC * 1000),
    }
    _record_usage_later(
        pool,
        user_id=auth["user_id"],
        product="predict",
        endpoint="/stage",
        units=len(rows),
        bytes_count=bytes_out,
        latency_ms=elapsed_ms,
    )
    return StageResponse(
        staged_id=_stage_id(cache_key),
        predictions=predictions_out,
        latency_ms=elapsed_ms,
        mode=mode,
        expires_in_sec=expires_in_sec,
        expires_at=expires_at,
        timing=timing,
    )


@app.post("/events")
async def events(
    req: EventsRequest,
    request: Request,
    authorization: str = Header(default=""),
) -> dict:
    pool = await _get_pool(request)
    auth_cache = _get_ttl_cache(request, "auth_cache", AUTH_CACHE_TTL_SEC)
    prediction_cache = _get_ttl_cache(request, "prediction_cache", PREDICTION_CACHE_TTL_SEC)
    auth = await authenticate(pool, authorization, auth_cache)

    rl = _get_rate_limiter(request)
    rl.check(f"events:{auth['user_id']}", limit=600, window_sec=60)

    if not req.events:
        return {"inserted": 0}

    now = time.time()
    user_id = auth["user_id"]
    end_user_id = req.end_user_id
    session_id = req.session_id
    platform = req.platform or auth.get("platform")

    rows = []
    for ev in req.events:
        rows.append(
            (
                user_id,
                end_user_id,
                session_id,
                platform,
                ev.role,
                ev.content,
                float(ev.ts) if ev.ts is not None else now,
                ev.source_path,
            )
        )

    text_concat = "\n".join(
        f"[{ev.role}{' source=' + ev.source_path if ev.source_path else ''}] {ev.content}"
        for ev in req.events
    )

    # Write raw text to blob storage; Postgres only stores metadata.
    storage_key, content_bytes = write_blob(
        user_id,
        session_id,
        text_concat,
        backend=HYDRATION_BACKEND,
        root=HYDRATION_ROOT,
        s3=S3_CONFIG,
    )
    preview = text_concat[:2000]
    top_terms = _top_terms(text_concat)

    async with pool.acquire() as conn:
        async with conn.transaction():
            # 1) Upsert session counter
            await conn.execute(
                """
                INSERT INTO ndp_sessions (user_id, end_user_id, session_id, platform, turn_count, last_active)
                VALUES ($1, $2, $3, $4, $5, NOW())
                ON CONFLICT (user_id, session_id) DO UPDATE
                SET turn_count  = ndp_sessions.turn_count + EXCLUDED.turn_count,
                    last_active = NOW(),
                    platform    = COALESCE(EXCLUDED.platform, ndp_sessions.platform)
                """,
                user_id, end_user_id, session_id, platform, len(rows),
            )
            # 2) Upsert conversation metadata only (no raw content in Postgres)
            await conn.execute(
                """
                INSERT INTO ndp_conversations
                  (user_id, end_user_id, session_id, platform,
                   content_preview, top_terms, storage_key, content_bytes, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
                ON CONFLICT (user_id, session_id) DO UPDATE
                SET content_preview = EXCLUDED.content_preview,
                    top_terms       = (
                      SELECT array_agg(DISTINCT term)
                      FROM unnest(COALESCE(ndp_conversations.top_terms, '{}') || EXCLUDED.top_terms) AS t(term)
                    ),
                    storage_key     = EXCLUDED.storage_key,
                    content_bytes   = EXCLUDED.content_bytes,
                    platform        = COALESCE(EXCLUDED.platform, ndp_conversations.platform),
                    updated_at      = NOW()
                """,
                user_id, end_user_id, session_id, platform,
                preview, top_terms, storage_key, content_bytes,
            )
    invalidated = _invalidate_prediction_scope(
        prediction_cache,
        user_id=user_id,
        end_user_id=end_user_id,
        session_id=session_id,
    )
    _record_usage_later(
        pool,
        user_id=user_id,
        product="ingest",
        endpoint="/events",
        units=len(rows),
        bytes_count=content_bytes,
        latency_ms=None,
    )
    return {"inserted": len(rows), "invalidated_predictions": invalidated}


@app.post("/hydrate", response_model=HydrateResponse)
async def hydrate(
    req: HydrateRequest,
    request: Request,
    authorization: str = Header(default=""),
) -> HydrateResponse:
    pool = await _get_pool(request)
    auth_cache = _get_ttl_cache(request, "auth_cache", AUTH_CACHE_TTL_SEC)
    auth = await authenticate(pool, authorization, auth_cache)

    rl = _get_rate_limiter(request)
    rl.check(f"hydrate:{auth['user_id']}", limit=120, window_sec=60)

    t0 = time.monotonic()
    handles = list(dict.fromkeys([*req.memory_handles, *req.session_ids]))
    contexts: list[dict[str, Any]] = []
    storage_keys = [(None, None, key) for key in dict.fromkeys(req.storage_keys)]

    if handles:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT session_id, storage_key
                FROM ndp_conversations
                WHERE user_id = $1
                  AND ($2::text IS NULL OR end_user_id = $2)
                  AND session_id = ANY($3::text[])
                """,
                auth["user_id"],
                req.end_user_id,
                handles,
            )
            found = {r["session_id"]: r["storage_key"] for r in rows if r["storage_key"]}
        for handle in handles:
            storage_keys.append((handle, handle, found.get(handle)))

    for memory_handle, session_id, storage_key in storage_keys:
        if storage_key:
            item = hydrate_blob(
                storage_key,
                backend=HYDRATION_BACKEND,
                root=HYDRATION_ROOT,
                s3=S3_CONFIG,
            )
        else:
            item = {
                "storage_key": "",
                "content": "",
                "content_bytes": 0,
                "found": False,
                "error": "storage_key_missing",
                "source": "local_fs",
            }
        item["memory_handle"] = memory_handle
        item["session_id"] = session_id
        contexts.append(item)

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    bytes_out = sum(int(c.get("content_bytes") or 0) for c in contexts)
    _record_usage_later(
        pool,
        user_id=auth["user_id"],
        product="hydration",
        endpoint="/hydrate",
        units=len(contexts),
        bytes_count=bytes_out,
        latency_ms=elapsed_ms,
    )
    return HydrateResponse(contexts=[HydratedContext(**c) for c in contexts], latency_ms=elapsed_ms)


@app.post("/reasoning", response_model=ReasoningResponse)
async def reasoning(
    req: ReasoningRequest,
    request: Request,
    authorization: str = Header(default=""),
    x_openai_api_key: str = Header(default=""),
) -> ReasoningResponse:
    """
    Premium answer layer: Memory -> optional Hydration -> LLM answer.

    Hosted mode uses OPENAI_API_KEY. BYOK mode uses X-OpenAI-API-Key for the
    model call and never stores that provider key.
    """
    pool = await _get_pool(request)
    auth_cache = _get_ttl_cache(request, "auth_cache", AUTH_CACHE_TTL_SEC)
    total_t0 = time.monotonic()
    auth_t0 = time.monotonic()
    auth = await authenticate(pool, authorization, auth_cache)
    auth_ms = int((time.monotonic() - auth_t0) * 1000)

    rl = _get_rate_limiter(request)
    rl.check(f"reasoning:{auth['user_id']}", limit=60, window_sec=60)

    provider_key = x_openai_api_key.strip() or OPENAI_API_KEY
    mode = "byok" if x_openai_api_key.strip() else "hosted"
    if not provider_key:
        raise HTTPException(
            status_code=503,
            detail="Reasoning is not configured. Set OPENAI_API_KEY or send X-OpenAI-API-Key.",
        )

    retrieve_t0 = time.monotonic()
    rows, _retrieval_mode, pred_timing = await predict_topk(
        pool,
        user_id=auth["user_id"],
        end_user_id=req.end_user_id,
        session_id=req.session_id,
        query=req.query,
        k=req.k,
    )
    retrieve_ms = int((time.monotonic() - retrieve_t0) * 1000)

    hydrate_t0 = time.monotonic()
    context_chunks: list[str] = []
    for row in rows:
        storage_key = row.get("storage_key")
        raw = ""
        if req.hydrate and storage_key:
            item = hydrate_blob(
                str(storage_key),
                backend=HYDRATION_BACKEND,
                root=HYDRATION_ROOT,
                s3=S3_CONFIG,
            )
            if item.get("found"):
                raw = str(item.get("content") or "")
        if not raw:
            raw = str(row.get("content_preview") or row.get("content") or "")
        if raw:
            context_chunks.append(
                f"[session_id={row.get('session_id')} score={float(row.get('score') or 0):.3f}]\n{raw}"
            )
    context_limit = max(1000, min(int(req.context_char_limit or REASONING_CONTEXT_CHARS), 60000))
    context = "\n\n---\n\n".join(context_chunks)[:context_limit]
    hydrate_ms = int((time.monotonic() - hydrate_t0) * 1000)
    if not context:
        raise HTTPException(status_code=404, detail="no relevant memory context found")

    model = req.model or REASONING_MODEL
    llm_t0 = time.monotonic()
    answer = await asyncio.to_thread(
        _call_openai_response,
        api_key=provider_key,
        model=model,
        query=req.query,
        context=context,
    )
    llm_ms = int((time.monotonic() - llm_t0) * 1000)
    elapsed_ms = int((time.monotonic() - total_t0) * 1000)

    _record_usage_later(
        pool,
        user_id=auth["user_id"],
        product="reasoning",
        endpoint="/reasoning",
        units=1,
        bytes_count=len(context.encode("utf-8")),
        latency_ms=elapsed_ms,
    )
    return ReasoningResponse(
        answer=answer,
        model=model,
        mode=mode,
        latency_ms=elapsed_ms,
        contexts_used=len(context_chunks),
        memory_handles=[str(r.get("memory_handle") or r.get("session_id")) for r in rows],
        timing={
            "auth_ms": auth_ms,
            "retrieve_ms": retrieve_ms,
            "db_candidate_ms": int(pred_timing.get("db_candidate_ms", 0)),
            "rank_ms": int(pred_timing.get("rank_ms", 0)),
            "hydrate_ms": hydrate_ms,
            "llm_ms": llm_ms,
            "total_ms": elapsed_ms,
            "byok": mode == "byok",
        },
    )


def _top_terms(text: str, limit: int = 300) -> list[str]:
    counts: dict[str, int] = {}
    for word in _WORD_RE.findall(text.lower()):
        if word not in _STOP and len(word) >= 3:
            counts[word] = counts.get(word, 0) + 1
    return [word for word, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]]
