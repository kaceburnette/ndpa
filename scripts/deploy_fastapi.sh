#!/usr/bin/env bash
# Deploy the NDPA FastAPI server to Fly.io.
# Prereqs: fly CLI (`brew install flyctl`), Fly account, DATABASE_URL from Supabase.
set -euo pipefail

REPO_ROOT="/Users/kaceburnette/Desktop/ndp"
APP_NAME="ndpa-api"
REGION="${FLY_REGION:-iad}"

cd "$REPO_ROOT"

# 1) Make sure fly CLI exists
if ! command -v fly >/dev/null 2>&1; then
  echo "fly CLI not installed. Run: brew install flyctl"
  exit 1
fi

# 2) Make sure user is logged in
if ! fly auth whoami >/dev/null 2>&1; then
  echo "Not logged in to Fly.io. Run: fly auth login"
  exit 1
fi

# 3) Read DATABASE_URL from .env if not already in env
if [ -z "${DATABASE_URL:-}" ]; then
  if grep -q '^DATABASE_URL=' "$REPO_ROOT/.env" 2>/dev/null; then
    DATABASE_URL="$(grep '^DATABASE_URL=' "$REPO_ROOT/.env" | head -1 | cut -d= -f2-)"
    export DATABASE_URL
  else
    echo "DATABASE_URL not set and not found in $REPO_ROOT/.env"
    echo "Get it from: Supabase Dashboard → Project → Settings → Database → Connection Pooling (Transaction mode)"
    exit 1
  fi
fi

# 4) Create app if it doesn't exist
if ! fly apps list 2>/dev/null | grep -q "^$APP_NAME"; then
  echo "Creating Fly app: $APP_NAME"
  fly apps create "$APP_NAME" --org personal
fi

# 5) Set secrets. Optional launch envs are picked up if present.
echo "Setting DATABASE_URL secret..."
secret_args=(DATABASE_URL=-)
if [ -n "${NDPA_HYDRATION_ROOT:-}" ]; then secret_args+=(NDPA_HYDRATION_ROOT="$NDPA_HYDRATION_ROOT"); fi
if [ -n "${NDPA_HYDRATION_BACKEND:-}" ]; then secret_args+=(NDPA_HYDRATION_BACKEND="$NDPA_HYDRATION_BACKEND"); fi
if [ -n "${NDPA_S3_ENDPOINT:-}" ]; then secret_args+=(NDPA_S3_ENDPOINT="$NDPA_S3_ENDPOINT"); fi
if [ -n "${NDPA_S3_BUCKET:-}" ]; then secret_args+=(NDPA_S3_BUCKET="$NDPA_S3_BUCKET"); fi
if [ -n "${NDPA_S3_REGION:-}" ]; then secret_args+=(NDPA_S3_REGION="$NDPA_S3_REGION"); fi
if [ -n "${NDPA_S3_ACCESS_KEY_ID:-}" ]; then secret_args+=(NDPA_S3_ACCESS_KEY_ID="$NDPA_S3_ACCESS_KEY_ID"); fi
if [ -n "${NDPA_S3_SECRET_ACCESS_KEY:-}" ]; then secret_args+=(NDPA_S3_SECRET_ACCESS_KEY="$NDPA_S3_SECRET_ACCESS_KEY"); fi
if [ -n "${NDPA_PUBLIC_API_BASE_URL:-}" ]; then secret_args+=(NDPA_PUBLIC_API_BASE_URL="$NDPA_PUBLIC_API_BASE_URL"); fi
if [ -n "${NDPA_CORS_ORIGINS:-}" ]; then secret_args+=(NDPA_CORS_ORIGINS="$NDPA_CORS_ORIGINS"); fi
if [ -n "${NDPA_ADMIN_TOKEN:-}" ]; then secret_args+=(NDPA_ADMIN_TOKEN="$NDPA_ADMIN_TOKEN"); fi
if [ -n "${NDPA_STRIPE_PAYMENT_LINK:-}" ]; then secret_args+=(NDPA_STRIPE_PAYMENT_LINK="$NDPA_STRIPE_PAYMENT_LINK"); fi
if [ -n "${OPENAI_API_KEY:-}" ]; then secret_args+=(OPENAI_API_KEY="$OPENAI_API_KEY"); fi
if [ -n "${NDPA_REASONING_MODEL:-}" ]; then secret_args+=(NDPA_REASONING_MODEL="$NDPA_REASONING_MODEL"); fi
echo "$DATABASE_URL" | fly secrets set "${secret_args[@]}" --app "$APP_NAME" --stage

# 6) Deploy
echo "Deploying..."
fly deploy --config "$REPO_ROOT/server/fly.toml" --app "$APP_NAME" --remote-only

# 7) Health check
echo ""
echo "Health check:"
sleep 5
curl -sf "https://$APP_NAME.fly.dev/health" && echo "  OK" || (echo "FAILED"; exit 1)

echo ""
echo "Deployed: https://$APP_NAME.fly.dev"
echo ""
echo "Point the SDK at it:"
echo "  export NDPA_BASE_URL=\"https://$APP_NAME.fly.dev\""
