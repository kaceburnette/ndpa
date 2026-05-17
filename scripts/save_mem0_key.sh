#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="/Users/kaceburnette/Desktop/ndp/.env"

read -s -p "Paste your Mem0 API key then press Enter: " K
echo

# Append or update MEM0_API_KEY in .env
if grep -q '^MEM0_API_KEY=' "$ENV_FILE" 2>/dev/null; then
    # Replace existing line (BSD sed compatible)
    sed -i '' "s|^MEM0_API_KEY=.*|MEM0_API_KEY=$K|" "$ENV_FILE"
else
    echo "MEM0_API_KEY=$K" >> "$ENV_FILE"
fi

chmod 600 "$ENV_FILE"

echo "Saved. Prefix check:"
grep '^MEM0_API_KEY=' "$ENV_FILE" | head -c 30
echo "..."
