#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_ENV="${DEPLOY_ENV:-$ROOT_DIR/config/deployment.env}"
if [[ -f "$DEPLOY_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$DEPLOY_ENV"
fi
: "${WORKER_HOST:?set WORKER_HOST or create config/deployment.env}"
HEAD_CONTAINER="dsv4-nvfp4-ds-mla-vllm027-tp2-head"
WORKER_CONTAINER="dsv4-nvfp4-ds-mla-vllm027-tp2-worker"

echo "--- head ---"
docker ps -a --filter "name=^/${HEAD_CONTAINER}$" --format '{{.Names}} {{.Status}} {{.Image}}'
docker logs --tail 40 "$HEAD_CONTAINER" 2>&1 || true
echo "--- worker ---"
ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_HOST" \
  "docker ps -a --filter 'name=^/${WORKER_CONTAINER}$' --format '{{.Names}} {{.Status}} {{.Image}}'; docker logs --tail 40 '$WORKER_CONTAINER' 2>&1 || true"
