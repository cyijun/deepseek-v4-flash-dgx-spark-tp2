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

docker rm -f "$HEAD_CONTAINER" >/dev/null 2>&1 || true
ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_HOST" \
  "docker rm -f '$WORKER_CONTAINER' >/dev/null 2>&1 || true"
echo "Stopped $HEAD_CONTAINER and $WORKER_CONTAINER"
