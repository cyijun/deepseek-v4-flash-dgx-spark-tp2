#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_ENV="${DEPLOY_ENV:-$ROOT_DIR/config/deployment.env}"
if [[ -f "$DEPLOY_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$DEPLOY_ENV"
fi
: "${WORKER_HOST:?set WORKER_HOST or create config/deployment.env}"
MIN_RUNTIME_AVAILABLE_GIB="${MIN_RUNTIME_AVAILABLE_GIB:-10}"
MAX_RUNTIME_SWAP_GROWTH_MIB="${MAX_RUNTIME_SWAP_GROWTH_MIB:-512}"
POLL_SECONDS="${POLL_SECONDS:-2}"
HEAD_CONTAINER="dsv4-nvfp4-ds-mla-vllm027-tp2-head"
WORKER_CONTAINER="dsv4-nvfp4-ds-mla-vllm027-tp2-worker"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10)

available_kib() {
  awk '/^MemAvailable:/ {print $2}' /proc/meminfo
}

worker_available_kib() {
  "${SSH[@]}" "$WORKER_HOST" cat /proc/meminfo |
    awk '/^MemAvailable:/ {print $2}'
}

swap_free_kib() {
  awk '/^SwapFree:/ {print $2}' /proc/meminfo
}

worker_swap_free_kib() {
  "${SSH[@]}" "$WORKER_HOST" cat /proc/meminfo |
    awk '/^SwapFree:/ {print $2}'
}

stop_with_diagnostics() {
  local reason="$1" stamp
  stamp="$(date '+%Y%m%dT%H%M%S%z')"
  mkdir -p "$ROOT_DIR/logs"
  docker logs "$HEAD_CONTAINER" >"$ROOT_DIR/logs/${stamp}-runtime-head.log" 2>&1 || true
  "${SSH[@]}" "$WORKER_HOST" docker logs "$WORKER_CONTAINER" \
    >"$ROOT_DIR/logs/${stamp}-runtime-worker.log" 2>&1 || true
  "$ROOT_DIR/scripts/stop.sh"
  printf '%s %s\n' "$(date -Ins)" "$reason" \
    >"$ROOT_DIR/logs/${stamp}-runtime-guard.txt"
  echo "RUNTIME GUARD STOP: $reason; logs=$ROOT_DIR/logs/${stamp}-runtime-*" >&2
  exit 1
}

minimum_kib="$((MIN_RUNTIME_AVAILABLE_GIB * 1024 * 1024))"
maximum_swap_growth_kib="$((MAX_RUNTIME_SWAP_GROWTH_MIB * 1024))"
head_swap_free_start_kib="$(swap_free_kib)"
worker_swap_free_start_kib="$(worker_swap_free_kib)"
last_report=0
while true; do
  head_kib="$(available_kib)"
  worker_kib="$(worker_available_kib)"
  head_swap_growth_kib="$((head_swap_free_start_kib - $(swap_free_kib)))"
  worker_swap_growth_kib="$((worker_swap_free_start_kib - $(worker_swap_free_kib)))"
  ((head_kib >= minimum_kib)) || stop_with_diagnostics \
    "head MemAvailable=$((head_kib / 1024 / 1024)) GiB < ${MIN_RUNTIME_AVAILABLE_GIB} GiB"
  ((worker_kib >= minimum_kib)) || stop_with_diagnostics \
    "worker MemAvailable=$((worker_kib / 1024 / 1024)) GiB < ${MIN_RUNTIME_AVAILABLE_GIB} GiB"
  ((head_swap_growth_kib <= maximum_swap_growth_kib)) || stop_with_diagnostics \
    "head swap grew by $((head_swap_growth_kib / 1024)) MiB > ${MAX_RUNTIME_SWAP_GROWTH_MIB} MiB"
  ((worker_swap_growth_kib <= maximum_swap_growth_kib)) || stop_with_diagnostics \
    "worker swap grew by $((worker_swap_growth_kib / 1024)) MiB > ${MAX_RUNTIME_SWAP_GROWTH_MIB} MiB"

  [[ "$(docker inspect --format '{{.State.Running}}' "$HEAD_CONTAINER")" == "true" ]] || exit 0
  [[ "$("${SSH[@]}" "$WORKER_HOST" docker inspect --format '{{.State.Running}}' "$WORKER_CONTAINER")" == "true" ]] || exit 0

  if ((SECONDS - last_report >= 30)); then
    echo "runtime guard: MemAvailable head=$((head_kib / 1024 / 1024)) GiB worker=$((worker_kib / 1024 / 1024)) GiB; swap growth head=$((head_swap_growth_kib / 1024)) MiB worker=$((worker_swap_growth_kib / 1024)) MiB"
    last_report="$SECONDS"
  fi
  sleep "$POLL_SECONDS"
done
