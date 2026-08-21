#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_ENV="${DEPLOY_ENV:-$ROOT_DIR/config/deployment.env}"
if [[ -f "$DEPLOY_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$DEPLOY_ENV"
fi

: "${WORKER_HOST:?set WORKER_HOST or create config/deployment.env}"
: "${NSYS_OUTPUT_DIR:?set NSYS_OUTPUT_DIR to the absolute profiling output directory}"
[[ "$NSYS_OUTPUT_DIR" == /* ]] || {
  echo "ERROR: NSYS_OUTPUT_DIR must be absolute" >&2
  exit 1
}
NSYS_SESSION_PREFIX="${NSYS_SESSION_PREFIX:-dsv4}"
NSYS_HOST_ROOT="${NSYS_HOST_ROOT:-/opt/nvidia/nsight-systems/2025.3.2}"
NSYS_GPU_METRICS_FREQUENCY="${NSYS_GPU_METRICS_FREQUENCY:-1000}"
HCA="${HCA:-rocep1s0f1}"
NIC="${NIC:-enp1s0f1np1}"
HEAD_CONTAINER="dsv4-nvfp4-ds-mla-vllm027-tp2-head"
WORKER_CONTAINER="dsv4-nvfp4-ds-mla-vllm027-tp2-worker"
CONTAINER_NSYS=/opt/nsys/bin/nsys
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10)

if (($# == 0)); then
  echo "usage: NSYS_OUTPUT_DIR=/absolute/path $0 <workload command> [args...]" >&2
  exit 2
fi

mkdir -p "$NSYS_OUTPUT_DIR"
counter_file="$NSYS_OUTPUT_DIR/${NSYS_SESSION_PREFIX}-rdma-counters.tsv"
printf 'node\tphase\tsource\tcounter\tvalue\n' >"$counter_file"

snapshot_local() {
  local node="$1" phase="$2" file source
  for file in \
    "/sys/class/infiniband/$HCA/ports/1/counters/"* \
    "/sys/class/infiniband/$HCA/ports/1/hw_counters/"* \
    "/sys/class/net/$NIC/statistics/"*; do
    [[ -f "$file" ]] || continue
    case "$file" in
      */hw_counters/*) source=rdma_hw ;;
      */counters/*) source=rdma_port ;;
      *) source=netdev ;;
    esac
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "$node" "$phase" "$source" "${file##*/}" "$(<"$file")"
  done >>"$counter_file"
}

snapshot_worker() {
  local phase="$1"
  "${SSH[@]}" "$WORKER_HOST" bash -s -- "$HCA" "$NIC" "$phase" <<'REMOTE' >>"$counter_file"
set -Eeuo pipefail
hca="$1"
nic="$2"
phase="$3"
for file in \
  "/sys/class/infiniband/$hca/ports/1/counters/"* \
  "/sys/class/infiniband/$hca/ports/1/hw_counters/"* \
  "/sys/class/net/$nic/statistics/"*; do
  [[ -f "$file" ]] || continue
  case "$file" in
    */hw_counters/*) source=rdma_hw ;;
    */counters/*) source=rdma_port ;;
    *) source=netdev ;;
  esac
  printf 'rank1\t%s\t%s\t%s\t%s\n' \
    "$phase" "$source" "${file##*/}" "$(<"$file")"
done
REMOTE
}

head_app_started=0
worker_app_started=0
head_metrics_started=0
worker_metrics_started=0

stop_collection() {
  set +e
  local pids=()
  if ((head_app_started)); then
    docker exec "$HEAD_CONTAINER" "$CONTAINER_NSYS" stop \
      --session="${NSYS_SESSION_PREFIX}rank0" &
    pids+=("$!")
    head_app_started=0
  fi
  if ((worker_app_started)); then
    "${SSH[@]}" "$WORKER_HOST" docker exec "$WORKER_CONTAINER" \
      "$CONTAINER_NSYS" stop --session="${NSYS_SESSION_PREFIX}rank1" &
    pids+=("$!")
    worker_app_started=0
  fi
  if ((head_metrics_started)); then
    sudo -n "$NSYS_HOST_ROOT/bin/nsys" stop \
      --session="${NSYS_SESSION_PREFIX}gpu0" &
    pids+=("$!")
    head_metrics_started=0
  fi
  if ((worker_metrics_started)); then
    "${SSH[@]}" "$WORKER_HOST" sudo -n "$NSYS_HOST_ROOT/bin/nsys" stop \
      --session="${NSYS_SESSION_PREFIX}gpu1" &
    pids+=("$!")
    worker_metrics_started=0
  fi
  if ((${#pids[@]})); then
    wait "${pids[@]}"
  fi
  set -e
}
trap stop_collection EXIT INT TERM

[[ "$(docker inspect --format '{{.State.Running}}' "$HEAD_CONTAINER")" == true ]]
[[ "$("${SSH[@]}" "$WORKER_HOST" docker inspect --format '{{.State.Running}}' "$WORKER_CONTAINER")" == true ]]
docker exec "$HEAD_CONTAINER" "$CONTAINER_NSYS" sessions list
"${SSH[@]}" "$WORKER_HOST" docker exec "$WORKER_CONTAINER" \
  "$CONTAINER_NSYS" sessions list

snapshot_local rank0 before
snapshot_worker before

sudo -n "$NSYS_HOST_ROOT/bin/nsys" start \
  --session-new="${NSYS_SESSION_PREFIX}gpu0" \
  --gpu-metrics-devices=all \
  --gpu-metrics-frequency="$NSYS_GPU_METRICS_FREQUENCY" \
  --sample=none --cpuctxsw=none \
  --output="$NSYS_OUTPUT_DIR/${NSYS_SESSION_PREFIX}-rank0-gpu-metrics"
head_metrics_started=1
"${SSH[@]}" "$WORKER_HOST" sudo -n "$NSYS_HOST_ROOT/bin/nsys" start \
  --session-new="${NSYS_SESSION_PREFIX}gpu1" \
  --gpu-metrics-devices=all \
  --gpu-metrics-frequency="$NSYS_GPU_METRICS_FREQUENCY" \
  --sample=none --cpuctxsw=none \
  --output="$NSYS_OUTPUT_DIR/${NSYS_SESSION_PREFIX}-rank1-gpu-metrics"
worker_metrics_started=1

docker exec "$HEAD_CONTAINER" "$CONTAINER_NSYS" start \
  --session="${NSYS_SESSION_PREFIX}rank0" &
head_start_pid=$!
head_app_started=1
"${SSH[@]}" "$WORKER_HOST" docker exec "$WORKER_CONTAINER" \
  "$CONTAINER_NSYS" start --session="${NSYS_SESSION_PREFIX}rank1" &
worker_start_pid=$!
worker_app_started=1
wait "$head_start_pid"
wait "$worker_start_pid"

workload_status=0
"$@" || workload_status=$?
stop_collection
trap - EXIT INT TERM

snapshot_local rank0 after
snapshot_worker after

mkdir -p "$NSYS_OUTPUT_DIR/rank1-artifacts"
for suffix in rank1.nsys-rep rank1-gpu-metrics.nsys-rep; do
  "${SSH[@]}" "$WORKER_HOST" test -f \
    "$NSYS_OUTPUT_DIR/${NSYS_SESSION_PREFIX}-$suffix" || continue
  scp -q "$WORKER_HOST:$NSYS_OUTPUT_DIR/${NSYS_SESSION_PREFIX}-$suffix" \
    "$NSYS_OUTPUT_DIR/rank1-artifacts/"
done

echo "Nsight artifacts: $NSYS_OUTPUT_DIR"
exit "$workload_status"
