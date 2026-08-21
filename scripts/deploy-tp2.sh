#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_ENV="${DEPLOY_ENV:-$ROOT_DIR/config/deployment.env}"
if [[ -f "$DEPLOY_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$DEPLOY_ENV"
fi
# shellcheck disable=SC1091
source "$ROOT_DIR/config/versions.env"

IMAGE="${IMAGE:-ghcr.io/cyijun/deepseek-v4-flash-dgx-spark-tp2:vllm027-sm121}"
: "${MODEL_REPO:?set MODEL_REPO to the identical Hugging Face cache root on both nodes}"
: "${MODEL_REV:?set MODEL_REV to an immutable checkpoint revision}"
MODEL_SUBPATH="${MODEL_SUBPATH:-snapshots/${MODEL_REV}}"
MODEL_PATH="/model/${MODEL_SUBPATH}"
EXPECTED_MODEL_WEIGHT_FILES="${EXPECTED_MODEL_WEIGHT_FILES:-48}"
SERVED_MODEL="${SERVED_MODEL:-deepseek-v4-flash}"

PULL_IMAGE="${PULL_IMAGE:-0}"
VERIFY_SOURCE_LABELS="${VERIFY_SOURCE_LABELS:-1}"
: "${WORKER_HOST:?set WORKER_HOST, for example user@dgx-spark-worker.local}"
HEAD_IP="${HEAD_IP:-10.100.32.1}"
WORKER_IP="${WORKER_IP:-10.100.32.2}"
NIC="${NIC:-enp1s0f1np1}"
HCA="${HCA:-rocep1s0f1}"
MASTER_PORT="${MASTER_PORT:-29627}"
API_PORT="${API_PORT:-8888}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-6}"
# Match the validated DSpark serving profile while keeping concurrency capped
# at C6 below.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
# DGX Spark GPU allocations share system DRAM.  Leave enough memory outside
# vLLM to prevent the Python workers and JIT compiler from swap thrashing.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.76}"
# A fixed KV budget is safer than a utilization ratio on unified-memory GPUs.
# The repository default is the final 6 GiB C6 safety envelope; a 10 GiB trial
# crossed the host-reserve guard during graph warmup.
# Set this empty to restore gpu_memory_utilization-based sizing.
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-6442450944}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-nvfp4_ds_mla}"
MTP_NUM_TOKENS="${MTP_NUM_TOKENS:-5}"
DRAFT_MOE_BACKEND="${DRAFT_MOE_BACKEND:-marlin}"
TARGET_MOE_BACKEND="${TARGET_MOE_BACKEND:-flashinfer_b12x}"
# Use the checkpoint's NVFP4 weights with B12X's packed W4A16 execution path.
# This keeps the weight footprint at FP4 while avoiding per-token activation
# quantization in every target-model expert layer.
B12X_NVFP4_W4A16="${B12X_NVFP4_W4A16-1}"
# Use b12x's caller-owned-scratch path for native MXFP4 checkpoints. The
# selector controls are ignored by newer b12x runtimes and bounded to decode M.
B12X_STANDALONE_MXFP4="${B12X_STANDALONE_MXFP4-}"
B12X_W4A16_FORCE_TILE_CONFIG="${B12X_W4A16_FORCE_TILE_CONFIG-}"
B12X_W4A16_FORCE_BLOCKS_PER_SM="${B12X_W4A16_FORCE_BLOCKS_PER_SM-}"
B12X_W4A16_FORCE_BLOCKS_MAX_M="${B12X_W4A16_FORCE_BLOCKS_MAX_M-36}"
LINEAR_BACKEND="${LINEAR_BACKEND:-auto}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-auto}"
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-1}"
NCCL_IGNORE_CPU_AFFINITY="${NCCL_IGNORE_CPU_AFFINITY:-0}"
# Optional server-wide DeepSeek-V4 prompt mode. Keep empty for the checkpoint's
# native vLLM default; aligned launchers set the exact JSON used by their
# comparison baseline.
DEFAULT_CHAT_TEMPLATE_KWARGS="${DEFAULT_CHAT_TEMPLATE_KWARGS:-}"
# Full decode verifies one target token plus the DSpark block per sequence.
# Match the validated C6 deployment profile: 6 * (5 + 1) = 36 tokens.
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-$((MAX_NUM_SEQS * (MTP_NUM_TOKENS + 1)))}"
# The validated serving profile needs regular full decode graphs through C6.
# Mixed/prefill graphs above eight tokens currently trip an SM121 kernel fault,
# so keep those paths eager instead of using FULL_AND_PIECEWISE.
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_DECODE_ONLY}"
# Keep the capture set used by the stable DSpark deployment. The V2 runner
# rounds these to uniform target-verification batches of 6, 24, 30, and 36.
CUDAGRAPH_CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES:-5,6,24,25,30,36}"
# The complete SM121 forward is not graph-safe at the C2 target shape even
# though its individual kernels are. Execute that exact token count eagerly
# instead of padding it to the next (24-token) graph. Other sizes retain CUDA
# graph acceleration.
# Use `${VAR-default}` so an aligned control lane can explicitly request no
# eager-only sizes by setting the variable to an empty string.
CUDAGRAPH_EAGER_SIZES="${CUDAGRAPH_EAGER_SIZES-12}"
# A CUDA allocation on DGX Spark consumes the same physical DRAM as the host.
# Keep a hard cgroup ceiling below total RAM so a bad warmup is killed before
# it can make either node unresponsive.  Do not allow the container to spill
# into swap: swapping CUDA-backed pages makes recovery substantially worse.
CONTAINER_MEMORY_LIMIT="${CONTAINER_MEMORY_LIMIT:-108g}"
CONTAINER_MEMORY_SWAP_LIMIT="${CONTAINER_MEMORY_SWAP_LIMIT:-108g}"
MIN_START_AVAILABLE_GIB="${MIN_START_AVAILABLE_GIB:-110}"
MIN_RUNTIME_AVAILABLE_GIB="${MIN_RUNTIME_AVAILABLE_GIB:-10}"
MAX_RUNTIME_SWAP_GROWTH_MIB="${MAX_RUNTIME_SWAP_GROWTH_MIB:-512}"
STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-1800}"
ENABLE_FLASHINFER_AUTOTUNE="${ENABLE_FLASHINFER_AUTOTUNE:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-1}"
# Optional bounded torch-profiler output. Keeping this empty has no runtime cost.
# The profiler itself is enabled with /start_profile and should only be used for
# a few decode iterations because traces can grow quickly.
PROFILE_DIR="${PROFILE_DIR:-}"
# Optional Nsight Systems application trace. The host installation is mounted
# read-only; collection stays delayed until nsys-capture.sh starts both ranks.
NSYS_OUTPUT_DIR="${NSYS_OUTPUT_DIR:-}"
NSYS_HOST_ROOT="${NSYS_HOST_ROOT:-/opt/nvidia/nsight-systems/2025.3.2}"
NSYS_SESSION_PREFIX="${NSYS_SESSION_PREFIX:-dsv4}"
NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx,osrt,cublas,cudnn}"
NSYS_CUDA_GRAPH_TRACE="${NSYS_CUDA_GRAPH_TRACE:-node}"
NSYS_OSRT_THRESHOLD_NS="${NSYS_OSRT_THRESHOLD_NS:-10000}"

LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
FLASHINFER_CACHE_DIR="${FLASHINFER_CACHE_DIR:-$ROOT_DIR/.cache/flashinfer}"
VLLM_CACHE_DIR="${VLLM_CACHE_DIR:-$ROOT_DIR/.cache/vllm}"
HEAD_CONTAINER="dsv4-nvfp4-ds-mla-vllm027-tp2-head"
WORKER_CONTAINER="dsv4-nvfp4-ds-mla-vllm027-tp2-worker"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10)

die() {
  echo "ERROR: $*" >&2
  exit 1
}

run_on_worker() {
  local command
  printf -v command '%q ' "$@"
  "${SSH[@]}" "$WORKER_HOST" "$command"
}

available_memory_kib() {
  awk '/^MemAvailable:/ {print $2}' /proc/meminfo
}

worker_available_memory_kib() {
  run_on_worker cat /proc/meminfo | awk '/^MemAvailable:/ {print $2}'
}

swap_free_kib() {
  awk '/^SwapFree:/ {print $2}' /proc/meminfo
}

worker_swap_free_kib() {
  run_on_worker cat /proc/meminfo | awk '/^SwapFree:/ {print $2}'
}

gib_to_kib() {
  echo "$(( $1 * 1024 * 1024 ))"
}

detect_rocev2_gid() {
  local hca="$1" nic="$2" f idx type ndev gid
  for f in "/sys/class/infiniband/${hca}/ports/1/gid_attrs/types/"*; do
    idx="${f##*/}"
    type="$(cat "$f" 2>/dev/null || true)"
    ndev="$(cat "/sys/class/infiniband/${hca}/ports/1/gid_attrs/ndevs/${idx}" 2>/dev/null || true)"
    gid="$(cat "/sys/class/infiniband/${hca}/ports/1/gids/${idx}" 2>/dev/null || true)"
    if [[ "$type" == "RoCE v2" && "$ndev" == "$nic" && "$gid" == *":ffff:"* ]]; then
      echo "$idx"
      return 0
    fi
  done
  return 1
}

remote_gid_command="for f in /sys/class/infiniband/${HCA}/ports/1/gid_attrs/types/*; do idx=\${f##*/}; type=\$(cat \"\$f\" 2>/dev/null || true); ndev=\$(cat /sys/class/infiniband/${HCA}/ports/1/gid_attrs/ndevs/\"\$idx\" 2>/dev/null || true); gid=\$(cat /sys/class/infiniband/${HCA}/ports/1/gids/\"\$idx\" 2>/dev/null || true); if [ \"\$type\" = \"RoCE v2\" ] && [ \"\$ndev\" = \"${NIC}\" ] && printf '%s' \"\$gid\" | grep -q ':ffff:'; then echo \"\$idx\"; break; fi; done"

echo "[1/4] Preflight"
command -v docker >/dev/null || die "docker is not installed"
"${SSH[@]}" "$WORKER_HOST" true || die "cannot SSH to $WORKER_HOST"
[[ "$TENSOR_PARALLEL_SIZE" == "2" ]] || die "this repository supports TP=2 only"
[[ "$PIPELINE_PARALLEL_SIZE" == "1" ]] || die "this repository intentionally does not deploy PP=2"
if [[ "$PULL_IMAGE" == "1" ]]; then
  docker pull "$IMAGE"
  run_on_worker docker pull "$IMAGE"
fi
mkdir -p "$FLASHINFER_CACHE_DIR" "$VLLM_CACHE_DIR"
run_on_worker mkdir -p "$FLASHINFER_CACHE_DIR" "$VLLM_CACHE_DIR"
if [[ -n "$PROFILE_DIR" ]]; then
  mkdir -p "$PROFILE_DIR"
  run_on_worker mkdir -p "$PROFILE_DIR"
fi
if [[ -n "$NSYS_OUTPUT_DIR" ]]; then
  [[ "$NSYS_OUTPUT_DIR" == /* ]] || die "NSYS_OUTPUT_DIR must be an absolute path"
  [[ "$NSYS_SESSION_PREFIX" =~ ^[A-Za-z][A-Za-z0-9_-]*$ ]] || die \
    "NSYS_SESSION_PREFIX must start with a letter and contain only letters, digits, _ or -"
  [[ -x "$NSYS_HOST_ROOT/bin/nsys" ]] || die \
    "Nsight Systems CLI is missing: $NSYS_HOST_ROOT/bin/nsys"
  worker_nsys="$(run_on_worker test -x "$NSYS_HOST_ROOT/bin/nsys" && echo yes)"
  [[ "$worker_nsys" == "yes" ]] || die \
    "worker Nsight Systems CLI is missing: $NSYS_HOST_ROOT/bin/nsys"
  mkdir -p "$NSYS_OUTPUT_DIR"
  run_on_worker mkdir -p "$NSYS_OUTPUT_DIR"
fi

head_available_kib="$(available_memory_kib)"
worker_available_kib="$(worker_available_memory_kib)"
head_swap_free_start_kib="$(swap_free_kib)"
worker_swap_free_start_kib="$(worker_swap_free_kib)"
minimum_start_kib="$(gib_to_kib "$MIN_START_AVAILABLE_GIB")"
((head_available_kib >= minimum_start_kib)) || die \
  "head has only $((head_available_kib / 1024 / 1024)) GiB available; need ${MIN_START_AVAILABLE_GIB} GiB"
((worker_available_kib >= minimum_start_kib)) || die \
  "worker has only $((worker_available_kib / 1024 / 1024)) GiB available; need ${MIN_START_AVAILABLE_GIB} GiB"

model_source_path="$MODEL_REPO/$MODEL_SUBPATH"
[[ -f "$model_source_path/config.json" ]] || die "model config is missing on head: $model_source_path"
worker_config="$(run_on_worker test -f "$model_source_path/config.json" && echo yes)"
[[ "$worker_config" == "yes" ]] || die "model config is missing on worker: $model_source_path"
head_shards="$(find "$model_source_path" -maxdepth 1 -name 'model*.safetensors' | wc -l)"
worker_shards="$(run_on_worker find "$model_source_path" -maxdepth 1 -name 'model*.safetensors' | wc -l)"
[[ "$head_shards" == "$EXPECTED_MODEL_WEIGHT_FILES" ]] || die \
  "head has ${head_shards}/${EXPECTED_MODEL_WEIGHT_FILES} model weight files"
[[ "$worker_shards" == "$EXPECTED_MODEL_WEIGHT_FILES" ]] || die \
  "worker has ${worker_shards}/${EXPECTED_MODEL_WEIGHT_FILES} model weight files"

case "$VERIFY_SOURCE_LABELS" in
  1)
    vllm_label="$(docker image inspect --format '{{index .Config.Labels "io.deepseek-v4-dgx-spark.vllm-source-commit"}}' "$IMAGE")"
    worker_vllm_label="$(run_on_worker docker image inspect --format '{{index .Config.Labels "io.deepseek-v4-dgx-spark.vllm-source-commit"}}' "$IMAGE")"
    [[ "$vllm_label" == "$worker_vllm_label" ]] || die "vLLM source labels differ"
    [[ "$vllm_label" == "$VLLM_REF" ]] || die "image vLLM source is $vllm_label, expected $VLLM_REF"
    flashinfer_label="$(docker image inspect --format '{{index .Config.Labels "io.deepseek-v4-dgx-spark.flashinfer-source-commit"}}' "$IMAGE")"
    worker_flashinfer_label="$(run_on_worker docker image inspect --format '{{index .Config.Labels "io.deepseek-v4-dgx-spark.flashinfer-source-commit"}}' "$IMAGE")"
    [[ "$flashinfer_label" == "$worker_flashinfer_label" ]] || die "FlashInfer source labels differ"
    [[ "$flashinfer_label" == "$FLASHINFER_REF" ]] || die "image FlashInfer source is $flashinfer_label, expected $FLASHINFER_REF"
    ;;
  0)
    head_image_id="$(docker image inspect --format '{{.Id}}' "$IMAGE")"
    worker_image_id="$(run_on_worker docker image inspect --format '{{.Id}}' "$IMAGE")"
    [[ "$head_image_id" == "$worker_image_id" ]] || die \
      "image IDs differ: head=$head_image_id worker=$worker_image_id"
    vllm_label="unverified-stock-image"
    flashinfer_label="unverified-stock-image"
    echo "  source-label verification disabled; identical image ID=$head_image_id"
    ;;
  *) die "VERIFY_SOURCE_LABELS must be 0 or 1" ;;
esac

head_gid="$(detect_rocev2_gid "$HCA" "$NIC")" || die "cannot find head IPv4 RoCEv2 GID"
worker_gid="$("${SSH[@]}" "$WORKER_HOST" "$remote_gid_command" 2>/dev/null)"
[[ -n "$worker_gid" ]] || die "cannot find worker IPv4 RoCEv2 GID"
[[ "$(cat "/sys/class/infiniband/${HCA}/ports/1/state")" == "4: ACTIVE" ]] || die "head RDMA link is not ACTIVE"
worker_state="$(run_on_worker cat "/sys/class/infiniband/${HCA}/ports/1/state")"
[[ "$worker_state" == "4: ACTIVE" ]] || die "worker RDMA link is not ACTIVE"

docker container inspect "$HEAD_CONTAINER" >/dev/null 2>&1 && die "$HEAD_CONTAINER already exists"
run_on_worker docker container inspect "$WORKER_CONTAINER" >/dev/null 2>&1 && die "$WORKER_CONTAINER already exists"

echo "  model=$MODEL_REV shards=$head_shards/$worker_shards"
echo "  vllm_source=$vllm_label"
echo "  flashinfer_source=$flashinfer_label"
echo "  RoCEv2 GID head=$head_gid worker=$worker_gid"
echo "  memory budget: measured runtime=89.2 GiB/node before KV, KV=${KV_CACHE_MEMORY_BYTES:-auto} bytes, container<=${CONTAINER_MEMORY_LIMIT}, host reserve>=${MIN_RUNTIME_AVAILABLE_GIB} GiB, swap growth<=${MAX_RUNTIME_SWAP_GROWTH_MIB} MiB"
echo "  bootstrap: tp=${TENSOR_PARALLEL_SIZE}, pp=${PIPELINE_PARALLEL_SIZE}, max_model_len=${MAX_MODEL_LEN}, gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}, kv_cache_dtype=${KV_CACHE_DTYPE}, max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}, max_cudagraph_capture_size=${MAX_CUDAGRAPH_CAPTURE_SIZE}, cudagraph_mode=${CUDAGRAPH_MODE}, cudagraph_capture_sizes=${CUDAGRAPH_CAPTURE_SIZES}, cudagraph_eager_sizes=${CUDAGRAPH_EAGER_SIZES}, autotune=${ENABLE_FLASHINFER_AUTOTUNE}, enforce_eager=${ENFORCE_EAGER}, deep_gemm=${VLLM_USE_DEEP_GEMM}, target_w4a16=${B12X_NVFP4_W4A16}, standalone_mxfp4=${B12X_STANDALONE_MXFP4:-0}, b12x_tile=${B12X_W4A16_FORCE_TILE_CONFIG:-auto}, b12x_tile_max_m=${B12X_W4A16_FORCE_BLOCKS_MAX_M:-default}, target_moe_backend=${TARGET_MOE_BACKEND}, draft_moe_backend=${DRAFT_MOE_BACKEND}, linear_backend=${LINEAR_BACKEND}, attention_backend=${ATTENTION_BACKEND}, default_chat_template_kwargs=${DEFAULT_CHAT_TEMPLATE_KWARGS:-native}"

common_vllm_args=(
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --pipeline-parallel-size "$PIPELINE_PARALLEL_SIZE"
  --distributed-executor-backend mp
  --nnodes 2
  --master-addr "$HEAD_IP"
  --master-port "$MASTER_PORT"
  --served-model-name "$SERVED_MODEL"
  --tokenizer-mode deepseek_v4
  --reasoning-parser deepseek_v4
  --kv-cache-dtype "$KV_CACHE_DTYPE"
  --block-size 256
  --trust-remote-code
  --max-model-len "$MAX_MODEL_LEN"
  # vLLM still uses this for its startup free-memory sanity check even when a
  # fixed KV cache size is supplied; fixed KV sizing only bypasses its use for
  # calculating the cache allocation.
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --max-cudagraph-capture-size "$MAX_CUDAGRAPH_CAPTURE_SIZE"
  --enable-prefix-caching
  --enable-chunked-prefill
  --async-scheduling
  --moe-backend "$TARGET_MOE_BACKEND"
  --generation-config vllm
  --distributed-timeout-seconds 1800
)

compilation_config="{\"cudagraph_mode\":\"${CUDAGRAPH_MODE}\",\"cudagraph_capture_sizes\":[${CUDAGRAPH_CAPTURE_SIZES}]"
if [[ -n "$CUDAGRAPH_EAGER_SIZES" ]]; then
  compilation_config+=",\"cudagraph_eager_sizes\":[${CUDAGRAPH_EAGER_SIZES}]"
fi
compilation_config+="}"
common_vllm_args+=(--compilation-config "$compilation_config")

if [[ "$DISABLE_CUSTOM_ALL_REDUCE" == "1" ]]; then
  common_vllm_args+=(--disable-custom-all-reduce)
fi

if [[ "$LINEAR_BACKEND" != "auto" ]]; then
  common_vllm_args+=(--linear-backend "$LINEAR_BACKEND")
fi

if [[ "$ATTENTION_BACKEND" != "auto" ]]; then
  common_vllm_args+=(--attention-backend "$ATTENTION_BACKEND")
fi

if [[ -n "$DEFAULT_CHAT_TEMPLATE_KWARGS" ]]; then
  common_vllm_args+=(
    --default-chat-template-kwargs "$DEFAULT_CHAT_TEMPLATE_KWARGS"
  )
fi

if ((MTP_NUM_TOKENS > 0)); then
  common_vllm_args+=(
    --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":${MTP_NUM_TOKENS},\"draft_sample_method\":\"probabilistic\",\"moe_backend\":\"${DRAFT_MOE_BACKEND}\"}"
  )
fi

if [[ -n "$PROFILE_DIR" ]]; then
  common_vllm_args+=(
    --profiler-config '{"profiler":"torch","torch_profiler_dir":"/profiles","torch_profiler_with_stack":false,"torch_profiler_record_shapes":false,"torch_profiler_with_memory":false,"ignore_frontend":true,"max_iterations":8}'
  )
fi

if [[ -n "$KV_CACHE_MEMORY_BYTES" ]]; then
  common_vllm_args+=(--kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES")
fi

if [[ "$ENABLE_FLASHINFER_AUTOTUNE" == "1" ]]; then
  common_vllm_args+=(--enable-flashinfer-autotune)
else
  common_vllm_args+=(--no-enable-flashinfer-autotune)
fi

if [[ "$ENFORCE_EAGER" == "1" ]]; then
  common_vllm_args+=(--enforce-eager)
fi

common_docker_args=(
  --gpus all
  --memory "$CONTAINER_MEMORY_LIMIT"
  --memory-swap "$CONTAINER_MEMORY_SWAP_LIMIT"
  --network host
  --ipc host
  --shm-size 64g
  --cap-add IPC_LOCK
  --ulimit memlock=-1:-1
  --ulimit stack=67108864:67108864
  --device /dev/infiniband:/dev/infiniband
  --mount "type=bind,src=${MODEL_REPO},dst=/model,readonly"
  --mount "type=bind,src=${FLASHINFER_CACHE_DIR},dst=/root/.cache/flashinfer"
  --mount "type=bind,src=${VLLM_CACHE_DIR},dst=/root/.cache/vllm"
  --env HF_HUB_OFFLINE=1
  --env TRANSFORMERS_OFFLINE=1
  --env FLASHINFER_DISABLE_VERSION_CHECK=1
  --env TORCH_CUDA_ARCH_LIST=12.1a
  --env FLASHINFER_CUDA_ARCH_LIST=12.1a
  --env CUTE_DSL_ARCH=sm_121a
  --env VLLM_USE_FLASHINFER_SAMPLER=1
  --env "VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM}"
  --env VLLM_MOE_USE_DEEP_GEMM=0
  --env DG_JIT_USE_NVRTC=0
  --env DG_JIT_NVCC_COMPILER=/usr/local/cuda/bin/nvcc
  --env VLLM_USE_BREAKABLE_CUDAGRAPH=0
  --env VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
  --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
  --env VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  --env "GLOO_SOCKET_IFNAME=${NIC}"
  --env "TP_SOCKET_IFNAME=${NIC}"
  --env "NCCL_SOCKET_IFNAME=${NIC}"
  --env "NCCL_IB_HCA=${HCA}"
  --env NCCL_IB_DISABLE=0
  --env NCCL_NET=IB
  --env NCCL_IB_ADDR_FAMILY=AF_INET
  --env NCCL_IB_ROCE_VERSION_NUM=2
  --env NCCL_CROSS_NIC=1
  --env NCCL_CUMEM_ENABLE=0
  --env NCCL_NVLS_ENABLE=0
  --env "NCCL_IGNORE_CPU_AFFINITY=${NCCL_IGNORE_CPU_AFFINITY}"
  --env NCCL_DEBUG=WARN
  --env TORCH_NCCL_ASYNC_ERROR_HANDLING=1
)

if [[ -n "$B12X_NVFP4_W4A16" ]]; then
  common_docker_args+=(--env "VLLM_B12X_NVFP4_W4A16=${B12X_NVFP4_W4A16}")
fi

if [[ -n "$B12X_STANDALONE_MXFP4" ]]; then
  common_docker_args+=(
    --env "VLLM_B12X_STANDALONE_MXFP4=${B12X_STANDALONE_MXFP4}"
  )
fi

if [[ -n "$B12X_W4A16_FORCE_TILE_CONFIG" ]]; then
  common_docker_args+=(
    --env "VLLM_B12X_W4A16_FORCE_TILE_CONFIG=${B12X_W4A16_FORCE_TILE_CONFIG}"
  )
fi

if [[ -n "$B12X_W4A16_FORCE_BLOCKS_PER_SM" ]]; then
  common_docker_args+=(
    --env "VLLM_B12X_W4A16_FORCE_BLOCKS_PER_SM=${B12X_W4A16_FORCE_BLOCKS_PER_SM}"
  )
fi

if [[ -n "$B12X_W4A16_FORCE_BLOCKS_MAX_M" ]]; then
  common_docker_args+=(
    --env "VLLM_B12X_W4A16_FORCE_BLOCKS_MAX_M=${B12X_W4A16_FORCE_BLOCKS_MAX_M}"
  )
fi

if [[ -n "$PROFILE_DIR" ]]; then
  common_docker_args+=(--mount "type=bind,src=${PROFILE_DIR},dst=/profiles")
fi

container_entrypoint=(--entrypoint vllm)
worker_command=(serve "$MODEL_PATH")
head_command=(serve "$MODEL_PATH")
if [[ -n "$NSYS_OUTPUT_DIR" ]]; then
  common_docker_args+=(
    --mount "type=bind,src=${NSYS_HOST_ROOT},dst=/opt/nsys,readonly"
    --mount "type=bind,src=${NSYS_OUTPUT_DIR},dst=/nsys-output"
  )
  container_entrypoint=(--entrypoint /opt/nsys/bin/nsys)
  nsys_common_args=(
    profile
    --start-later=true
    --trace="$NSYS_TRACE"
    --sample=none
    --cpuctxsw=process-tree
    --cuda-graph-trace="$NSYS_CUDA_GRAPH_TRACE"
    --osrt-threshold="$NSYS_OSRT_THRESHOLD_NS"
    --force-overwrite=true
  )
  worker_command=(
    "${nsys_common_args[@]}"
    --session-new="${NSYS_SESSION_PREFIX}rank1"
    --output="/nsys-output/${NSYS_SESSION_PREFIX}-rank1"
    vllm serve "$MODEL_PATH"
  )
  head_command=(
    "${nsys_common_args[@]}"
    --session-new="${NSYS_SESSION_PREFIX}rank0"
    --output="/nsys-output/${NSYS_SESSION_PREFIX}-rank0"
    vllm serve "$MODEL_PATH"
  )
fi

# Invoked indirectly through the EXIT trap below.
# shellcheck disable=SC2317,SC2329
cleanup_on_error() {
  status=$?
  if ((status != 0)); then
    set +e
    mkdir -p "$LOG_DIR"
    failure_stamp="$(date '+%Y%m%dT%H%M%S%z')"
    docker inspect "$HEAD_CONTAINER" >"$LOG_DIR/${failure_stamp}-head-inspect.json" 2>&1
    docker logs "$HEAD_CONTAINER" >"$LOG_DIR/${failure_stamp}-head.log" 2>&1
    run_on_worker docker inspect "$WORKER_CONTAINER" >"$LOG_DIR/${failure_stamp}-worker-inspect.json" 2>&1
    run_on_worker docker logs "$WORKER_CONTAINER" >"$LOG_DIR/${failure_stamp}-worker.log" 2>&1
    docker rm -f "$HEAD_CONTAINER" >/dev/null 2>&1 || true
    run_on_worker docker rm -f "$WORKER_CONTAINER" >/dev/null 2>&1 || true
    echo "  startup diagnostics saved under $LOG_DIR/${failure_stamp}-*" >&2
  fi
  exit "$status"
}
trap cleanup_on_error EXIT

echo "[2/4] Starting worker rank 1"
run_on_worker docker run -d \
  --name "$WORKER_CONTAINER" \
  --label io.deepseek-v4-dgx-spark.role=worker \
  "${common_docker_args[@]}" \
  "${container_entrypoint[@]}" \
  --env "VLLM_HOST_IP=${WORKER_IP}" \
  --env "NCCL_IB_GID_INDEX=${worker_gid}" \
  "$IMAGE" "${worker_command[@]}" \
  "${common_vllm_args[@]}" \
  --node-rank 1 \
  --headless

echo "[3/4] Starting head rank 0"
docker run -d \
  --name "$HEAD_CONTAINER" \
  --label io.deepseek-v4-dgx-spark.role=head \
  "${common_docker_args[@]}" \
  "${container_entrypoint[@]}" \
  --env "VLLM_HOST_IP=${HEAD_IP}" \
  --env "NCCL_IB_GID_INDEX=${head_gid}" \
  "$IMAGE" "${head_command[@]}" \
  "${common_vllm_args[@]}" \
  --node-rank 0 \
  --host 127.0.0.1 \
  --port "$API_PORT"

echo "[4/4] Waiting for health with unified-memory guard"
minimum_runtime_kib="$(gib_to_kib "$MIN_RUNTIME_AVAILABLE_GIB")"
startup_deadline="$((SECONDS + STARTUP_TIMEOUT_SECONDS))"
last_report=0
while ((SECONDS < startup_deadline)); do
  head_available_kib="$(available_memory_kib)"
  worker_available_kib="$(worker_available_memory_kib)"
  head_swap_growth_kib="$((head_swap_free_start_kib - $(swap_free_kib)))"
  worker_swap_growth_kib="$((worker_swap_free_start_kib - $(worker_swap_free_kib)))"
  if ((head_available_kib < minimum_runtime_kib)); then
    die "head MemAvailable fell below ${MIN_RUNTIME_AVAILABLE_GIB} GiB during startup"
  fi
  if ((worker_available_kib < minimum_runtime_kib)); then
    die "worker MemAvailable fell below ${MIN_RUNTIME_AVAILABLE_GIB} GiB during startup"
  fi
  if ((head_swap_growth_kib > MAX_RUNTIME_SWAP_GROWTH_MIB * 1024)); then
    die "head swap grew by $((head_swap_growth_kib / 1024)) MiB during startup"
  fi
  if ((worker_swap_growth_kib > MAX_RUNTIME_SWAP_GROWTH_MIB * 1024)); then
    die "worker swap grew by $((worker_swap_growth_kib / 1024)) MiB during startup"
  fi

  head_running="$(docker inspect --format '{{.State.Running}}' "$HEAD_CONTAINER")"
  worker_running="$(run_on_worker docker inspect --format '{{.State.Running}}' "$WORKER_CONTAINER")"
  [[ "$head_running" == "true" ]] || die "head container exited during startup"
  [[ "$worker_running" == "true" ]] || die "worker container exited during startup"

  if curl --fail --silent --max-time 2 "http://127.0.0.1:${API_PORT}/health" >/dev/null; then
    echo "  healthy; MemAvailable head=$((head_available_kib / 1024 / 1024)) GiB worker=$((worker_available_kib / 1024 / 1024)) GiB"
    echo "  API=http://127.0.0.1:${API_PORT}/v1"
    trap - EXIT
    exit 0
  fi
  if ((SECONDS - last_report >= 30)); then
    echo "  starting; MemAvailable head=$((head_available_kib / 1024 / 1024)) GiB worker=$((worker_available_kib / 1024 / 1024)) GiB"
    last_report="$SECONDS"
  fi
  sleep 2
done

die "startup did not become healthy within ${STARTUP_TIMEOUT_SECONDS} seconds"
