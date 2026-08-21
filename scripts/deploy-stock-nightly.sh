#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_ENV="${DEPLOY_ENV:-$ROOT_DIR/config/deployment.env}"
if [[ -f "$DEPLOY_ENV" ]]; then
  # Export the complete cluster configuration, then prevent deploy-tp2.sh from
  # sourcing it again over the stock-image overrides below.
  set -a
  # shellcheck disable=SC1090
  source "$DEPLOY_ENV"
  set +a
fi

: "${MODEL_REPO:?set MODEL_REPO to the deepseek-ai/DeepSeek-V4-Flash-0731 cache root on both nodes}"

stock_mtp_num_tokens="${STOCK_MTP_NUM_TOKENS:-5}"
[[ "$stock_mtp_num_tokens" =~ ^[0-9]+$ ]] || {
  echo "ERROR: STOCK_MTP_NUM_TOKENS must be a non-negative integer" >&2
  exit 1
}
if ((stock_mtp_num_tokens > 0)); then
  stock_cudagraph_capture_sizes="${STOCK_CUDAGRAPH_CAPTURE_SIZES:-5,6,12,24,25,30,36}"
else
  stock_cudagraph_capture_sizes="${STOCK_CUDAGRAPH_CAPTURE_SIZES:-1,2,3,4,5,6}"
fi

# Stock vLLM images do not carry this repository's source-contract labels.
# The common preflight still requires the immutable Docker image ID to be
# identical on both nodes and keeps every memory/RoCE guard enabled.
IMAGE="${STOCK_NIGHTLY_IMAGE:-vllm/vllm-openai:nightly}" \
PULL_IMAGE=0 \
VERIFY_SOURCE_LABELS=0 \
MODEL_REPO="$MODEL_REPO" \
MODEL_REV="${STOCK_MODEL_REV:-9e165c30e2704aec5d9d593cce3eebd58bbef1cb}" \
SERVED_MODEL="${SERVED_MODEL:-deepseek-v4-flash-stock-nightly}" \
EXPECTED_MODEL_WEIGHT_FILES="${EXPECTED_MODEL_WEIGHT_FILES:-48}" \
FLASHINFER_CACHE_DIR="${STOCK_FLASHINFER_CACHE_DIR:-$ROOT_DIR/.cache/flashinfer-stock-nightly}" \
VLLM_CACHE_DIR="${STOCK_VLLM_CACHE_DIR:-$ROOT_DIR/.cache/vllm-stock-nightly}" \
KV_CACHE_DTYPE=fp8_ds_mla \
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-6442450944}" \
TARGET_MOE_BACKEND="${STOCK_MOE_BACKEND:-auto}" \
DRAFT_MOE_BACKEND="${STOCK_MOE_BACKEND:-auto}" \
MTP_NUM_TOKENS="$stock_mtp_num_tokens" \
DEFAULT_CHAT_TEMPLATE_KWARGS='{"thinking":true,"reasoning_effort":"low"}' \
B12X_NVFP4_W4A16='' \
B12X_STANDALONE_MXFP4='' \
B12X_W4A16_FORCE_TILE_CONFIG='' \
B12X_W4A16_FORCE_BLOCKS_PER_SM='' \
B12X_W4A16_FORCE_BLOCKS_MAX_M='' \
ENABLE_FLASHINFER_AUTOTUNE=1 \
CUDAGRAPH_CAPTURE_SIZES="$stock_cudagraph_capture_sizes" \
CUDAGRAPH_EAGER_SIZES='' \
MAX_NUM_SEQS=6 \
NSYS_OUTPUT_DIR='' \
DEPLOY_ENV=/dev/null \
"$ROOT_DIR/scripts/deploy-tp2.sh"
