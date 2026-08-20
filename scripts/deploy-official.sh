#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_ENV="${DEPLOY_ENV:-$ROOT_DIR/../config/deployment.env}"
if [[ -f "$DEPLOY_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$DEPLOY_ENV"
fi
: "${MODEL_REPO:?set MODEL_REPO to the deepseek-ai/DeepSeek-V4-Flash-0731 cache root on both nodes}"
DEFAULT_CHAT_TEMPLATE_KWARGS="${DEFAULT_CHAT_TEMPLATE_KWARGS:-}"
if [[ -z "$DEFAULT_CHAT_TEMPLATE_KWARGS" ]]; then
  DEFAULT_CHAT_TEMPLATE_KWARGS='{"thinking":true,"reasoning_effort":"low"}'
fi

# The official checkpoint uses FP8 dense/linear weights and native MXFP4
# routed experts. Both target and MTP experts are forced through B12X W4A16,
# matching Anemll's expert-compute choice. Explicit fp8_ds_mla avoids Anemll
# 0.1.1's misleading nvfp4_ds_mla alias for the same physical FP8 layout.
MODEL_REPO="$MODEL_REPO" \
MODEL_REV="${MODEL_REV:-9e165c30e2704aec5d9d593cce3eebd58bbef1cb}" \
SERVED_MODEL="${SERVED_MODEL:-deepseek-v4-flash-0731-vllm027}" \
EXPECTED_MODEL_WEIGHT_FILES="${EXPECTED_MODEL_WEIGHT_FILES:-48}" \
FLASHINFER_CACHE_DIR="${FLASHINFER_CACHE_DIR:-$ROOT_DIR/../.cache/flashinfer-official}" \
VLLM_CACHE_DIR="${VLLM_CACHE_DIR:-$ROOT_DIR/../.cache/vllm-official}" \
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_ds_mla}" \
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-6442450944}" \
TARGET_MOE_BACKEND="${TARGET_MOE_BACKEND:-flashinfer_b12x}" \
DRAFT_MOE_BACKEND="${DRAFT_MOE_BACKEND:-flashinfer_b12x}" \
ATTENTION_BACKEND="${ATTENTION_BACKEND:-auto}" \
DEFAULT_CHAT_TEMPLATE_KWARGS="$DEFAULT_CHAT_TEMPLATE_KWARGS" \
B12X_NVFP4_W4A16=0 \
B12X_STANDALONE_MXFP4=1 \
B12X_W4A16_FORCE_TILE_CONFIG="${B12X_W4A16_FORCE_TILE_CONFIG:-128,64,128}" \
B12X_W4A16_FORCE_BLOCKS_MAX_M="${B12X_W4A16_FORCE_BLOCKS_MAX_M:-36}" \
ENABLE_FLASHINFER_AUTOTUNE="${ENABLE_FLASHINFER_AUTOTUNE:-1}" \
CUDAGRAPH_CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES:-5,6,12,24,25,30,36}" \
CUDAGRAPH_EAGER_SIZES="${CUDAGRAPH_EAGER_SIZES-}" \
MAX_NUM_SEQS="${MAX_NUM_SEQS:-6}" \
"$ROOT_DIR/deploy-tp2.sh"
