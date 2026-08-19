#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_ENV="${DEPLOY_ENV:-$ROOT_DIR/config/deployment.env}"
if [[ -f "$DEPLOY_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$DEPLOY_ENV"
fi
: "${MODEL_REPO:?set MODEL_REPO to the NVFP4 checkpoint cache root on both nodes}"
: "${MODEL_REV:?set MODEL_REV to the NVFP4 checkpoint revision}"

# NVFP4 checkpoint storage, B12X W4A16 target execution, and a real packed
# NVFP4 DS-MLA KV row. Marlin remains the measured faster draft default for
# this abliterated checkpoint; set DRAFT_MOE_BACKEND=flashinfer_b12x to match
# the official-checkpoint/Anemll expert-compute strategy.
MODEL_REPO="$MODEL_REPO" \
MODEL_REV="$MODEL_REV" \
SERVED_MODEL="${SERVED_MODEL:-deepseek-v4-flash-nvfp4}" \
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-nvfp4_ds_mla}" \
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-6442450944}" \
TARGET_MOE_BACKEND="${TARGET_MOE_BACKEND:-flashinfer_b12x}" \
DRAFT_MOE_BACKEND="${DRAFT_MOE_BACKEND:-marlin}" \
B12X_NVFP4_W4A16="${B12X_NVFP4_W4A16:-1}" \
MAX_NUM_SEQS="${MAX_NUM_SEQS:-6}" \
"$ROOT_DIR/scripts/deploy-tp2.sh"
