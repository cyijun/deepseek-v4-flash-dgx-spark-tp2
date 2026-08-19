#!/usr/bin/env bash
set -Eeuo pipefail

if (($# != 3)); then
  echo "usage: $0 VLLM_SOURCE FLASHINFER_SOURCE OUTPUT_DIR" >&2
  exit 2
fi

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_SOURCE="$(realpath "$1")"
FLASHINFER_SOURCE="$(realpath "$2")"
OUTPUT_DIR="$3"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -f "$VLLM_SOURCE/vllm/config/vllm.py" ]] || die "invalid vLLM checkout: $VLLM_SOURCE"
[[ -f "$FLASHINFER_SOURCE/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_activation.py" ]] || \
  die "invalid FlashInfer checkout: $FLASHINFER_SOURCE"
command -v rsync >/dev/null || die "rsync is required"

if [[ -e "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  die "output directory must be absent or empty: $OUTPUT_DIR"
fi

mkdir -p \
  "$OUTPUT_DIR/vllm" \
  "$OUTPUT_DIR/flashinfer" \
  "$OUTPUT_DIR/flashinfer_data/csrc" \
  "$OUTPUT_DIR/flashinfer_data/include/flashinfer/attention/sparse_mla_sm120"

cp "$ROOT_DIR/container/Dockerfile" "$OUTPUT_DIR/Dockerfile"
cp "$ROOT_DIR/container/.dockerignore" "$OUTPUT_DIR/.dockerignore"
rsync -a --exclude '__pycache__/' --exclude '*.pyc' "$VLLM_SOURCE/vllm/" "$OUTPUT_DIR/vllm/"
rsync -a --exclude '__pycache__/' --exclude '*.pyc' "$FLASHINFER_SOURCE/flashinfer/" "$OUTPUT_DIR/flashinfer/"
cp \
  "$FLASHINFER_SOURCE/csrc/sparse_mla_sm120_decode_dsv4.cu" \
  "$FLASHINFER_SOURCE/csrc/sparse_mla_sm120_jit_binding.cu" \
  "$FLASHINFER_SOURCE/csrc/sparse_mla_sm120_prefill.cu" \
  "$OUTPUT_DIR/flashinfer_data/csrc/"
cp \
  "$FLASHINFER_SOURCE/include/flashinfer/attention/sparse_mla_sm120/prefill_kernel.cuh" \
  "$OUTPUT_DIR/flashinfer_data/include/flashinfer/attention/sparse_mla_sm120/"

echo "prepared build context: $OUTPUT_DIR"
