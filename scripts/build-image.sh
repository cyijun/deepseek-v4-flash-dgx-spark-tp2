#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
set -a
# shellcheck disable=SC1091
source "$ROOT_DIR/config/versions.env"
set +a

IMAGE="${IMAGE:-ghcr.io/cyijun/deepseek-v4-flash-dgx-spark-tp2:vllm027-sm121}"
VLLM_SOURCE="${VLLM_SOURCE:-}"
FLASHINFER_SOURCE="${FLASHINFER_SOURCE:-}"
PUSH_IMAGE="${PUSH_IMAGE:-0}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

command -v docker >/dev/null || die "docker is required"
command -v git >/dev/null || die "git is required"

task_dir="$(mktemp -d)"
trap 'rm -rf -- "$task_dir"' EXIT

checkout_commit() {
  local repository="$1" ref="$2" destination="$3"
  git init -q "$destination"
  git -C "$destination" remote add origin "$repository"
  git -C "$destination" fetch -q --depth=1 origin "$ref"
  git -C "$destination" checkout -q --detach FETCH_HEAD
}

if [[ -z "$VLLM_SOURCE" ]]; then
  VLLM_SOURCE="$task_dir/vllm-source"
  checkout_commit "$VLLM_REPOSITORY" "$VLLM_REF" "$VLLM_SOURCE"
fi
if [[ -z "$FLASHINFER_SOURCE" ]]; then
  FLASHINFER_SOURCE="$task_dir/flashinfer-source"
  checkout_commit "$FLASHINFER_REPOSITORY" "$FLASHINFER_REF" "$FLASHINFER_SOURCE"
fi

vllm_commit="$(git -C "$VLLM_SOURCE" rev-parse HEAD)"
flashinfer_commit="$(git -C "$FLASHINFER_SOURCE" rev-parse HEAD)"
[[ "$vllm_commit" == "$VLLM_REF" ]] || die "vLLM checkout is $vllm_commit, expected $VLLM_REF"
[[ "$flashinfer_commit" == "$FLASHINFER_REF" ]] || die "FlashInfer checkout is $flashinfer_commit, expected $FLASHINFER_REF"

context="$task_dir/context"
"$ROOT_DIR/scripts/prepare-build-context.sh" "$VLLM_SOURCE" "$FLASHINFER_SOURCE" "$context"

docker build \
  --file "$context/Dockerfile" \
  --platform linux/arm64 \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --build-arg "B12X_VERSION=$B12X_VERSION" \
  --build-arg "B12X_SOURCE_IMAGE=$B12X_SOURCE_IMAGE" \
  --build-arg "VLLM_SOURCE_COMMIT=$vllm_commit" \
  --build-arg "FLASHINFER_SOURCE_COMMIT=$flashinfer_commit" \
  --tag "$IMAGE" \
  "$context"

if [[ "$PUSH_IMAGE" == "1" ]]; then
  docker push "$IMAGE"
fi

echo "image=$IMAGE"
echo "vllm_source=$vllm_commit"
echo "flashinfer_source=$flashinfer_commit"
