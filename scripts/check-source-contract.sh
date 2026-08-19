#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/config/versions.env"

check_ref() {
  local name="$1" repository="$2" ref="$3"
  [[ "$ref" =~ ^[0-9a-f]{40}$ ]] || {
    echo "ERROR: $name ref is not a full commit: $ref" >&2
    return 1
  }
  if ! git ls-remote "$repository" | awk -v expected="$ref" '$1 == expected {found=1} END {exit !found}'; then
    echo "ERROR: $name commit is not advertised by $repository: $ref" >&2
    return 1
  fi
  echo "$name=$ref"
}

check_ref vllm "$VLLM_REPOSITORY" "$VLLM_REF"
check_ref flashinfer "$FLASHINFER_REPOSITORY" "$FLASHINFER_REF"
