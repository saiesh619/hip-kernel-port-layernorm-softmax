#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

if [[ ! -f "${BUILD_DIR}/layernorm_cuda" ]]; then
  echo "CUDA binaries not found. Run scripts/build_cuda.sh first."
  exit 1
fi

echo "[PROFILE] Nsight Systems placeholder for CUDA LayerNorm..."
# Example (uncomment and adjust when Nsight is available):
# nsys profile -o "${BUILD_DIR}/layernorm_cuda_profile" "${BUILD_DIR}/layernorm_cuda"

echo "[PROFILE] Nsight Systems placeholder for CUDA Softmax..."
# nsys profile -o "${BUILD_DIR}/softmax_cuda_profile" "${BUILD_DIR}/softmax_cuda"

echo "Profiling commands are placeholders. Update with your local Nsight setup."
