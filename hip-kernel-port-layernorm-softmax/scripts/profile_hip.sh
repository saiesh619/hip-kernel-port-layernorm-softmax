#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

if [[ ! -f "${BUILD_DIR}/layernorm_hip" ]]; then
  echo "HIP binaries not found. Run scripts/build_hip.sh first."
  exit 1
fi

echo "[PROFILE] rocprof placeholder for HIP LayerNorm..."
# Example (uncomment and adjust when rocprof is available):
# rocprof --stats --hip-trace \
#   --hsa-trace \
#   --out "${BUILD_DIR}/layernorm_hip_profile.json" \
#   "${BUILD_DIR}/layernorm_hip"

echo "[PROFILE] rocprof placeholder for HIP Softmax..."
# rocprof --stats --hip-trace \
#   --hsa-trace \
#   --out "${BUILD_DIR}/softmax_hip_profile.json" \
#   "${BUILD_DIR}/softmax_hip"

echo "Profiling commands are placeholders. Update with your local ROCm setup."
