#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

mkdir -p "${BUILD_DIR}"

echo "[BUILD] Compiling HIP LayerNorm..."
hipcc -O2 -std=c++17 \
    -I"${ROOT_DIR}/include" \
    "${ROOT_DIR}/hip/layernorm.hip.cpp" \
    -o "${BUILD_DIR}/layernorm_hip"

echo "[BUILD] Compiling HIP Softmax..."
hipcc -O2 -std=c++17 \
    -I"${ROOT_DIR}/include" \
    "${ROOT_DIR}/hip/softmax.hip.cpp" \
    -o "${BUILD_DIR}/softmax_hip"

echo "[BUILD] HIP binaries built in ${BUILD_DIR}"
