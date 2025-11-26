#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

mkdir -p "${BUILD_DIR}"

echo "[BUILD] Compiling CUDA LayerNorm..."
nvcc -O2 -std=c++17 \
    -I"${ROOT_DIR}/include" \
    "${ROOT_DIR}/cuda/layernorm.cu" \
    -o "${BUILD_DIR}/layernorm_cuda"

echo "[BUILD] Compiling CUDA Softmax..."
nvcc -O2 -std=c++17 \
    -I"${ROOT_DIR}/include" \
    "${ROOT_DIR}/cuda/softmax.cu" \
    -o "${BUILD_DIR}/softmax_cuda"

echo "[BUILD] CUDA binaries built in ${BUILD_DIR}"
