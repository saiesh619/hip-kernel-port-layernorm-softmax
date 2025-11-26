# HIP Kernel Port: LayerNorm & Softmax

This repository explores porting CUDA kernels to HIP for AMD GPUs, focusing on
correctness, performance, and cross-platform behavior for LayerNorm and Softmax.

## Layout

- `cuda/` – CUDA implementations of LayerNorm & Softmax
- `hip/` – HIP ports for AMD ROCm
- `include/` – public headers for kernel launchers
- `benchmarks/` – Python benchmarking scripts + results
- `scripts/` – build and profiling helpers
- `docs/` – notes, findings, and design decisions

## Goals

- Establish a clean CUDA baseline implementation
- Port kernels to HIP and validate numerical correctness
- Compare performance on NVIDIA vs AMD GPUs using standard benchmarks
