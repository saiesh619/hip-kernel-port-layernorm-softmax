# Findings & Notes

This document will track:

## 1. Correctness

- [ ] Compare CUDA vs HIP outputs for LayerNorm
- [ ] Compare CUDA vs HIP outputs for Softmax
- [ ] Document any numerical drift and its causes (precision, order of ops, etc.)

## 2. Performance

- [ ] Record latency and throughput for different input sizes
- [ ] Note differences between NVIDIA and AMD hardware
- [ ] Capture profiling snapshots (Nsight / rocprof) and summarize bottlenecks

## 3. Implementation Notes

- [ ] Alignment and memory access patterns
- [ ] Block and grid configuration choices
- [ ] Any ROCm-specific behavior or workarounds
