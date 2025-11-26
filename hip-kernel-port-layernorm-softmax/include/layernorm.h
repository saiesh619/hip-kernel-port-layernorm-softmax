#pragma once

#include <cstddef>

// Launches the CUDA LayerNorm implementation on the current device.
// input/output: device pointers, size: number of elements in the vector.
// mean, inv_std: precomputed scalar values on host, passed by value.
void launch_layernorm_cuda(const float* d_input,
                           float* d_output,
                           std::size_t size,
                           float mean,
                           float inv_std);

// Launches the HIP LayerNorm implementation on the current device.
void launch_layernorm_hip(const float* d_input,
                          float* d_output,
                          std::size_t size,
                          float mean,
                          float inv_std);