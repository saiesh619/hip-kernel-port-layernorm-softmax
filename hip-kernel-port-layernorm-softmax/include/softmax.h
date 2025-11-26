#pragma once

#include <cstddef>

// Launches the CUDA Softmax implementation on the current device.
// Computes softmax over a 1D vector of length `size`.
void launch_softmax_cuda(const float* d_input,
                         float* d_output,
                         std::size_t size);

// Launches the HIP Softmax implementation on the current device.
void launch_softmax_hip(const float* d_input,
                        float* d_output,
                        std::size_t size);
