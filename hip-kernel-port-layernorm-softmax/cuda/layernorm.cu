#include <cuda_runtime.h>
#include <cstdio>
#include "layernorm.h"

// Very simple LayerNorm-style kernel over a 1D vector.
// Assumes mean and inv_std (1 / sqrt(var + eps)) are precomputed on CPU.
__global__ void layernorm_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 std::size_t size,
                                 float mean,
                                 float inv_std) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float norm = (x - mean) * inv_std;
        output[idx] = norm; // no gamma/beta for now
    }
}

void launch_layernorm_cuda(const float* d_input,
                           float* d_output,
                           std::size_t size,
                           float mean,
                           float inv_std) {
    const int blockSize = 256;
    int gridSize = static_cast<int>((size + blockSize - 1) / blockSize);

    layernorm_kernel<<<gridSize, blockSize>>>(d_input, d_output, size, mean, inv_std);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[CUDA] LayerNorm launch failed: %s\n",
                     cudaGetErrorString(err));
    }
}
