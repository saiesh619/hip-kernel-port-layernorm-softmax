#include <hip/hip_runtime.h>
#include <cstdio>
#include "layernorm.h"

// HIP LayerNorm kernel mirroring the CUDA version.
__global__ void layernorm_kernel_hip(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     std::size_t size,
                                     float mean,
                                     float inv_std) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float norm = (x - mean) * inv_std;
        output[idx] = norm;
    }
}

void launch_layernorm_hip(const float* d_input,
                          float* d_output,
                          std::size_t size,
                          float mean,
                          float inv_std) {
    const int blockSize = 256;
    int gridSize = static_cast<int>((size + blockSize - 1) / blockSize);

    hipLaunchKernelGGL(layernorm_kernel_hip,
                       dim3(gridSize),
                       dim3(blockSize),
                       0,
                       0,
                       d_input,
                       d_output,
                       size,
                       mean,
                       inv_std);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::fprintf(stderr, "[HIP] LayerNorm launch failed: %s\n",
                     hipGetErrorString(err));
    }
}
