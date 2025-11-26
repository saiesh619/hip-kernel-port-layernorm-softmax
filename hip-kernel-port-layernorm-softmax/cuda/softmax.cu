#include <cuda_runtime.h>
#include <cstdio>
#include "softmax.h"

// Naive 1D softmax over `size` elements.
// For simplicity, a single-block implementation; good enough for a demo.
__global__ void softmax_kernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               std::size_t size) {
    extern __shared__ float shm[];

    // Step 1: find max
    float local_max = -1e30f;
    for (std::size_t i = threadIdx.x; i < size; i += blockDim.x) {
        local_max = fmaxf(local_max, input[i]);
    }

    // Reduce max across block
    shm[threadIdx.x] = local_max;
    __syncthreads();

    // Simple parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shm[threadIdx.x] = fmaxf(shm[threadIdx.x], shm[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    float max_val = shm[0];

    // Step 2: compute exp(x - max)
    float local_sum = 0.0f;
    for (std::size_t i = threadIdx.x; i < size; i += blockDim.x) {
        float e = expf(input[i] - max_val);
        output[i] = e;
        local_sum += e;
    }

    // Reduce sum across block
    shm[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shm[threadIdx.x] += shm[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float sum = shm[0];

    // Step 3: normalize
    for (std::size_t i = threadIdx.x; i < size; i += blockDim.x) {
        output[i] = output[i] / sum;
    }
}

void launch_softmax_cuda(const float* d_input,
                         float* d_output,
                         std::size_t size) {
    const int blockSize = 256;
    const int gridSize = 1; // naive single-block implementation for demo
    const std::size_t shmSize = blockSize * sizeof(float);

    softmax_kernel<<<gridSize, blockSize, shmSize>>>(d_input, d_output, size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[CUDA] Softmax launch failed: %s\n",
                     cudaGetErrorString(err));
    }
}
