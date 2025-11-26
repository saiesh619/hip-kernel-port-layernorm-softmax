#include <hip/hip_runtime.h>
#include <cstdio>
#include "softmax.h"

// HIP softmax kernel mirroring the CUDA version.
__global__ void softmax_kernel_hip(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   std::size_t size) {
    extern __shared__ float shm[];

    float local_max = -1e30f;
    for (std::size_t i = threadIdx.x; i < size; i += blockDim.x) {
        local_max = fmaxf(local_max, input[i]);
    }

    shm[threadIdx.x] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shm[threadIdx.x] = fmaxf(shm[threadIdx.x], shm[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    float max_val = shm[0];

    float local_sum = 0.0f;
    for (std::size_t i = threadIdx.x; i < size; i += blockDim.x) {
        float e = expf(input[i] - max_val);
        output[i] = e;
        local_sum += e;
    }

    shm[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shm[threadIdx.x] += shm[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float sum = shm[0];

    for (std::size_t i = threadIdx.x; i < size; i += blockDim.x) {
        output[i] = output[i] / sum;
    }
}

void launch_softmax_hip(const float* d_input,
                        float* d_output,
                        std::size_t size) {
    const int blockSize = 256;
    const int gridSize = 1;
    const std::size_t shmSize = blockSize * sizeof(float);

    hipLaunchKernelGGL(softmax_kernel_hip,
                       dim3(gridSize),
                       dim3(blockSize),
                       shmSize,
                       0,
                       d_input,
                       d_output,
                       size);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::fprintf(stderr, "[HIP] Softmax launch failed: %s\n",
                     hipGetErrorString(err));
    }
}
