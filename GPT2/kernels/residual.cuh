#ifndef RESIDUAL_KERNEL_CUH_
#define RESIDUAL_KERNEL_CUH_

#include <cuda_runtime.h>
#include "../utils/cuda_utils.cuh"

__global__ void residual_forward_kernel(float* out, float* inp1, float* inp2, int N) {
    // Implement this
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)  {
        out[idx] = inp1[idx] + inp2[idx];
    }
    

}

// Launch kernel here
void residual_forward(float* out, float* inp1, float* inp2, int N) {
    // Implement this
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    residual_forward_kernel<<<blocks, threads>>>(out, inp1, inp2, N);
    cudaDeviceSynchronize();

}

#endif // RESIDUAL_KERNEL_CUH_