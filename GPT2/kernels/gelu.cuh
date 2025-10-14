#ifndef GELU_KERNEL_CUH_
#define GELU_KERNEL_CUH_

#include <cuda_runtime.h>
#include "../utils/cuda_utils.cuh"

__global__ void gelu_forward_kernel(float* out, const float* inp, int N) {
    // Implement this
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {

        float x = inp[idx];

        float x3 = x * x * x;
        float inner = 0.7978845608f * (x + 0.044715f * x3);
        float tanh_inner = tanhf(inner);

        out[idx] = 0.5f * x * (1.0f + tanh_inner);

    }

}

// Launch kernel here
void gelu_forward(float* out, const float* inp, int N) {
    // Implement this
    int blockSize = 256;
    int gridSize = (N+blockSize-1)/blockSize;

    gelu_forward_kernel<<<gridSize, blockSize>>>(out, inp, N);
    cudaDeviceSynchronize();
    
}

#endif // GELU_KERNEL_CUH_