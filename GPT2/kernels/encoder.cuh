#ifndef ENCODER_FORWARD_KERNEL_CUH
#define ENCODER_FORWARD_KERNEL_CUH

#include <cuda_runtime.h>
#include "../utils/cuda_utils.cuh"

__global__ void encoder_forward_kernel(float* out, const int* inp, const float* wte, const float* wpe,
                                       int B, int T, int C) {
    // Implement this
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * T * C;
    if (idx < N) {
        int b = idx / (T * C);
        int t = (idx / C) % T;
        int c = idx % C;
        int token_id = inp[b * T + t];
        float token_emb = wte[token_id * C + c];
        float pos_emb = wpe[t * C + c]; 
        out[idx] = token_emb + pos_emb;
    }
}

// Launch kernel here
void encoder_forward(float* out, const int* inp, const float* wte, const float* wpe, int B, int T, int C) {
    // Implement this
    int N = B * T * C;
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    encoder_forward_kernel<<<gridSize, blockSize>>>(out, inp, wte, wpe, B, T, C);
    cudaDeviceSynchronize();
}


#endif // ENCODER_FORWARD_KERNEL_CUH