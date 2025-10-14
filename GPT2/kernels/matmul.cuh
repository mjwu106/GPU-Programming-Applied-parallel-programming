#ifndef __MATMUL_KERNEL_CUH__
#define __MATMUL_KERNEL_CUH__

#include <cuda_runtime.h>
#include "../utils/cuda_utils.cuh"

#define TILE_SIZE 32

__global__ void matmul_forward_kernel(float* out, const float* inp, 
                                      const float* weight, const float* bias,
                                      int M, int C, int OC) {
    // NOTE: Weight is stored as [OC, C] (each row is one output dimension)
    
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];  
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE]; 
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;  // output row (M dimension)
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;  // output col (OC dimension)
    
    float sum = 0.0f;
    int numTiles = (C + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; t++) {
        int tileStart = t * TILE_SIZE;
        
        // Load tile from inp [M, C]
        int aCol = tileStart + threadIdx.x;
        if (row < M && aCol < C) {
            tile_A[threadIdx.y][threadIdx.x] = inp[row * C + aCol];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load tile from weight [OC, C]
        int bRow = tileStart + threadIdx.y; 
        if (col < OC && bRow < C) {
            tile_B[threadIdx.y][threadIdx.x] = weight[col * C + bRow];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < OC) {
        if (bias != nullptr) {
            sum += bias[col];
        }
        out[row * OC + col] = sum;
    }
}

void matmul_forward(float* out, const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC) {
    int M = B * T;
    
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((OC + TILE_SIZE - 1) / TILE_SIZE, 
                 (M + TILE_SIZE - 1) / TILE_SIZE);
    
    matmul_forward_kernel<<<gridDim, blockDim>>>(out, inp, weight, bias, M, C, OC);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("MatMul error: %s\n", cudaGetErrorString(err));
    }
}

#endif // __MATMUL_KERNEL_CUH__