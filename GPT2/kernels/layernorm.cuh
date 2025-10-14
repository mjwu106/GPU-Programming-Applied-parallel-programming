#ifndef __LAYERNORM_KERNEL_CUH__
#define __LAYERNORM_KERNEL_CUH__

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include "../utils/cuda_utils.cuh"

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void layernorm_forward_kernel(float* out, float* mean, float* rstd, const float* inp, const float* weight,
                                         const float* bias, int B, int T, int C) {
    // Implement this
  
    int row = blockIdx.x; 
    
    // Check if this thread is within bounds
    if (row >= B * T) {
        return;
    }

    const int tid = threadIdx.x;
    const int warp_idx = tid >> 5;  // tid / 32
    const int lane = tid & 31;      // tid % 32
    const int num_warps = blockDim.x >> 5;
    
    __shared__ float smem_mean[32];
    __shared__ float smem_variance[32];
    
    const float* input_row = inp + row * C;
    float* output_row = out + row * C;
    
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    for (int col = tid; col < C; col += blockDim.x) {
        float val = input_row[col];
        local_sum += val;
        local_sum_sq += val * val;
    }
    
    float warp_sum = warp_reduce_sum(local_sum);
    float warp_sum_sq = warp_reduce_sum(local_sum_sq);
    
    if (lane == 0) {
        smem_mean[warp_idx] = warp_sum;
        smem_variance[warp_idx] = warp_sum_sq;
    }
    __syncthreads();
    
   
    float total_sum = 0.0f;
    float total_sum_sq = 0.0f;
    
    if (warp_idx == 0) {
        float val1 = (lane < num_warps) ? smem_mean[lane] : 0.0f;
        float val2 = (lane < num_warps) ? smem_variance[lane] : 0.0f;
        
        total_sum = warp_reduce_sum(val1);
        total_sum_sq = warp_reduce_sum(val2);
    }
    
    if (tid == 0) {
        smem_mean[0] = total_sum / C;
        smem_variance[0] = total_sum_sq / C;
    }
    __syncthreads();
    
    float row_mean = smem_mean[0];
    float mean_sq = smem_variance[0];
    
    // Compute standard deviation
    float variance = mean_sq - row_mean * row_mean;
    float inv_std = rsqrtf(variance + 1e-5f);
    
  
    if (tid == 0) {
        mean[row] = row_mean;
        rstd[row] = inv_std;
    }
    
    // Apply normalization and affine transform
    for (int col = tid; col < C; col += blockDim.x) {
        float normalized = (input_row[col] - row_mean) * inv_std;
        output_row[col] = normalized * weight[col] + bias[col];
    }
}

// Launch kernel here
void layernorm_forward(float* out, float* mean, float* rstd, float* inp, float* weight, float* bias,
                       int B, int T, int C) {
    // Implement this
    int num_rows = B * T;

    int block_size = 256;

    // int num_blocks = num_rows;

    // int shared_mem_size = 2 * block_size * sizeof(float);
    // launch kernel
    layernorm_forward_kernel<<<num_rows, block_size>>>(
        out, mean, rstd, inp, weight, bias, B, T, C
    );

    
}

#endif // __LAYERNORM_KERNEL_CUH__