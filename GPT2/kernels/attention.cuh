#ifndef __ATTENTION_CUH__
#define __ATTENTION_CUH__

#include <cuda_runtime.h>
#include <cmath>
#include "../utils/cuda_utils.cuh"

__global__ void permute_kernel(float* q, float* k, float* v, 
                               const float* inp, 
                               int B, int T, int NH, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int C = NH * d;
    
    if (idx < B * T * C) {
        int d_idx = idx % d;
        int t_idx = (idx / d) % T;
        int nh_idx = (idx / (d * T)) % NH;
        int b_idx = idx / (d * T * NH);
        
        int c_idx = nh_idx * d + d_idx;
        int inp_base = b_idx * T * 3 * C + t_idx * 3 * C;
        
        int out_idx = b_idx * NH * T * d + nh_idx * T * d + t_idx * d + d_idx;
        
        q[out_idx] = inp[inp_base + c_idx];
        k[out_idx] = inp[inp_base + C + c_idx];
        v[out_idx] = inp[inp_base + 2 * C + c_idx];
    }
}

__global__ void unpermute_kernel(float* out, const float* inp, 
                                 int B, int T, int NH, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int C = NH * d;
    
    if (idx < B * T * C) {
        int c_idx = idx % C;
        int t_idx = (idx / C) % T;
        int b_idx = idx / (C * T);
        
        int d_idx = c_idx % d;
        int nh_idx = c_idx / d;
        int inp_idx = b_idx * NH * T * d + nh_idx * T * d + t_idx * d + d_idx;
        
        out[idx] = inp[inp_idx];
    }
}

__global__ void softmax_kernel(float* att, int B, int NH, int T) {
    int row = blockIdx.x;  
    
    if (row < B * NH * T) {
        float* row_ptr = att + row * T;
        
        float max_val = -INFINITY;
        for (int i = 0; i < T; i++) {
            if (row_ptr[i] > max_val) {
                max_val = row_ptr[i];
            }
        }
        
        float sum = 0.0f;
        for (int i = 0; i < T; i++) {
            row_ptr[i] = expf(row_ptr[i] - max_val);
            sum += row_ptr[i];
        }
        
        float inv_sum = 1.0f / (sum + 1e-12f);  
        for (int i = 0; i < T; i++) {
            row_ptr[i] *= inv_sum;
        }
    }
}

__global__ void qk_matmul_kernel(float* att, const float* q, const float* k,
                                 int B, int NH, int T, int d, float scale) {
    int t_k = blockIdx.x * blockDim.x + threadIdx.x;
    int t_q = blockIdx.y * blockDim.y + threadIdx.y;
    int b_nh = blockIdx.z;
    
    if (t_q < T && t_k < T && b_nh < B * NH) {
        int b = b_nh / NH;
        int nh = b_nh % NH;
        
        if (t_k > t_q) {
            int att_idx = b * NH * T * T + nh * T * T + t_q * T + t_k;
            att[att_idx] = -INFINITY;
        } else {
            float dot = 0.0f;
            for (int i = 0; i < d; i++) {
                int q_idx = b * NH * T * d + nh * T * d + t_q * d + i;
                int k_idx = b * NH * T * d + nh * T * d + t_k * d + i;
                dot += q[q_idx] * k[k_idx];
            }
            
            int att_idx = b * NH * T * T + nh * T * T + t_q * T + t_k;
            att[att_idx] = dot * scale;
        }
    }
}

__global__ void attv_matmul_kernel(float* out, const float* att, const float* v,
                                   int B, int NH, int T, int d) {
    int d_out = blockIdx.x * blockDim.x + threadIdx.x;
    int t_out = blockIdx.y * blockDim.y + threadIdx.y;
    int b_nh = blockIdx.z;
    
    if (t_out < T && d_out < d && b_nh < B * NH) {
        int b = b_nh / NH;
        int nh = b_nh % NH;
        
        float sum = 0.0f;
        for (int t_v = 0; t_v < T; t_v++) {
            int att_idx = b * NH * T * T + nh * T * T + t_out * T + t_v;
            int v_idx = b * NH * T * d + nh * T * d + t_v * d + d_out;
            sum += att[att_idx] * v[v_idx];
        }
        
        int out_idx = b * NH * T * d + nh * T * d + t_out * d + d_out;
        out[out_idx] = sum;
    }
}

void attention_forward(float* out, float* qkvr, float* att, 
                       float* inp, int B, int T, int C, int NH) {
    int d = C / NH;
    
    float* q = qkvr;
    float* k = qkvr + B * NH * T * d;
    float* v = qkvr + 2 * B * NH * T * d;
    
    int threads = 256;
    
    int total = B * NH * T * d;
    int blocks = (total + threads - 1) / threads;
    permute_kernel<<<blocks, threads>>>(q, k, v, inp, B, T, NH, d);
    cudaDeviceSynchronize();
    
    float scale = 1.0f / sqrtf((float)d);
    dim3 block_qk(16, 16);
    dim3 grid_qk((T + 15) / 16, (T + 15) / 16, B * NH);
    qk_matmul_kernel<<<grid_qk, block_qk>>>(att, q, k, B, NH, T, d, scale);
    cudaDeviceSynchronize();
    
    blocks = B * NH * T;
    softmax_kernel<<<blocks, 1>>>(att, B, NH, T);
    cudaDeviceSynchronize();
    
    dim3 block_av(16, 16);
    dim3 grid_av((d + 15) / 16, (T + 15) / 16, B * NH);
    attv_matmul_kernel<<<grid_av, block_av>>>(q, att, v, B, NH, T, d);
    cudaDeviceSynchronize();
    
    blocks = (B * T * C + threads - 1) / threads;
    unpermute_kernel<<<blocks, threads>>>(out, q, B, T, NH, d);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Attention error: %s\n", cudaGetErrorString(err));
    }
}

#endif // __ATTENTION_CUH__