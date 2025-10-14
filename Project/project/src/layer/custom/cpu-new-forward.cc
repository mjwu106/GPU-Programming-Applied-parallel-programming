#include "cpu-new-forward.h"
#include <algorithm> 
#define TILE_SIZE_BATCH 16
#define TILE_SIZE_M 16
#define TILE_SIZE_H 16
#define TILE_SIZE_W 16

void conv_forward_cpu(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
  /*
  Modify this function to implement the forward pass described in Chapter 16.
  The code in 16 is for a single image.
  We have added an additional dimension to the tensors to support an entire mini-batch
  The goal here is to be correct, not fast (this is the CPU implementation.)

  Function paramters:
  output - output
  input - input
  mask - convolution kernel
  Batch - batch_size (number of images in x)
  Map_out - number of output feature maps
  Channel - number of input feature maps
  Height - input height dimension
  Width - input width dimension
  K - kernel height and width (K x K)
  */

  const int Height_out = Height - K + 1;
  const int Width_out = Width - K + 1;

  // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
  // An example use of these macros:
  // float a = in_4d(0,0,0,0)
  // out_4d(0,0,0,0) = a
  
  #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
  #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
  #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

  // Insert your CPU convolution kernel code here
  for (int b_s = 0; b_s < Batch; b_s += TILE_SIZE_BATCH) {
    for (int m_s = 0; m_s < Map_out; m_s += TILE_SIZE_M) {
        for (int h_s = 0; h_s < Height_out; h_s += TILE_SIZE_H) {
            for (int w_s = 0; w_s < Width_out; w_s += TILE_SIZE_W) {
                
                for (int b = b_s; b < std::min(b_s + TILE_SIZE_BATCH, Batch); ++b) {
                    for (int m = m_s; m < std::min(m_s + TILE_SIZE_M, Map_out); ++m) {
                        for (int h = h_s; h < std::min(h_s + TILE_SIZE_H, Height_out); ++h) {
                            for (int w = w_s; w < std::min(w_s + TILE_SIZE_W, Width_out); ++w) {
                                
                                out_4d(b, m, h, w) = 0;
                                
                                for (int c = 0; c < Channel; ++c) {
                                    for (int p = 0; p < K; ++p) {
                                        for (int q = 0; q < K; ++q) {
                                            out_4d(b, m, h, w) += in_4d(b, c, h + p, w + q) * mask_4d(m, c, p, q);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

  #undef out_4d
  #undef in_4d
  #undef mask_4d

}