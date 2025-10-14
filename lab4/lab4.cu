#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define TILE_WIDTH 4
#define Radius 1
#define MASK_WIDTH 3
//@@ Define constant memory for device kernel here
__constant__ float MASK[MASK_WIDTH * MASK_WIDTH * MASK_WIDTH];   //constant memory for the mask
__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  int tx = threadIdx.x; // thread x index within the block
  int ty = threadIdx.y; // thread y index within the block
  int tz = threadIdx.z; // thread z index within the block
  int bx = blockIdx.x * TILE_WIDTH; // thread block x offset within the grid
  int by = blockIdx.y * TILE_WIDTH; // thread block y offset within the grid
  int bz = blockIdx.z * TILE_WIDTH; // thread block z offset within the grid
  int x = bx + tx; // global x index
  int y = by + ty; // global y index
  int z = bz + tz; // global z index

  __shared__ float subtileM[TILE_WIDTH + 2*Radius][TILE_WIDTH + 2*Radius][TILE_WIDTH + 2*Radius];
  for (int dz = -Radius; dz <= Radius; dz++) {
    for (int dy = -Radius; dy <= Radius; dy++) {
      for (int dx = -Radius; dx <= Radius; dx++) {
        int lx = x + dx;
        int ly = y + dy;
        int lz = z + dz;    

        int shared_x = tx + dx + Radius;
        int shared_y = ty + dy + Radius;
        int shared_z = tz + dz + Radius;

        if (lx >= 0 && lx < x_size && 
            ly >= 0 && ly < y_size && 
            lz >= 0 && lz < z_size) {
          subtileM[shared_z][shared_y][shared_x] = 
            input[lz * y_size * x_size + ly * x_size + lx];
        } else {
          subtileM[shared_z][shared_y][shared_x] = 0.0f; // padding with zero
        }   
      }
    }
  }
  __syncthreads();

  float result = 0.0f;
  if (x < x_size && y < y_size && z < z_size) {
    for (int k = 0; k < MASK_WIDTH; k++) {
      for (int j = 0; j < MASK_WIDTH; j++) {
        for (int i = 0; i < MASK_WIDTH; i++) {
          
          result += subtileM[tz + k][ty + j][tx + i] * 
                    MASK[k * MASK_WIDTH * MASK_WIDTH + j * MASK_WIDTH + i];
        }
      }
    }
    output[z * y_size * x_size + y * x_size + x] = result;
  }


}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  //@@ Initial deviceInput and deviceOutput here.
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel = (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  

  int mLen = inputLength - 3;
  int mSize = mLen * sizeof(float);


  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  // host_input += 3;
  cudaMalloc((void**)&deviceInput, x_size*y_size*z_size*sizeof(float));
  cudaMalloc((void**)&deviceOutput, x_size*y_size*z_size*sizeof(float));

  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, hostInput + 3, mSize, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(MASK, hostKernel, MASK_WIDTH*MASK_WIDTH*MASK_WIDTH*sizeof(float));
  dim3 GridDim(ceil((1.0 * x_size) / TILE_WIDTH), 
                ceil((1.0 * y_size) / TILE_WIDTH), 
                ceil((1.0 * z_size) / TILE_WIDTH));
  dim3 BlockDim(TILE_WIDTH, TILE_WIDTH, TILE_WIDTH);


  //@@ Initialize grid and block dimensions here

  //@@ Launch the GPU kernel here
  conv3d<<<GridDim, BlockDim>>>(deviceInput, deviceOutput, z_size, y_size, x_size);

  cudaDeviceSynchronize();



  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
 cudaMemcpy(hostOutput + 3, deviceOutput,
             (inputLength - 3) * sizeof(float), cudaMemcpyDeviceToHost);



  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  //@@ Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  free(hostKernel);
  return 0;
}

